"""
Accuracy vs Efficiency: EBM-LSTM vs Vanilla LSTM
=================================================

Task
----
Binary classification on sequences of length SEQ_LEN=100.
A rare "marker" event is injected somewhere in the DISTANT PAST
(steps 0..BUFFER_LEN-1).  Steps [BUFFER_LEN..SEQ_LEN-1] are pure noise.
Label = 1 iff a marker was injected.

Models
------
1. VanillaLSTM (full window=100)  — sees the distant past, expensive
2. VanillaLSTM (short window=20)  — cheap, but BLIND to the distant past
3. EBM-LSTM    (buf=80, lstm=20)  — Phase-1: cheap event-detection-only pass
                                     over distant 80 steps fills event slots;
                                     Phase-2: full LSTM cell on recent 20 steps.

Metrics
-------
  - Test accuracy
  - Inference latency (µs / batch)
  - Trainable parameter count
"""

import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt


# ── Config ─────────────────────────────────────────────────────────────────────
SEQ_LEN     = 100
LSTM_WIN    = 20
BUFFER_LEN  = SEQ_LEN - LSTM_WIN   # 80 — distant-past range

INPUT_DIM   = 16
HIDDEN      = 64
N_SLOTS     = 10
TAU         = 0.5

MARKER_PROB = 0.5       # P(label=1) — balanced classes
N_TRAIN     = 6000
N_TEST      = 1500
EPOCHS      = 25
LR          = 1e-3
BATCH       = 128
N_LAT_ITERS = 300
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"


# ── Dataset ────────────────────────────────────────────────────────────────────
def make_dataset(n, seq_len, buffer_len, input_dim, marker_prob):
    """
    Returns X (n, seq_len, input_dim) and y (n,) long.
    When label=1, a strong spike (+5 * unit vector) is added to a random
    position in [0, buffer_len-1] so the event detector reliably fires.
    """
    X = torch.randn(n, seq_len, input_dim) * 0.3
    y = torch.zeros(n, dtype=torch.long)
    mask = torch.rand(n) < marker_prob
    positions = torch.randint(0, buffer_len, (n,))
    spike = torch.zeros(input_dim)
    spike[0] = 5.0          # fixed direction so detector can learn it
    for i in range(n):
        if mask[i]:
            X[i, positions[i]] += spike
            y[i] = 1
    return X, y


# ── Models ─────────────────────────────────────────────────────────────────────
class VanillaLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, window, out_dim=2):
        super().__init__()
        self.window = window
        self.lstm   = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.head   = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):                       # x: (B, T, D)
        x_win = x[:, -self.window:, :]
        _, (h, _) = self.lstm(x_win)
        return self.head(h.squeeze(0))


class EBM_LSTM(nn.Module):
    """
    Phase 1 — buffer summary (steps 0..buffer_len-1):
        Fully vectorized: learned attention over all buffer steps in one pass.
        event_detector scores each step → softmax weights → weighted sum of
        value projections → single summary vector.  No Python loop, no
        sequential state — just two matrix multiplies.

    Phase 2 — LSTM (steps buffer_len..seq_len-1):
        nn.LSTM (fused C++ kernel) over the short recent window, initialised
        with the buffer summary projected into h0.
    """
    def __init__(self, input_dim, hidden_dim, n_slots, buffer_len, out_dim=2, tau=0.5):
        super().__init__()
        self.buffer_len     = buffer_len
        self.event_detector = nn.Linear(input_dim, 1)          # attention scores
        self.value_proj     = nn.Linear(input_dim, input_dim)  # value transform
        self.summary_proj   = nn.Linear(input_dim, hidden_dim) # → h0
        self.lstm           = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.head           = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):                       # x: (B, T, D)
        # Phase 1 — vectorised buffer summary
        buf     = x[:, :self.buffer_len, :]                    # (B, buf, D)
        scores  = self.event_detector(buf)                     # (B, buf, 1)
        weights = torch.softmax(scores, dim=1)                 # (B, buf, 1)
        summary = (weights * self.value_proj(buf)).sum(dim=1)  # (B, D)
        h0      = self.summary_proj(summary).unsqueeze(0)      # (1, B, H)
        c0      = torch.zeros_like(h0)

        # Phase 2 — fused LSTM over recent window
        win        = x[:, self.buffer_len:, :]                 # (B, win, D)
        _, (h, _)  = self.lstm(win, (h0, c0))
        return self.head(h.squeeze(0))


# ── Training & evaluation ───────────────────────────────────────────────────────
def train_and_eval(model, train_loader, test_loader, epochs, lr, device):
    model.to(device)
    opt     = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            correct += (model(xb).argmax(1) == yb).sum().item()
            total   += yb.size(0)
    return correct / total


def measure_latency(model, x_sample, n_warmup=50, n_iters=N_LAT_ITERS, device=DEVICE):
    model.eval()
    x_sample = x_sample.to(device)
    with torch.no_grad():
        for _ in range(n_warmup):
            model(x_sample)
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_iters):
            model(x_sample)
        if device == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
    return (t1 - t0) / n_iters * 1e6   # µs / batch


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ── Plotting ───────────────────────────────────────────────────────────────────
def plot_results(results, save_dir="figures"):
    os.makedirs(save_dir, exist_ok=True)
    names   = list(results.keys())
    accs    = [results[n]["accuracy"]  for n in names]
    lats    = [results[n]["latency"]   for n in names]
    params  = [results[n]["params"]    for n in names]

    colors  = ["#2196F3", "#F44336", "#4CAF50"]     # blue, red, green
    x       = range(len(names))
    short_names = [
        f"LSTM\n(full win={SEQ_LEN})",
        f"LSTM\n(short win={LSTM_WIN})",
        f"EBM-LSTM\n(buf={BUFFER_LEN}, lstm={LSTM_WIN})",
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle(
        f"EBM-LSTM vs Vanilla LSTM  |  seq={SEQ_LEN}, "
        f"buf={BUFFER_LEN}, lstm_win={LSTM_WIN}, slots={N_SLOTS}",
        fontsize=12,
    )

    # Accuracy
    ax = axes[0]
    bars = ax.bar(x, accs, color=colors, edgecolor="black", width=0.5)
    ax.set_xticks(x); ax.set_xticklabels(short_names, fontsize=9)
    ax.set_ylim(0, 1.05); ax.set_ylabel("Test Accuracy"); ax.set_title("Accuracy")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, label="random baseline")
    ax.legend(fontsize=8)
    for bar, v in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01, f"{v:.3f}",
                ha="center", va="bottom", fontsize=9)

    # Latency
    ax = axes[1]
    bars = ax.bar(x, lats, color=colors, edgecolor="black", width=0.5)
    ax.set_xticks(x); ax.set_xticklabels(short_names, fontsize=9)
    ax.set_ylabel("Latency (µs / batch)"); ax.set_title("Inference Latency")
    for bar, v in zip(bars, lats):
        ax.text(bar.get_x() + bar.get_width() / 2, v + max(lats) * 0.01,
                f"{v:.0f}", ha="center", va="bottom", fontsize=9)

    # Accuracy vs Latency scatter (the tradeoff view)
    ax = axes[2]
    for i, (name, c) in enumerate(zip(short_names, colors)):
        ax.scatter(lats[i], accs[i], color=c, s=120, zorder=3, label=name)
    ax.set_xlabel("Latency (µs / batch)"); ax.set_ylabel("Test Accuracy")
    ax.set_title("Accuracy–Efficiency Tradeoff")
    ax.set_ylim(0.3, 1.05)
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "accuracy_efficiency.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"\nPlot saved → {path}")


# ── Main ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    torch.manual_seed(42)
    print(f"Device      : {DEVICE}")
    print(f"Sequence    : {SEQ_LEN} steps  |  Buffer: {BUFFER_LEN}  |  LSTM win: {LSTM_WIN}")
    print(f"EBM slots   : {N_SLOTS}  |  tau: {TAU}")
    print(f"Marker prob : {MARKER_PROB}  |  N_train: {N_TRAIN}  |  N_test: {N_TEST}\n")

    X_tr, y_tr = make_dataset(N_TRAIN, SEQ_LEN, BUFFER_LEN, INPUT_DIM, MARKER_PROB)
    X_te, y_te = make_dataset(N_TEST,  SEQ_LEN, BUFFER_LEN, INPUT_DIM, MARKER_PROB)
    train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=BATCH, shuffle=True)
    test_loader  = DataLoader(TensorDataset(X_te, y_te), batch_size=BATCH)
    x_sample = X_te[:BATCH]

    model_specs = {
        f"VanillaLSTM(win={SEQ_LEN})": VanillaLSTM(INPUT_DIM, HIDDEN, window=SEQ_LEN),
        f"VanillaLSTM(win={LSTM_WIN})": VanillaLSTM(INPUT_DIM, HIDDEN, window=LSTM_WIN),
        f"EBM-LSTM(buf={BUFFER_LEN},lstm={LSTM_WIN},slots={N_SLOTS})":
            EBM_LSTM(INPUT_DIM, HIDDEN, N_SLOTS, BUFFER_LEN, tau=TAU),
    }

    results = {}
    for name, model in model_specs.items():
        print(f"Training: {name} ...")
        acc    = train_and_eval(model, train_loader, test_loader, EPOCHS, LR, DEVICE)
        lat    = measure_latency(model, x_sample, device=DEVICE)
        params = count_params(model)
        results[name] = {"accuracy": acc, "latency": lat, "params": params}
        print(f"  acc={acc:.3f}  latency={lat:6.1f} µs  params={params:,}")

    print("\n" + "=" * 70)
    print(f"{'Model':<48} {'Accuracy':>8} {'Latency µs':>11} {'Params':>9}")
    print("-" * 70)
    for name, r in results.items():
        print(f"{name:<48} {r['accuracy']:>8.3f} {r['latency']:>11.1f} {r['params']:>9,}")

    full_lat = results[f"VanillaLSTM(win={SEQ_LEN})"]["latency"]
    ebm_lat  = results[f"EBM-LSTM(buf={BUFFER_LEN},lstm={LSTM_WIN},slots={N_SLOTS})"]["latency"]
    print(f"\nSpeedup (full LSTM → EBM-LSTM): {full_lat / ebm_lat:.2f}×")

    plot_results(results)
