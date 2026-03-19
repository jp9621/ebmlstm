"""
Metrics collection for the EBM-LSTM paper.
Compares EBM-LSTM against Vanilla LSTM variants across:

  1. Core comparison  — accuracy, precision, recall, F1, latency, params
  2. Latency scaling  — how latency grows with sequence length

Task
----
Binary classification. A marker event (moderate-amplitude spike in a fixed
direction) is injected at a UNIFORMLY RANDOM position anywhere in [0, T).
Neither model has a positional structural advantage: the full LSTM must
propagate the signal through sequential state; the EBM-LSTM must detect it
via attention when it falls in the buffer or via LSTM when it falls in the
recent window.

Amplitude is kept moderate (SNR ≈ 4) so models must actually learn rather
than trivially threshold a single feature.

All results are written to saved/metrics.json.
"""

import os
import json
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

SAVE_DIR = "saved"
os.makedirs(SAVE_DIR, exist_ok=True)

# ── Config ───────────────────────────────────────────────────────────────────
SEQ_LEN     = 100
LSTM_WIN    = 20
BUFFER_LEN  = SEQ_LEN - LSTM_WIN   # 80
INPUT_DIM   = 16
HIDDEN      = 64
N_SLOTS     = 10
TAU         = 0.5
MARKER_PROB = 0.5
N_TRAIN     = 6000
N_TEST      = 1500
EPOCHS      = 30
LR          = 1e-3
BATCH       = 128
N_LAT_ITERS = 500
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"


# ── Models ───────────────────────────────────────────────────────────────────
class VanillaLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, window, out_dim=2):
        super().__init__()
        self.window = window
        self.lstm   = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.head   = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):              # x: (B, T, D)
        _, (h, _) = self.lstm(x[:, -self.window:, :])
        return self.head(h.squeeze(0))


class EBM_LSTM(nn.Module):
    """
    Phase 1 — vectorised attention over distant buffer steps (no loop).
    Phase 2 — fused LSTM over recent window, initialised with buffer summary.
    """
    def __init__(self, input_dim, hidden_dim, n_slots, buffer_len, out_dim=2, tau=0.5):
        super().__init__()
        self.buffer_len     = buffer_len
        self.event_detector = nn.Linear(input_dim, 1)
        self.value_proj     = nn.Linear(input_dim, input_dim)
        self.summary_proj   = nn.Linear(input_dim, hidden_dim)
        self.lstm           = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.head           = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):              # x: (B, T, D)
        buf     = x[:, :self.buffer_len, :]
        weights = torch.softmax(self.event_detector(buf), dim=1)   # (B, buf, 1)
        summary = (weights * self.value_proj(buf)).sum(dim=1)      # (B, D)
        h0      = self.summary_proj(summary).unsqueeze(0)          # (1, B, H)
        c0      = torch.zeros_like(h0)
        _, (h, _) = self.lstm(x[:, self.buffer_len:, :], (h0, c0))
        return self.head(h.squeeze(0))


# ── Dataset ──────────────────────────────────────────────────────────────────
def make_dataset(n, seq_len, buffer_len, input_dim, marker_prob=0.5, seed=None):
    """
    Marker position is uniformly random in [0, seq_len) — NOT constrained to
    the buffer. Amplitude is moderate (2.0) against background noise (std 0.5),
    giving SNR ≈ 4 so both models must actually learn to detect the event.
    """
    if seed is not None:
        torch.manual_seed(seed)
    X    = torch.randn(n, seq_len, input_dim) * 0.5
    y    = torch.zeros(n, dtype=torch.long)
    mask = torch.rand(n) < marker_prob
    pos  = torch.randint(0, seq_len, (n,))     # anywhere in the full sequence
    spike = torch.zeros(input_dim)
    spike[0] = 2.0                              # fixed direction, moderate amplitude
    for i in range(n):
        if mask[i]:
            X[i, pos[i]] += spike
            y[i] = 1
    return X, y


# ── Utilities ────────────────────────────────────────────────────────────────
def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def measure_latency(model, x, n_warmup=50, n_iters=N_LAT_ITERS, device=DEVICE):
    model.eval()
    x = x.to(device)
    lats = []
    with torch.no_grad():
        for _ in range(n_warmup):
            model(x)
        if device == "cuda":
            torch.cuda.synchronize()
        if device == "cuda":
            for _ in range(n_iters):
                e0 = torch.cuda.Event(enable_timing=True)
                e1 = torch.cuda.Event(enable_timing=True)
                e0.record(); model(x); e1.record()
                torch.cuda.synchronize()
                lats.append(e0.elapsed_time(e1) * 1000)  # ms → µs
        else:
            for _ in range(n_iters):
                t0 = time.perf_counter(); model(x); t1 = time.perf_counter()
                lats.append((t1 - t0) * 1e6)
    return lats


def latency_stats(lats, store_all=False):
    sv = sorted(lats); n = len(sv)
    mean = sum(sv) / n
    out  = {
        "mean": mean,
        "std":  (sum((v - mean)**2 for v in sv) / n) ** 0.5,
        "min":  sv[0],
        "p50":  sv[int(n * 0.50)],
        "p95":  sv[min(int(n * 0.95), n-1)],
        "p99":  sv[min(int(n * 0.99), n-1)],
        "max":  sv[-1],
    }
    if store_all:
        out["all"] = sv
    return out


def train_and_eval(model, train_loader, test_loader, epochs, lr, device):
    """Returns (final_accuracy, curves_dict)."""
    model.to(device)
    opt     = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    curves  = {"train_loss": [], "train_acc": [], "val_acc": []}

    for _ in range(epochs):
        model.train()
        tot_loss = tot_correct = tot_n = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss   = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            tot_loss    += loss.item() * xb.size(0)
            tot_correct += (logits.argmax(1) == yb).sum().item()
            tot_n       += xb.size(0)

        curves["train_loss"].append(tot_loss / tot_n)
        curves["train_acc"].append(tot_correct / tot_n)

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(device), yb.to(device)
                correct += (model(xb).argmax(1) == yb).sum().item()
                total   += yb.size(0)
        curves["val_acc"].append(correct / total)

    return correct / total, curves


def classification_metrics(model, test_loader, device):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            preds.extend(model(xb).argmax(1).cpu().tolist())
            labels.extend(yb.cpu().tolist())
    tp = sum(p == 1 and l == 1 for p, l in zip(preds, labels))
    fp = sum(p == 1 and l == 0 for p, l in zip(preds, labels))
    fn = sum(p == 0 and l == 1 for p, l in zip(preds, labels))
    tn = sum(p == 0 and l == 0 for p, l in zip(preds, labels))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec  = tp / (tp + fn) if tp + fn else 0.0
    f1   = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {
        "accuracy":  (tp + tn) / len(preds),
        "precision": prec,
        "recall":    rec,
        "f1":        f1,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def to_json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_json_safe(v) for v in obj]
    if isinstance(obj, (bool, int, float, str)):
        return obj
    return float(obj)


# ══════════════════════════════════════════════════════════════════════════════
# 1. CORE COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
def run_core_comparison():
    print("\n" + "="*60)
    print("1. CORE COMPARISON (accuracy, latency, params)")
    print("="*60)

    torch.manual_seed(42)
    X_tr, y_tr = make_dataset(N_TRAIN, SEQ_LEN, BUFFER_LEN, INPUT_DIM, MARKER_PROB)
    X_te, y_te = make_dataset(N_TEST,  SEQ_LEN, BUFFER_LEN, INPUT_DIM, MARKER_PROB)
    train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=BATCH, shuffle=True)
    test_loader  = DataLoader(TensorDataset(X_te, y_te), batch_size=BATCH)
    x_sample     = X_te[:BATCH]

    specs = [
        ("VanillaLSTM_full",  VanillaLSTM(INPUT_DIM, HIDDEN, window=SEQ_LEN)),
        ("VanillaLSTM_short", VanillaLSTM(INPUT_DIM, HIDDEN, window=LSTM_WIN)),
        ("EBM_LSTM",          EBM_LSTM(INPUT_DIM, HIDDEN, N_SLOTS, BUFFER_LEN)),
    ]

    core, curves = {}, {}
    for name, model in specs:
        print(f"\n  Training {name} ...")
        acc, crv = train_and_eval(model, train_loader, test_loader, EPOCHS, LR, DEVICE)
        model.to(DEVICE)
        clf    = classification_metrics(model, test_loader, DEVICE)
        lats   = measure_latency(model, x_sample, device=DEVICE)
        lstats = latency_stats(lats, store_all=True)
        params = count_params(model)

        core[name]   = {**clf, "latency": lstats, "params": params}
        curves[name] = crv
        print(f"    acc={acc:.4f}  lat_mean={lstats['mean']:.1f}µs  params={params:,}")

    fl = core["VanillaLSTM_full"]["latency"]["mean"]
    el = core["EBM_LSTM"]["latency"]["mean"]
    core["speedup_full_vs_ebm"] = fl / el
    print(f"\n  Speedup (full→EBM): {fl/el:.2f}×")

    return core, curves


# ══════════════════════════════════════════════════════════════════════════════
# 2. LATENCY SCALING vs SEQUENCE LENGTH
# ══════════════════════════════════════════════════════════════════════════════
def run_latency_scaling_seq():
    print("\n" + "="*60)
    print("2. LATENCY SCALING vs SEQUENCE LENGTH")
    print("="*60)

    seq_lens = [25, 50, 100, 200, 500, 1000, 2000]
    results  = {}

    for T in seq_lens:
        print(f"  T={T}", end="", flush=True)
        buf = max(1, int(T * 0.8))
        win = T - buf

        models = {
            "VanillaLSTM_full":  VanillaLSTM(INPUT_DIM, HIDDEN, window=T).to(DEVICE),
            "VanillaLSTM_short": VanillaLSTM(INPUT_DIM, HIDDEN, window=win).to(DEVICE),
            "EBM_LSTM":          EBM_LSTM(INPUT_DIM, HIDDEN, N_SLOTS, buf).to(DEVICE),
        }
        x = torch.randn(BATCH, T, INPUT_DIM, device=DEVICE)
        entry = {}
        for name, m in models.items():
            lats = measure_latency(m, x, n_iters=300, device=DEVICE)
            entry[name] = latency_stats(lats)
            print(".", end="", flush=True)

        fl = entry["VanillaLSTM_full"]["mean"]
        el = entry["EBM_LSTM"]["mean"]
        entry["speedup_full_vs_ebm"] = fl / el
        results[str(T)] = entry
        print(f"  speedup={fl/el:.2f}×")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print(f"Device : {DEVICE}")
    print(f"Seq    : {SEQ_LEN}  |  Buffer: {BUFFER_LEN}  |  LSTM win: {LSTM_WIN}")
    print(f"Slots  : {N_SLOTS}  |  Tau: {TAU}")
    print(f"Train  : {N_TRAIN}  |  Test: {N_TEST}  |  Epochs: {EPOCHS}")
    print(f"Task   : marker anywhere in [0,{SEQ_LEN})  |  amplitude=2.0  |  noise_std=0.5")

    all_metrics = {}

    core, curves = run_core_comparison()
    all_metrics["core_comparison"] = core
    all_metrics["training_curves"] = curves
    all_metrics["latency_scaling_seq"] = run_latency_scaling_seq()

    all_metrics["config"] = {
        "SEQ_LEN": SEQ_LEN, "LSTM_WIN": LSTM_WIN, "BUFFER_LEN": BUFFER_LEN,
        "INPUT_DIM": INPUT_DIM, "HIDDEN": HIDDEN, "N_SLOTS": N_SLOTS,
        "TAU": TAU, "N_TRAIN": N_TRAIN, "N_TEST": N_TEST,
        "EPOCHS": EPOCHS, "LR": LR, "BATCH": BATCH, "DEVICE": DEVICE,
        "task": {
            "marker_position": "uniform random in [0, seq_len)",
            "spike_amplitude": 2.0,
            "noise_std": 0.5,
            "snr_approx": 4.0,
        },
    }

    out = os.path.join(SAVE_DIR, "metrics.json")
    with open(out, "w") as f:
        json.dump(to_json_safe(all_metrics), f, indent=2)

    print(f"\n{'='*60}")
    print(f"All metrics saved → {out}")
