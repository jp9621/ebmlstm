import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

from slinglstm import EventAugmentedLSTM
from plots import (
    plot_pointer_trajectory,
    plot_event_write_overlay,
    plot_slot_recency_heatmap,
    plot_h_mem_heatmap,
)

T            = 100
HIDDEN       = 16
OUT          = 1
MEM_SLOTS    = 3
P_PRIMARY    = 0.05
P_DISTRACTOR = 0.10
N_SEQS       = 1000
BS           = 64
LR           = 1e-3
EPOCHS       = 10
DEVICE       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def make_complex_dataset(n,
                         baseline_mu=1.0, baseline_sigma=0.2,
                         event_rate=0.2,
                         spike_scale=10.0,
                         label_margin=1.2):
    """
    n           : # sequences
    T           : seq length
    event_rate  : prob of a 'big' event on any step (~Poisson)
    spike_scale : scale parameter for exponential spikes
    """
    X = np.random.normal(baseline_mu, baseline_sigma, size=(n, T))
    is_event = np.random.rand(n, T) < event_rate
    is_event[:, :3] = True
    spikes = np.random.exponential(scale=spike_scale, size=(n, T))
    X[is_event] = spikes[is_event]
    ys = []
    for seq in X:
        idx = np.where(seq > baseline_mu + 3 * baseline_sigma)[0]
        if len(idx) < 3:
            ys.append(0.0)
            continue
        i1, i2, i3 = idx[-3], idx[-2], idx[-1]
        a1, a2, a3 = seq[i1], seq[i2], seq[i3]
        d1, d2 = i2 - i1, i3 - i2
        s1 = seq[i1+1:i2].sum()
        s2 = seq[i2+1:i3].sum()
        cond = (a1 < a2 < a3) and (d2 > d1) and (s2 > label_margin * s1)
        ys.append(1.0 if cond else 0.0)

    Xt = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)
    yt = torch.tensor(ys, dtype=torch.float32).unsqueeze(-1)
    return Xt, yt


def visualize_memory_dynamics(model, X, device):
    model.eval()
    seq = X.permute(1, 0, 2).to(device)
    T, B, _ = seq.size()

    h_lstm, c_lstm, h_mem, slots, ptr = \
        model.cell.init_state(batch_size=1, device=device)
    mem = model.cell.mem_cell

    ptr_history = []
    e_history = []
    slot_ages = []
    h_mem_history = []
    n_slots = mem.n_slots
    last_write = [-1] * n_slots

    for t in range(T):
        x_t = seq[t]
        e_t = torch.sigmoid(mem.event_detector(x_t))
        e_val = e_t.item()
        e_history.append(e_val)

        ptr_history.append(ptr.item())

        if e_val > mem.tau:
            last_write[ptr.item()] = t

        ages = [(t - lw) if lw >= 0 else (t + 1) for lw in last_write]
        slot_ages.append(ages)

        h_new, (h_lstm, c_lstm, h_mem, slots, ptr) = \
            model.cell(x_t, (h_lstm, c_lstm, h_mem, slots, ptr))
        h_mem_history.append(h_mem.squeeze(0).detach().cpu().numpy())

    plot_pointer_trajectory(ptr_history)
    plot_event_write_overlay(e_history, mem.tau)
    plot_slot_recency_heatmap(slot_ages)
    plot_h_mem_heatmap(h_mem_history)


def main():
    X, y = make_complex_dataset(N_SEQS)
    ds = TensorDataset(X, y)
    loader = DataLoader(ds, batch_size=BS, shuffle=True)

    model = EventAugmentedLSTM(
        input_dim=1,
        mem_slots=MEM_SLOTS,
        hidden_dim=HIDDEN,
        out_dim=OUT
    ).to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=LR)
    crit = nn.BCEWithLogitsLoss()

    for ep in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            xb = xb.permute(1, 0, 2)
            logits = model(xb)[-1]
            loss = crit(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * xb.size(1)
        print(f"[Hybrid] Epoch {ep:2d} | Loss: {total_loss / N_SEQS:.4f}")

    model.eval()
    with torch.no_grad():
        xb, yb = X[:1000], y[:1000]
        xb = xb.permute(1, 0, 2).to(DEVICE)
        yb = yb.to(DEVICE)
        pred = torch.sigmoid(model(xb)[-1]).round()
        acc = (pred.cpu() == yb.cpu()).float().mean().item()
        print(f"[Hybrid] Eval Acc: {acc * 100:.2f}%")

    X_vis, _ = make_complex_dataset(1)
    visualize_memory_dynamics(model, X_vis, DEVICE)


if __name__ == "__main__":
    main()
