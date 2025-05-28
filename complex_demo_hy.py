import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

from slinglstm import EventAugmentedLSTM
from plots import (
    plot_slot_activation_heatmap,
    plot_attention_line,
    plot_attention_stack,
    plot_commit_scatter,
    plot_delta_t_evolution,
    plot_slot_delta_heatmap
)

# --- hyperparameters ---
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
    n        : # sequences
    T        : seq length
    event_rate : prob of a 'big' event on any step (~Poisson)
    spike_scale: scale parameter for exponential spikes
    """
    # 1) baseline noise around ~1.0
    X = np.random.normal(baseline_mu, baseline_sigma, size=(n, T))
        # 2) draw event mask
    is_event = np.random.rand(n, T) < event_rate

    # # ──> force the first 3 bars of every sequence to be events
    # is_event[:, :3] = True

    # 3) heavy-tailed spikes
    spikes   = np.random.exponential(scale=spike_scale, size=(n, T))
    X[is_event] = spikes[is_event]

    # 4) labels: look at last 3 events and compare gaps & sums
    ys = []
    for seq in X:
        idx = np.where(seq > baseline_mu + 3 * baseline_sigma)[0]
        if len(idx) < 3:
            ys.append(0.0)
            continue
        i1, i2, i3 = idx[-3], idx[-2], idx[-1]
        a1, a2, a3 = seq[i1], seq[i2], seq[i3]
        d1, d2 = i2 - i1, i3 - i2
        s1 = seq[i1 + 1:i2].sum()
        s2 = seq[i2 + 1:i3].sum()
        cond = (a1 < a2 < a3) and (d2 > d1) and (s2 > label_margin * s1)
        ys.append(1.0 if cond else 0.0)

    Xt = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)  # (n,T,1)
    yt = torch.tensor(ys, dtype=torch.float32).unsqueeze(-1)
    return Xt, yt


def visualize_memory_dynamics(model, X, device):
    """
    Runs one sequence through the trained model to extract and plot:
      - slot activations
      - attention weights
      - commit strengths
      - delta_t evolution
    """
    model.eval()
    seq = X.permute(1, 0, 2).to(device)  # (T, 1, 1)
    T, B, _ = seq.size()

    # --- DECOMPRESS INITIAL STATE (now 7 values: +filled) ---
    h_lstm, c_lstm, h_mem, slots, cum_feats, delta_t, filled = \
        model.cell.init_state(batch_size=1, device=device)

    # buffers for plotting
    slot_vals, deltas, alphas, gammas = [], [], [], []

    for t in range(T):
        x_t = seq[t]  # (1, 1)

        # recompute α and g for plotting
        em = model.cell.event_cell
        delta_t = delta_t + 1
        q    = em.query(x_t)
        keys = em.key(slots)
        sims = (keys * q.unsqueeze(1)).sum(dim=-1)
        alpha = torch.softmax(sims, dim=1)
        max_sim, _ = sims.max(dim=1)
        g = torch.sigmoid(max_sim.unsqueeze(1))

        # --- STEP THROUGH THE CELL (now pass & return 'filled') ---
        h_new, full_state = model.cell(
            x_t,
            (h_lstm, c_lstm, h_mem, slots, cum_feats, delta_t, filled)
        )
        h_lstm, c_lstm, h_mem, slots, cum_feats, delta_t, filled = full_state

        # record for plots
        alphas.append(alpha.detach().cpu().numpy().squeeze())
        gammas.append(g.detach().cpu().numpy().squeeze())
        slot_vals.append(slots.detach().cpu().numpy().squeeze())
        deltas.append(delta_t.detach().cpu().numpy().squeeze())

    # stack and visualize
    slot_vals = np.stack(slot_vals)    # (T, n_slots, feature_dim)
    deltas     = np.stack(deltas)      # (T, n_slots)
    alphas     = np.stack(alphas)      # (T, n_slots)
    gammas     = np.array(gammas)      # (T,)
    delta_vals = slot_vals[1:] - slot_vals[:-1]   # shape (T-1, n_slots)

    plot_slot_activation_heatmap(slot_vals)
    plot_attention_line        (alphas)
    plot_attention_stack       (alphas)
    plot_commit_scatter        (gammas, event_mask=None)
    plot_delta_t_evolution     (deltas)
    plot_slot_delta_heatmap(delta_vals)


def main():
    # prepare data
    X, y = make_complex_dataset(N_SEQS)
    ds   = TensorDataset(X, y)
    loader = DataLoader(ds, batch_size=BS, shuffle=True)

    # initialize model
    model = EventAugmentedLSTM(
        input_dim=1,
        mem_slots=MEM_SLOTS,
        hidden_dim=HIDDEN,
        out_dim=OUT
    ).to(DEVICE)
    opt  = optim.Adam(model.parameters(), lr=LR)
    crit = nn.BCEWithLogitsLoss()

    # training loop
    for ep in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0

        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            xb = xb.permute(1, 0, 2)               # (T, B, 1)
            logits = model(xb)[-1]                # final timestep
            loss = crit(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * xb.size(1)

        print(f"[Hybrid] Epoch {ep:2d} | Loss: {total_loss / N_SEQS:.4f}")

    # evaluation
    model.eval()
    with torch.no_grad():
        xb, yb = X[:1000], y[:1000]
        xb = xb.permute(1, 0, 2).to(DEVICE)
        yb = yb.to(DEVICE)
        pred = torch.sigmoid(model(xb)[-1]).round()
        acc  = (pred.cpu() == yb.cpu()).float().mean().item()
        print(f"[Hybrid] Eval Acc: {acc * 100:.2f}%")

    # visualize internals
    X_vis, _ = make_complex_dataset(1)
    visualize_memory_dynamics(model, X_vis, DEVICE)


if __name__ == "__main__":
    main()
