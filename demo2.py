import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

# --- same hyperparams as before, plus memory slots ---
T            = 100
HIDDEN       = 16
OUT          = 1
MEM_SLOTS    = 5    # number of slots in your EventMemoryCell
P_PRIMARY    = 0.05
P_DISTRACTOR = 0.10
N_SEQS       = 5000
BS           = 64
LR           = 1e-3
EPOCHS       = 10
DEVICE       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def make_complex_dataset(n):
    X = np.random.rand(n, T) * 0.5
    is_primary = np.random.rand(n, T) < P_PRIMARY
    is_distr  = (np.random.rand(n, T) < P_DISTRACTOR) & ~is_primary

    X[is_primary] = 1.0 + np.random.rand(is_primary.sum()) * 0.5
    X[is_distr]  = 0.8 + np.random.rand(is_distr.sum()) * 0.2

    ys = []
    for seq in X:
        idx = np.where(seq >= 1.0)[0]
        if len(idx) < 3:
            ys.append(0.0); continue
        i1, i2, i3 = idx[-3], idx[-2], idx[-1]
        a1, a2, a3 = seq[i1], seq[i2], seq[i3]
        d_prev = i2 - i1
        d_last = i3 - i2
        sum_prev = seq[i1+1:i2].sum()
        sum_last = seq[i2+1:i3].sum()
        cond = (a1 < a2 < a3) and (d_last > d_prev) and (sum_last > 1.2 * sum_prev)
        ys.append(1.0 if cond else 0.0)

    Xt = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)
    yt = torch.tensor(ys, dtype=torch.float32).unsqueeze(-1)
    return Xt, yt

# import your hybrid LSTM
from lstmhy import EventAugmentedLSTM

def main():
    # prepare data
    X, y = make_complex_dataset(N_SEQS)
    ds   = TensorDataset(X, y)
    loader = DataLoader(ds, batch_size=BS, shuffle=True)

    # instantiate hybrid cell model
    model = EventAugmentedLSTM(input_dim=1,
                               mem_slots=MEM_SLOTS,
                               hidden_dim=HIDDEN,
                               out_dim=OUT).to(DEVICE)
    opt   = optim.Adam(model.parameters(), lr=LR)
    crit  = nn.BCEWithLogitsLoss()

    # training loop
    for ep in range(1, EPOCHS+1):
        model.train()
        total_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            # time-first for your cell: (T, B, 1)
            xb = xb.permute(1,0,2)
            logits = model(xb)[-1]            # take final-timestep logits
            loss = crit(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * xb.size(1)
        print(f"[Hybrid]    Epoch {ep:2d} | Loss: {total_loss/N_SEQS:.4f}")

    # evaluation
    model.eval()
    with torch.no_grad():
        xb, yb = X[:1000], y[:1000]
        xb = xb.permute(1,0,2).to(DEVICE)
        yb = yb.to(DEVICE)
        pred = torch.sigmoid(model(xb)[-1]).round()
        acc  = (pred.cpu() == yb.cpu()).float().mean().item()
        print(f"[Hybrid]    Eval Acc: {acc*100:.2f}%")

if __name__ == "__main__":
    main()
