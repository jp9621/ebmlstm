import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from main import EventRNN  # your code

# --- hyperparams ---
T, SLOTS, D, OUT = 30, 3, 8, 1
EVENT_THRESH = 90
N_SEQS, BS, LR, EPOCHS = 10000, 64, 1e-3, 10

def make_dataset(n):
    X = np.random.rand(n, T)*50
    mask = np.random.rand(n, T) < 0.2
    X[mask] = 90 + np.random.rand(mask.sum())*30
    ys = []
    for seq in X:
        ev = seq[seq>=EVENT_THRESH]
        # cond1: last 3 events form a peak
        c1 = len(ev)>=3 and ev[-3] < ev[-2] > ev[-1]
        # cond2: last 3 raw vols form a valley
        v = seq[-3:]
        c2 = v[0] > v[1] < v[2]
        ys.append(1.0 if (c1 and c2) else 0.0)
    Xt = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)  # (n,T,1)
    yt = torch.tensor(ys, dtype=torch.float32).unsqueeze(-1)
    return Xt, yt

# 1) data
X, y = make_dataset(N_SEQS)
ds = TensorDataset(X, y)
loader = DataLoader(ds, batch_size=BS, shuffle=True)

# 2) model
model = EventRNN(input_dim=1, mem_slots=SLOTS, mem_dim=D, out_dim=OUT)
opt   = optim.Adam(model.parameters(), lr=LR)
crit  = nn.BCEWithLogitsLoss()

# 3) train
for ep in range(1, EPOCHS+1):
    tot = 0
    for xb, yb in loader:
        xb = xb.permute(1,0,2)                # (T,B,1)
        out = model(xb)[-1]                  # last step logits
        loss = crit(out, yb)
        opt.zero_grad(); loss.backward(); opt.step()
        tot += loss.item()*xb.size(1)
    print(f"Epoch {ep:2d} | Loss: {tot/len(ds):.4f}")

# 4) eval
with torch.no_grad():
    xb, yb = X[:200].permute(1,0,2), y[:200]
    pred = torch.sigmoid(model(xb)[-1]).round()
    acc  = (pred==yb).float().mean().item()
    print(f"Accuracy (200 ex.): {acc*100:.1f}%")
