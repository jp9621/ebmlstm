import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from slingshot import EventRNN  # your implementation

# --- hyperparameters ---
T, SLOTS, D, OUT = 30, 3, 8, 1
EVENT_THRESH = 90
N_SEQS, BS, LR, EPOCHS = 10000, 64, 1e-3, 10
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def make_dataset(n):
    # generate sequences with occasional "event spikes"
    X = np.random.rand(n, T) * 50
    mask = np.random.rand(n, T) < 0.2
    X[mask] = EVENT_THRESH + np.random.rand(mask.sum()) * 30

    ys = []
    for seq in X:
        ev = seq[seq >= EVENT_THRESH]
        # cond1: last 3 events form a peak
        c1 = len(ev) >= 3 and (ev[-3] < ev[-2] > ev[-1])
        # cond2: last 3 raw vols form a valley
        v = seq[-3:]
        c2 = (v[0] > v[1] < v[2])
        ys.append(1.0 if (c1 and c2) else 0.0)

    # shape to (n, T, 1)
    Xt = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)
    # labels shape (n, 1)
    yt = torch.tensor(ys, dtype=torch.float32).unsqueeze(-1)
    return Xt, yt

def main():
    # 1) prepare data
    X, y = make_dataset(N_SEQS)
    ds = TensorDataset(X, y)
    loader = DataLoader(ds, batch_size=BS, shuffle=True)

    # 2) build model
    model = EventRNN(input_dim=1, mem_slots=SLOTS, mem_dim=D, out_dim=OUT).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()

    # 3) training loop
    for epoch in range(1, EPOCHS+1):
        model.train()
        total_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            # xb: (batch, T, 1) → permute to (T, batch, 1)
            xb = xb.permute(1, 0, 2)
            logits = model(xb)[-1]            # get last‐step logits
            loss = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(1)

        avg_loss = total_loss / N_SEQS
        print(f"Epoch {epoch:2d} | Loss: {avg_loss:.4f}")

    # 4) evaluation
    model.eval()
    with torch.no_grad():
        xb, yb = X[:200], y[:200]
        xb = torch.tensor(xb, dtype=torch.float32).permute(1, 0, 2).to(DEVICE)
        yb = torch.tensor(yb, dtype=torch.float32).to(DEVICE)
        preds = torch.sigmoid(model(xb)[-1]).round()
        acc = (preds.cpu() == yb.unsqueeze(-1)).float().mean().item()
        print(f"Accuracy on 200 examples: {acc*100:.1f}%")

if __name__ == "__main__":
    main()
    