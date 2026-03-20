# EBM-LSTM

An LSTM variant designed for lower inference latency on long sequences. Instead of processing the full sequence, it summarizes an early buffer via learned attention, then runs the LSTM over a short trailing window.

## Usage

```python
from metrics import EBM_LSTM

model = EBM_LSTM(
    input_dim=16,    # input feature dimension
    hidden_dim=64,   # LSTM hidden size
    n_slots=10,      # attention slots for buffer summary
    buffer_len=80,   # timesteps to summarize (remainder processed by LSTM)
    out_dim=2,       # output classes
)

# x: (batch, seq_len, input_dim)
logits = model(x)  # (batch, out_dim)
```

