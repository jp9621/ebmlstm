import torch
import torch.nn as nn

class EventMemoryCell(nn.Module):
    """
    A PyTorch recurrent cell with event-based memory:
      - Soft-attention commit (softmax over similarities).
      - FIFO buffer (drop oldest, append newest).
      - Sigmoid gating for commit strength and per-step leakage.
      - Per-timestep cumulative feature updates, reset on commit.
      - Interval-based timestamps (delta since last commit).
      - Slotwise RNN to fuse memory slots into a single vector.
    """
    def __init__(self, input_dim, hidden_dim, n_slots):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_slots = n_slots

        # Key, query, and value projections
        self.key = nn.Linear(input_dim, hidden_dim, bias=False)
        self.query = nn.Linear(input_dim, hidden_dim, bias=False)
        self.value = nn.Linear(input_dim, input_dim)

        # Gates: commit strength and per-step leakage
        self.commit_bias = nn.Parameter(torch.zeros(1))
        self.reabsorb_gate = nn.Linear(input_dim, 1)
        self.absorb_gate = nn.Linear(input_dim, 1)

        # Slotwise RNN for fusion
        feat_dim = input_dim * 2 + 1
        self.slot_rnn = nn.LSTM(input_size=feat_dim,
                                hidden_size=hidden_dim,
                                num_layers=1,
                                batch_first=True)

        # Output projection: combine slot-RNN, previous hidden, and current input
        self.out_proj = nn.Linear(hidden_dim + hidden_dim + input_dim, hidden_dim)

    def init_state(self, batch_size, device=None):
        """
        Initialize hidden state and memory buffers.
        Returns (h, slots, cum_feats, delta_t).
        """
        if device is None:
            device = next(self.parameters()).device
        h0 = torch.zeros(batch_size, self.hidden_dim, device=device)
        slots = torch.zeros(batch_size, self.n_slots, self.input_dim, device=device)
        cum_feats = torch.zeros_like(slots)
        delta_t = torch.zeros(batch_size, self.n_slots, device=device)
        return (h0, slots, cum_feats, delta_t)

    def forward(self, x_t, state):
        """
        x_t:        Tensor of shape (batch_size, input_dim)
        state:     Tuple (h_prev, slots, cum_feats, delta_t)
        Returns:
            h_new:    (batch_size, hidden_dim)
            new_state:(h_new, slots_new, cum_feats_new, delta_t_new)
        """
        h_prev, slots, cum_feats, delta_t = state
        batch_size = x_t.size(0)

        # 1. Update time intervals
        delta_t = delta_t + 1

        # 2. Compute similarities and soft-attention weights
        q = self.query(x_t)                               # (b, H)
        keys = self.key(slots)                            # (b, n, H)
        sims = torch.sum(keys * q.unsqueeze(1), dim=-1)   # (b, n)
        alpha = torch.softmax(sims, dim=1)                # (b, n)

        # 3. Commit strength (scalar per batch)
        max_sim, _ = sims.max(dim=1)                      # (b,)
        g = torch.sigmoid(max_sim.unsqueeze(1))           # (b, 1)

        # 4. Prepare new slot (reabsorb oldest)
        r = torch.sigmoid(self.reabsorb_gate(x_t))        # (b, 1)
        v = self.value(x_t)                               # (b, input_dim)
        oldest = slots[:, 0, :]                           # (b, input_dim)
        new_slot = r * oldest + (1 - r) * v               # (b, input_dim)
        new_slot = g * new_slot                           # gated commit

        # 5. FIFO buffer update
        slots = torch.cat([slots[:, 1:, :], new_slot.unsqueeze(1)], dim=1)

        # 6. Cumulative features: reset new slot, then add current input
        cum_feats = torch.cat(
            [cum_feats[:, 1:, :], torch.zeros(batch_size, 1, self.input_dim, device=slots.device)],
            dim=1
        )
        cum_feats = cum_feats + x_t.unsqueeze(1)

        # 7. Update delta_t: reset new slot
        delta_t = torch.cat([delta_t[:, 1:], torch.zeros(batch_size, 1, device=slots.device)], dim=1)

        # 8. Per-step leakage into all slots
        leak = torch.sigmoid(self.absorb_gate(x_t)).unsqueeze(1)  # (b,1)
        slots = slots + leak * x_t.unsqueeze(1)
        cum_feats = cum_feats + x_t.unsqueeze(1)

        # 9. Slotwise RNN fusion
        mem_seq = torch.cat([slots, cum_feats, delta_t.unsqueeze(2)], dim=2)  # (b, n, F)
        rnn_out, _ = self.slot_rnn(mem_seq)
        mem_out = rnn_out[:, -1, :]  # (b, hidden)

        # 10. Combine with h_prev and x_t
        h_new = torch.tanh(
            self.out_proj(torch.cat([mem_out, h_prev, x_t], dim=1))
        )

        return h_new, (h_new, slots, cum_feats, delta_t)

class EventRNN(nn.Module):
    """
    Sequence wrapper for EventMemoryCell + a linear readout.
    
    Args:
      input_dim:  Dimensionality of x_t.
      mem_slots:  Number of event‐memory slots.
      mem_dim:    Hidden/memory dimension of the cell.
      out_dim:    Number of output features per time step.
    """
    def __init__(self, input_dim, mem_slots, mem_dim, out_dim):
        super().__init__()
        # core cell
        self.cell = EventMemoryCell(input_dim, mem_dim, mem_slots)
        # map hidden state → output logits
        self.classifier = nn.Linear(mem_dim, out_dim)

    def forward(self, x):
        """
        x: Tensor of shape (T, B, input_dim)
        Returns:
          logits: Tensor of shape (T, B, out_dim)
        """
        T, B, _ = x.size()
        # initialize memory + hidden
        h, slots, cum_feats, delta_t = self.cell.init_state(B, device=x.device)
        state = (h, slots, cum_feats, delta_t)

        outs = []
        for t in range(T):
            h, state = self.cell(x[t], state)    # h: (B, mem_dim)
            logits = self.classifier(h)           # (B, out_dim)
            outs.append(logits.unsqueeze(0))

        return torch.cat(outs, dim=0)            # (T, B, out_dim)