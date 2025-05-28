import torch
import torch.nn as nn

class EventMemoryCell(nn.Module):
    """
    Core event-memory module:
      - Soft-attention commit into slots by content
      - No leak gate
      - Per-step cumulative features & Δt
      - ‘filled’ mask: use empty slots before overwriting
      - Slot-wise RNN fusion → h_mem
    """
    def __init__(self, input_dim, hidden_dim, n_slots):
        super().__init__()
        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim
        self.n_slots    = n_slots

        # Key / Query / Value projections
        self.key   = nn.Linear(input_dim, hidden_dim, bias=False)
        self.query = nn.Linear(input_dim, hidden_dim, bias=False)
        self.value = nn.Linear(input_dim, input_dim)

        # Slot-wise RNN for fusion → hidden_dim
        feat_dim = input_dim * 2 + 1
        self.slot_rnn = nn.LSTM(
            input_size=feat_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )

    def init_state(self, batch_size, device=None):
        if device is None:
            device = next(self.parameters()).device
        h_mem     = torch.zeros(batch_size, self.hidden_dim, device=device)
        slots     = torch.zeros(batch_size, self.n_slots, self.input_dim, device=device)
        cum_feats = torch.zeros_like(slots)
        delta_t   = torch.zeros(batch_size, self.n_slots, device=device)
        filled    = torch.zeros(batch_size, self.n_slots, dtype=torch.bool, device=device)
        return (h_mem, slots, cum_feats, delta_t, filled)

    def forward(self, x_t, state):
        """
        x_t:  (B, input_dim)
        state: (h_mem_prev, slots, cum_feats, delta_t, filled)
        Returns:
          h_mem, (h_mem, slots, cum_feats, delta_t, filled)
        """
        h_mem_prev, slots, cum_feats, delta_t, filled = state
        B = x_t.size(0)

        # 1) Age & accumulate
        delta_t   = delta_t + 1
        cum_feats = cum_feats + x_t.unsqueeze(1)

        # 2) Soft-attention over slots
        q    = self.query(x_t)                               # (B, H)
        keys = self.key(slots)                               # (B, n_slots, H)
        sims = torch.einsum('bnh,bh->bn', keys, q)           # (B, n_slots)
        alpha = torch.softmax(sims, dim=1)                   # (B, n_slots)

        # 3) Commit strength (optional)
        max_sim, _ = sims.max(dim=1, keepdim=True)           # (B,1)
        g = torch.sigmoid(max_sim)                           # (B,1)

        # 4) New slot candidate
        v = self.value(x_t)                                  # (B, input_dim)

        # 5) Choose which slot to write:
        #    first empty, else the most-similar
        empty_any   = (~filled).any(dim=1)                   # (B,)
        empty_mask  = (~filled).float()                      # (B, n_slots)
        idx_empty   = empty_mask.argmax(dim=1)               # (B,) first empty or 0
        _, idx_cont = sims.max(dim=1)                        # (B,)
        idx         = torch.where(empty_any, idx_empty, idx_cont)

        # prepare for scatter updates
        arange = torch.arange(B, device=slots.device)
        idx_exp = idx.view(B,1,1).expand(-1,1,self.input_dim)

        # 6) Overwrite that slot
        slots     = slots.scatter(1, idx_exp, v.unsqueeze(1))
        cum_feats = cum_feats.scatter(1, idx_exp, x_t.unsqueeze(1))
        # 7) Reset its timer & mark it filled
        delta_t   = delta_t.clone()
        delta_t[arange, idx] = 0
        filled    = filled.clone()
        filled[arange, idx] = True

        # 8) Fuse slots → h_mem via slot-wise LSTM
        mem_seq = torch.cat([slots, cum_feats, delta_t.unsqueeze(2)], dim=2)
        rnn_out, _ = self.slot_rnn(mem_seq)  # (B, n_slots, hidden_dim)
        h_mem = rnn_out[:, -1, :]            # (B, hidden_dim)

        return h_mem, (h_mem, slots, cum_feats, delta_t, filled)


class EventAugmentedLSTMCell(nn.Module):
    """
    Wraps EventMemoryCell + an LSTM-from-scratch over [x_t; h_mem].
    State tuple: (h_lstm, c_lstm, h_mem, slots, cum_feats, delta_t, filled)
    """
    def __init__(self, input_dim, hidden_dim, n_slots):
        super().__init__()
        self.event_cell = EventMemoryCell(input_dim, hidden_dim, n_slots)
        self.W_ih = nn.Linear(input_dim + hidden_dim, 4 * hidden_dim)
        self.W_hh = nn.Linear(hidden_dim,             4 * hidden_dim, bias=False)
        self.hidden_dim = hidden_dim

    def init_state(self, batch_size, device=None):
        if device is None:
            device = next(self.parameters()).device
        h_lstm, c_lstm = (torch.zeros(batch_size, self.hidden_dim, device=device),
                          torch.zeros(batch_size, self.hidden_dim, device=device))
        h_mem, slots, cum_feats, delta_t, filled = \
            self.event_cell.init_state(batch_size, device)
        return (h_lstm, c_lstm, h_mem, slots, cum_feats, delta_t, filled)

    def forward(self, x_t, state):
        h_lstm, c_lstm, h_mem_prev, slots, cum_feats, delta_t, filled = state

        # 1) memory update
        h_mem, (h_mem, slots, cum_feats, delta_t, filled) = \
            self.event_cell(x_t, (h_mem_prev, slots, cum_feats, delta_t, filled))

        # 2) LSTM-from-scratch on [x_t; h_mem]
        gates = self.W_ih(torch.cat([x_t, h_mem], dim=1)) + self.W_hh(h_lstm)
        i, f, g_tilde, o = gates.chunk(4, dim=1)
        i, f, o = map(torch.sigmoid, (i, f, o))
        g_tilde = torch.tanh(g_tilde)

        c_new = f * c_lstm + i * g_tilde
        h_new = o * torch.tanh(c_new)

        return h_new, (h_new, c_new, h_mem, slots, cum_feats, delta_t, filled)


class EventAugmentedLSTM(nn.Module):
    """
    Sequence wrapper: steps through EventAugmentedLSTMCell + final readout.
    """
    def __init__(self, input_dim, mem_slots, hidden_dim, out_dim):
        super().__init__()
        self.cell       = EventAugmentedLSTMCell(input_dim, hidden_dim, mem_slots)
        self.classifier = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        T, B, _ = x.size()
        state = self.cell.init_state(B, device=x.device)

        outputs = []
        for t in range(T):
            h_t, state = self.cell(x[t], state)
            outputs.append(self.classifier(h_t).unsqueeze(0))

        return torch.cat(outputs, dim=0)
