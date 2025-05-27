# import torch
# import torch.nn as nn

# class EventMemoryCell(nn.Module):
#     """
#     Core event‐memory module:
#       - Soft‐attention commit into FIFO slots
#       - Per‐step leakage and cumulative features
#       - Per‐slot timestamp Δt
#       - Slot‐wise RNN fusion → h_mem
#     """
#     def __init__(self, input_dim, hidden_dim, n_slots):
#         super().__init__()
#         self.input_dim  = input_dim
#         self.hidden_dim = hidden_dim
#         self.n_slots    = n_slots

#         # Key / Query / Value projections
#         self.key   = nn.Linear(input_dim, hidden_dim, bias=False)
#         self.query = nn.Linear(input_dim, hidden_dim, bias=False)
#         self.value = nn.Linear(input_dim, input_dim)

#         # Gates: reabsorb oldest & per‐step leak
#         self.reabsorb_gate = nn.Linear(input_dim, 1)
#         self.absorb_gate   = nn.Linear(input_dim, 1)

#         # Slot‐wise RNN for fusion → hidden_dim
#         feat_dim = input_dim * 2 + 1
#         self.slot_rnn = nn.LSTM(
#             input_size=feat_dim,
#             hidden_size=hidden_dim,
#             num_layers=1,
#             batch_first=True
#         )

#     def init_state(self, batch_size, device=None):
#         if device is None:
#             device = next(self.parameters()).device
#         h_mem     = torch.zeros(batch_size, self.hidden_dim, device=device)
#         slots     = torch.zeros(batch_size, self.n_slots, self.input_dim, device=device)
#         cum_feats = torch.zeros_like(slots)
#         delta_t   = torch.zeros(batch_size, self.n_slots, device=device)
#         return (h_mem, slots, cum_feats, delta_t)

#     def forward(self, x_t, state):
#         """
#         x_t:  (B, input_dim)
#         state: (h_mem_prev, slots, cum_feats, delta_t)
#         Returns:
#           h_mem, (h_mem, slots, cum_feats, delta_t)
#         """
#         h_mem_prev, slots, cum_feats, delta_t = state
#         B = x_t.size(0)

#         # 1) Age all slots
#         delta_t = delta_t + 1

#         # 2) Soft‐attention over slots
#         q    = self.query(x_t)                       # (B, H)
#         keys = self.key(slots)                       # (B, n_slots, H)
#         sims = torch.einsum('bnh,bh->bn', keys, q)   # (B, n_slots)
#         alpha = torch.softmax(sims, dim=1)           # (B, n_slots)

#         # 3) Commit strength (scalar)
#         max_sim, _ = sims.max(dim=1, keepdim=True)   # (B,1)
#         g = torch.sigmoid(max_sim)                   # (B,1)

#         # 4) Form new slot via reabsorb + value projection
#         r      = torch.sigmoid(self.reabsorb_gate(x_t))   # (B,1)
#         v      = self.value(x_t)                          # (B, input_dim)
#         oldest = slots[:, 0, :]                           # (B, input_dim)
#         new_slot = r * oldest + (1 - r) * v               # (B, input_dim)
#         new_slot = g * new_slot                           # gated commit

#         # 5) FIFO replace
#         slots = torch.cat([slots[:, 1:, :], new_slot.unsqueeze(1)], dim=1)

#         # 6) Cum‐features: reset newest, then add x_t
#         cum_feats = torch.cat([
#             cum_feats[:, 1:, :],
#             torch.zeros(B, 1, self.input_dim, device=slots.device)
#         ], dim=1)
#         cum_feats = cum_feats + x_t.unsqueeze(1)

#         # 7) Reset Δt for newest
#         delta_t = torch.cat([
#             delta_t[:, 1:],
#             torch.zeros(B, 1, device=delta_t.device)
#         ], dim=1)

#         # 8) Leak into all slots & cum_feats
#         leak = torch.sigmoid(self.absorb_gate(x_t)).unsqueeze(1)  # (B,1)
#         slots     = slots     + leak * x_t.unsqueeze(1)
#         cum_feats = cum_feats + x_t.unsqueeze(1)

#         # 9) Fuse slots → h_mem
#         mem_seq = torch.cat([
#             slots,
#             cum_feats,
#             delta_t.unsqueeze(2)
#         ], dim=2)  # (B, n_slots, 2*input_dim + 1)
#         rnn_out, _ = self.slot_rnn(mem_seq)  # (B, n_slots, hidden_dim)
#         h_mem = rnn_out[:, -1, :]            # (B, hidden_dim)

#         return h_mem, (h_mem, slots, cum_feats, delta_t)


# class EventAugmentedLSTMCell(nn.Module):
#     """
#     Wraps EventMemoryCell + an LSTM‐from‐scratch over [x_t; h_mem].
#     State tuple: (h_lstm, c_lstm, h_mem, slots, cum_feats, delta_t)
#     """
#     def __init__(self, input_dim, hidden_dim, n_slots):
#         super().__init__()
#         self.input_dim  = input_dim
#         self.hidden_dim = hidden_dim

#         # core memory submodule
#         self.event_cell = EventMemoryCell(input_dim, hidden_dim, n_slots)

#         # LSTM‐from‐scratch parameters
#         # input = [x_t; h_mem]
#         self.W_ih = nn.Linear(input_dim + hidden_dim, 4 * hidden_dim)
#         self.W_hh = nn.Linear(hidden_dim,             4 * hidden_dim, bias=False)

#     def init_state(self, batch_size, device=None):
#         if device is None:
#             device = next(self.parameters()).device
#         # LSTM state
#         h_lstm = torch.zeros(batch_size, self.hidden_dim, device=device)
#         c_lstm = torch.zeros(batch_size, self.hidden_dim, device=device)
#         # memory state
#         h_mem, slots, cum_feats, delta_t = \
#             self.event_cell.init_state(batch_size, device)
#         return (h_lstm, c_lstm, h_mem, slots, cum_feats, delta_t)

#     def forward(self, x_t, state):
#         """
#         x_t:  (B, input_dim)
#         state: (h_lstm, c_lstm, h_mem_prev, slots, cum_feats, delta_t)
#         Returns:
#           h_new, (h_new, c_new, h_mem, slots, cum_feats, delta_t)
#         """
#         h_lstm, c_lstm, h_mem_prev, slots, cum_feats, delta_t = state

#         # --- 1) memory update ---
#         h_mem, (h_mem, slots, cum_feats, delta_t) = \
#             self.event_cell(x_t, (h_mem_prev, slots, cum_feats, delta_t))

#         # --- 2) LSTM‐from‐scratch on [x_t; h_mem] ---
#         gates = self.W_ih(torch.cat([x_t, h_mem], dim=1)) \
#               + self.W_hh(h_lstm)
#         i, f, g_tilde, o = gates.chunk(4, dim=1)
#         i = torch.sigmoid(i)
#         f = torch.sigmoid(f)
#         g_tilde = torch.tanh(g_tilde)
#         o = torch.sigmoid(o)

#         c_new = f * c_lstm + i * g_tilde
#         h_new = o * torch.tanh(c_new)

#         return h_new, (h_new, c_new, h_mem, slots, cum_feats, delta_t)


# class EventAugmentedLSTM(nn.Module):
#     """
#     Sequence wrapper: steps through EventAugmentedLSTMCell + final readout.
#     """
#     def __init__(self, input_dim, mem_slots, hidden_dim, out_dim):
#         super().__init__()
#         self.cell       = EventAugmentedLSTMCell(input_dim, hidden_dim, mem_slots)
#         self.classifier = nn.Linear(hidden_dim, out_dim)

#     def forward(self, x):
#         """
#         x: Tensor[T, B, input_dim]
#         returns: Tensor[T, B, out_dim]
#         """
#         T, B, _ = x.size()
#         state = self.cell.init_state(B, device=x.device)

#         outputs = []
#         for t in range(T):
#             h_t, state = self.cell(x[t], state)
#             outputs.append(self.classifier(h_t).unsqueeze(0))

#         return torch.cat(outputs, dim=0)


import torch
import torch.nn as nn

class EventMemoryCell(nn.Module):
    """
    Core event‑memory module:
      - Soft‑attention commit into FIFO slots
      - Removed reabsorb & leak gates, now commit-only replacement
      - Per‑step cumulative features & Δt
      - Slot‑wise RNN fusion → h_mem
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

        # Slot‑wise RNN for fusion → hidden_dim
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
        return (h_mem, slots, cum_feats, delta_t)

    def forward(self, x_t, state):
        """
        x_t:  (B, input_dim)
        state: (h_mem_prev, slots, cum_feats, delta_t)
        Returns:
          h_mem, (h_mem, slots, cum_feats, delta_t)
        """
        h_mem_prev, slots, cum_feats, delta_t = state
        B = x_t.size(0)

        # 1) Age all slots
        delta_t = delta_t + 1

        # 2) Soft‑attention over slots
        q    = self.query(x_t)                              # (B, H)
        keys = self.key(slots)                              # (B, n_slots, H)
        sims = torch.einsum('bnh,bh->bn', keys, q)          # (B, n_slots)
        alpha = torch.softmax(sims, dim=1)                  # (B, n_slots)

        # 3) Commit strength (scalar)
        max_sim, _ = sims.max(dim=1, keepdim=True)          # (B,1)
        g = torch.sigmoid(max_sim)                          # (B,1)

        # 4) Simplified commit-only: project value and gate by g
        v        = self.value(x_t)                          # (B, input_dim)
        new_slot = g * v                                    # (B, input_dim)

        # 5) FIFO replace _only_ when commit exceeds threshold
        tau   = 0.1
        mask  = (g > tau).float().view(B,1,1)               # (B,1,1)
        # candidate FIFO-shifted slots + new slot
        cand  = torch.cat([slots[:, 1:, :], new_slot.unsqueeze(1)], dim=1)
        # apply conditional replace
        slots = mask * cand + (1 - mask) * slots

        # 6) Cum‑features: reset newest, then add x_t
        cum_feats = torch.cat([
            cum_feats[:, 1:, :],
            torch.zeros(B, 1, self.input_dim, device=slots.device)
        ], dim=1)
        cum_feats = cum_feats + x_t.unsqueeze(1)

        # 7) Reset Δt for newest
        delta_t = torch.cat([
            delta_t[:, 1:],
            torch.zeros(B, 1, device=delta_t.device)
        ], dim=1)

        # 8) (removed) no leak/absorb gates

        # 9) Fuse slots → h_mem
        mem_seq = torch.cat([
            slots,
            cum_feats,
            delta_t.unsqueeze(2)
        ], dim=2)  # (B, n_slots, 2*input_dim + 1)
        rnn_out, _ = self.slot_rnn(mem_seq)  # (B, n_slots, hidden_dim)
        h_mem = rnn_out[:, -1, :]            # (B, hidden_dim)

        return h_mem, (h_mem, slots, cum_feats, delta_t)


class EventAugmentedLSTMCell(nn.Module):
    """
    Wraps EventMemoryCell + an LSTM-from-scratch over [x_t; h_mem].
    State tuple: (h_lstm, c_lstm, h_mem, slots, cum_feats, delta_t)
    """
    def __init__(self, input_dim, hidden_dim, n_slots):
        super().__init__()
        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim

        # core memory submodule
        self.event_cell = EventMemoryCell(input_dim, hidden_dim, n_slots)

        # LSTM-from-scratch parameters
        # input = [x_t; h_mem]
        self.W_ih = nn.Linear(input_dim + hidden_dim, 4 * hidden_dim)
        self.W_hh = nn.Linear(hidden_dim,             4 * hidden_dim, bias=False)

    def init_state(self, batch_size, device=None):
        if device is None:
            device = next(self.parameters()).device
        # LSTM state
        h_lstm = torch.zeros(batch_size, self.hidden_dim, device=device)
        c_lstm = torch.zeros(batch_size, self.hidden_dim, device=device)
        # memory state
        h_mem, slots, cum_feats, delta_t = \
            self.event_cell.init_state(batch_size, device)
        return (h_lstm, c_lstm, h_mem, slots, cum_feats, delta_t)

    def forward(self, x_t, state):
        """
        x_t:  (B, input_dim)
        state:
         (h_lstm, c_lstm, h_mem_prev, slots, cum_feats, delta_t)
        Returns:
          h_new, (h_new, c_new, h_mem, slots, cum_feats, delta_t)
        """
        h_lstm, c_lstm, h_mem_prev, slots, cum_feats, delta_t = state

        # --- 1) memory update ---
        h_mem, (h_mem, slots, cum_feats, delta_t) = \
            self.event_cell(x_t, (h_mem_prev, slots, cum_feats, delta_t))

        # --- 2) LSTM-from-scratch on [x_t; h_mem] ---
        gates = self.W_ih(torch.cat([x_t, h_mem], dim=1)) \
              + self.W_hh(h_lstm)
        i, f, g_tilde, o = gates.chunk(4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g_tilde = torch.tanh(g_tilde)
        o = torch.sigmoid(o)

        c_new = f * c_lstm + i * g_tilde
        h_new = o * torch.tanh(c_new)

        return h_new, (h_new, c_new, h_mem, slots, cum_feats, delta_t)


class EventAugmentedLSTM(nn.Module):
    """
    Sequence wrapper: steps through EventAugmentedLSTMCell + final readout.
    """
    def __init__(self, input_dim, mem_slots, hidden_dim, out_dim):
        super().__init__()
        self.cell       = EventAugmentedLSTMCell(input_dim, hidden_dim, mem_slots)
        self.classifier = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        """
        x: Tensor[T, B, input_dim]
        returns: Tensor[T, B, out_dim]
        """
        T, B, _ = x.size()
        state = self.cell.init_state(B, device=x.device)

        outputs = []
        for t in range(T):
            h_t, state = self.cell(x[t], state)
            outputs.append(self.classifier(h_t).unsqueeze(0))

        return torch.cat(outputs, dim=0)

