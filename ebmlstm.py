import torch
import torch.nn as nn

class SequenceMemoryCell(nn.Module):
    """
    Keeps the last n_slots events in a circular buffer.
    Writes only when event_detector(x_t) > tau.
    Adds positional embeddings before fusing slots with an LSTM.
    """
    def __init__(self, input_dim, hidden_dim, n_slots, tau=0.5):
        super().__init__()
        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim
        self.n_slots    = n_slots
        self.tau        = tau

        self.value = nn.Linear(input_dim, input_dim)
        self.event_detector = nn.Linear(input_dim, 1)
        self.pos_emb = nn.Parameter(torch.randn(n_slots, input_dim))
        self.slot_weights = nn.Parameter(torch.ones(n_slots))
        self.slot_proj = nn.Linear(input_dim, hidden_dim)

    def init_state(self, batch_size, device=None):
        if device is None:
            device = next(self.parameters()).device
        slots = torch.zeros(batch_size, self.n_slots, self.input_dim, device=device)
        ptr   = torch.zeros(batch_size, dtype=torch.long, device=device)
        return (slots, ptr)

    def forward(self, x_t, state):
        """
        x_t:    (B, input_dim)
        state:  (slots: (B, n_slots, input_dim), ptr: (B,))
        → h_mem: (B, hidden_dim), (new_slots, new_ptr)
        """
        slots, ptr = state
        B = x_t.size(0)

        # e_t and v depend only on x_t and are independent of each other
        e_t        = torch.sigmoid(self.event_detector(x_t))
        v          = self.value(x_t)

        event_mask = e_t > 0.85
        ptr_idx    = ptr.view(B, 1, 1).expand(-1, 1, self.input_dim)
        v_slot     = v.unsqueeze(1)

        new_slots = torch.where(
            event_mask.view(B, 1, 1),
            slots.scatter(1, ptr_idx, v_slot),
            slots
        )

        new_ptr = (ptr + event_mask.view(-1).long()) % self.n_slots

        w     = torch.softmax(self.slot_weights, dim=0)
        h_mem = self.slot_proj(
            ((new_slots + self.pos_emb.unsqueeze(0)) * w.view(1, -1, 1)).sum(dim=1)
        )

        return h_mem, (new_slots, new_ptr)


class EventAugmentedLSTMCell(nn.Module):
    """
    Wraps SequenceMemoryCell + a from-scratch LSTM on [x_t; h_mem].
    State tuple: (h_lstm, c_lstm, h_mem, slots, ptr)
    """
    def __init__(self, input_dim, hidden_dim, n_slots, tau=0.5):
        super().__init__()
        self.mem_cell   = SequenceMemoryCell(input_dim, hidden_dim, n_slots, tau)
        self.W_ih       = nn.Linear(input_dim + hidden_dim, 4 * hidden_dim)
        self.W_hh       = nn.Linear(hidden_dim,             4 * hidden_dim, bias=False)
        self.hidden_dim = hidden_dim

    def init_state(self, batch_size, device=None):
        if device is None:
            device = next(self.parameters()).device
        h_lstm = torch.zeros(batch_size, self.hidden_dim, device=device)
        c_lstm = torch.zeros(batch_size, self.hidden_dim, device=device)
        slots, ptr = self.mem_cell.init_state(batch_size, device)
        h_mem = torch.zeros(batch_size, self.hidden_dim, device=device)
        return (h_lstm, c_lstm, h_mem, slots, ptr)

    def forward(self, x_t, state):
        """
        x_t:   (B, input_dim)
        state: (h_lstm, c_lstm, h_mem_prev, slots, ptr)
        → h_new, (h_new, c_new, h_mem, slots, ptr)
        """
        h_lstm, c_lstm, _, slots, ptr = state

        h_mem, (slots, ptr) = self.mem_cell(x_t, (slots, ptr))

        gates = self.W_ih(torch.cat([x_t, h_mem], dim=1)) + self.W_hh(h_lstm)
        i, f, g_tilde, o = gates.chunk(4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g_tilde = torch.tanh(g_tilde)

        c_new  = f * c_lstm + i * g_tilde
        h_new  = o * torch.tanh(c_new)

        return h_new, (h_new, c_new, h_mem, slots, ptr)


class EventAugmentedLSTM(nn.Module):
    """
    Sequence wrapper: steps through EventAugmentedLSTMCell + final readout.
    """
    def __init__(self, input_dim, mem_slots, hidden_dim, out_dim, tau=0.5):
        super().__init__()
        self.cell       = EventAugmentedLSTMCell(input_dim, hidden_dim, mem_slots, tau)
        self.classifier = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        T, B, _ = x.size()
        state = self.cell.init_state(B, device=x.device)
        outputs = []
        for t in range(T):
            h_t, state = self.cell(x[t], state)
            outputs.append(self.classifier(h_t).unsqueeze(0))
        return torch.cat(outputs, dim=0)
