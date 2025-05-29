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

        # map x_t → slot content
        self.value = nn.Linear(input_dim, input_dim)
        # detect whether x_t is an “event”
        self.event_detector = nn.Linear(input_dim, 1)
        # positional embeddings for each slot index
        self.pos_emb = nn.Parameter(torch.randn(n_slots, input_dim))
        # fuse the sequence of slots → h_mem
        self.slot_fuser = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )

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

        # 1) detect event (boolean mask)
        e_t         = torch.sigmoid(self.event_detector(x_t))  # (B,1)
        event_mask  = e_t > 0.85 # self.tau                            # (B,1) boolean
        mask3       = event_mask.view(B, 1, 1)                  # (B,1,1) boolean

        # 2) prepare new content
        v       = self.value(x_t)                               # (B, input_dim)
        ptr_idx = ptr.view(B,1,1).expand(-1,1,self.input_dim)   # (B,1,input_dim)

        # 3) circular write only when event_mask=True
        new_slots = torch.where(
            mask3,
            slots.scatter(1, ptr_idx, v.unsqueeze(1)),
            slots
        )

        # 4) advance pointer mod n_slots (using event_mask.long())
        new_ptr = (ptr + event_mask.view(-1).long()) % self.n_slots

        # 5) add positional embeddings
        slots_pe = new_slots + self.pos_emb.unsqueeze(0)        # (B, n_slots, input_dim)

        # 6) fuse with slot-wise LSTM
        rnn_out, _ = self.slot_fuser(slots_pe)                  # (B, n_slots, hidden_dim)
        h_mem      = rnn_out[:, -1, :]                          # (B, hidden_dim)

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

        # 1) update circular memory → h_mem
        h_mem, (slots, ptr) = self.mem_cell(x_t, (slots, ptr))

        # 2) LSTM-from-scratch on [x_t; h_mem]
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
        # x: (T, B, input_dim)
        T, B, _ = x.size()
        state = self.cell.init_state(B, device=x.device)
        outputs = []
        for t in range(T):
            h_t, state = self.cell(x[t], state)
            outputs.append(self.classifier(h_t).unsqueeze(0))
        return torch.cat(outputs, dim=0)
