import torch
import torch.nn as nn

class EventAugmentedLSTMCell(nn.Module):
    """
    LSTM cell augmented with a circular event memory.
    Writes x_t into a slot when event_detector(x_t) > tau; slots are
    fused via learned positional embeddings + slot weights to produce h_mem,
    which is concatenated with x_t as input to the LSTM gates.
    State tuple: (h_lstm, c_lstm, slots, ptr)
    """
    def __init__(self, input_dim, hidden_dim, n_slots, tau=0.5):
        super().__init__()
        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim
        self.n_slots    = n_slots
        self.tau        = tau

        self.mem_value    = nn.Linear(input_dim, input_dim)
        self.mem_detector = nn.Linear(input_dim, 1)
        self.mem_pos_emb  = nn.Parameter(torch.randn(n_slots, input_dim))
        self.mem_weights  = nn.Parameter(torch.ones(n_slots))
        self.mem_proj     = nn.Linear(input_dim, hidden_dim)

        self.W_ih = nn.Linear(input_dim + hidden_dim, 4 * hidden_dim)
        self.W_hh = nn.Linear(hidden_dim,             4 * hidden_dim, bias=False)

    def init_state(self, batch_size, device=None):
        if device is None:
            device = next(self.parameters()).device
        return (
            torch.zeros(batch_size, self.hidden_dim, device=device),
            torch.zeros(batch_size, self.hidden_dim, device=device),
            torch.zeros(batch_size, self.n_slots, self.input_dim, device=device),
            torch.zeros(batch_size, dtype=torch.long, device=device),
        )

    def forward(self, x_t, state):
        """
        x_t:   (B, input_dim)
        state: (h_lstm, c_lstm, slots, ptr)
        → h_new, (h_new, c_new, slots, ptr)
        """
        h_lstm, c_lstm, slots, ptr = state
        B = x_t.size(0)

        e_t        = torch.sigmoid(self.mem_detector(x_t))
        event_mask = e_t > self.tau
        v          = self.mem_value(x_t)
        ptr_idx    = ptr.view(B, 1, 1).expand(-1, 1, self.input_dim)
        slots = torch.where(
            event_mask.view(B, 1, 1),
            slots.scatter(1, ptr_idx, v.unsqueeze(1)),
            slots
        )
        ptr = (ptr + event_mask.view(-1).long()) % self.n_slots

        w     = torch.softmax(self.mem_weights, dim=0)
        h_mem = self.mem_proj(
            ((slots + self.mem_pos_emb.unsqueeze(0)) * w.view(1, -1, 1)).sum(dim=1)
        )

        gates   = self.W_ih(torch.cat([x_t, h_mem], dim=1)) + self.W_hh(h_lstm)
        i, f, g, o = gates.chunk(4, dim=1)
        c_new  = torch.sigmoid(f) * c_lstm + torch.sigmoid(i) * torch.tanh(g)
        h_new  = torch.sigmoid(o) * torch.tanh(c_new)

        return h_new, (h_new, c_new, slots, ptr)


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
