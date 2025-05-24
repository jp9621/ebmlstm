import torch
import torch.nn as nn

# assumes EventMemoryCell is defined exactly as in your snippet
from slingshot import EventMemoryCell

class EventAugmentedLSTMCell(nn.Module):
    """
    Wraps an LSTMCell with your EventMemoryCell:
      - each step: update event‐memory → get h_mem
      - then run LSTMCell on [x_t; h_mem]
    """
    def __init__(self, input_dim, hidden_dim, mem_slots):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.event_cell = EventMemoryCell(input_dim, hidden_dim, mem_slots)
        # LSTMCell now sees both x_t and the memory summary h_mem
        self.lstm = nn.LSTMCell(input_size=input_dim + hidden_dim,
                                 hidden_size=hidden_dim)

    def init_state(self, batch_size, device=None):
        # LSTM state
        if device is None:
            device = next(self.parameters()).device
        h_lstm = torch.zeros(batch_size, self.hidden_dim, device=device)
        c_lstm = torch.zeros(batch_size, self.hidden_dim, device=device)
        # Event‐memory state: (h_mem, slots, cum_feats, delta_t)
        h_mem, slots, cum_feats, delta_t = self.event_cell.init_state(batch_size, device)
        return (h_lstm, c_lstm, h_mem, slots, cum_feats, delta_t)

    def forward(self, x_t, state):
        """
        x_t:   (batch, input_dim)
        state: (h_lstm, c_lstm, h_mem, slots, cum_feats, delta_t)
        returns:
          h_new, c_new, h_mem_new, slots_new, cum_feats_new, delta_t_new
        """
        h_lstm, c_lstm, h_mem, slots, cum_feats, delta_t = state

        # 1) update event memory → new h_mem, new (slots, cum_feats, delta_t)
        h_mem, (h_mem, slots, cum_feats, delta_t) =\
            self.event_cell(x_t, (h_mem, slots, cum_feats, delta_t))

        # 2) run LSTMCell on concatenated [x_t; h_mem]
        lstm_in = torch.cat([x_t, h_mem], dim=1)
        h_new, c_new = self.lstm(lstm_in, (h_lstm, c_lstm))

        return h_new, c_new, h_mem, slots, cum_feats, delta_t


class EventAugmentedLSTM(nn.Module):
    """
    Sequence wrapper: event‐augmented LSTMCell + final readout.
    """
    def __init__(self, input_dim, mem_slots, hidden_dim, out_dim):
        super().__init__()
        self.cell = EventAugmentedLSTMCell(input_dim, hidden_dim, mem_slots)
        self.classifier = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        """
        x: (T, B, input_dim)
        returns logits: (T, B, out_dim)
        """
        T, B, _ = x.size()
        # init all states
        state = self.cell.init_state(B, device=x.device)

        outputs = []
        for t in range(T):
            h, c, h_mem, slots, cum_feats, delta_t = self.cell(x[t], state)
            state = (h, c, h_mem, slots, cum_feats, delta_t)
            logits = self.classifier(h)        # use LSTM hidden h for readout
            outputs.append(logits.unsqueeze(0))

        return torch.cat(outputs, dim=0)      # (T, B, out_dim)
