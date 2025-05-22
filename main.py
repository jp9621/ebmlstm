import torch
import torch.nn as nn
import torch.nn.functional as F

class EventMemoryCell(nn.Module):
    """
    An event-based memory cell with continuous drift and event-context return.

    - input_dim:  number of input features per time-step (e.g. volume, range)
    - mem_slots:  how many event-slots we keep alive
    - mem_dim:    size of each slot’s embedding vector

    Learnable parameters:
      threshold_param → controls how “different” a bar must be to be a new event
      alpha_param     → rate at which we absorb the old slot’s content when overwriting
      beta_param      → rate at which we blend a matching slot toward the new bar
      gamma_param     → continuous drift rate: nudges all slots toward every new bar
    """
    def __init__(self, input_dim: int, mem_slots: int, mem_dim: int):
        super().__init__()
        self.input_dim = input_dim       # raw feature size (e.g. 1 for volume only)
        self.mem_slots = mem_slots       # number of prototypes/outlier slots
        self.mem_dim   = mem_dim         # embedding size for each slot

        # a linear layer that turns raw input → mem_dim embedding
        self.embed = nn.Linear(input_dim, mem_dim)

        # initialize all gating scalars at zero → sigmoid(0)=0.5
        self.threshold_param = nn.Parameter(torch.zeros(1))
        self.alpha_param     = nn.Parameter(torch.zeros(1))
        self.beta_param      = nn.Parameter(torch.zeros(1))
        self.gamma_param     = nn.Parameter(torch.zeros(1))

    def forward(self, x_t: torch.Tensor, state: tuple):
        """
        x_t:   tensor of shape (batch, input_dim): the raw input at time t
        state: a tuple (M, idx):
          M:   tensor (batch, mem_slots, mem_dim) holding slot embeddings
          idx: tensor (batch,) pointing to which slot to overwrite next

        Returns:
          r_t:      (batch, mem_dim)   → the memory read-out vector
          new_state:(M_new, idx_new)   → updated memory + pointers
          is_new:   (batch,)           → binary flag: 1 if x_t was a new event
        """
        M, idx = state          # unpack previous memory slots & pointer
        B = x_t.size(0)         # batch size

        # 1) embed current raw input into mem_dim space
        new_e = self.embed(x_t)  # shape: (B, mem_dim)

        # 2) compute cosine similarity between this embed and each slot
        M_norm = F.normalize(M, dim=2)                 # (B, S, D)
        e_norm = F.normalize(new_e, dim=1).unsqueeze(2)  # (B, D, 1)
        sim    = torch.bmm(M_norm, e_norm).squeeze(2)   # (B, S) similarities

        # 3) decide if it's a “new event” vs. matching an existing slot
        threshold = torch.sigmoid(self.threshold_param)  # a learned cutoff in [0,1]
        max_sim, match_idx = sim.max(dim=1)              # best sim per batch
        is_new = (max_sim < threshold).float()           # 1.0 = new event

        # 4) read gating rates via sigmoid for [0,1]
        alpha = torch.sigmoid(self.alpha_param)  # absorption on overwrite
        beta  = torch.sigmoid(self.beta_param)   # match-blend rate
        gamma = torch.sigmoid(self.gamma_param)  # continuous drift rate

        # 5) clone old memory & prepare new pointer
        M_new   = M.clone()
        idx_new = (idx + is_new.long()) % self.mem_slots

        # 6a) NEW EVENT: overwrite the slot at idx
        #    - extract old slot
        old_e    = M[torch.arange(B), idx]       # (B, mem_dim)
        #    - absorb old content into new bar embed
        absorbed = new_e + alpha * old_e         # (B, mem_dim)
        #    - if it’s new, replace that slot; else leave unchanged
        M_new[torch.arange(B), idx] = (
            is_new.unsqueeze(1) * absorbed +
            (1 - is_new).unsqueeze(1) * M_new[torch.arange(B), idx]
        )

        # 6b) MATCHED EVENT: blend new embed into the matched slot
        matched_e = M_new[torch.arange(B), match_idx]          # (B, mem_dim)
        updated   = (1 - beta) * matched_e + beta * new_e      # (B, mem_dim)
        M_new[torch.arange(B), match_idx] = updated

        # 7) CONTINUOUS DRIFT: nudge all slots toward new_e
        drift_target = new_e.unsqueeze(1).expand_as(M_new)     # (B, S, D)
        M_new = (1 - gamma) * M_new + gamma * drift_target

        # 8) READ-OUT: soft-attention over slots → single vector r_t
        attn = F.softmax(sim, dim=1).unsqueeze(1)              # (B,1,S)
        r_t  = torch.bmm(attn, M_new).squeeze(1)               # (B, mem_dim)

        return r_t, (M_new, idx_new), is_new


class EventRNN(nn.Module):
    """
    Sequence model combining:
     - EventMemoryCell (long-term event memory)
     - raw short-term input
     - event-context metrics (cum_vol & delta_t)

    At each time t, it:
      1. updates accumulators for non-events
      2. steps the EventMemoryCell
      3. captures [cum_vol, delta_t] as context
      4. resets those when an event occurred
      5. concatenates [r_t, x_t, context] → final linear output
    """
    def __init__(self,
                 input_dim: int,
                 mem_slots: int,
                 mem_dim: int,
                 out_dim: int,
                 ctx_dim: int = 2):
        super().__init__()
        self.cell   = EventMemoryCell(input_dim, mem_slots, mem_dim)
        # output sees: [mem_readout | raw_input | context_feats]
        self.output = nn.Linear(mem_dim + input_dim + ctx_dim, out_dim)

    def forward(self, x_seq: torch.Tensor):
        """
        x_seq: (T, batch, input_dim) sequence of raw inputs

        returns:
          y_seq: (T, batch, out_dim) per-step outputs
        """
        T, B, D = x_seq.size()  # sequence length, batch size, feature dim

        # initialize empty memory slots + pointer
        M   = x_seq.new_zeros(B, self.cell.mem_slots, self.cell.mem_dim)
        idx = x_seq.new_zeros(B, dtype=torch.long)
        state = (M, idx)

        # initialize accumulators for context features
        cum_vol = x_seq.new_zeros(B, 1)  # sum of volumes since last event
        delta_t = x_seq.new_zeros(B, 1)  # number of bars since last event

        outputs = []
        for t in range(T):
            x_t = x_seq[t]                # raw input at step t (B, D)

            # 1) update context accumulators for *all* bars
            cum_vol = cum_vol + x_t      # add this bar’s volume
            delta_t = delta_t + 1        # increment bar count

            # 2) memory cell update: get r_t and event flag
            r_t, state, is_new = self.cell(x_t, state)

            # 3) form context vector and reset where event happened
            ctx = torch.cat([cum_vol, delta_t], dim=1)     # (B, 2)
            mask = is_new.unsqueeze(1)                     # (B, 1)
            cum_vol = cum_vol * (1 - mask)                 # zero on events
            delta_t = delta_t * (1 - mask)

            # 4) concatenate long-term + raw + context → head
            joint = torch.cat([r_t, x_t, ctx], dim=1)      # (B, mem_dim+D+2)
            outputs.append(self.output(joint))             # (B, out_dim)

        return torch.stack(outputs, dim=0)  # (T, B, out_dim)
