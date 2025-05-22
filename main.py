import torch
import torch.nn as nn
import torch.nn.functional as F

class EventMemoryCell(nn.Module):
    """
    An event-based memory cell with continuous drift.

    - input_dim: size of each input feature vector
    - mem_slots: number of event slots (n)
    - mem_dim: dimensionality of each slot

    Learnable params:
      - threshold_param → event threshold t
      - alpha_param     → absorption rate α (on eject)
      - beta_param      → match-update rate β
      - gamma_param     → continuous drift rate γ
    """
    def __init__(self, input_dim: int, mem_slots: int, mem_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.mem_slots = mem_slots
        self.mem_dim = mem_dim

        # maps raw input → embedding space
        self.embed = nn.Linear(input_dim, mem_dim)

        # initialize all to 0 → sigmoid(0)=0.5
        self.threshold_param = nn.Parameter(torch.zeros(1))
        self.alpha_param     = nn.Parameter(torch.zeros(1))
        self.beta_param      = nn.Parameter(torch.zeros(1))
        self.gamma_param     = nn.Parameter(torch.zeros(1))  # new drift rate

    def forward(self, x_t: torch.Tensor, state: tuple):
        """
        x_t:    (batch, input_dim)
        state:  (M, idx)
          M:   (batch, mem_slots, mem_dim)
          idx: (batch,) oldest-slot pointers
        returns:
          r_t:     (batch, mem_dim)
          new_state: (M_new, idx_new)
        """
        M, idx = state
        B = x_t.size(0)

        # 1) embed current input
        new_e = self.embed(x_t)                     # (B, mem_dim)

        # 2) cosine similarities against each slot
        M_norm = F.normalize(M, dim=2)              # (B, S, D)
        e_norm = F.normalize(new_e, dim=1).unsqueeze(2)  # (B, D, 1)
        sim = torch.bmm(M_norm, e_norm).squeeze(2)  # (B, S)

        # 3) event vs. match
        threshold = torch.sigmoid(self.threshold_param)
        max_sim, match_idx = sim.max(dim=1)         # best sim per batch
        is_new = (max_sim < threshold).float()      # 1.0=new event

        # 4) rates
        alpha = torch.sigmoid(self.alpha_param)
        beta  = torch.sigmoid(self.beta_param)
        gamma = torch.sigmoid(self.gamma_param)

        # 5) prepare new memory & pointer
        M_new   = M.clone()
        idx_new = (idx + is_new.long()) % self.mem_slots

        # 6a) new-event logic: eject oldest, absorb it, write new
        old_e = M[torch.arange(B), idx]                                 # (B, D)
        absorbed = new_e + alpha * old_e                                # (B, D)
        M_new[torch.arange(B), idx] = (                               
            is_new.unsqueeze(1) * absorbed                             # replace if new
          + (1 - is_new).unsqueeze(1) * M_new[torch.arange(B), idx]     # else keep old
        )

        # 6b) match logic: blend into matched slot
        matched_e = M_new[torch.arange(B), match_idx]                   # (B, D)
        updated   = (1 - beta) * matched_e + beta * new_e
        M_new[torch.arange(B), match_idx] = updated

        # 7) continuous drift: all slots → towards new_e
        #    M_new = (1 - γ)·M_new + γ·new_e.unsqueeze(1)
        drift_target = new_e.unsqueeze(1).expand_as(M_new)             # (B, S, D)
        M_new = (1 - gamma) * M_new + gamma * drift_target

        # 8) read via attention
        attn = F.softmax(sim, dim=1).unsqueeze(1)                       # (B,1,S)
        r_t  = torch.bmm(attn, M_new).squeeze(1)                        # (B, D)

        return r_t, (M_new, idx_new)


class EventRNN(nn.Module):
    """
    Sequence model wrapping EventMemoryCell:
    - input:  x_seq (T, batch, input_dim)
    - output: y_seq (T, batch, out_dim)
    """
    def __init__(self, input_dim: int, mem_slots: int, mem_dim: int, out_dim: int):
        super().__init__()
        self.cell   = EventMemoryCell(input_dim, mem_slots, mem_dim)
        self.output = nn.Linear(mem_dim, out_dim)

    def forward(self, x_seq: torch.Tensor):
        T, B, _ = x_seq.size()
        # init zero memory + pointer
        M   = x_seq.new_zeros(B, self.cell.mem_slots, self.cell.mem_dim)
        idx = x_seq.new_zeros(B, dtype=torch.long)
        state = (M, idx)

        outputs = []
        for t in range(T):
            r_t, state = self.cell(x_seq[t], state)
            outputs.append(self.output(r_t))

        return torch.stack(outputs, dim=0)  # (T, B, out_dim)
