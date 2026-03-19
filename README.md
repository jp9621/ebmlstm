# Event-Augmented LSTM with Sequence Memory (EBM-LSTM)

Event-Augmented LSTM architecture that combines traditional LSTM processing with an event-based sequence memory mechanism. The model is particularly useful for tasks requiring selective memory of important events in sequential data.

## Architecture Overview

The implementation consists of three main components:

### 1. SequenceMemoryCell
- Maintains a circular buffer of the last `n_slots` events
- Writes to memory only when event detection threshold is exceeded (hardcoded at 0.85)
- Adds positional embeddings to slots, then aggregates via a learned weighted sum (parallelizable)
- Projects the aggregated slot representation to `hidden_dim` via `slot_proj`

### 2. EventAugmentedLSTMCell
- Combines SequenceMemoryCell with a standard LSTM
- Processes input by concatenating current input with memory state
- Maintains both LSTM state and memory cell state
- State tuple consists of (h_lstm, c_lstm, h_mem, slots, ptr)

### 3. EventAugmentedLSTM
- Main sequence processing module
- Wraps EventAugmentedLSTMCell for processing entire sequences
- Includes final classification layer for output predictions

## Usage

```python
import torch
from ebmlstm import EventAugmentedLSTM

# Initialize model
model = EventAugmentedLSTM(
    input_dim=64,    # Dimension of input features
    mem_slots=10,    # Number of memory slots
    hidden_dim=128,  # Hidden state dimension
    out_dim=10,      # Output dimension
    tau=0.5         # Event detection threshold
)

# Forward pass
# x shape: (sequence_length, batch_size, input_dim)
outputs = model(x)  # outputs shape: (sequence_length, batch_size, out_dim)
```

## Key Features

1. **Selective Memory**: The model only stores events that exceed an importance threshold, optimizing memory usage.
2. **Positional Encoding**: Incorporates positional information in the memory slots.
3. **Circular Buffer**: Efficient memory management using a circular buffer mechanism.
4. **Hybrid Architecture**: Combines benefits of LSTM with event-based memory.
5. **Parallel Slot Aggregation**: Slots are fused via a learned softmax-weighted sum, with no sequential dependency across slots.

## Model Parameters

- `input_dim`: Dimension of input features
- `hidden_dim`: Dimension of hidden states
- `n_slots` / `mem_slots`: Number of memory slots in the sequence memory
- `tau`: Passed to `SequenceMemoryCell` (default: 0.5); event write threshold is hardcoded at 0.85
- `out_dim`: Dimension of output predictions