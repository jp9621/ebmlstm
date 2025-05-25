import numpy as np
import matplotlib.pyplot as plt

def plot_slot_activation_heatmap(slot_values):
    """
    slot_values: numpy array of shape (seq_len, n_slots)
    """
    activation = slot_values.T  # shape: (n_slots, seq_len)
    plt.figure()
    plt.imshow(activation, aspect='auto', origin='lower')
    plt.xlabel('Time step')
    plt.ylabel('Slot index')
    plt.title('Slot Activation Heatmap')
    plt.colorbar(label='Activation value')
    plt.savefig('figures/slot_activation_heatmap.png')

def plot_attention_line(attention_weights):
    """
    attention_weights: numpy array of shape (seq_len, n_slots)
    """
    seq_len, n_slots = attention_weights.shape
    plt.figure()
    for i in range(n_slots):
        plt.plot(range(seq_len), attention_weights[:, i], label=f'Slot {i}')
    plt.xlabel('Time step')
    plt.ylabel('Attention weight')
    plt.title('Attention Weights Over Time')
    plt.legend()
    plt.savefig('figures/attention_weights_over_time.png')

def plot_attention_stack(attention_weights):
    """
    attention_weights: numpy array of shape (seq_len, n_slots)
    """
    plt.figure()
    x = range(attention_weights.shape[0])
    plt.stackplot(x, attention_weights.T)
    plt.xlabel('Time step')
    plt.ylabel('Attention weight')
    plt.title('Stacked Attention Weights')
    plt.savefig('figures/stacked_attention_weights.png')

def plot_commit_scatter(commit_strengths, event_mask=None):
    """
    commit_strengths: numpy array of shape (seq_len,)
    event_mask: optional boolean array of shape (seq_len,) indicating event occurrences
    """
    plt.figure()
    x = np.arange(len(commit_strengths))
    plt.scatter(x, commit_strengths)
    if event_mask is not None:
        # highlight commit strengths at event steps
        event_indices = np.where(event_mask)[0]
        plt.scatter(event_indices, commit_strengths[event_mask], marker='x')
    plt.xlabel('Time step')
    plt.ylabel('Commit strength')
    plt.title('Commit Strength Over Time')
    plt.savefig('figures/commit_strength_over_time.png')

def plot_delta_t_evolution(delta_t):
    """
    delta_t: numpy array of shape (seq_len, n_slots)
    """
    seq_len, n_slots = delta_t.shape
    plt.figure()
    for i in range(n_slots):
        plt.plot(range(seq_len), delta_t[:, i], label=f'Slot {i}')
    plt.xlabel('Time step')
    plt.ylabel('Delta t')
    plt.title('Delta t Evolution Over Time')
    plt.legend()
    plt.savefig('figures/dt_over_time.png')
