import matplotlib.pyplot as plt
import numpy as np
import os

# Ensure the figures directory exists
os.makedirs('figures', exist_ok=True)


def plot_pointer_trajectory(ptr_history):
    """
    Plot the circular buffer write pointer index over time.

    Args:
        ptr_history (List[int]): Sequence of pointer indices (0...n_slots-1).
    """
    plt.figure()
    plt.plot(ptr_history, marker='o')
    plt.xlabel('Time step')
    plt.ylabel('Write pointer index')
    plt.title('Pointer Trajectory')
    plt.grid(True)
    plt.savefig(f'figures/pointer_trajectory.png')
    plt.close()


def plot_event_write_overlay(e_history, threshold):
    """
    Plot commit strength over time with an overlay showing events above the threshold.

    Args:
        e_history (List[float]): Sequence of commit strengths.
        threshold (float): Write threshold \u03c4.
    """
    fig, ax = plt.subplots()
    ax.plot(e_history, label='Commit strength')
    ax.axhline(threshold, color='red', linestyle='--', label=f'Threshold={threshold}')

    # Shade regions where commit strength exceeds threshold
    e_arr = np.array(e_history)
    above = e_arr > threshold
    ax.fill_between(
        range(len(e_history)),
        e_arr,
        threshold,
        where=above,
        color='red',
        alpha=0.3
    )

    ax.set_xlabel('Time step')
    ax.set_ylabel('Commit strength')
    ax.set_title('Event Write Overlay')
    ax.legend()
    ax.grid(True)
    plt.savefig(f'figures/event_write_overlay.png')
    plt.close()


def plot_slot_recency_heatmap(slot_ages):
    """
    Heatmap of slot "age" (steps since last write) over time.

    Args:
        slot_ages (np.ndarray): 2D array shaped (timesteps, n_slots).
    """
    # transpose so rows=slots, cols=time
    ages = np.array(slot_ages)
    plt.figure()
    plt.imshow(ages.T, aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar(label='Age (steps since last write)')
    plt.xlabel('Time step')
    plt.ylabel('Slot index')
    plt.title('Slot Recency Heatmap')
    plt.savefig(f'figures/slot_recency_heatmap.png')
    plt.close()


def plot_h_mem_heatmap(h_mem_history):
    """
    Heatmap of the fused memory vector h_mem over time.

    Args:
        h_mem_history (np.ndarray): 2D array shaped (timesteps, hidden_dim).
    """
    mem = np.array(h_mem_history)
    plt.figure()
    plt.imshow(mem.T, aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar(label='h_mem values')
    plt.xlabel('Time step')
    plt.ylabel('Hidden dimension index')
    plt.title('Fused Memory Evolution Heatmap')
    plt.savefig(f'figures/fused_memory_evolution_heatmap.png')
    plt.close()


def visualize_memory_dynamics(ptr_history, e_history, threshold, slot_ages, h_mem_history):
    """
    Run all sequence-memory diagnostics plots in sequence.

    Args:
        ptr_history (List[int]): Write pointer indices over time.
        e_history (List[float]): Commit strengths over time.
        threshold (float): Write threshold used to detect events.
        slot_ages (List[List[int]] or np.ndarray): Ages per slot per timestep.
        h_mem_history (List[List[float]] or np.ndarray): Fused memory content per timestep.
    """
    # 1. Pointer trajectory
    plot_pointer_trajectory(ptr_history)

    # 2. Commit strength with event overlay
    plot_event_write_overlay(e_history, threshold)

    # 3. Slot recency heatmap
    plot_slot_recency_heatmap(slot_ages)

    # 4. Fused memory evolution
    plot_h_mem_heatmap(h_mem_history)
