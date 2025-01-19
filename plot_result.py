import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_result(tensors: torch.tensor, save_path, size=1024):
    # Example tensors (replace these with your actual 1024x1024 data arrays)
    # tensors = [np.random.rand(1024, 1024) for _ in range(6)]

    # Number of rows and columns
    rows, cols = 2, 3

    width = cols * size // 8 * 10 + (cols - 1) * 50
    height = rows * size + (rows - 1) * 50

    # Create the figure
    fig, axs = plt.subplots(rows, cols, figsize=(width / 300, height / 300), constrained_layout=True)

    for ax, tensor in zip(axs.flat, tensors):
        # Display the tensor image
        img = ax.imshow(tensor[:size, :size], cmap='viridis', interpolation='none')  # No resampling
        # Add a colorbar that matches the height of the image
        cbar = fig.colorbar(img, ax=ax, orientation='vertical', fraction=0.2, pad=0.01)
        # Disable axes for a clean look
        ax.axis('off')

    plt.show()
    fig.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
