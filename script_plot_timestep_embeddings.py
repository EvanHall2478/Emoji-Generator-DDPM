import matplotlib.pyplot as plt
from network import TimestepEmbedding
import torch


def plot_timestep_embeddings(dim=128, max_timestep=128):
    """Plot and save the timestep embeddings visualization."""
    # Create the embedding layer
    embedding = TimestepEmbedding(dim)

    # Get embeddings
    timesteps = torch.arange(1, max_timestep + 1, dtype=torch.float32)
    embeddings = embedding(timesteps)
    embeddings_np = embeddings.detach().numpy()

    # Create the plot
    plt.figure(figsize=(8, 6))
    plt.imshow(
        X=embeddings_np.T,
        aspect='auto',
        cmap='RdBu',
        extent=[1, max_timestep, 0, dim],
    )

    # Colorbar
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=14)

    # Axis labels and title
    plt.xlabel('Timestep', fontsize=16, labelpad=12)
    plt.ylabel('Embedding Dimension', fontsize=16, labelpad=12)
    plt.title(f'Timestep Embeddings (dim={dim}, timesteps=1-{max_timestep})',
              fontsize=16,
              pad=15)

    # Tick labels
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    plt.tight_layout()
    plt.savefig('timestep_embeddings.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved as 'timestep_embeddings.png'")


if __name__ == "__main__":
    plot_timestep_embeddings()
