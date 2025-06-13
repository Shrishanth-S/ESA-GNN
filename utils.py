from scipy.spatial.distance import cdist
import numpy as np
import torch

def build_graph(positions, threshold=2.0):
    # positions: [num_peds, 2] tensor or numpy
    if isinstance(positions, torch.Tensor):
        positions = positions.cpu().numpy()

    dist = cdist(positions, positions)
    adj = (dist < threshold).astype(int)
    np.fill_diagonal(adj, 0)
    edge_index = np.array(np.nonzero(adj), dtype=np.int64)
    return torch.tensor(edge_index, dtype=torch.long)

def social_force_loss(positions, min_dist=0.5):
    """
    Computes a social force loss to penalize predicted positions that are too close.

    Args:
        positions: Tensor of shape [N, 2] — predicted positions
        min_dist: Minimum comfortable distance between pedestrians

    Returns:
        A scalar loss penalizing too-close positions
    """
    N = positions.size(0)
    loss = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            dist = torch.norm(positions[i] - positions[j])
            if dist < min_dist:
                loss += (min_dist - dist) ** 2
    return loss / N