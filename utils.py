from scipy.spatial.distance import cdist
import numpy as np
import torch

def build_graph(positions, threshold=2.0):
    """
    Constructs edges between pedestrians closer than threshold.

    Args:
        positions: Tensor [N, 2]
        threshold: Distance threshold for edge creation

    Returns:
        edge_index: Tensor [2, num_edges]
    """
    if isinstance(positions, torch.Tensor):
        positions = positions.cpu().numpy()

    dist = cdist(positions, positions)  # [N, N]
    adj = (dist < threshold).astype(int)
    np.fill_diagonal(adj, 0)  # Remove self-loops
    edge_index = np.array(np.nonzero(adj), dtype=np.int64)
    return torch.tensor(edge_index, dtype=torch.long)

def social_force_loss(positions, min_dist=0.5):
    """
    Penalizes pedestrians getting too close to each other.

    Args:
        positions: Tensor [N, 2] (flattened predicted positions at t)
        min_dist: Threshold for discomfort

    Returns:
        Tensor scalar loss
    """
    N = positions.size(0)
    if N < 2:
        return torch.tensor(0.0, dtype=positions.dtype, device=positions.device)

    loss = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            dist = torch.norm(positions[i] - positions[j])
            if dist < min_dist:
                loss += (min_dist - dist) ** 2

    return loss / N
