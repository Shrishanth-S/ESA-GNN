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
