import torch
import numpy as np
from torch_geometric.data import Data
from utils import build_graph

def generate_synthetic_trajectories(num_agents=3, obs_len=8, pred_len=12):
    """
    Generate linear trajectories for num_agents.
    Each agent moves in a different direction with small noise.
    """
    obs_trajs = []
    fut_trajs = []

    for _ in range(num_agents):
        x_start, y_start = np.random.uniform(0, 5), np.random.uniform(0, 5)
        dx, dy = np.random.uniform(0.1, 0.4), np.random.uniform(0.1, 0.4)

        obs = np.array([[x_start + dx*t, y_start + dy*t] for t in range(obs_len)])
        fut = np.array([[x_start + dx*(obs_len + t), y_start + dy*(obs_len + t)] for t in range(pred_len)])

        obs_trajs.append(obs)
        fut_trajs.append(fut)

    obs_tensor = torch.tensor(np.array(obs_trajs), dtype=torch.float32)  # [N, obs_len, 2]
    fut_tensor = torch.tensor(np.array(fut_trajs), dtype=torch.float32)  # [N, pred_len, 2]

    # Format like your real data: [obs_len, N, 2]
    obs_tensor = obs_tensor.permute(1, 0, 2)
    fut_tensor = fut_tensor.permute(1, 0, 2)

    return obs_tensor, fut_tensor
