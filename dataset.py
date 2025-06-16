import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset
from torch_geometric.data import Data
from utils import build_graph
from model import EncoderLSTM

class PedestrianDataset(Dataset):
    def __init__(self, path, obs_len=8, pred_len=12, min_agents=2, hidden_size=32):
        df = pd.read_csv(path, header=None).transpose()
        df.columns = ['frame', 'ped_id', 'y', 'x']
        df = df.sort_values(['frame', 'ped_id'])

        self.obs_len = obs_len
        self.pred_len = pred_len
        self.min_agents = min_agents
        self.frames = sorted(df['frame'].unique())
        self.df = df
        self.hidden_size = hidden_size
        self.lstm_encoder = EncoderLSTM(input_size=2, hidden_size=hidden_size)
        self.samples = self._extract_samples()

    def _extract_samples(self):
        samples = []
        total_frames = len(self.frames)

        for i in range(total_frames - self.obs_len - self.pred_len + 1):
            obs_frames = self.frames[i:i+self.obs_len]
            fut_frames = self.frames[i+self.obs_len:i+self.obs_len+self.pred_len]

            obs_data = self.df[self.df['frame'].isin(obs_frames)]
            fut_data = self.df[self.df['frame'].isin(fut_frames)]

            obs_counts = obs_data.groupby('ped_id')['frame'].nunique()
            fut_counts = fut_data.groupby('ped_id')['frame'].nunique()
            valid_agents = obs_counts[obs_counts == self.obs_len].index
            valid_agents = valid_agents.intersection(fut_counts[fut_counts == self.pred_len].index)

            if len(valid_agents) < self.min_agents:
                continue

            obs_trajs = []
            fut_trajs = []

            for ped_id in valid_agents:
                obs_traj = obs_data[obs_data['ped_id'] == ped_id][['x', 'y']].values
                fut_traj = fut_data[fut_data['ped_id'] == ped_id][['x', 'y']].values
                obs_trajs.append(obs_traj)
                fut_trajs.append(fut_traj)

            obs_trajs = torch.tensor(np.array(obs_trajs), dtype=torch.float32)  # [N, obs_len, 2]
            fut_trajs = torch.tensor(np.array(fut_trajs), dtype=torch.float32)  # [N, pred_len, 2]
            target = fut_trajs

            # Encode with LSTM
            node_input = obs_trajs.permute(0, 1, 2)  # [N, obs_len, 2]
            _, (h_n, _) = self.lstm_encoder.lstm(node_input)  # [1, N, hidden]
            lstm_features = h_n.squeeze(0)  # [N, hidden]

            # Add last position as part of node feature
            last_pos = obs_trajs[:, -1, :]  # [N, 2]
            node_features = torch.cat([last_pos, lstm_features], dim=1)  # [N, 34]

            # Build graph
            edge_index = build_graph(last_pos)  # [2, num_edges]

            data = Data(x=node_features, edge_index=edge_index, y=target)
            data.obs_seq = obs_trajs
            samples.append(data)

        return samples


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
