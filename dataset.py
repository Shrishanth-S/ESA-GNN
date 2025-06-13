import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset
from torch_geometric.data import Data
from utils import build_graph  # assumes you have a graph-building utility
from model import EncoderLSTM  # assumes you have an LSTM encoder defined

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

        # Create one shared encoder (you can replace with external)
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

            # Find agents that exist in all frames
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

            obs_trajs = torch.tensor(np.array(obs_trajs), dtype=torch.float32)

            fut_trajs = torch.tensor(np.array(fut_trajs), dtype=torch.float32)


            # Encode each agent's motion using LSTM
            obs_seq = obs_trajs  # [N, obs_len, 2]
            obs_seq = obs_seq.permute(0, 1, 2)  # already [N, obs_len, 2]
            _, (h_n, _) = self.lstm_encoder.lstm(obs_seq)  # h_n: [1, N, hidden]
            node_features = h_n.squeeze(0)  # [N, hidden]

            # Last position is used to build graph
            last_positions = obs_seq[:, -1, :]  # [N, 2]
            edge_index = build_graph(last_positions)

            target = fut_trajs[:, 0, :]  # First predicted step [N, 2]

            data = Data(x=node_features, edge_index=edge_index, y=target)
            data.obs_seq = obs_seq
            samples.append(data)

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
