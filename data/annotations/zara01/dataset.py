import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np  

class MultiPersonETHDataset(Dataset):
    def __init__(self, path, obs_len=8, pred_len=12, min_agents=2):
        df = pd.read_csv(path, header=None).transpose()
        df.columns = ['frame', 'ped_id', 'y', 'x']
        df = df.sort_values(['frame', 'ped_id'])

        self.obs_len = obs_len
        self.pred_len = pred_len
        self.min_agents = min_agents
        self.frames = sorted(df['frame'].unique())
        self.df = df

        self.samples = self._extract_samples()

    def _extract_samples(self):
        samples = []
        total_frames = len(self.frames)

        for i in range(total_frames - self.obs_len - self.pred_len + 1):
            obs_frames = self.frames[i:i+self.obs_len]
            fut_frames = self.frames[i+self.obs_len:i+self.obs_len+self.pred_len]

            obs_data = self.df[self.df['frame'].isin(obs_frames)]
            fut_data = self.df[self.df['frame'].isin(fut_frames)]

            # Keep only agents present in all frames
            obs_counts = obs_data.groupby('ped_id')['frame'].nunique()
            fut_counts = fut_data.groupby('ped_id')['frame'].nunique()
            valid_agents = obs_counts[obs_counts == self.obs_len].index
            valid_agents = valid_agents.intersection(fut_counts[fut_counts == self.pred_len].index)

            if len(valid_agents) < self.min_agents:
                continue  # skip small groups

            obs_trajs = []
            fut_trajs = []

            for ped_id in valid_agents:
                obs_traj = obs_data[obs_data['ped_id'] == ped_id][['x', 'y']].values
                fut_traj = fut_data[fut_data['ped_id'] == ped_id][['x', 'y']].values

                obs_trajs.append(obs_traj)
                fut_trajs.append(fut_traj)

            obs_trajs = torch.tensor(np.array(obs_trajs), dtype=torch.float32)
            fut_trajs = torch.tensor(np.array(fut_trajs), dtype=torch.float32)

            samples.append((obs_trajs.transpose(0, 1), fut_trajs.transpose(0, 1)))
            # Final shape: obs: [obs_len, num_agents, 2], fut: [pred_len, num_agents, 2]

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
