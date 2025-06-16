import torch
import torch.nn as nn
from torch_geometric.nn import GATConv

class GAT(nn.Module):
    def __init__(self, in_channels=34, hidden_channels=64, out_channels=64):
        super().__init__()
        self.gat1 = GATConv(in_channels, hidden_channels, heads=2, concat=True)
        self.gat2 = GATConv(hidden_channels * 2, out_channels, heads=1, concat=False)

    def forward(self, x, edge_index):
        x = self.gat1(x, edge_index)
        x = torch.relu(x)
        x = self.gat2(x, edge_index)
        return x  # shape: [N, hidden]

class EncoderLSTM(nn.Module):
    def __init__(self, input_size=2, hidden_size=32):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)

    def forward(self, obs_seq):
        _, (h_n, _) = self.lstm(obs_seq)  # h_n: [1, N, hidden]
        return h_n.squeeze(0)             # [N, hidden]

class DecoderLSTM(nn.Module):
    def __init__(self, input_size=66, hidden_size=64, pred_len=12):
        super().__init__()
        self.pred_len = pred_len
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, context, start_pos):
        """
        context: [N, 64]
        start_pos: [N, 2]
        """
        N = context.size(0)
        input = torch.cat([start_pos, context], dim=1).unsqueeze(1)  # [N, 1, 66]
        hidden = None
        outputs = []

        for _ in range(self.pred_len):
            out, hidden = self.lstm(input, hidden)       # [N, 1, hidden]
            delta = self.fc(out.squeeze(1))              # [N, 2]
            next_pos = input[:, 0, :2] + delta           # [N, 2]
            outputs.append(next_pos)
            input = torch.cat([next_pos, context], dim=1).unsqueeze(1)

        return torch.stack(outputs, dim=1)  # [N, 12, 2]
