import torch
import torch.nn as nn
from torch_geometric.nn import GATConv

class GAT(nn.Module):
    def __init__(self, in_channels=34, hidden_channels=64, dropout=0.1):
        super().__init__()
        self.gat1 = GATConv(in_channels, hidden_channels, heads=2, concat=True, dropout=dropout)
        self.gat2 = GATConv(hidden_channels * 2, hidden_channels, heads=1, concat=False, dropout=dropout)

    def forward(self, x, edge_index):
        x = self.gat1(x, edge_index)
        x = torch.relu(x)
        x = self.gat2(x, edge_index)
        return x  # [N, hidden]

class EncoderLSTM(nn.Module):
    def __init__(self, input_size=2, hidden_size=32):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)

    def forward(self, obs_seq):
        _, (h_n, _) = self.lstm(obs_seq)  # h_n: [1, N, hidden]
        return h_n.squeeze(0)             # [N, hidden]

class DecoderLSTM(nn.Module):
    def __init__(self, hidden_size=64, pred_len=12, output_size=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.pred_len = pred_len
        self.output_size = output_size

        self.lstm = nn.LSTM(output_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, context, start_pos):
        N = context.size(0)
        h = context.unsqueeze(0)         # [1, N, hidden]
        c = torch.zeros_like(h)

        inputs = start_pos.unsqueeze(1)  # [N, 1, 2]
        outputs = []

        for _ in range(self.pred_len):
            out, (h, c) = self.lstm(inputs, (h, c))  # [N, 1, hidden]
            pred = self.fc(out.squeeze(1))           # [N, 2]
            outputs.append(pred)
            inputs = pred.unsqueeze(1)

        return torch.stack(outputs, dim=1)  # [N, pred_len, 2]
