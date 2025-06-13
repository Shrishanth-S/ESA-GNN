import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
from model import GAT
from utils import build_graph
from dataset import MultiPersonETHDataset

def train():
    dataset = MultiPersonETHDataset("data/annotations/zara01/world_coordinate_inter.csv")
    print("Loaded dataset with", len(dataset), "samples")

    model = GAT()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    print("Training on data...")
    for epoch in range(200):  # train for a few epochs
        total_loss = 0
        for obs, fut in dataset:  # obs: [obs_len, N, 2], fut: [pred_len, N, 2]
            last_obs = obs[-1]    # [N, 2] — positions at last observed frame
            target = fut[0]       # [N, 2] — first future positions

            edge_index = build_graph(last_obs)

            data = Data(x=last_obs, edge_index=edge_index)

            optimizer.zero_grad()
            output = model(data.x, data.edge_index)  # shape: [N, 2]
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()


        print(f"Epoch {epoch+1}, Loss: {total_loss:.6f}")
