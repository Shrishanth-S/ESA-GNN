import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from model import GAT, EncoderLSTM
from utils import social_force_loss
from dataset import PedestrianDataset

def train():
    # 1. Dataset and DataLoader
    dataset = PedestrianDataset("data/annotations/zara01/world_coordinate_inter.csv")
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    print("Loaded dataset with", len(dataset), "samples")

    # 2. Models
    model = GAT(in_channels=32)
    encoder = EncoderLSTM()

    # 3. Optimizer and Loss
    optimizer = optim.Adam(list(model.parameters()) + list(encoder.parameters()), lr=0.01)
    loss_fn = nn.MSELoss()

    # 4. Training loop
    print("Training on data...")
    for epoch in range(30):
        total_loss = 0.0
        total_pred_loss = 0.0
        total_reg_loss = 0.0

        for batch in dataloader:
            # batch is a batch of `Data` objects, auto-collated by PyG
            obs = batch.obs_seq    # shape: [N_total, obs_len, 2]
            target = batch.y       # shape: [N_total, 2]

            # Encode history → node features
            node_features = encoder(obs)  # [N_total, hidden_size]

            optimizer.zero_grad()

            # GAT forward
            out = model(node_features, batch.edge_index)

            # Loss
            pred_loss = loss_fn(out, target)
            reg_loss = social_force_loss(out)
            loss = pred_loss + 0.1 * reg_loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_pred_loss += pred_loss.item()
            total_reg_loss += reg_loss.item()

        print(f"Epoch {epoch+1:2d} | Total Loss: {total_loss:.2f} | "
              f"Pred Loss: {total_pred_loss:.2f} | Reg Loss: {total_reg_loss:.2f}")
