import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from model import GAT, EncoderLSTM, DecoderLSTM
from utils import social_force_loss
from dataset import PedestrianDataset
from torch.utils.data import Subset
from visualize_prediction import predict_and_visualize

def train():
    full_dataset = PedestrianDataset("data/annotations/univ/world_coordinate_inter.csv")
    dataset = Subset(full_dataset, range(200))
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    print("Loaded dataset with", len(dataset), "samples")

    encoder = EncoderLSTM()
    gat = GAT(in_channels=34)
    decoder = DecoderLSTM(pred_len=12)

    optimizer = optim.Adam(
        list(encoder.parameters()) + 
        list(gat.parameters()) + 
        list(decoder.parameters()), 
        lr=0.01
    )

    loss_fn = nn.MSELoss()

    for epoch in range(1, 101):
        total_loss = 0.0
        total_pred_loss = 0.0
        total_reg_loss = 0.0

        for batch in dataloader:
            obs = batch.obs_seq              # [N, obs_len, 2]
            target = batch.y                 # [N, 12, 2]
            last_pos = obs[:, -1, :]         # [N, 2]

            encoded = encoder(obs)           # [N, 32]
            node_input = torch.cat([last_pos, encoded], dim=1)  # [N, 34]
            context = gat(node_input, batch.edge_index)         # [N, 64]
            pred = decoder(context, last_pos)                   # [N, 12, 2]

            pred_loss = loss_fn(pred, target)
            reg_loss = social_force_loss(pred.view(pred.size(0), -1))

            # Ensure reg_loss is a tensor
            if not isinstance(reg_loss, torch.Tensor):
                reg_loss = torch.tensor(reg_loss, dtype=pred.dtype, device=pred.device)

            loss = pred_loss + 0.1 * reg_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_pred_loss += pred_loss.item()
            total_reg_loss += reg_loss.item()

        print(f"Epoch {epoch:3d} | Total Loss: {total_loss:.2f} | "
              f"Pred Loss: {total_pred_loss:.2f} | Reg Loss: {total_reg_loss:.2f}")

    predict_and_visualize(gat, encoder, decoder, dataset, sample_index=0)
