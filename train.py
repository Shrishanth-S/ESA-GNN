import torch
import torch.nn as nn
import torch.optim as optim
import os
from torch_geometric.loader import DataLoader
from model import GAT, EncoderLSTM, DecoderLSTM
from utils import social_force_loss
from dataset import PedestrianDataset
from torch.utils.data import Subset
from visualize_prediction import predict_and_visualize
from visualize_uncertainty import visualize_uncertainty

def train():

    dataset = Subset(PedestrianDataset("data/annotations/zara01/world_coordinate_inter.csv"), range(200))
    video_folder = os.path.basename(os.path.dirname(dataset.dataset.path))

    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    encoder = EncoderLSTM()
    gat = GAT(in_channels=34)
    decoder = DecoderLSTM(pred_len=12)

    optimizer = optim.Adam(
        list(encoder.parameters()) + list(gat.parameters()) + list(decoder.parameters()),
        lr=0.01
    )
    loss_fn = nn.MSELoss()

    for epoch in range(1, 201):
        total_loss = total_pred_loss = total_reg_loss = 0.0

        for batch in dataloader:
            obs = batch.obs_seq              # [N, obs_len, 2]
            target = batch.y                 # [N, 12, 2]
            last_pos = obs[:, -1, :]         # [N, 2]

            encoded = encoder(obs)           # [N, 32]
            node_input = torch.cat([last_pos, encoded], dim=1)  # [N, 34]
            context = gat(node_input, batch.edge_index)         # [N, 64]

            pred = decoder(context, last_pos)  # [N, 12, 2]

            pred_loss = loss_fn(pred, target)
            reg_loss = social_force_loss(pred.view(pred.size(0), -1))
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
        
        
    model_info = {
    "total_loss": total_loss,
    "epoch": epoch,
    "encoder_state_dict": encoder.state_dict(),
    "gat_state_dict": gat.state_dict(),
    "decoder_state_dict": decoder.state_dict(),
}

        
    save_path = f"saved_models/model_{video_folder}_loss{total_loss:.4f}.pt"
    torch.save(model_info, save_path)
    print(f"✅ Saved model to {save_path}")

    predict_and_visualize(gat, encoder, decoder, dataset, sample_index=0)
    visualize_uncertainty(gat, encoder, decoder, dataset, sample_index=0)
