import torch
import torch.nn as nn
import torch.optim as optim
import os
import cv2
import numpy as np

from torch_geometric.loader import DataLoader
from model import GAT, EncoderLSTM, DecoderLSTM
from utils import social_force_loss, map_penalty_loss
from dataset import PedestrianDataset
from torch.utils.data import Subset
from visualize_prediction import predict_and_visualize
from visualize_uncertainty import visualize_uncertainty

def train():
    # Load dataset
    dataset_path = "data/annotations/seq_hotel/world_coordinate_inter.csv"
    dataset = Subset(PedestrianDataset(dataset_path), range(200))
    video_folder = os.path.basename(os.path.dirname(dataset.dataset.path))
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Using device: {device}")

    # Load map and homography
    map_path = "data/annotations/seq_hotel/map.png"
    map_image = cv2.imread(map_path, cv2.IMREAD_GRAYSCALE)
    H = np.linalg.inv(np.loadtxt("data/annotations/seq_hotel/H.txt"))

    # Initialize models
    encoder = EncoderLSTM().to(device)
    gat = GAT(in_channels=34).to(device)
    decoder = DecoderLSTM(pred_len=12).to(device)

    optimizer = optim.Adam(
        list(encoder.parameters()) + list(gat.parameters()) + list(decoder.parameters()),
        lr=0.01
    )
    loss_fn = nn.MSELoss()

    for epoch in range(1, 201):
        encoder.train()
        gat.train()
        decoder.train()

        total_loss = total_pred_loss = total_reg_loss = total_map_loss = 0.0

        for batch in dataloader:
            obs = batch.obs_seq.to(device)         # [N, obs_len, 2]
            target = batch.y.to(device)            # [N, 12, 2]
            last_pos = obs[:, -1, :]               # [N, 2]

            encoded = encoder(obs)                 # [N, 32]
            node_input = torch.cat([last_pos, encoded], dim=1)  # [N, 34]
            context = gat(node_input, batch.edge_index.to(device))  # [N, 64]

            pred = decoder(context, last_pos)      # [N, 12, 2]

            pred_loss = loss_fn(pred, target)
            reg_loss = social_force_loss(pred.view(pred.size(0), -1))

            # Map penalty loss
            last_preds = pred[:, -1, :]  # Final predicted position [N, 2]
            map_loss = map_penalty_loss(last_preds, map_image, H)

            if not isinstance(reg_loss, torch.Tensor):
                reg_loss = torch.tensor(reg_loss, dtype=pred.dtype, device=pred.device)

            loss = pred_loss + 0.1 * reg_loss +  map_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_pred_loss += pred_loss.item()
            total_reg_loss += reg_loss.item()
            total_map_loss += map_loss.item()

        print(f"Epoch {epoch:3d} | Total: {total_loss:.2f} | Pred: {total_pred_loss:.2f} | "
              f"Reg: {total_reg_loss:.2f} | Map: {total_map_loss:.2f}")

    # Save model
    os.makedirs("saved_models", exist_ok=True)
    model_info = {
        "total_loss": total_loss,
        "epoch": epoch,
        "encoder_state_dict": encoder.state_dict(),
        "gat_state_dict": gat.state_dict(),
        "decoder_state_dict": decoder.state_dict(),
    }
    save_path = f"saved_models/model_{video_folder}_loss{total_loss:.4f}_UsingMap.pt"
    torch.save(model_info, save_path)
    print(f"✅ Saved model to {save_path}")

    # Visualize predictions and uncertainty
    predict_and_visualize(gat, encoder, decoder, dataset, sample_index=0)
    visualize_uncertainty(gat, encoder, decoder, dataset, sample_index=0)


if __name__ == "__main__":
    print("🚀 Training started...")
    train()
