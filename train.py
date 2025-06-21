import torch
import torch.nn as nn
import torch.optim as optim
import os
import cv2
import numpy as np

from torch_geometric.loader import DataLoader
from model import GAT, EncoderLSTM, DecoderLSTM
from utils import social_force_loss, map_penalty_loss, compute_ade_fde
from dataset import PedestrianDataset

from torch.utils.data import Subset
from torch.utils.data import random_split
from visualize_prediction import predict_and_visualize
from visualize_uncertainty import visualize_uncertainty


def train():
    # === Load Full Dataset ===
    dataset_path = "data/annotations/seq_hotel/world_coordinate_inter.csv"
    full_dataset = PedestrianDataset(dataset_path)
    video_folder = os.path.basename(os.path.dirname(full_dataset.path))

    # === Split Train/Test (80/20) ===
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
    print(f"📊 Dataset split: {train_size} train, {test_size} test")

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64)

    # === Device ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Using device: {device}")

    # === Map & Homography ===
    map_image = cv2.imread("data/annotations/seq_hotel/map.png", cv2.IMREAD_GRAYSCALE)
    H = np.linalg.inv(np.loadtxt("data/annotations/seq_hotel/H.txt"))

    # === Model Init ===
    encoder = EncoderLSTM().to(device)
    gat = GAT(in_channels=34).to(device)
    decoder = DecoderLSTM(pred_len=12).to(device)

    optimizer = optim.Adam(
        list(encoder.parameters()) + list(gat.parameters()) + list(decoder.parameters()),
        lr=0.001
    )
    
    loss_fn = nn.MSELoss()

    for epoch in range(1, 101):
        encoder.train()
        gat.train()
        decoder.train()

        total_loss = total_pred_loss = total_reg_loss = total_map_loss = 0.0

        for batch in train_loader:
            obs = batch.obs_seq.to(device)
            target = batch.y.to(device)
            last_pos = obs[:, -1, :]

            encoded = encoder(obs)
            node_input = torch.cat([last_pos, encoded], dim=1)
            context = gat(node_input, batch.edge_index.to(device))
            pred = decoder(context, last_pos)

            pred_loss = loss_fn(pred, target)
            reg_loss = social_force_loss(pred.view(pred.size(0), -1))
            map_loss = map_penalty_loss(pred[:, -1, :], map_image, H)

            if not isinstance(reg_loss, torch.Tensor):
                reg_loss = torch.tensor(reg_loss, dtype=pred.dtype, device=pred.device)

            loss = pred_loss + 0.1 * reg_loss + map_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_pred_loss += pred_loss.item()
            total_reg_loss += reg_loss.item()
            total_map_loss += map_loss.item()

        # === Evaluation on Test Set ===
        encoder.eval()
        gat.eval()
        decoder.eval()
        all_preds, all_targets = [], []

        with torch.no_grad():
            for batch in test_loader:
                obs = batch.obs_seq.to(device)
                target = batch.y.to(device)
                last_pos = obs[:, -1, :]
                encoded = encoder(obs)
                node_input = torch.cat([last_pos, encoded], dim=1)
                context = gat(node_input, batch.edge_index.to(device))
                pred = decoder(context, last_pos)
                all_preds.append(pred)
                all_targets.append(target)

        preds = torch.cat(all_preds, dim=0)
        targets = torch.cat(all_targets, dim=0)
        ade, fde = compute_ade_fde(preds, targets)

        print(f"Epoch {epoch:3d} | Total: {total_loss:.2f} | Pred: {total_pred_loss:.2f} | "
              f"Reg: {total_reg_loss:.2f} | Map: {total_map_loss:.2f} | "
              f"ADE: {ade:.4f} | FDE: {fde:.4f}")

        # === Save Best Model ===
    os.makedirs("saved_models", exist_ok=True)
    save_path = f"saved_models/model_{video_folder}_epoch{epoch}_ade{ade:.4f}_fde{fde:.4f}.pt"
    torch.save({
            "epoch": epoch,
            "ade": ade,
            "fde": fde,
            "encoder_state_dict": encoder.state_dict(),
            "gat_state_dict": gat.state_dict(),
            "decoder_state_dict": decoder.state_dict(),
        }, save_path)

    # === Final Visualization on Test ===
    predict_and_visualize(gat, encoder, decoder, test_dataset, sample_index=0)
    visualize_uncertainty(gat, encoder, decoder, test_dataset, sample_index=0)

