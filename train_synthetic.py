import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim
from model import GAT, EncoderLSTM
from utils import build_graph, social_force_loss
from synthetic_dataset import generate_synthetic_trajectories

def visualize(obs, fut, pred):
    obs = obs.permute(1, 0, 2).numpy()  # [N, obs_len, 2]
    fut = fut.permute(1, 0, 2).numpy()  # [N, pred_len, 2]
    pred = pred.detach().numpy()       # [N, 2]

    for i in range(obs.shape[0]):
        plt.plot(obs[i, :, 0], obs[i, :, 1], 'bo-', label='Observed' if i == 0 else "")
        plt.plot(fut[i, :, 0], fut[i, :, 1], 'go--', label='True Future' if i == 0 else "")
        plt.plot([obs[i, -1, 0], pred[i, 0]], [obs[i, -1, 1], pred[i, 1]], 'rx-', label='Predicted' if i == 0 else "")

    plt.legend()
    plt.title("Trajectory Prediction (Synthetic)")
    plt.axis("equal")
    plt.grid()
    plt.show()


def train_synthetic():
    obs_len, pred_len = 8, 12
    model = GAT(in_channels=32)
    encoder = EncoderLSTM()

    optimizer = optim.Adam(list(model.parameters()) + list(encoder.parameters()), lr=0.01)
    loss_fn = nn.MSELoss()

    print("Training on synthetic data...")

    for epoch in range(200):
        total_loss = 0
        obs, fut = generate_synthetic_trajectories(num_agents=5, obs_len=obs_len, pred_len=pred_len)
        target = fut[0]              # [N, 2]
        last_pos = obs[-1]          # [N, 2]

        # Encode motion with LSTM
        node_input = obs.permute(1, 0, 2)  # [N, obs_len, 2]
        node_features = encoder(node_input)  # [N, hidden]

        # Build graph from last positions
        edge_index = build_graph(last_pos)

        # Forward pass
        output = model(node_features, edge_index)  # [N, 2]

        pred_loss = loss_fn(output, target)
        reg_loss = social_force_loss(output)
        loss = pred_loss + 0.1 * reg_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        print(f"Epoch {epoch+1:2d} | Total Loss: {total_loss:.4f} | Pred: {pred_loss:.4f} | Reg: {reg_loss:.4f}")
    visualize(obs, fut, output)     

train_synthetic()


