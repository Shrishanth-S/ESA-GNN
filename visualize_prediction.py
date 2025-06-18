import torch
import matplotlib.pyplot as plt
from utils import build_graph

def predict_and_visualize(gat, encoder, decoder, dataset, sample_index=0):
    gat.eval()
    encoder.eval()
    decoder.eval()

    with torch.no_grad():
        data = dataset[sample_index]
        obs = data.obs_seq            # [N, obs_len, 2]
        true_fut = data.y             # [N, 12, 2]
        last_pos = obs[:, -1, :]      # [N, 2]

        encoded = encoder(obs)
        node_input = torch.cat([last_pos, encoded], dim=1)
        edge_index = build_graph(last_pos)
        context = gat(node_input, edge_index)

        pred = decoder(context, last_pos)  # [N, 12, 2]

        obs_np = obs.cpu().numpy()
        true_np = true_fut.cpu().numpy()
        pred_np = pred.cpu().numpy()

        # Print all points
        print("\n📊 Future Trajectory Comparison (Predicted vs Ground Truth):")
        for i, (pred_traj, true_traj) in enumerate(zip(pred_np, true_np)):
            print(f"\n👤 Pedestrian {i}:")
            for t, (pred_point, true_point) in enumerate(zip(pred_traj, true_traj)):
                print(f"  t+{t+1}: Pred: (x={pred_point[0]:.2f}, y={pred_point[1]:.2f})  |  True: (x={true_point[0]:.2f}, y={true_point[1]:.2f})")

        # Plot
        plt.figure(figsize=(8, 6))
        for i in range(obs_np.shape[0]):
            plt.plot(obs_np[i, :, 0], obs_np[i, :, 1], 'bo-', label='Observed' if i == 0 else "")
            plt.plot(true_np[i, :, 0], true_np[i, :, 1], 'go--', label='True Future' if i == 0 else "")
            plt.plot(pred_np[i, :, 0], pred_np[i, :, 1], 'r.-', label='Predicted Future' if i == 0 else "")

        plt.title("12-Step Future Trajectory Prediction")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend()
        plt.grid(True)
        plt.show()
        plt.close()
