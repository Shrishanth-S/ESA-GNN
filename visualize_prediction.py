import torch
import matplotlib.pyplot as plt
import cv2
import numpy as np
from utils import build_graph, world_to_pixel
from pathlib import Path

def predict_and_visualize(gat, encoder, decoder, dataset, sample_index=0):
    device = next(gat.parameters()).device  # Automatically get model's device
    gat.eval()
    encoder.eval()
    decoder.eval()

    with torch.no_grad():
        data = dataset[sample_index]

        # 🚀 Move tensors to device
        obs = data.obs_seq.to(device)       # [N, obs_len, 2]
        true_fut = data.y.to(device)        # [N, 12, 2]
        last_pos = obs[:, -1, :]            # [N, 2]

        # === Model Forward ===
        encoded = encoder(obs)
        node_input = torch.cat([last_pos, encoded], dim=1)
        edge_index = data.edge_index.to(device)
        context = gat(node_input, edge_index)
        pred = decoder(context, last_pos)  # [N, 12, 2]

        # === Move back to CPU for visualization
        obs_np = obs.cpu().numpy()
        true_np = true_fut.cpu().numpy()
        pred_np = pred.cpu().numpy()

        # === Load map + homography ===
        H = np.linalg.inv(np.loadtxt("data/annotations/seq_hotel/H.txt"))
        map_img = cv2.imread("data/annotations/seq_hotel/map.png", cv2.IMREAD_GRAYSCALE)

        def to_pixel(trajs):
            N, T, _ = trajs.shape
            flat = trajs.reshape(-1, 2)
            pixels = world_to_pixel(flat, H, map_img.shape)  # [N*T, 2]
            return pixels.reshape(N, T, 2)

        obs_pix = to_pixel(obs_np)
        true_pix = to_pixel(true_np)
        pred_pix = to_pixel(pred_np)

        # === Print comparison ===
        print("\n📊 Future Trajectory Comparison (Predicted vs Ground Truth):")
        for i in range(pred_np.shape[0]):
            print(f"\n👤 Pedestrian {i}:")
            for t in range(12):
                px, py = pred_np[i, t]
                tx, ty = true_np[i, t]
                print(f"  t+{t+1}: Pred: (x={px:.2f}, y={py:.2f})  |  True: (x={tx:.2f}, y={ty:.2f})")

        # === Plot on map ===
        plt.figure(figsize=(10, 8))
        plt.imshow(map_img, cmap='gray')

        for i in range(obs_pix.shape[0]):
            plt.plot(obs_pix[i, :, 0], obs_pix[i, :, 1], 'bo-', label='Observed' if i == 0 else "")
            plt.plot(true_pix[i, :, 0], true_pix[i, :, 1], 'go--', label='True Future' if i == 0 else "")
            plt.plot(pred_pix[i, :, 0], pred_pix[i, :, 1], 'y.-', label='Predicted Future' if i == 0 else "")

        plt.title("Trajectory Prediction on Map (Pixel Coordinates)")
        plt.axis('equal')
        plt.legend(loc="upper right")
        plt.tight_layout()
        plt.grid(True)
        plt.show()
        plt.close()
