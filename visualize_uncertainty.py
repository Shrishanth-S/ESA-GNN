import matplotlib.pyplot as plt
import torch
from uncertainty_utils import predict_with_uncertainty

def visualize_uncertainty(gat, encoder, decoder, dataset, sample_index=0, T=100):
    # Set models to eval mode
    gat.eval()
    encoder.eval()
    decoder.eval()

    # Get one sample
    data = dataset[sample_index]
    obs = data.obs_seq               # [N, obs_len, 2]
    true_fut = data.y                # [N, 12, 2]
    last_pos = obs[:, -1, :]        # [N, 2]

    # Predict T stochastic samples
    preds = predict_with_uncertainty(gat, encoder, decoder, data, T=T)  # [T, N, 12, 2]

    # Mean and std
    pred_mean = preds.mean(dim=0)   # [N, 12, 2]
    pred_std = preds.std(dim=0)     # [N, 12, 2]

    # Convert to numpy
    obs_np = obs.detach().numpy()
    true_np = true_fut.detach().numpy()
    mean_np = pred_mean.detach().numpy()
    std_np = pred_std.detach().numpy()
    preds_np = preds.detach().numpy()  # [T, N, 12, 2]

    # 🖨️ Print prediction vs true
    print("\n📊 Future Prediction with Uncertainty:")
    for i in range(mean_np.shape[0]):
        print(f"\n👤 Pedestrian {i}")
        for t in range(12):
            print(f"  t+{t+1}: Pred=({mean_np[i,t,0]:.2f},{mean_np[i,t,1]:.2f}) ± ({std_np[i,t,0]:.2f},{std_np[i,t,1]:.2f}) | True=({true_np[i,t,0]:.2f},{true_np[i,t,1]:.2f})")

    # 📉 Plot
    plt.figure(figsize=(8, 6))
    for i in range(mean_np.shape[0]):
        plt.plot(obs_np[i,:,0], obs_np[i,:,1], 'bo-', label='Observed' if i==0 else "")
        plt.plot(true_np[i,:,0], true_np[i,:,1], 'go--', label='Ground Truth' if i==0 else "")

        # 🔁 Multiple sampled predictions
        for t in range(T):
            plt.plot(preds_np[t, i,:,0], preds_np[t, i,:,1], 'r-', alpha=0.35, linewidth=1, label='Sampled Prediction' if (i==0 and t==0) else "")

        # 🔴 Mean trajectory
        plt.plot(mean_np[i,:,0], mean_np[i,:,1], 'rx--', label='Predicted Mean' if i==0 else "")

    plt.title("Trajectory Prediction with Uncertainty")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.close()
