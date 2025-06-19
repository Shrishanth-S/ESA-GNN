import matplotlib.pyplot as plt
import torch
from uncertainty_utils import predict_with_uncertainty

def visualize_uncertainty(
    gat, encoder, decoder, dataset,
    sample_index, T, save=False, alpha=5
):
    """
    Visualizes uncertainty-aware trajectory prediction using GPU if available.

    Args:
        gat, encoder, decoder: Trained models.
        dataset: Dataset object.
        sample_index: Index of sample to visualize.
        T: Number of Monte Carlo samples (dropout passes).
        save: If True, saves the plot as PNG.
        alpha: Scaling factor for predicted std dev (for calibration).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Move models to device
    gat = gat.to(device).eval()
    encoder = encoder.to(device).eval()
    decoder = decoder.to(device).eval()

    # Load data and move to device
    data = dataset[sample_index]
    obs = data.obs_seq.to(device)           # [N, obs_len, 2]
    true_fut = data.y.to(device)            # [N, 12, 2]
    last_pos = obs[:, -1, :]                # [N, 2]

    # Predict T stochastic samples
    preds = predict_with_uncertainty(gat, encoder, decoder, data, T=T)  # [T, N, 12, 2]

    # Mean and std of predictions
    pred_mean = preds.mean(dim=0)           # [N, 12, 2]
    pred_std = preds.std(dim=0)             # [N, 12, 2]
    scaled_std = pred_std * alpha           # Apply calibration scaling

    # Convert to NumPy (after moving to CPU and detaching)
    obs_np = obs.cpu().numpy()
    true_np = true_fut.cpu().numpy()
    mean_np = pred_mean.cpu().detach().numpy()
    std_np = scaled_std.cpu().detach().numpy()
    preds_np = preds.cpu().detach().numpy()  # [T, N, 12, 2]

    # Print predictions
    print("\n📊 Future Prediction with Uncertainty:")
    for i in range(mean_np.shape[0]):
        print(f"\n👤 Pedestrian {i}")
        for t in range(12):
            print(f"  t+{t+1}: Pred=({mean_np[i,t,0]:.2f},{mean_np[i,t,1]:.2f}) ± "
                  f"({std_np[i,t,0]:.2f},{std_np[i,t,1]:.2f}) | "
                  f"True=({true_np[i,t,0]:.2f},{true_np[i,t,1]:.2f})")

    # Try calibration metrics
    try:
        from calibration_metrics import evaluate_calibration
        metrics = evaluate_calibration(pred_mean.cpu(), true_fut.cpu(), scaled_std.cpu())
        print("\n📏 Calibration:")
        for k, v in metrics.items():
            print(f"  {k} → {v*100:.2f}%")
    except ImportError:
        print("\n⚠️ calibration_metrics not found, skipping calibration evaluation.")

    # Plotting
    plt.figure(figsize=(8, 6))
    for i in range(mean_np.shape[0]):
        plt.plot(obs_np[i,:,0], obs_np[i,:,1], 'bo-', label='Observed' if i == 0 else "")
        plt.plot(true_np[i,:,0], true_np[i,:,1], 'go--', label='Ground Truth' if i == 0 else "")
        for t in range(T):
            plt.plot(preds_np[t,i,:,0], preds_np[t,i,:,1], 'r-', alpha=0.2, linewidth=1,
                     label='Sampled Prediction' if (i == 0 and t == 0) else "")
        plt.plot(mean_np[i,:,0], mean_np[i,:,1], 'y.-', label='Predicted Mean' if i == 0 else "")

    plt.title(f"Trajectory Prediction with Uncertainty (Sample {sample_index})")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis("equal")
    plt.grid(True)
    plt.legend(loc='upper right')
    plt.tight_layout()

    if save:
        fname = f"uncertainty_pred_sample{sample_index}.png"
        plt.savefig(fname)
        print(f"✅ Saved plot to: {fname}")

    plt.show()
    plt.close()
