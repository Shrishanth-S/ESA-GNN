import torch
import numpy as np
import matplotlib.pyplot as plt

def compute_ece(preds, targets, stds, n_bins=10):
    """
    Compute Expected Calibration Error (ECE).
    preds: [N, T, 2] or [N, 2]
    targets: [N, 2]
    stds: [N, 2]
    """
    preds = preds.reshape(-1, 2)
    targets = targets.reshape(-1, 2)
    stds = stds.reshape(-1, 2)

    errors = torch.norm(preds - targets, dim=1)
    conf = torch.norm(stds, dim=1)

    bins = torch.linspace(0, conf.max(), steps=n_bins + 1)
    ece = 0.0
    total = len(errors)

    for i in range(n_bins):
        in_bin = (conf >= bins[i]) & (conf < bins[i + 1])
        prop = in_bin.float().mean()
        if prop > 0:
            acc_in_bin = errors[in_bin].mean()
            conf_in_bin = conf[in_bin].mean()
            ece += torch.abs(conf_in_bin - acc_in_bin) * prop

    return ece.item()

def compute_mce(preds, targets, stds, n_bins=10):
    """
    Maximum Calibration Error (MCE)
    """
    preds = preds.reshape(-1, 2)
    targets = targets.reshape(-1, 2)
    stds = stds.reshape(-1, 2)

    errors = torch.norm(preds - targets, dim=1)
    conf = torch.norm(stds, dim=1)

    bins = torch.linspace(0, conf.max(), steps=n_bins + 1)
    mce = 0.0

    for i in range(n_bins):
        in_bin = (conf >= bins[i]) & (conf < bins[i + 1])
        if in_bin.any():
            acc_in_bin = errors[in_bin].mean()
            conf_in_bin = conf[in_bin].mean()
            mce = max(mce, torch.abs(acc_in_bin - conf_in_bin).item())

    return mce

def compute_brier_score(pred_mean, true, std):
    """
    Brier Score (simplified for Gaussian model).
    Lower is better.
    """
    mse = (pred_mean - true).pow(2)
    var = std.pow(2)
    brier = mse + var
    return brier.mean().item()

def plot_reliability_diagram(preds, targets, stds, n_bins=10, save=False, fname="reliability_diagram.png"):
    """
    Plots a reliability diagram comparing confidence (std) vs actual error.
    """
    preds = preds.reshape(-1, 2)
    targets = targets.reshape(-1, 2)
    stds = stds.reshape(-1, 2)

    errors = torch.norm(preds - targets, dim=1).detach().cpu().numpy()
    confs = torch.norm(stds, dim=1).detach().cpu().numpy()


    bins = np.linspace(0, np.max(confs), n_bins + 1)
    bin_centers = 0.5 * (bins[1:] + bins[:-1])
    accs, conf_avgs = [], []

    for i in range(n_bins):
        mask = (confs >= bins[i]) & (confs < bins[i + 1])
        if np.sum(mask) > 0:
            accs.append(errors[mask].mean())
            conf_avgs.append(confs[mask].mean())
        else:
            accs.append(0)
            conf_avgs.append(0)

    plt.figure(figsize=(6, 6))
    plt.plot(conf_avgs, accs, 'o-', label="Model")
    plt.plot([0, max(confs)], [0, max(confs)], 'k--', label="Perfect Calibration")
    plt.xlabel("Predicted Uncertainty (Norm of Std)")
    plt.ylabel("Actual Error")
    plt.title("Reliability Diagram")
    plt.grid(True)
    plt.legend()

    if save:
        plt.savefig(fname)
        print(f"✅ Saved reliability diagram to {fname}")
    plt.show()
