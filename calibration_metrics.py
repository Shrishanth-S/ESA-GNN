import numpy as np
import torch

def evaluate_calibration(preds, true_fut, stds, thresholds=[0.5, 1.0, 1.5, 2.0]):
    """
    Evaluate how many ground-truth points fall within predicted confidence intervals.

    Args:
        preds: [N, T, 2] predicted means for T time steps
        true_fut: [N, T, 2] ground truth
        stds: [N, T, 2] predicted stddevs
        thresholds: how many std devs to check coverage

    Returns:
        A dictionary mapping threshold to empirical coverage
    """
    errors = torch.abs(preds - true_fut)  # [N, T, 2]
    results = {}

    for t in thresholds:
        within_x = (errors[:,:,0] <= t * stds[:,:,0])
        within_y = (errors[:,:,1] <= t * stds[:,:,1])
        covered = (within_x & within_y).float()
        coverage = covered.mean().item()
        results[f"{t}σ"] = coverage

    return results
