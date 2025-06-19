from scipy.spatial.distance import cdist
import numpy as np
import torch
import cv2

def build_graph(positions, threshold=2.0):
    """
    Constructs edges between pedestrians closer than threshold.
    """
    if isinstance(positions, torch.Tensor):
        positions = positions.cpu().numpy()

    dist = cdist(positions, positions)  # [N, N]
    adj = (dist < threshold).astype(int)
    np.fill_diagonal(adj, 0)
    edge_index = np.array(np.nonzero(adj), dtype=np.int64)
    return torch.tensor(edge_index, dtype=torch.long)


def social_force_loss(positions, min_dist=0.5):
    """
    Penalizes pedestrians getting too close to each other.
    """
    N = positions.size(0)
    if N < 2:
        return torch.tensor(0.0, dtype=positions.dtype, device=positions.device)

    loss = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            dist = torch.norm(positions[i] - positions[j])
            if dist < min_dist:
                loss += (min_dist - dist) ** 2

    return loss / N


def world_to_pixel(world_coords, H, map_shape):
    """
    Convert [N, 2] world coords to pixel coords using homography H.
    Returns integer pixel coordinates: [N, 2]
    """
    N = world_coords.shape[0]
    world_homo = np.hstack([world_coords, np.ones((N, 1))])  # [N, 3]
    pixel_coords = (H @ world_homo.T).T  # [N, 3]
    pixel_coords = pixel_coords[:, :2] / pixel_coords[:, 2:]  # [N, 2]

    # Swap x and y axes and vertically flip
    x_t, y_t = pixel_coords[:, 0], pixel_coords[:, 1]
    pixel_x = y_t
    pixel_y = x_t

    return np.round(np.stack([pixel_x, pixel_y], axis=1)).astype(int)  # [N, 2]

def map_penalty_loss(predicted_positions, map_image, homography_matrix, penalty_value=5.0, dilation_radius=3):
    """
    Penalize predicted positions that fall near or inside non-walkable regions (white pixels).

    Args:
        predicted_positions: Tensor [N, 2] in world coordinates.
        map_image: np.array, grayscale map (0 = walkable, 255 = obstacle).
        homography_matrix: 3x3 numpy array to convert world → pixel.
        penalty_value: Penalty for each invalid prediction.
        dilation_radius: Radius (in pixels) to dilate non-walkable areas.

    Returns:
        Scalar tensor loss
    """
    device = predicted_positions.device

    # Convert predicted world positions to pixel coordinates
    pred_np = predicted_positions.detach().cpu().numpy()  # [N, 2]
    pixel_coords = world_to_pixel(pred_np, homography_matrix, map_image.shape)  # [N, 2]

    # Clamp pixel coordinates to stay inside image
    pixel_coords[:, 0] = np.clip(pixel_coords[:, 0], 0, map_image.shape[1] - 1)
    pixel_coords[:, 1] = np.clip(pixel_coords[:, 1], 0, map_image.shape[0] - 1)

    # 1️⃣ Dilate white regions to create a "buffer zone"
    kernel = np.ones((2 * dilation_radius + 1, 2 * dilation_radius + 1), np.uint8)
    dilated_map = cv2.dilate((map_image > 127).astype(np.uint8), kernel)

    # 2️⃣ Check if any predicted point lies in the danger zone
    penalty = 0.0
    for x, y in pixel_coords:
        if dilated_map[y, x]:  # 1 if inside white zone or close to it
            penalty += penalty_value

    return torch.tensor(penalty, dtype=torch.float32, device=device) / predicted_positions.size(0)

