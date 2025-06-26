from scipy.spatial.distance import cdist
import numpy as np
import torch
import cv2
import math

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
    Penalizes pedestrians getting too close to each other using vectorized computation.
    positions: Tensor of shape [N, 2]
    """
    N = positions.size(0)
    if N < 2:
        return torch.tensor(0.0, dtype=positions.dtype, device=positions.device)

    # Compute pairwise distances
    diff = positions.unsqueeze(1) - positions.unsqueeze(0)  # [N, N, 2]
    dist = torch.norm(diff, dim=2)  # [N, N]

    # Mask out self-distances safely
    mask = torch.eye(N, device=positions.device).bool()
    dist = dist.masked_fill(mask, float('inf'))  # ✅ No in-place

    # Penalize distances below threshold
    close_mask = (dist < min_dist)
    penalty = ((min_dist - dist[close_mask]) ** 2).sum()

    return penalty / N


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

def world_to_pixel_univ_zara(world_coords, H, map_shape):
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
    pixel_x = x_t
    pixel_y = y_t

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

def map_penalty_loss_univ_zara(predicted_positions, map_image, homography_matrix, penalty_value=5.0, dilation_radius=3):
    device = predicted_positions.device

    # Convert predicted world positions to pixel coordinates
    pred_np = predicted_positions.detach().cpu().numpy()  # [N, 2]
    pixel_coords = world_to_pixel_univ_zara(pred_np, homography_matrix, map_image.shape)  # [N, 2]

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

def compute_ade_fde(pred, target):
    """
    Compute Average Displacement Error and Final Displacement Error.
    Args:
        pred: [N, T, 2]
        target: [N, T, 2]
    Returns:
        (ade, fde)
    """
    ade = torch.norm(pred - target, dim=2).mean().item()
    fde = torch.norm(pred[:, -1, :] - target[:, -1, :], dim=1).mean().item()
    return ade, fde

# === Earth / Drone Parameters ===
DRONE_LAT = 13.009162
DRONE_LON = 74.795902
ALTITUDE = 10  # meters
FOV_DEG = 84
IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720

def world_to_gps(x, y):
    """
    Convert (x, y) world in meters to (lat, lon) using WGS84 ellipsoid.
    """
    R = 6378137  # Earth's radius in meters
    lat1_rad = math.radians(DRONE_LAT)

    dlat = y / R
    dlon = x / (R * math.cos(lat1_rad))

    lat = DRONE_LAT + math.degrees(dlat)
    lon = DRONE_LON + math.degrees(dlon)
    return lat, lon

def gps_to_pixel(lat, lon):
    """
    Convert GPS coordinates to pixel coordinates in drone camera view.
    """
    a = 6378137.0
    f = 1 / 298.257223563
    lat_rad = math.radians(DRONE_LAT)
    sin_lat = math.sin(lat_rad)
    cos_lat = math.cos(lat_rad)
    N = a / math.sqrt(1 - f * (2 - f) * sin_lat**2)

    dlat = math.radians(lat - DRONE_LAT)
    dlon = math.radians(lon - DRONE_LON)

    dy = -dlat * a
    dx = dlon * N * cos_lat

    # Convert to pixel coordinates
    aspect_ratio = IMAGE_WIDTH / IMAGE_HEIGHT
    fov_h_rad = math.radians(FOV_DEG)
    fov_v_rad = math.atan(math.tan(fov_h_rad / 2) / aspect_ratio) * 2

    x_angle = math.atan(dx / ALTITUDE)
    y_angle = math.atan(dy / ALTITUDE)

    u = IMAGE_WIDTH / 2 + (x_angle / (fov_h_rad / 2)) * (IMAGE_WIDTH / 2)
    v = IMAGE_HEIGHT / 2 + (y_angle / (fov_v_rad / 2)) * (IMAGE_HEIGHT / 2)

    return int(u), int(v)

def world_to_pixel_via_gps(world_coords):
    """
    Convert [N, 2] world coords (meters) → GPS → pixel coords.
    Returns [N, 2] pixel coords
    """
    result = []
    for x, y in world_coords:
        lat, lon = world_to_gps(x, y)
        u, v = gps_to_pixel(lat, lon)
        result.append([u, v])
    return np.array(result)

def map_penalty_loss_via_gps(predicted_positions, map_image, penalty_value=5.0, dilation_radius=3):
    """
    Penalty using GPS → pixel mapping, without homography.
    """
    device = predicted_positions.device
    pred_np = predicted_positions.detach().cpu().numpy()  # [N, 2]
    pixel_coords = world_to_pixel_via_gps(pred_np)

    pixel_coords[:, 0] = np.clip(pixel_coords[:, 0], 0, map_image.shape[1] - 1)
    pixel_coords[:, 1] = np.clip(pixel_coords[:, 1], 0, map_image.shape[0] - 1)

    kernel = np.ones((2 * dilation_radius + 1, 2 * dilation_radius + 1), np.uint8)
    dilated_map = cv2.dilate((map_image > 127).astype(np.uint8), kernel)

    penalty = 0.0
    for x, y in pixel_coords:
        if dilated_map[y, x]:
            penalty += penalty_value

    return torch.tensor(penalty, dtype=torch.float32, device=device) / predicted_positions.size(0)





