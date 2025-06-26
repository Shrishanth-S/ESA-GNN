import torch
import matplotlib.pyplot as plt
import time
import numpy as np
import cv2
import math

# === DRONE CAMERA PARAMETERS ===
DRONE_LAT = 13.009162
DRONE_LON = 74.795902
ALTITUDE = 10  # meters
FOV_DEG = 84
IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720

# === UTILS ===
def gps_to_pixel(lat, lon):
    """
    Convert GPS coordinates back to pixel coordinates using camera geometry.
    """
    # Earth model
    a = 6378137.0
    f = 1 / 298.257223563
    lat_rad = math.radians(DRONE_LAT)
    sin_lat = math.sin(lat_rad)
    cos_lat = math.cos(lat_rad)

    N = a / math.sqrt(1 - f * (2 - f) * sin_lat**2)

    # Delta in radians
    dlat = math.radians(lat - DRONE_LAT)
    dlon = math.radians(lon - DRONE_LON)

    dy = -dlat * a  # north-south offset in meters
    dx = dlon * N * cos_lat  # east-west offset in meters

    # Convert to pixel offsets
    aspect_ratio = IMAGE_WIDTH / IMAGE_HEIGHT
    fov_h_rad = math.radians(FOV_DEG)
    fov_v_rad = math.atan(math.tan(fov_h_rad / 2) / aspect_ratio) * 2

    x_angle = math.atan(dx / ALTITUDE)
    y_angle = math.atan(dy / ALTITUDE)

    x_pixel = IMAGE_WIDTH / 2 + (x_angle / (fov_h_rad / 2)) * (IMAGE_WIDTH / 2)
    y_pixel = IMAGE_HEIGHT / 2 + (y_angle / (fov_v_rad / 2)) * (IMAGE_HEIGHT / 2)

    return int(x_pixel), int(y_pixel)

def meters_to_gps(x_meter, y_meter):
    """
    Convert world coordinates in meters back to GPS.
    """
    R = 6378137
    lat1_rad = math.radians(DRONE_LAT)
    dlat = y_meter / R
    dlon = x_meter / (R * math.cos(lat1_rad))

    lat = DRONE_LAT + math.degrees(dlat)
    lon = DRONE_LON + math.degrees(dlon)
    return lat, lon

# === MAIN FUNCTION ===
def predict_and_visualize(gat, encoder, decoder, dataset, sample_index, map_path="manual_mask.png"):
    device = next(gat.parameters()).device
    gat.eval()
    encoder.eval()
    decoder.eval()

    with torch.no_grad():
        data = dataset[sample_index]

        obs = data.obs_seq.to(device)       # [N, obs_len, 2]  ← world meters
        true_fut = data.y.to(device)        # [N, pred_len, 2]
        last_pos = obs[:, -1, :]
        edge_index = data.edge_index.to(device)

        # === Inference
        start_time = time.time()
        encoded = encoder(obs)
        node_input = torch.cat([last_pos, encoded], dim=1)
        context = gat(node_input, edge_index)
        pred = decoder(context, last_pos)
        end_time = time.time()
        print(f"\n⏱ Inference Time: {(end_time - start_time) * 1000:.2f} ms")

        # Convert to numpy
        obs_np = obs.cpu().numpy()
        true_np = true_fut.cpu().numpy()
        pred_np = pred.cpu().numpy()

        # === Load map
        map_img = cv2.imread(map_path)
        if map_img is None:
            raise FileNotFoundError(f"❌ Map image not found at {map_path}")
        map_img = cv2.resize(map_img, (IMAGE_WIDTH, IMAGE_HEIGHT))

        # === Project each point to pixel
        def world_to_pixel_coords(coords):  # coords: [N, T, 2]
            pixels = []
            for traj in coords:
                traj_pix = []
                for x_meter, y_meter in traj:
                    lat, lon = meters_to_gps(x_meter, y_meter)
                    u, v = gps_to_pixel(lat, lon)
                    traj_pix.append((u, v))
                pixels.append(traj_pix)
            return np.array(pixels)

        obs_pix = world_to_pixel_coords(obs_np)
        true_pix = world_to_pixel_coords(true_np)
        pred_pix = world_to_pixel_coords(pred_np)

        # === Overlay on map
        overlay = map_img.copy()
        for i in range(obs_pix.shape[0]):
            for (u, v) in obs_pix[i]:
                cv2.circle(overlay, (u, v), 4, (255, 0, 0), -1)  # 🔵 Observed
            for (u, v) in true_pix[i]:
                cv2.circle(overlay, (u, v), 4, (0, 255, 0), -1)  # 🟢 True
            for (u, v) in pred_pix[i]:
                cv2.circle(overlay, (u, v), 4, (0, 255, 255), -1)  # 🟡 Predicted

        # === Display
        plt.figure(figsize=(12, 8))
        plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        plt.title("Trajectory Prediction on Map")
        plt.axis("off")
        plt.tight_layout()
        plt.show()
