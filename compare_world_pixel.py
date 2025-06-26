import pandas as pd
import math
import cv2
import matplotlib.pyplot as plt

# ======= CONFIG =======
WORLD_CSV = "pedestrian_detector/world_coordinates/world_coordinates.csv"
PIXEL_CSV = "pedestrian_detector/pixel_coordinates/pixel_coordinates.csv"
MAP_IMAGE = "manual_mask.png"

DRONE_LAT = 13.009162
DRONE_LON = 74.795902
ALTITUDE = 10
FOV_DEG = 84
IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720

# ======= Conversion Functions =======
def meters_to_gps(dx, dy):
    R = 6378137
    dlat = dy / R
    dlon = dx / (R * math.cos(math.radians(DRONE_LAT)))
    lat = DRONE_LAT + math.degrees(dlat)
    lon = DRONE_LON + math.degrees(dlon)
    return lat, lon

def gps_to_pixel(lat, lon):
    a = 6378137.0
    f = 1 / 298.257223563
    lat_rad = math.radians(DRONE_LAT)
    sin_lat = math.sin(lat_rad)
    cos_lat = math.cos(lat_rad)

    N = a / math.sqrt(1 - f * (2 - f) * sin_lat ** 2)
    M = a * (1 - f * (2 - f)) / (1 - f * (2 - f) * sin_lat ** 2) ** 1.5

    dlat = math.radians(lat - DRONE_LAT)
    dlon = math.radians(lon - DRONE_LON)

    dy = -dlat * M
    dx = dlon * N * cos_lat

    fov_h_rad = math.radians(FOV_DEG)
    aspect_ratio = IMAGE_WIDTH / IMAGE_HEIGHT
    fov_v_rad = math.atan(math.tan(fov_h_rad / 2) / aspect_ratio) * 2

    x_angle = math.atan(dx / ALTITUDE)
    y_angle = math.atan(dy / ALTITUDE)

    x_norm = math.tan(x_angle) / math.tan(fov_h_rad / 2)
    y_norm = math.tan(y_angle) / math.tan(fov_v_rad / 2)

    x_pixel = int((x_norm * IMAGE_WIDTH / 2) + IMAGE_WIDTH / 2)
    y_pixel = int((y_norm * IMAGE_HEIGHT / 2) + IMAGE_HEIGHT / 2)
    return x_pixel, y_pixel

# ======= Load Data =======
df_pixel = pd.read_csv(PIXEL_CSV)
df_world = pd.read_csv(WORLD_CSV)

df_merge = pd.merge(df_pixel, df_world, on=["frame", "ped_id"], suffixes=('_orig', '_world'))

# ======= Filter Subset (Example: First 10 points or specific conditions) =======
selected_df = df_merge.iloc[:10]  # 🔁 You can filter with conditions like: df_merge[df_merge['frame'] == 50]

# ======= Load map image =======
img = cv2.imread(MAP_IMAGE)
if img is None:
    raise FileNotFoundError(f"Map image not found: {MAP_IMAGE}")
img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# ======= Overlay Points =======
for _, row in selected_df.iterrows():
    x_orig = int(row["x_pixel"])
    y_orig = int(row["y_pixel"])

    lat, lon = meters_to_gps(row["x_meter"], row["y_meter"])
    x_recon, y_recon = gps_to_pixel(lat, lon)

    ped_id = int(row["ped_id"])
    cv2.circle(img, (x_orig, y_orig), 5, (0, 0, 255), -1)  # 🔴 Original
    cv2.circle(img, (x_recon, y_recon), 5, (0, 255, 255), -1)  # 🟡 Reconstructed
    cv2.line(img, (x_orig, y_orig), (x_recon, y_recon), (0, 255, 0), 1)
    cv2.putText(img, f'ID {ped_id}', (x_orig + 5, y_orig - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

# ======= Display Image =======
plt.figure(figsize=(10, 8))
plt.imshow(img)
plt.title("Pixel vs Reconstructed Coordinates (Subset)")
plt.axis("off")
plt.tight_layout()
plt.show()
