import pandas as pd
import math

# ==== CONFIG ====
INPUT_CSV = "pedestrian_detector/pixel_coordinates/pixel_coordinates.csv"
OUTPUT_CSV = "pedestrian_detector/world_coordinates/world_coordinates.csv"


# Drone & Camera Parameters
DRONE_LAT = 13.009162
DRONE_LON = 74.795902
ALTITUDE = 10  # meters
FOV_DEG = 84  # Horizontal FOV
IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720

def pixel_to_gps(x_pixel, y_pixel):
    x_offset = x_pixel - IMAGE_WIDTH / 2
    y_offset = y_pixel - IMAGE_HEIGHT / 2

    # Field of view (radians)
    fov_h_rad = math.radians(FOV_DEG)
    aspect_ratio = IMAGE_WIDTH / IMAGE_HEIGHT
    fov_v_rad = math.atan(math.tan(fov_h_rad / 2) / aspect_ratio) * 2

    # Angle offsets
    x_angle = math.atan(math.tan(fov_h_rad / 2) * (2 * x_offset / IMAGE_WIDTH))
    y_angle = math.atan(math.tan(fov_v_rad / 2) * (2 * y_offset / IMAGE_HEIGHT))

    # Ground distance offsets
    dx = ALTITUDE * math.tan(x_angle)
    dy = ALTITUDE * math.tan(y_angle)

    # Convert distance to lat/lon deltas using WGS84 Earth model
    a = 6378137.0        # Earth's equatorial radius
    f = 1 / 298.257223563
    lat_rad = math.radians(DRONE_LAT)

    sin_lat = math.sin(lat_rad)
    cos_lat = math.cos(lat_rad)

    N = a / math.sqrt(1 - f * (2 - f) * sin_lat ** 2)  # Radius of curvature in prime vertical
    M = a * (1 - f * (2 - f)) / (1 - f * (2 - f) * sin_lat ** 2) ** 1.5

    dlat = -dy / M
    dlon = dx / (N * cos_lat)

    lat = DRONE_LAT + math.degrees(dlat)
    lon = DRONE_LON + math.degrees(dlon)

    return lat, lon

def gps_to_relative_meters(lat, lon):
    # Convert lat/lon to local (x, y) in meters w.r.t. drone position
    R = 6378137  # Radius of Earth in meters
    dlat = math.radians(lat - DRONE_LAT)
    dlon = math.radians(lon - DRONE_LON)
    lat1 = math.radians(DRONE_LAT)

    dy = dlat * R
    dx = dlon * R * math.cos(lat1)
    return dx, dy

# ==== Process CSV ====
df = pd.read_csv(INPUT_CSV)  # Columns: frame, ped_id, y_pixel, x_pixel

x_meters = []
y_meters = []

for _, row in df.iterrows():
    x_pix = row['x_pixel']
    y_pix = row['y_pixel']

    lat, lon = pixel_to_gps(x_pix, y_pix)
    dx, dy = gps_to_relative_meters(lat, lon)

    x_meters.append(dx)
    y_meters.append(dy)

# Final Output
df_out = pd.DataFrame({
    "frame": df["frame"],
    "ped_id": df["ped_id"],
    "y_meter": y_meters,
    "x_meter": x_meters
})

df_out.to_csv(OUTPUT_CSV, index=False)
print(f"✅ Saved: {OUTPUT_CSV}")
