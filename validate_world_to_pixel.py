import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

# === Load Homography Matrix ===
H = np.linalg.inv(np.loadtxt("data/annotations/univ/H.txt"))


# === Load World and Pixel Coordinates ===
world_df = pd.read_csv("data/annotations/univ/world_coordinate_inter.csv", header=None).T
world_df.columns = ['frame', 'ped_id', 'y', 'x']
world_df = world_df.sort_values(['frame', 'ped_id'])

pixel_df = pd.read_csv("data/annotations/univ/pixel_coordinate_inter.csv", header=None).T
pixel_df.columns = ['frame', 'ped_id', 'y', 'x']
pixel_df = pixel_df.sort_values(['frame', 'ped_id'])

# === Merge on frame & ped_id ===
merged = pd.merge(world_df, pixel_df, on=['frame', 'ped_id'], suffixes=('_world', '_pixel'))
sample = merged.sample(100)  # Random 100 points

# === Convert world → pixel using H ===
def world_to_pixel(world_pts, H):
    N = world_pts.shape[0]
    world_h = np.hstack([world_pts, np.ones((N, 1))])
    pixel_h = (H @ world_h.T).T
    return pixel_h[:, :2] / pixel_h[:, 2:3]

world_pts = sample[['x_world', 'y_world']].values
actual_pixel_pts = sample[['x_pixel', 'y_pixel']].values
transformed_pts = world_to_pixel(world_pts, H)

# === Load Map ===
map_img = cv2.imread("data/annotations/univ/map.png", cv2.IMREAD_GRAYSCALE)

# === Plot ===
plt.figure(figsize=(8, 6))
plt.imshow(map_img, cmap='gray')

x_t, y_t = transformed_pts[:, 0], transformed_pts[:, 1]


# 🔴 Transformed from world → pixel
plt.scatter(x_t, y_t, c='red', label='Transformed (H × World)', s=12)

# 🔵 Actual annotated pixel points
plt.scatter(actual_pixel_pts[:, 0], actual_pixel_pts[:, 1], c='blue', label='Actual Pixel', s=12, alpha=0.6)

plt.title("World → Pixel Projection Validation")
plt.legend()
plt.grid(True)
plt.axis("equal")
plt.tight_layout()
plt.show()
