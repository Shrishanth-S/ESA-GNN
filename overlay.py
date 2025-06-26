import cv2
import pandas as pd

# === Config ===
csv_path = "pedestrian_detector/pixel_coordinates/pixel_coordinates.csv"
video_path = "pedestrian_detector/Canteen_Dense.mp4"
save_video_path = "pedestrian_detector/results/overlayed.mp4"  # Set to None if not saving
circle_radius = 5

# === Load CSV ===
df = pd.read_csv(csv_path)
df['x_pixel'] = df['x_pixel'].astype(int)
df['y_pixel'] = df['y_pixel'].astype(int)
df['frame'] = df['frame'].astype(int)
df['ped_id'] = df['ped_id'].astype(int)

# === Build frame-wise dictionary ===
frame_dict = {}
for _, row in df.iterrows():
    frame_id = row['frame']
    ped_id = row['ped_id']
    x = row['x_pixel']
    y = row['y_pixel']
    if frame_id not in frame_dict:
        frame_dict[frame_id] = []
    frame_dict[frame_id].append((ped_id, x, y))

# === Read video ===
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
if fps is None or fps <= 1.0:
    print("⚠️ Warning: Invalid FPS detected, defaulting to 30.")
    fps = 30.0

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# === Optional: Initialize video writer ===
writer = None
if save_video_path:
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(save_video_path, fourcc, fps, (width, height))

# === Overlay ===
frame_idx = 1  # CSV uses 1-indexed frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_idx in frame_dict:
        for ped_id, x, y in frame_dict[frame_idx]:
            cv2.circle(frame, (x, y), circle_radius, (0, 255, 0), -1)
            cv2.putText(frame, str(ped_id), (x + 5, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow("Overlay", frame)
    if writer:
        writer.write(frame)

    if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
        break

    frame_idx += 1

cap.release()
if writer:
    writer.release()
cv2.destroyAllWindows()
print("✅ Done")
