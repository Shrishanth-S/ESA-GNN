import cv2
import pandas as pd

# Load the pixel coordinate CSV
df = pd.read_csv("D:/pedestrian_trajectory_detection/DataExtraction-master/DataExtraction-master/zara01/pixel_coordinate_inter.csv", header=None)
df = df.transpose()
df.columns = ['frame', 'ped_id', 'u', 'v']

# Convert coordinates to integers
df['u'] = df['u'].astype(int)
df['v'] = df['v'].astype(int)

# Load the video
video_path = "D:/pedestrian_trajectory_detection/videos/videos/crowds_zara01.avi"
cap = cv2.VideoCapture(video_path)

# Prepare frame dictionary (key = frame index starting from 1)
frame_dict = {}
for _, row in df.iterrows():
    frame_id = int(row['frame'])  # 1-based in dataset
    ped_id = int(row['ped_id'])
    u = int(row['u'])
    v = int(row['v'])

    if frame_id not in frame_dict:
        frame_dict[frame_id] = []
    frame_dict[frame_id].append((ped_id, u, v))

# Read and overlay on each frame
frame_num = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    current_frame_id = frame_num + 1  # Dataset is 1-indexed

    if current_frame_id in frame_dict:
        for ped_id, u, v in frame_dict[current_frame_id]:
            # ✅ Swap (u, v) to (v, u) to fix rotation
            cv2.circle(frame, (v, u), 5, (0, 255, 0), -1)
            cv2.putText(frame, str(ped_id), (v + 5, u - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow('Trajectory Overlay', frame)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

    frame_num += 1

cap.release()
cv2.destroyAllWindows()
