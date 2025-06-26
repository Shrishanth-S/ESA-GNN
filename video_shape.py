import cv2

video_path = "pedestrian_detector/Canteen_Dense.mp4"

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Failed to open video")
else:
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video resolution:Width={width} px, Height={height}px")
    
cap.release()
    