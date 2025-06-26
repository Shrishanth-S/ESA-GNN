import cv2
import numpy as np

# === CONFIG ===
video_path = "pedestrian_detector/Canteen_Dense.mp4"  # Escape backslash
frame_number = 100
draw_radius = 10

# === Load frame from video ===
cap = cv2.VideoCapture(video_path)
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
ret, frame = cap.read()
cap.release()

if not ret:
    print("❌ Failed to read frame.")
    exit()

# === Drawing canvas ===
drawing_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
drawing_mode = 'brush'
drawing = False
start_point = None

def draw(event, x, y, flags, param):
    global drawing, start_point, drawing_mask

    if drawing_mode == 'brush':
        if event == cv2.EVENT_LBUTTONDOWN or flags & cv2.EVENT_FLAG_LBUTTON:
            cv2.circle(drawing_mask, (x, y), draw_radius, 255, -1)

    else:
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            start_point = (x, y)

        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            param['preview'] = (start_point, (x, y))

        elif event == cv2.EVENT_LBUTTONUP:
            end_point = (x, y)
            drawing = False

            if drawing_mode == 'line':
                cv2.line(drawing_mask, start_point, end_point, 255, thickness=2)

            elif drawing_mode == 'rect':
                cv2.rectangle(drawing_mask, start_point, end_point, 255, -1)

            elif drawing_mode == 'circle':
                radius = int(np.hypot(end_point[0]-start_point[0], end_point[1]-start_point[1]))
                cv2.circle(drawing_mask, start_point, radius, 255, -1)

            param['preview'] = None

preview_state = {'preview': None}

cv2.namedWindow("Draw Mask")
cv2.setMouseCallback("Draw Mask", draw, preview_state)

# === Display loop ===
while True:
    display = frame.copy()
    display[drawing_mask > 0] = (255, 255, 255)

    # Draw preview shapes
    if preview_state['preview']:
        pt1, pt2 = preview_state['preview']
        if drawing_mode == 'line':
            cv2.line(display, pt1, pt2, (0, 255, 0), 1)
        elif drawing_mode == 'rect':
            cv2.rectangle(display, pt1, pt2, (0, 255, 0), 1)
        elif drawing_mode == 'circle':
            radius = int(np.hypot(pt2[0] - pt1[0], pt2[1] - pt1[1]))
            cv2.circle(display, pt1, radius, (0, 255, 0), 1)

    cv2.putText(display, f"Mode: {drawing_mode.upper()}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("Draw Mask", display)
    key = cv2.waitKey(1) & 0xFF

    if key == 27:  # ESC
        break
    elif key == ord('s'):
        binary_mask = cv2.threshold(drawing_mask, 127, 255, cv2.THRESH_BINARY)[1]
        black_white_image = cv2.merge([binary_mask]*3)
        cv2.imwrite("manual_mask.png", binary_mask)
        cv2.imwrite("black_white_overlay.png", black_white_image)
        print("✅ Saved black-and-white mask.")
    elif key == ord('x'):
        drawing_mask[:] = 0
        print("🔄 Reset drawing.")
    elif key == ord('b'):
        drawing_mode = 'brush'
        print("🖌️ Mode: Brush")
    elif key == ord('l'):
        drawing_mode = 'line'
        print("📏 Mode: Line")
    elif key == ord('r'):
        drawing_mode = 'rect'
        print("🔳 Mode: Rectangle")
    elif key == ord('c'):
        drawing_mode = 'circle'
        print("⚪ Mode: Circle")

cv2.destroyAllWindows()
