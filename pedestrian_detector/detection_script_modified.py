#!/usr/bin/env python3
import cv2
import numpy as np
import time
import argparse
import torch
import pandas as pd
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from pathlib import Path

class PedestrianDetector:
    def __init__(self, args):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"✅ Running on {self.device.upper()}")

        self.model = YOLO(args.model, task='detect').to(self.device)
        self.tracker = DeepSort(max_age=50)

        self.cap = cv2.VideoCapture(args.source if not args.source.startswith('usb') 
                                    else int(args.source[3:]))
        if not self.cap.isOpened():
            raise RuntimeError(f"Unable to open source '{args.source}'")

        self.cap.set(3, args.width)
        self.cap.set(4, args.height)

        self.input_fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.input_fps <= 1.0:
            self.input_fps = 30.0

        self.video_writer = None
        if args.save_video:
            output_path = Path(args.save_video)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(
                str(output_path), fourcc, self.input_fps, (args.width, args.height)
            )

        self.thresh = args.thresh
        self.show_display = args.display
        self.image_width = args.width
        self.image_height = args.height
        self.bbox_colors = [
            (164,120,87), (68,148,228), (93,97,209), (178,182,133),
            (88,159,106), (96,202,231), (159,124,168), (169,162,241),
            (98,118,150), (172,176,184)
        ]

        # CSV storage: frame, ped_id, y_pixel, x_pixel
        self.detections = []
        self.frame_index = 0

    def process_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return False

        vis_frame = frame.copy()
        results = self.model(frame, device=self.device, verbose=False)
        detections = results[0].boxes

        dets = []
        for i in range(len(detections)):
            xyxy = detections[i].xyxy.cpu().numpy().squeeze()
            xmin, ymin, xmax, ymax = xyxy.astype(int)
            conf = detections[i].conf.item()

            if conf > self.thresh:
                width = xmax - xmin
                height = ymax - ymin
                dets.append(([xmin, ymin, width, height], conf, int(detections[i].cls.item())))

        tracked_objects = self.tracker.update_tracks(dets, frame=frame)

        for track in tracked_objects:
            if not track.is_confirmed():
                continue

            track_id = int(track.track_id)
            x, y, w, h = map(int, track.to_ltwh())

            # 🧠 HEAD CENTER COORDINATE
            x_center = x + w // 2
            y_head = y  # top of the box

            # Save in [frame, ped_id, y_pixel, x_pixel]
            self.detections.append([self.frame_index, track_id, y_head, x_center])

            # Optional: show on video
            color = self.bbox_colors[track_id % len(self.bbox_colors)]
            cv2.rectangle(vis_frame, (x, y), (x + w, y + h), color, 2)
            label = f'ID {track_id}'
            cv2.putText(vis_frame, label, (x, y - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        if self.video_writer:
            self.video_writer.write(vis_frame)

        if self.show_display:
            cv2.imshow('Tracking Results', vis_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                return False

        self.frame_index += 1
        return True

    def run(self, save_csv_path="head_coordinates.csv"):
        start_time = time.time()
        frame_count = 0
        try:
            while True:
                if not self.process_frame():
                    break
                frame_count += 1
        finally:
            self.cap.release()
            if self.video_writer:
                self.video_writer.release()
            if self.show_display:
                cv2.destroyAllWindows()

            elapsed = time.time() - start_time
            print(f"\n🟢 Processed {frame_count} frames in {elapsed:.2f} seconds")
            print(f"📈 Average FPS: {frame_count / elapsed:.2f}")

            # Save to CSV
            df = pd.DataFrame(self.detections, columns=["frame", "ped_id", "y_pixel", "x_pixel"])
            df.to_csv(save_csv_path, index=False)
            print(f"📁 Saved head coordinates to {save_csv_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='Path to YOLO model (e.g., yolov8n.pt)')
    parser.add_argument('--source', required=True, help='Video file or "usb0", "usb1" for webcam')
    parser.add_argument('--thresh', default=0.5, type=float, help='Confidence threshold')
    parser.add_argument('--width', default=1280, type=int, help='Frame width')
    parser.add_argument('--height', default=720, type=int, help='Frame height')
    parser.add_argument('--display', action='store_true', help='Show live video')
    parser.add_argument('--save-video', type=str, help='Path to save output video')
    parser.add_argument('--save-csv', type=str, help='Output CSV filename')
    args = parser.parse_args()

    detector = PedestrianDetector(args)
    detector.run(save_csv_path=args.save_csv)

if __name__ == '__main__':
    main()
