# object_detector.py

from ultralytics import YOLO
import cv2
import numpy as np
import os
import uuid
import time
import json
import threading


class ObjectDetector:
    def __init__(self):
        self.model = YOLO('yolov8n.pt')
        self.known_objects = {'person', 'cell phone', 'laptop'}
        self.unknown_objects = {}
        self.photo_dir = "unknown_objects"
        os.makedirs(self.photo_dir, exist_ok=True)
        self.photos_per_object = 100
        self.capture_interval = 0.05
        self.confidence_threshold = 0.2
        self.capturing = False
        self.capture_start_time = 0
        self.current_unknown_box = None
        self.current_unknown_class = None
        self.frame = None
        self.annotated_frame = None
        self.capture_thread = None

    def generate_name(self):
        return f"Unknown_{uuid.uuid4().hex[:8]}"

    def capture_photos(self, name):
        object_dir = os.path.join(self.photo_dir, name)
        os.makedirs(object_dir, exist_ok=True)

        for i in range(self.photos_per_object):
            if self.frame is None:
                continue

            frame = self.frame.copy()
            results = self.model(frame, conf=self.confidence_threshold)
            updated_box = None
            for r in results:
                for box, conf, cls in zip(r.boxes.xyxy, r.boxes.conf, r.boxes.cls):
                    if self.model.names[int(cls)] == self.current_unknown_class:
                        updated_box = box.tolist()
                        break
                if updated_box:
                    break

            if updated_box:
                self.current_unknown_box = updated_box

            metadata = {
                "box": self.current_unknown_box,
                "timestamp": time.time()
            }

            photo_name = f"{name}_{i + 1}.jpg"
            path = os.path.join(object_dir, photo_name)
            cv2.imwrite(path, frame)

            metadata_path = os.path.join(object_dir, f"{name}_{i + 1}_metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f)

            print(f"Captured photo: {path}")
            time.sleep(self.capture_interval)

        self.capturing = False
        self.current_unknown_box = None
        self.current_unknown_class = None

    def detect_and_capture(self, frame):
        self.frame = frame
        results = self.model(frame, conf=self.confidence_threshold)
        annotated_frame = frame.copy()

        for r in results:
            for box, conf, cls in zip(r.boxes.xyxy, r.boxes.conf, r.boxes.cls):
                detected_class = self.model.names[int(cls)]
                print(f"Detected: {detected_class}, Confidence: {conf:.2f}")

                if conf > self.confidence_threshold:
                    x1, y1, x2, y2 = map(int, box)
                    if detected_class not in self.known_objects:
                        if detected_class not in self.unknown_objects and not self.capturing:
                            new_name = self.generate_name()
                            self.unknown_objects[detected_class] = new_name
                            self.capturing = True
                            self.capture_start_time = time.time()
                            self.current_unknown_box = box.tolist()
                            self.current_unknown_class = detected_class
                            self.capture_thread = threading.Thread(target=self.capture_photos, args=(new_name,))
                            self.capture_thread.start()
                            print(f"New unknown object detected: {detected_class}")

                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(annotated_frame, f"Unknown: {detected_class} ({conf:.2f})", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                    else:
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(annotated_frame, f"{detected_class} ({conf:.2f})", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        if self.capturing:
            elapsed_time = time.time() - self.capture_start_time
            remaining_time = max(0, self.photos_per_object * self.capture_interval - elapsed_time)
            cv2.putText(annotated_frame, f"Unknown object detected! Capturing images... {remaining_time:.1f}s",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        self.annotated_frame = annotated_frame

    def run(self):
        cap = cv2.VideoCapture(0)

        while cap.isOpened():
            success, frame = cap.read()
            if success:
                self.detect_and_capture(frame)
                if self.annotated_frame is not None:
                    cv2.imshow("Object Detection", self.annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    detector = ObjectDetector()
    detector.run()