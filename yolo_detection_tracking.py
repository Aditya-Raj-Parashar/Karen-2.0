import cv2
from ultralytics import YOLO
from deep_sort_tracker import DeepSortTracker

model = YOLO("yolov8n.pt")
tracker = DeepSortTracker()

cap = cv2.VideoCapture("run.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]

    detections = []
    for box in results.boxes.data:
        x1, y1, x2, y2, conf, cls = box.tolist()
        detections.append([x1, y1, x2, y2, conf, cls])

    tracks = tracker.update(detections, frame)

    for t in tracks:
        x1, y1, x2, y2 = t["bbox"]
        track_id = t["track_id"]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(frame, f"ID {track_id}", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    cv2.imshow("Tracking", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()