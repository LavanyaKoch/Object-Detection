import cv2
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO('yolov8n.pt')  # You can use 'yolov8s.pt' for a larger model


def detect_objects(frame):
    # Perform object detection on the frame
    results = model.predict(frame, stream=False)

    # Annotate detected objects on the frame
    annotated_frame = frame.copy()
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
        for box in boxes:
            x1, y1, x2, y2 = map(int, box[:4])
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return annotated_frame

