from ultralytics import YOLO
import cv2
from ultralytics.models.yolo.yoloe.val import YOLOEDetectValidator

model = YOLO("yolov8n.pt")
# Start webcam
cap = cv2.VideoCapture(0)

TARGET_CLASSES = [
    "person", "cell phone", "chair", "bottle", "dining table"
]

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break


    results = model(frame)[0]


    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = model.names[cls_id]


        if label in TARGET_CLASSES:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Show frame
    cv2.imshow("Object Detection", frame)

    # Exit on ESC
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()