
import cv2
from ultralytics import YOLO

model = YOLO('yolov8n.pt')

cap = cv2.VideoCapture('videos/39211.avi')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    vehicle_count = {}
    vehicle_count['car'] = 0
    vehicle_count['bus'] = 0

    # Iterate through detected objects
    for result in results:
        boxes = result.boxes
        for box, cls in zip(boxes.xyxy, result.boxes.cls):
            if int(cls) == 2:
                vehicle_count['car'] += 1
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, 'car', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            elif int(cls) == 5:
                vehicle_count['bus'] += 1
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, 'bus', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Show total cars currently in frame
    cv2.putText(frame, f'cars in frame: {vehicle_count['car']}, buses in frame: {vehicle_count['bus']}', (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Current car and bus count', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()