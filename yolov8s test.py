from ultralytics import YOLO
import cv2

# Load your trained model
model = YOLO("yolov8s best.pt")

# Run detection (LOW confidence to ensure detection)
results = model("cat.jpeg", conf=0.05)

# Get image with bounding boxes
img = results[0].plot()

# Show image
cv2.imshow("Detection Result", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save output image
cv2.imwrite("output.jpg", img)

# Print detections
for r in results:
    print("Classes:", r.boxes.cls)
    print("Confidence:", r.boxes.conf)