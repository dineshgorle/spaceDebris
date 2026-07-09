from ultralytics import YOLO
import cv2
model = YOLO("yolov8s best.pt")
results = model("cat.jpeg", conf=0.05)
img = results[0].plot()
cv2.imshow("Detection Result", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("output.jpg", img)
for r in results:
    print("Classes:", r.boxes.cls)
    print("Confidence:", r.boxes.conf)
