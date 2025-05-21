# Object-Detection-Using-Webcam
## AIM: 
To write a Python code to Object Detection Using Webcam.

## PROCEDURE:
STEP-1 Load the pre-trained YOLOv4 network (.weights and .cfg) using cv2.dnn.readNet().

STEP-2 Read class labels (COCO dataset) from the coco.names file.

STEP-3 Get the output layer names from the YOLO network using getLayerNames() and getUnconnectedOutLayers().

STEP-4 Start webcam video capture using cv2.VideoCapture(0).

STEP-5 Process each frame:

Convert the frame to a YOLO-compatible input using cv2.dnn.blobFromImage().
Pass the blob into the network (net.setInput()) and run forward pass to get detections (net.forward()).
Parse the output to extract bounding boxes, confidence scores, and class IDs for detected objects.
STEP-6 Use NMS to remove overlapping bounding boxes and retain the best ones.

STEP-7 Draw bounding boxes and labels on detected objects using cv2.rectangle() and cv2.putText().

STEP-8 Show the processed video frames with object detections using cv2.imshow().

STEP-9 Exit the loop if the 'q' key is pressed.

STEP-10 Release the video capture and close any OpenCV windows (cap.release() and cv2.destroyAllWindows()).

## PROGRAM:
#### NAME :  NISHA D
#### REG.NO : 212223230143

```
import cv2
import numpy as np

# Load YOLOv4 model
net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]
classes = open("coco.names").read().strip().split("\n")

# Open webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    height, width, _ = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    detections = net.forward(output_layers)

    for output in detections:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x, center_y, w, h = map(int, detection[:4] * [width, height, width, height])
                x, y = center_x - w // 2, center_y - h // 2
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"{classes[class_id]} {confidence:.2f}", (x, y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display output
    cv2.imshow("YOLOv4 Real-Time Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

```
## OUTPUT:

![WhatsApp Image 2025-05-21 at 11 42 14_10118436](https://github.com/user-attachments/assets/8fe35977-e089-490b-9214-b88d06319ad2)


## RESULT:
Thus, the Python Program to detect object using web camera as been successfully executed.



