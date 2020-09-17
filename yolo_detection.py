import cv2
import numpy as np

#load yolo
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

classes = []

#open names file
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_name = net.getLayerNames()
output_layer = [layer_name[i[0]-1] for i in net.getUnconnectedOutLayers()]

colors = np.random.uniform(0,255, size=(len(classes), 3))

# load image
img = cv2.imread("dog.jpg")
img = cv2.resize(img, (600, 600))


width, height, channel = img.shape

#detecting objects
blob = cv2.dnn.blobFromImage(img, 0.0015, (416, 416), (0, 0, 0), True, crop = False)

# for b in blob:
#     for n, img_blob in enumerate(b):
#         cv2.imshow(str(n), img_blob)

net.setInput(blob)
outs = net.forward(output_layer)

# Showing information on the screen

boxes = []
confidences = []
class_ids = []

for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            center_x = int(detection[0]*width)
            center_y = int(detection[1]*height)
            w = int(detection[2]*width)
            h = int(detection[3]*height)

            x = int(center_x-w/2)
            y = int(center_y-h/2)

            boxes.append([x,y,w,h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

# draw rectangle around objects

for i in range(len(boxes)):
    if i in indexes:
        x,y,w,h = boxes[i]
        label = classes[class_ids[i]]
        color = colors[i]
        cv2.rectangle(img, (x,y),(x+w ,y+h), color, 2)
        
        # showing name of objects

        Y = y - 15 if y - 15 > 15 else y + 15
        cv2.rectangle(img, (x,y), (x+len(label)*12, y+15), (255,255,255), -1)
        cv2.putText(img, label, (x,y+10), cv2.FONT_HERSHEY_SIMPLEX , 0.5, (0,0,0), 2)

cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()