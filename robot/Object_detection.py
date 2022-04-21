import cv2
import numpy as np

# Sets the classes from coco and names for objects to be detected, split line makes a string turn into a list.
net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')
classes = []
with open('coco.names', 'r') as f:
    classes = f.read().splitlines()

cap = cv2.VideoCapture(0)

while True:
                    # names each frame (_) as img
                _, img = cap.read()
                height, width, _ = img.shape

                blob = cv2.dnn.blobFromImage(img, 1 / 255, (416, 416), (0, 0, 0), swapRB=True, crop=False)
                net.setInput(blob)
                # Used to obtain output at output layer and assign names
                output_layers_names = net.getUnconnectedOutLayersNames()
                LayerOutputs = net.forward(output_layers_names)

                # extracts bounding boxes as list
                boxes = []
                # extracts confidence e.g. probability that thing detected is what we want it to be
                confidences = []
                # represent predicted classes
                class_ids = []

                # 1st for loop extracts all information from layers output 2nd extracts information from each detection, score starts from 6th element
                for output in LayerOutputs:
                    for detection in output:
                        # array of scores that contain predictions
                        scores = detection[5:]
                        # extracts highest scores location
                        class_id = np.argmax(scores)
                        confidence = scores[class_id]
                        if confidence > 0.2:
                            centre_x = int(detection[0] * width)
                            centre_y = int(detection[1] * height)
                            w = int(detection[2] * width)
                            h = int(detection[3] * height)
                            # 0,1,2,3 represent where they are in the list and multiplied by height or width so it is readable to cv
                            x = int(centre_x - w / 2)
                            y = int(centre_y - h / 2)
                            # using append to add these newly defined values to the already defined list
                            boxes.append([x, y, w, h])
                            confidences.append((float(confidence)))
                            class_ids.append(class_id)

                    # Building a threshold to filter out/ go for the highest probability tin
                indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.1)

                font = cv2.FONT_HERSHEY_PLAIN
                colours = np.random.uniform(0, 255, size=(len(boxes), 3))

                if len(indexes) > 0:
                    for i in indexes.flatten():
                        x, y, w, h = boxes[i]
                        label = str(classes[class_ids[i]])
                        confidence = str(round(confidences[i], 2))
                        colour = colours[i]
                        cv2.rectangle(img, (x, y), (x + w, y + h), colour, 2)
                        cv2.putText(img, label + ' ' + confidence, (x, y + 20), font, 2, (255, 255, 255), 2)

                cv2.imshow('Image', img)
                key = cv2.waitKey(1)
                if key == 27:
                    break

cap.release()
cv2.destroyAllWindows()