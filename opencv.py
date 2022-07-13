from statistics import mode
import cv2 as cv
import numpy as np
import time
import os

weights = 'plastic/yolov4-obj_last.weights'
model = 'plastic/yolov4-obj.cfg'
names = 'plastic/obj.names'

# yolov4
# weights = 'yolov4/yolov4.weights'
# model = 'yolov4/yolov4.cfg'
# names = 'yolov4/coco.names'

net = cv.dnn.readNet(weights, model)
classes = []
with  open(names, 'r') as f:
    classes = [i.strip() for i in f.readlines()]

layer_names = net.getLayerNames()
output_layers = [layer_names[i-1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size = (len(classes), 3))

def predict(img, boxsize=2):
    height, width, _ = img.shape
    # Detecting object                                          # if it is beeing read from the image itself maybay change it to false
    blob = cv.dnn.blobFromImage(img,0.00392, (416, 416), (0,0,0), True, crop= False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    confidences = []
    boxes = []
    class_ids = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    indexes = cv.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    n_obj_detected = len(indexes)
    for i in indexes:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        color = colors[class_ids[i]]
        cv.rectangle(img, (x,y), (x + w, y + h), color, boxsize) # 15
        cv.putText(img, label, (x + 5, y- 10 ), cv.FONT_HERSHEY_DUPLEX, boxsize, color, boxsize) # 5
        # cv.putText(img, str(n_obj_detected), (25, 25), cv.FONT_HERSHEY_DUPLEX, boxsize, (255,0,0)) # 5

def apiImage(path):
    img = cv.imread(path)
    predict(img)
    cv.imwrite(path, img)

def image(path, factor_resize = 1):
    img = cv.imread(path)
    img = cv.resize(img, None, fx = factor_resize, fy = factor_resize)
    predict(img)
    cv.imshow('img', img )
    cv.waitKey(0)

def apiVideo(path):
    video = cv.VideoCapture(path)
    size = (int(video.get(3)), int(video.get(4)))
    fileName = path.split('.')[0]
    fileName = f'{fileName}.avi'
    result = cv.VideoWriter(fileName, cv.VideoWriter_fourcc(*'MJPG'), 10, size)
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
             break        
        predict(frame)
        result.write(frame)
        if cv.waitKey(1) == ord('q'):
            break
    video.release()
    result.release()
    os.remove(path)
    return fileName

def camera():
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    prev_frame_time = 0
    new_frame_time = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv.flip(frame, 1)
        predict(frame, 2)
        cv.imshow('frame', frame)
        if cv.waitKey(1) == ord('q'):
            break
        new_frame_time = time.time() 
        fps = 1/(new_frame_time-prev_frame_time)
        print(int(fps))
        prev_frame_time = new_frame_time
    cap.release()
    cv.destroyAllWindows()
    
if __name__ == '__main__':
    camera()
    # apiVideo("files/UNyntcWVzt.MOV")