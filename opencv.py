from math import ceil
import cv2 as cv
import numpy as np
import os
import base64

from utilities import loadColors, getModel

modelCfg = os.getenv('modelcfg')

colors, classes = loadColors()


def createNet(model):
    weights = getModel(model)
    net = cv.dnn.readNet(weights, modelCfg)
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i-1] for i in net.getUnconnectedOutLayers()]
    return net, output_layers

def predict(net, output_layers, img, confidenceTreshold):
    height, width, _ = img.shape
    boxsize = ((height + width) / 2)* 0.002
    boxsize = ceil(boxsize)
    blob = cv.dnn.blobFromImage(img, 0.00392, (512, 512), (0,0,0), True, crop= False)
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
    indexes = cv.dnn.NMSBoxes(boxes, confidences, confidenceTreshold, 0.4)
    n_obj_detected = len(indexes)
    for i in indexes:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        color = colors[class_ids[i]]
        cv.rectangle(img, (x,y), (x + w, y + h), color, boxsize)
        cv.putText(img, label, (x + 5, y- 10 ), cv.FONT_HERSHEY_DUPLEX, boxsize, color, boxsize)

def apiImage(path, model, confidence=0.5):
    img = cv.imread(path)
    net, output_layers = createNet(model)
    predict(net, output_layers,img,confidence)
    cv.imwrite(path, img)

def apiVideo(path, newVideoPath, model, confidence=0.5):
    video = cv.VideoCapture(path)
    size = (int(video.get(3)), int(video.get(4)))
    fileName = f'{newVideoPath}.mp4'
    print(fileName)
    # MPV4 works but not in docker
    # X264 with errors but works for web 
    # H264 with errors  but works for web

    net, output_layers = createNet(model)
    result = cv.VideoWriter(fileName, cv.VideoWriter_fourcc(*'H264'), video.get(cv.CAP_PROP_FPS), size)
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
             break        
        predict(net, output_layers, frame, confidence)
        result.write(frame)
    video.release()
    result.release()
    os.remove(path)
    return fileName

async def livevideo(websocket):
    model, lastmodel = '', '';
    net, output_layers = None, None
    try:
        await websocket.accept()
        while True:
            data = await websocket.receive_json()
            lastmodel = model 
            model = data['model']
            if model != lastmodel:
                net, output_layers = createNet(model)
            if model == '':
                continue

            data_img = data['image']
            confidece = float(data['confidence'])
            confidece /= 100;
            img_b64 = data_img[22:]
            bin_data = base64.b64decode(img_b64)
            image = np.asarray(bytearray(bin_data), dtype=np.uint8)
            img = cv.imdecode(image, cv.IMREAD_COLOR)
            img  = cv.flip(img, 1)
            predict(net, output_layers, img, confidece)
            img_str = cv.imencode('.jpg', img)[1].tostring()
            jpg_as_text = base64.b64encode(img_str)
            
            await websocket.send_text(jpg_as_text.decode('ascii'))
            if not data['keepconection']:
                break
        await websocket.close()
    except:
        pass