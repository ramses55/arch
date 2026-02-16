import base64
import json
import requests
import cv2
import numpy as np
import pathlib
import time



def extract(vert):
    res = np.zeros((4,2), dtype=np.int32)
    for i,v in enumerate(vert):
        a = np.array([v['x'],v['y']], dtype=np.int32)
        res[i] = a
    return res 
    



def inside(box, box1):

    if box1 is None:
        return True


    x_min = box1[:,0].min()
    x_max = box1[:,0].max()

    
    y_min = box1[:,1].min()
    y_max = box1[:,1].max()

    x = box[:,0]
    y = box[:,1]


    a1 = (x > x_min).all()
    a2 = (x < x_max).all()
    a3 = (y > y_min).all()
    a4 = (y < y_max).all()

    #print(a1,a2,a3,a4)
    res  = a1 * a2 * a3 * a4

    return res



def get_text(blocks, label_box):

    text = list()
    boxes = list()
    for block in blocks:
        for line in block['lines']:
            box = extract(line['boundingBox']['vertices'])
            if inside(box,label_box):
                #l = list()
                word = line['text']
                #l.append(word)
                
                boxes.append(box)
                #text.append(l)
                text.append(word)

    return text, boxes




def index1(text: list) -> str:

    '''
        This function tries to find ind1



        Args:
            text (list) : output of OCR



        Returns:
           ind1 (str) : index1
    '''


    for a in text:
        if a.isnumeric() and len(a)>2:

            return a
    
    return "no_index1"






def index2(text: list) -> str:

    '''
        This function tries to find ind1



        Args:
            text (list) : output of OCR



        Returns:
           ind2 (str) : index2
    '''

    for a in text:
        if not a[0].isnumeric() and a[-1].isnumeric():
            return a


    return "no_index2"





def date(text: list) -> str:

    '''
        This function tries to find date



        Args:
            text (list) : output of OCR



        Returns:
           date (str) : index1
    '''

    for a in text:
        a = a.replace(".", "").replace(" ","")
        if a.isnumeric() and len(a)==8:
            return a


    return ""




def ocr(image: np.ndarray,
        apiKey_path: pathlib.PosixPath,
        folderId_path: pathlib.PosixPath,
        packet_box=None,
        label_box=None
        ) -> list:
    '''

        Read text from label image using yandex cloud ocr. api key and folder
        id are essentail



        Args:
            apiKey_path (pathlib.PosixPath) : path to file with api key
            folderId_path (pathlib.PosixPath) : path to file with folder id



        Returns:
           text (list) : filter text read by ocr


    '''

    with open(apiKey_path, "r") as f:
        token = f.read().strip('\n')
    
    
    with open(folderId_path, "r") as f:
        folder_id = f.read().strip('\n')
    
    
    # yandex cloud ocr expects encoded image as input
    success, buffer = cv2.imencode(".jpg", image)
    
    if not success:
        raise RuntimeError("Image encoding failed")
    
    content = base64.b64encode(buffer).decode("utf-8")
    
    
    
    data1 = {"mimeType": "PNG",
            "languageCodes": ["ru"],
            "model": "table", 
            "content": content}
    
    
    
    data2 = {"mimeType": "PNG",
            "languageCodes": ["ru"],
            "model": "handwritten", 
            "content": content}
    
    url = "https://ocr.api.cloud.yandex.net/ocr/v1/recognizeText"
    
    headers= {"Content-Type": "application/json",
              "Authorization": f"Api-Key {token}",
              "x-folder-id": folder_id,
              "x-data-logging-enabled": "true"}
      
    w1 = requests.post(url=url,
                       headers=headers,
                       data=json.dumps(data1),
                       timeout=10)
    
    time.sleep(1)
    
    #w2 = requests.post(url=url,
    #                   headers=headers,
    #                   data=json.dumps(data2),
    #                   timeout=10)



    #if w2.status_code != 200:
    #    raise RuntimeError(f"OCR failed!: {w2.status_code}")
    
    
    if w1.status_code != 200:
        raise RuntimeError(f"OCR failed!: {w1.status_code}")



    blocks1 = w1.json()['result']['textAnnotation']['blocks']
    #blocks2 = w2.json()['result']['textAnnotation']['blocks']

    text1, boxes1 = get_text(blocks1, label_box)
    #text2, boxes2 = get_text(blocks2, label_box)

    #text = list(set(text1 + text2))
    text = list(set(text1))


    return text


    
    
    

    
