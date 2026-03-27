#import boto3

#with open("./keys/s3-key-id", "r") as f:
#    access_key = f.readline().rstrip()
#
#with open("./keys/s3-key", "r") as f:
#    secret_key = f.readline().rstrip()
#
#
#
#
#QUEUE_URL = "https://message-queue.api.cloud.yandex.net/b1gn2cep80rfd8mos87v/dj60000000ifp1rm0516/queue"
#
#sqs = boto3.client(
#    "sqs",
#    endpoint_url="https://message-queue.api.cloud.yandex.net",
#    region_name="ru-central1",
#    aws_access_key_id=access_key,
#    aws_secret_access_key=secret_key
#)
#
#
#
#response = sqs.list_queues()
#
#print(response['QueueUrls'])


import ultralytics
import cv2
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from utils import crop_rect, orient_label, find_label
from utils import ocr, index1, index2

from pathlib import Path
import torch
import torchvision.io as io
import torchvision.transforms.v2 as v2
from torchvision.models.resnet import resnet18
import time


from PIL import Image, ImageDraw, ImageFont



model = ultralytics.YOLO("./best.pt")



import glob



class dirty(torch.nn.Module):
    def __init__(self):
        super().__init__()
        resnet = resnet18(weights='DEFAULT')
        resnet.fc = torch.nn.Linear(in_features=512, out_features=2, bias=True)

        checkpoint = torch.load("./data-yolo/best-dirty.pth", weights_only=True)
        resnet.load_state_dict(checkpoint["model_state"])
        self.model = resnet
        self.mean=[0.485, 0.456, 0.406] #it's for ImageNet
        self.std=[0.229, 0.224, 0.225]
        self.transforms = v2.Compose([v2.ToImage(),
                                 v2.ToDtype(torch.float32, scale=True),
                                 v2.Normalize(self.mean, self.std),
                                 v2.Resize((128,128)),
                                ])

    def forward(self,x):
        res = torch.stack(self.transforms(x))
        y = self.model(res).argmax(dim=1)
        return y





class char(torch.nn.Module):
    def __init__(self):
        super().__init__()
        resnet = resnet18(weights='DEFAULT')
        resnet.fc = torch.nn.Linear(in_features=512, out_features=3, bias=True)

        checkpoint = torch.load("./data-yolo/best-char.pth", weights_only=True)
        resnet.load_state_dict(checkpoint["model_state"])
        self.model = resnet
        self.mean=[0.485, 0.456, 0.406] #it's for ImageNet
        self.std=[0.229, 0.224, 0.225]
        self.transforms = v2.Compose([v2.ToImage(),
                                 v2.ToDtype(torch.float32, scale=True),
                                 v2.Normalize(self.mean, self.std),
                                 v2.Resize((128,128)),
                                ])

    def forward(self,x):
        res = torch.stack(self.transforms(x))
        y = self.model(res).argmax(dim=1)
        return y





class result:
    def __init__(self, res: ultralytics.engine.results.Results):
        #label  indices
        self.label_ind = torch.argwhere(res.obb.cls == 0).squeeze() 
        #fragments indices
        self.frag_ind = torch.argwhere(res.obb.cls != 0) 
        #fragments box
        self.frag_box = res.obb.xyxyxyxy[self.frag_ind].detach().cpu().numpy().astype(int)
        #label box
        self.label_box = res.obb.xyxyxyxy[self.label_ind].detach().cpu().numpy().astype(int)
        #fragments size in pixels
        self.pixel_size = res.obb.xywhr[self.frag_ind].detach().cpu().numpy().reshape((-1,5))[:,2:4]
        
        self.orig_image = res.orig_img
        
        label_rect = res.obb.xywhr[self.label_ind].detach().cpu().numpy().reshape(-1,5)
        label_rect = [(rect[0:2], rect[2:4], rect[4]/np.pi * 180) for rect in label_rect]
        label = [crop_rect(self.orig_image, rect) for rect in label_rect]
        self.label = [orient_label(l) for l in label] 
        self.mes = np.array(self.label)[:,1].mean()/19
        
        frag_rect = res.obb.xywhr[self.frag_ind].detach().cpu().numpy().reshape(-1,5)
        frag_rect = [(rect[0:2], rect[2:4], rect[4]/np.pi * 180) for rect in frag_rect]
        frags = [cv2.cvtColor(crop_rect(self.orig_image, rect), cv2.COLOR_BGR2RGB) for rect in frag_rect]
        
        self.frag = [cv2.rotate(frag,cv2.ROTATE_90_CLOCKWISE)
                     if frag.shape[1] > frag.shape[0]
                     else frag 
                     for frag in frags]
        
        size = np.round((self.pixel_size / self.mes).reshape(-1,2)).astype(int)
        self.size = np.sort(size, axis=-1)[:,::-1]
        
        self.names = [res.names[res.obb.cls[i.detach().item()].item()] for i in self.frag_ind]
        self.dirty_model = dirty()
        self.char_model = char()
        self.file_name = []
        self.charred = []
        self.dirty = []

        self.old_filename = Path(res.path).stem
        

    
    def use_ocr(self,
                apiKey_path = "./keys/api_key",
                folderId_path = "./keys/folder_id",
                packet_box=None,
                label_box=None
                ) -> list:
        for label in self.label:
            text  = ocr(label, apiKey_path, folderId_path)
            if text is not None:
                self.file_name.append(index1(text) + "_" + index2(text))
                
    def check_charred(self):
        if len(self.frag) > 0:
            self.charred = self.char_model(self.frag).detach().cpu().numpy()

    def check_dirty(self):
        if len(self.frag) > 0:
            self.dirty = self.dirty_model(self.frag).detach().cpu().numpy()
        
    def csv_res(self):
        size = [f'{s[0]}*{s[1]}' for s in self.size]
        dr = {0: "нет загрязнения", 1: "загрязнение"}
        ch = {0: "нет обугливания", 1: "частичное обугливание", 2: "обугливание"}
        dirty = [dr[x] for x in self.dirty]
        charred = [ch[x] for x in self.charred]
        #print(self.old_filename)
        #print(self.names)
        #print(size)
        #print(dirty)
        #print(charred)
        #print(len(self.names))
        #print(self.file_name)
        message = f"{self.old_filename}|{self.names}|{size}|{dirty}|{charred}|{len(self.names)}|{self.file_name}"
        return message

    def all(self):
        self.use_ocr()
        self.check_charred()
        self.check_dirty()


    def draw(self):
        image = self.orig_image.copy()
        for box,size,cl in zip(self.frag_box,self.size,self.names):
            cv2.drawContours(image, [box], 0, (0,255,0), 2)

        for label in self.label_box.reshape(-1,4,2):
            cv2.drawContours(image, [label], 0, (0,0,255), 2)

        img_pil = Image.fromarray(image)
        draw = ImageDraw.Draw(img_pil)
        font = ImageFont.truetype("DejaVuSans.ttf",30)


        sizes = [f'({s[0]}*{s[1]})' for s in self.size]
        dr = {0: "нет загр", 1: "загр"}
        ch = {0: "нет обугл", 1: "част обугл", 2: "обугл"}
        dirty = [dr[x] for x in self.dirty]
        charred = [ch[x] for x in self.charred]
        for box,size,cl,d,c in zip(self.frag_box,sizes,self.names, dirty, charred):
            #print("box:", box)
            box = box.squeeze()
            position = (box[:,0].min()-30, box[:,1].min()-30)
            #print("pos:",position)
            text = cl + '-' + size + '-' + d + '-' + c
            draw.text(position, text, font=font, fill=(0,0,0))



        for label,fl in zip(self.label_box.reshape(-1,4,2), self.file_name):
            label = label
            position = (label[:,0].min()-30, label[:,1].min()-30)
            #print("pos:",position)
            text = fl
            draw.text(position, text, font=font, fill=(0,0,0))

        return np.array(img_pil)
        


f = open("extra-result.csv", "w")
names = glob.glob("./extra/*")

for i,name in enumerate(names):
    time.sleep(1.5)
    print(i, name)
    output = model(name,
                conf=0.1,
                save=False,
                show=False,
                verbose=False)
    
    try:
        r = result(output[0])
        r.all()
        m = r.csv_res()
        im = r.draw()
        success = cv2.imwrite(f"m-{os.path.basename(name)}", im)
        print(success)
        print(m,file=f)
    except Exception as e:
        print(f"Failed file_name:{name}")
        print(f"Error occurred: {e}") 

f.close()
