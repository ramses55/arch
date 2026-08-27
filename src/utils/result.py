import cv2
import numpy as np
import torch
import ultralytics
import io



from PIL import Image, ImageDraw, ImageFont
from pathlib import Path


from utils import crop_rect, orient_label, find_label
from utils import ocr, index1, index2, fix_cyr
from utils.config import settings
#from .dirt import dirty
#from .char import charred

class result:
    def __init__(self, res, filename):
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
        #to counter multiple detected labels, 19 mm is the size of label
        self.mes = np.array(np.array(self.label[0]).shape[1:3]).mean()/19
        
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
        #self.dirty_model = dirty()
        #self.char_model = charred()
        self.file_name = []
        self.charred = []
        self.dirty = []

        self.old_filename = filename        

    
    def use_ocr(self,
                session,
                apiKey = settings.api_key,
                folderId = settings.folder_id,
                packet_box=None,
                label_box=None
                ) -> list:
        for label in self.label:
            text, code  = ocr(label, apiKey, folderId, session)
            ind1 = index1(text)
            text.remove(ind1)
            ind2 = index2(text)
            if text is not None:
                self.file_name.append(fix_cyr(ind1 + "_" + ind2))
            else:
                self.file_name.append(f"OCR failed!!!: {code}")
                
    #def check_charred(self):
    #    if len(self.frag) > 0:
    #        self.charred = self.char_model(self.frag).detach().cpu().numpy()

    #def check_dirty(self):
    #    if len(self.frag) > 0:
    #        self.dirty = self.dirty_model(self.frag).detach().cpu().numpy()
        

    def all(self, session):
        self.use_ocr(session=session)
        #self.check_charred()
        #self.check_dirty()


    def csv_res(self):
        size = [f'{s[0]}*{s[1]}' for s in self.size]
        #dr = {0: "нет загрязнения", 1: "загрязнение"}
        #ch = {0: "нет обугливания", 1: "частичное обугливание", 2: "обугливание"}
        #dirty = [dr[x] for x in self.dirty]
        #charred = [ch[x] for x in self.charred]

        #print(self.old_filename)
        #print(self.names)
        #print(size)
        #print(dirty)
        #print(charred)
        #print(len(self.names))
        #print(self.file_name)

        #message = f"{self.old_filename}|{self.names}|{size}|{dirty}|{charred}|{len(self.names)}|{self.file_name}"
        message = f"{self.old_filename}|{self.names}|{size}|{len(self.names)}|{self.file_name}"
        return message


    def draw(self):
        image = self.orig_image.copy()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        for box,size,cl in zip(self.frag_box,self.size,self.names):
            cv2.drawContours(image, [box], 0, (0,255,0), 2)

        for label in self.label_box.reshape(-1,4,2):
            cv2.drawContours(image, [label], 0, (0,0,255), 2)

        img_pil = Image.fromarray(image)
        draw = ImageDraw.Draw(img_pil)
        font = ImageFont.truetype("DejaVuSans.ttf",30)

        sizes = [f'({s[0]}*{s[1]})' for s in self.size]
        #dr = {0: "нет загр", 1: "загр"}
        #ch = {0: "нет обугл", 1: "част обугл", 2: "обугл"}
        #dirty = [dr[x] for x in self.dirty]
        #charred = [ch[x] for x in self.charred]
        #for box,size,cl,d,c in zip(self.frag_box,sizes,self.names, dirty, charred):
        for box,size,cl in zip(self.frag_box,sizes,self.names):
            #print("box:", box)
            box = box.squeeze()
            position = (box[:,0].min()-30, box[:,1].min()-30)
            #print("pos:",position)
            text = cl + '-' + size
            draw.text(position, text, font=font, fill=(0,0,0))



        for label,fl in zip(self.label_box.reshape(-1,4,2), self.file_name):
            label = label
            position = (label[:,0].min()-30, label[:,1].min()-30)
            #print("pos:",position)
            text = fl
            draw.text(position, text, font=font, fill=(0,0,0))

        #img_pil.thumbnail((800,800))  # keeps aspect ratio
            
        buffer = io.BytesIO()
        img_pil.save(buffer, format="JPEG", quality=70, optimize=True)
        buffer.seek(0)
        return buffer
