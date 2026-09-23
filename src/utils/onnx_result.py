import cv2
import numpy as np
import io

from PIL import Image, ImageDraw, ImageFont
from pathlib import Path


from utils import crop_rect, orient_label, find_label
from utils import ocr, index1, index2, fix_cyr
from utils.config import settings



from itertools import combinations, compress

class onnx_result:
    def __init__(self, onnx_res, filename, orig_image, new_shape, conf,
                 use_nms=True, thres=0.5):
        #drop results with low confidence
        self.class_names = {
            0: 'этикетка', 
            1: 'зуб',
            2: 'кость',
            3: 'ноготь',
            4: 'горелый'
        }
        self.old_filename = filename
        self.orig_image = orig_image
        self.file_name = []



        #makes sure results are sorted by confidence for nms
        onnx_res = onnx_res[np.where(onnx_res[:,4] > conf)]
        ind = np.argsort(onnx_res[:,4])[::-1]
        self.res = onnx_res[ind]
        
        self.res_frag = self.res[np.where(self.res[:,5] != 0)]
        self.res_label = self.res[np.where(self.res[:,5] == 0)]

        
        self.cls = self.res_frag[:,5]
        self.conf_label = self.res_label[:,4]
        self.conf_frag = self.res_frag[:,4]
        
        self.names = [self.class_names[cls] for cls in self.cls]

        #scale back coordinates for original image size
        self.res_frag = onnx_result.scale_boxes(new_shape[-2:],
                                    self.res_frag,
                                    orig_image.shape[:2],
                                    xywh=True)
        
        self.res_label = onnx_result.scale_boxes(new_shape[-2:],
                                    self.res_label,
                                    orig_image.shape[:2],
                                    xywh=True)

        #makes rects out of onnx output
        self.label_rects = [onnx_result.get_rect(a) for a in self.res_label]
        self.frag_rects = [onnx_result.get_rect(a) for a in self.res_frag]

        #apply nms for label and all frags 
        if use_nms:
            self.label_rects, self.conf_label, _ = onnx_result.nms(self.label_rects,
                                                                   self.conf_label,
                                                                   thres)

            self.frag_rects, self.conf_frag, name_ind = onnx_result.nms(self.frag_rects,
                                                                        self.conf_frag,
                                                                        thres)

            self.names = [self.names[i] for i in name_ind]



        self.label = [orient_label(crop_rect(self.orig_image,l)) for l in self.label_rects]
        self.frag = [crop_rect(self.orig_image,f) for f in self.frag_rects]

        
        #makes boxes out of onnx output
        self.label_box = np.array([np.int32(cv2.boxPoints(a)) for a in self.label_rects])
        self.frag_box = np.array([np.int32(cv2.boxPoints(a)) for a in self.frag_rects])
        


        self.mes = (np.array(np.array(self.label[0]).shape[1:3]).mean()/19).item()

        self.size = np.array([np.sort(np.array(r[1]))[::-1] / self.mes for r in self.frag_rects]).astype(int)


    def use_ocr(self,
                session,
                apiKey = settings.api_key,
                folderId = settings.folder_id,
                packet_box=None,
                label_box=None
                ) -> list:
        for label in self.label:
            text, code  = ocr(label, apiKey, folderId, session)
            if text is not None:
                ind1 = index1(text)
                text.remove(ind1)
                ind2 = index2(text)
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






    @staticmethod
    def get_box(onnx_out):
        center = onnx_out[:2].tolist()
        size = onnx_out[2:4].tolist()
        angle = onnx_out[-1].item()/np.pi * 180

        return np.int32(cv2.boxPoints((center, size, angle)))

    @staticmethod
    def get_rect(onnx_out):
        center = onnx_out[:2].tolist()
        size = onnx_out[2:4].tolist()
        angle = onnx_out[-1].item()/np.pi * 180

        return (center, size, angle)



    #maybe too slow
    @staticmethod
    def nms(rects, confs, thres = 0.5):

        ind = range(len(confs))
        pairs = list(combinations(ind, 2))
        to_drop = {y for x,y in compress(pairs, [onnx_result.rect_iou(rects[f],rects[s]) > thres for (f,s) in pairs])}
        final_ind = list(set(ind) - to_drop)
             

        final_rects = [rects[i] for i in final_ind]
        final_confs = [confs[i] for i in final_ind]
        return final_rects, final_confs, final_ind

    @staticmethod
    def rect_iou(rect1, rect2):
        status, points = cv2.rotatedRectangleIntersection(rect1, rect2)
        if status != 0:
            i = cv2.contourArea(points)
            u = cv2.contourArea(cv2.boxPoints(rect1)) + \
                cv2.contourArea(cv2.boxPoints(rect2)) - \
                i
            return i/u
        else:
            return 0
        

    # source --- ultralytics/utils/ops.py 
    @staticmethod
    def scale_boxes(
        img1_shape: tuple[int, int],
        boxes:  np.ndarray,
        img0_shape: tuple[int, int],
        ratio_pad: tuple | None = None,
        padding: bool = True,
        xywh: bool = False,
    ) ->  np.ndarray:
        """Rescale bounding boxes from one image shape to another.
    
        Rescales bounding boxes from img1_shape to img0_shape, accounting for padding and aspect ratio changes. Supports
        both xyxy and xywh box formats.
    
        Args:
            img1_shape (tuple[int, int]): Shape of the source image (height, width).
            boxes (torch.Tensor | np.ndarray): Bounding boxes to rescale in format (N, 4).
            img0_shape (tuple[int, int]): Shape of the target image (height, width).
            ratio_pad (tuple, optional): Ratio and padding as ((ratio_h, ratio_w), (pad_w, pad_h)).
            padding (bool): Whether boxes are based on YOLO-style augmented images with padding.
            xywh (bool): Whether box format is xywh (True) or xyxy (False).
    
        Returns:
            (torch.Tensor | np.ndarray): Rescaled bounding boxes in the same format as input.
        """
        if ratio_pad is None:  # calculate from img0_shape
            gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
            gain_y = gain_x = gain
            pad_x = round((img1_shape[1] - round(img0_shape[1] * gain)) / 2 - 0.1)
            pad_y = round((img1_shape[0] - round(img0_shape[0] * gain)) / 2 - 0.1)
        else:
            gain_y, gain_x = ratio_pad[0]
            pad_x, pad_y = ratio_pad[1]
    
        if padding:
            boxes[..., 0] -= pad_x  # x padding
            boxes[..., 1] -= pad_y  # y padding
            if not xywh:
                boxes[..., 2] -= pad_x  # x padding
                boxes[..., 3] -= pad_y  # y padding
        boxes[..., 0] /= gain_x
        boxes[..., 1] /= gain_y
        boxes[..., 2] /= gain_x
        boxes[..., 3] /= gain_y
        return boxes if xywh else onnx_result.clip_boxes(boxes, img0_shape)


    #source is also ultralytics
    @staticmethod
    def clip_boxes(boxes, shape):
       """Clip bounding boxes to image boundaries.
    
       Args:
           boxes (torch.Tensor | np.ndarray): Bounding boxes to clip.
           shape (tuple): Image shape as HWC or HW (supports both).
    
       Returns:
           (torch.Tensor | np.ndarray): Clipped bounding boxes.
       """
       h, w = shape[:2]  # supports both HWC or HW shapes
       boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, w)  # x1, x2
       boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, h)  # y1, y2
       
       return boxes








#source ultralytics/data/augment.py
def get_params(img, new_shape = (1024, 1024),
               scaleup = 1, auto = False, stride = 32,
               scale_fill = False, center = True):
      """Compute letterboxing parameters.

      Args:
          labels (dict[str, Any]): Input labels dictionary containing 'img'.

      Returns:
          (dict): Parameters including 'orig_shape', 'new_shape', 'ratio', padding, and resize info.
      """
      #img = labels["img"]
      shape = img.shape[:2]  # current shape [height, width]
      #new_shape = labels.pop("rect_shape", new_shape)
      #if isinstance(new_shape, int):
      #    new_shape = (new_shape, new_shape)

      # Scale ratio (new / old)
      r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
      if not scaleup:  # only scale down, do not scale up (for better val mAP)
          r = min(r, 1.0)

      # Compute padding
      ratio = r, r  # width, height ratios
      new_unpad = round(shape[1] * r), round(shape[0] * r)
      dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
      if auto:  # minimum rectangle
          dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
      elif scale_fill:  # stretch
          dw, dh = 0.0, 0.0
          new_unpad = (new_shape[1], new_shape[0])
          ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

      if center:
          dw /= 2  # divide padding into 2 sides
          dh /= 2

      top, bottom = round(dh - 0.1) if center else 0, round(dh + 0.1)
      left, right = round(dw - 0.1) if center else 0, round(dw + 0.1)

      return {
          "orig_shape": shape,
          "new_shape": new_shape,
          "ratio": ratio,
          "new_unpad": new_unpad,
          "top": top,
          "bottom": bottom,
          "left": left,
          "right": right,
      }


#source ultralytics/data/augment.py
def apply_image(img, params, interpolation = 1,
               padding_value = 114):
    """Resize and pad the image.

    Args:
        labels (dict[str, Any]): Dictionary containing 'img'.
        params (dict): Parameters from get_params.

    Returns:
        (dict): Updated labels with resized and padded image.
    """
    shape = img.shape[:2]
    new_unpad = params["new_unpad"]

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=interpolation)
        if img.ndim == 2:
            img = img[..., None]

    h, w, c = img.shape
    top, bottom = params["top"], params["bottom"]
    left, right = params["left"], params["right"]
    if c == 3:
        img = cv2.copyMakeBorder(
            img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(padding_value,) * 3
        )
    else:  # multispectral
        pad_img = np.full((h + top + bottom, w + left + right, c), fill_value=padding_value, dtype=img.dtype)
        pad_img[top : top + h, left : left + w] = img
        img = pad_img

    return img

def preprosses(img):
    params = get_params(img)
    r = apply_image(img, params).astype(np.float32)
    r = r[...,::-1] / 255.0
    r = r.transpose(2,0,1)
    r = r[np.newaxis]
    return r
