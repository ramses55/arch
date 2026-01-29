from utils import find_packet,  find_ruler_sift, find_region, fragments_contours
#import glob
from pathlib import Path
import cv2
import os
import numpy as np


directory = Path("./data/raw/")
jpg_files = list(directory.glob("*.jpg")) + list(directory.glob("*.jpeg"))
template = cv2.imread('./ruler_template-v.png', cv2.IMREAD_GRAYSCALE)
if  template is None:
    raise IOError('failed to read ./ruler_template_v.png')
files = jpg_files[:]


for i in files:
    img = cv2.imread(i)
    img_copy = img.copy()
    img1, packet = find_packet(img)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cnt1 = find_ruler_sift(img_gray, template)

    cv2.drawContours(img, [cnt1], 0, (255,255,255), -1)
    cv2.drawContours(img_gray, [cnt1], 0, (255,255,255), -1)


    cnt2 = find_ruler_sift(img_gray, template)
    cv2.drawContours(img, [cnt2], 0, (255,255,255), -1)



    cv2.drawContours(img, [packet], 0, (255,255,255), -1)
    
    

    final = find_region(img,cnt1, cnt2)



    if (max(cv2.minAreaRect(cnt1)[1])==0):
        print(f'failed {i}')
        continue

    mes = 255 / max(cv2.minAreaRect(cnt1)[1])

    cnt = fragments_contours(final)

    cv2.drawContours(img_copy, [packet], 0, (0,255,0), 5)
    cv2.drawContours(img_copy, [cnt1], 0, (0,0,255), 5)
    cv2.drawContours(img_copy, [cnt2], 0, (0,0,255), 5)

    for c in cnt:
        rect = cv2.minAreaRect(c)
        box = np.intp(cv2.boxPoints(rect))
        width, height = rect[1]
        width = round(width * mes)
        height = round(height * mes)

        x_min = box[:,0].min()-10
        y_min = box[:,1].min()-10


        cv2.putText(img_copy, f'w:{width} h:{height}', (x_min, y_min),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2)

        cv2.drawContours(img_copy, [box], 0, (0,255,0),5)





    cv2.imwrite(os.path.join('./output/',os.path.basename(i)), img_copy)
    pass


