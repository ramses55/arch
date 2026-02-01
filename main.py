from utils import find_packet,  find_ruler_sift, \
                    find_region, fragments_contours, \
                    save_template, load_template, \
                    filter_features

from pathlib import Path
import cv2
import os
import numpy as np


directory = Path("./data/raw/")
jpg_files = list(directory.glob("*.jpg")) + list(directory.glob("*.jpeg"))

#template = cv2.imread("./ruler_template-v.png")
#save_template(template)

# get features of template with its height width
template_filename = "./template.npz"
kp_t, des_t, h, w = load_template(template_filename)


files = jpg_files[:]


for i in files:
    img = cv2.imread(i)
    img_copy = img.copy()

    mask_packet, packet = find_packet(img)
    packet = np.intp(cv2.boxPoints(packet))

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)



    feat = cv2.SIFT_create()
    

    # find the keypoints and descriptors of image with SIFT
    kp, des = feat.detectAndCompute(img_gray,
                                    cv2.bitwise_not(mask_packet))

    #cnt1 is the contour of first ruler
    cnt1 = find_ruler_sift(kp, des, kp_t, des_t, h, w)

    #fill area of first ruler with white in the original image
    cv2.drawContours(img, [cnt1], 0, (255,255,255), -1)


    #mask_cnt1 is the mask of first ruler
    mask_cnt1 = np.zeros_like(img_gray, dtype=np.uint8)
    cv2.drawContours(mask_cnt1, [cnt1], 0, 255, -1)

    mask = cv2.bitwise_or(mask_packet, mask_cnt1)


    kp_new, des_new = filter_features(mask,kp, des)

    cnt2 = find_ruler_sift(kp_new, des_new, kp_t, des_t, h, w)
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
    cv2.drawContours(img_copy, [cnt2], 0, (255,0,0), 5)

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


