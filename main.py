from utils import find_packet,  find_ruler_sift, \
                    find_region, fragments_contours, \
                    save_template, load_template, \
                    filter_features, find_label, \
                    crop_rect, ocr, index1, \
                    index2, date

from pathlib import Path
import cv2
import os
import numpy as np
from cv2 import dnn_superres


def order_points(pts):
    rect = np.zeros((4, 2), dtype="int32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left

    return rect


directory = Path("./data/raw/")
jpg_files = list(directory.glob("*.jpg")) + list(directory.glob("*.jpeg"))

#template = cv2.imread("./ruler_template-v.png")
#save_template(template)

# get features of template with its height width
template_filename = "./template.npz"
kp_t, des_t, h, w = load_template(template_filename)

f = open("new-res", 'a')


files = jpg_files[-4:]

 
###########################

for i in files:
    img = cv2.imread(i)
    #print(i)
    #print(img.shape)
    img_copy = img.copy()
    img_copy1 = img.copy()

    mask_packet, packet_rect = find_packet(img)
    packet_crop = crop_rect(img, packet_rect)

    img_copy_label = img.copy()
    packet_box = np.intp(cv2.boxPoints(packet_rect))
    mask = np.zeros(img.shape[0:2], dtype=np.uint8)
    cv2.drawContours(mask,
                     [packet_box],
                     0,
                     255,
                     -1)
    img_copy_label[mask==0] = np.array([255,255,255])
    label_rect, rects = find_label(img_copy_label)
    label_box = np.intp(cv2.boxPoints(label_rect))
    label_crop = crop_rect(img, label_rect)



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



    cv2.drawContours(img, [packet_box], 0, (255,255,255), -1)
    
    

    final = find_region(img,cnt1, cnt2)



    if (max(cv2.minAreaRect(cnt1)[1])==0):
        print(f'failed {i}')
        continue

    mes = 255 / max(cv2.minAreaRect(cnt1)[1])

    cnt = fragments_contours(final)



####################################
    center, _, angle = label_rect

    height, width = img.shape[0:2]

    M = cv2.getRotationMatrix2D(center, angle,1)


    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_width = int((height * sin) + (width * cos))
    new_height = int((height * cos) + (width * sin))

    M[0, 2] += (new_width / 2) - center[0]
    M[1, 2] += (new_height / 2) - center[1]


    rotated = cv2.warpAffine(img_copy1, M, (new_width, new_height)) 


    new_label_box = np.hstack((label_box, np.ones((4,1)))) @ M.T


    new_boxes = list()
    for rect in rects:
        box0 = cv2.boxPoints(rect)
        new_box0 =  np.hstack((box0, np.ones((4,1)))) @ M.T
        new_boxes.append(np.intp(new_box0))




    blacks = np.zeros(4)
    for j,new_box in enumerate(new_boxes):

        x_min = new_box[:,0].min()
        x_max = new_box[:,0].max()
        y_min = new_box[:,1].min()
        y_max = new_box[:,1].max()

        crop = rotated[y_min:y_max, x_min:x_max]
        black=cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).sum()
        blacks[j] = black


    ma = np.argmax(blacks)
    mi = np.argmin(blacks)



    qr  = np.intp(new_boxes[mi])
    arg = np.argmin(((new_label_box - qr)**2).sum(axis=1))
    ori = ["top_left", "top_right", "bottom_right", "bottom_left"]
    #ori = ["bottom_left", "top_left", "top_right", "bottom_right"]
    rota_arg = [cv2.ROTATE_90_CLOCKWISE,
                None,
                cv2.ROTATE_90_COUNTERCLOCKWISE,
                cv2.ROTATE_180]

    rota_arg1 = [180,
                -90,
                0,
                90]



    #res = cv2.rotate(rotated, rota_arg[arg])
    #res = cv2.rotate(res, cv2.ROTATE_90_CLOCKWISE)



    center= new_label_box.mean(axis=0)

    height, width = img.shape[0:2]

    M = cv2.getRotationMatrix2D(center, rota_arg1[arg],1)


    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_width = int((height * sin) + (width * cos))
    new_height = int((height * cos) + (width * sin))

    M[0, 2] += (new_width / 2) - center[0]
    M[1, 2] += (new_height / 2) - center[1]


    res = cv2.warpAffine(rotated, M, (new_width, new_height)) 


    new_label_box = np.hstack((new_label_box, np.ones((4,1)))) @ M.T





    res_z = np.zeros_like(res)
    mask1 = cv2.drawContours(res_z, [np.intp(new_label_box)], 0, (255,255,255), -1)
    mask1 = cv2.bitwise_not(mask1)
    res = res + mask1




    new_label_box = np.intp(new_label_box)
    x_min = new_label_box[:,0].min()
    x_max = new_label_box[:,0].max()
    y_min = new_label_box[:,1].min()
    y_max = new_label_box[:,1].max()

    new_label_crop = res[y_min:y_max, x_min:x_max]
    #print(new_label_crop.shape)
    if new_label_crop.shape[0] < 100:
        scale = 2
        new_label_crop = cv2.resize(
            new_label_crop,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_LANCZOS4  # or INTER_CUBIC
            #interpolation=cv2.INTER_CUBIC  # or INTER_CUBIC
        )



    #cv2.drawContours(res, [np.intp(new_label_box)], 0, (0,255,0),5)
    #for b in new_boxes:
    #    cv2.drawContours(res, [b], 0, (0,0,0), 7)

    #cv2.putText(res, 'QR', np.intp(rects[mi][0]),
    #                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)



    #cv2.putText(res, 'W', np.intp(rects[ma][0]),
    #                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

######################################################################



    #text1  = ocr(img_copy,"./api_key", "folder_id", label_box=label_box)     
    #text2  = ocr(img_copy_label,"./api_key", "folder_id", label_box=label_box)     
    text3  = ocr(new_label_crop,"./api_key", "folder_id", label_box=None)     

    #text = list(set(text1 + text2 + text3))
    text = text3


    d = date(text)
    ind2 = index2(text)
    ind1 = index1(text)





    cv2.drawContours(img_copy, [packet_box], 0, (0,255,0), 5)
    cv2.drawContours(img_copy, [cnt1], 0, (0,0,255), 5)
    cv2.drawContours(img_copy, [cnt2], 0, (255,0,0), 5)
    cv2.drawContours(img_copy, [label_box], 0, (0,0,255), 5)


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

        #cv2.drawContours(img_copy, [box], 0, (0,255,0),5)





    cv2.imwrite(os.path.join('./output4/',Path(i).stem + '.png'), new_label_crop)
    cv2.imwrite(os.path.join('./output3/',Path(i).stem + '.png'), img_copy)
    cv2.imwrite(os.path.join('./output2/',Path(i).stem + '.png'), packet_crop)
    cv2.imwrite(os.path.join('./output1/',Path(i).stem + '.png'), label_crop)

    print(f'{i}\t{ind1}_{ind2}\t{img.shape[0:2]}\t{text}', file=f)

f.close()
