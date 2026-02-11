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




directory = Path("./data/raw/")
jpg_files = list(directory.glob("*.jpg")) + list(directory.glob("*.jpeg"))

#template = cv2.imread("./ruler_template-v.png")
#save_template(template)

# get features of template with its height width
template_filename = "./template.npz"
kp_t, des_t, h, w = load_template(template_filename)

f = open("res", 'w+')


files = jpg_files[26:]


for i in files:
    img = cv2.imread(i)
    img_copy = img.copy()

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
    label_rect = find_label(img_copy_label)
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

    text1  = ocr(img_copy,"./api_key", "folder_id", label_box=label_box)     
    text2  = ocr(img_copy_label,"./api_key", "folder_id", label_box=label_box)     
    text3  = ocr(label_crop,"./api_key", "folder_id", label_box=None)     

    text = list(set(text1 + text2 + text3))


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

        cv2.drawContours(img_copy, [box], 0, (0,255,0),5)





    cv2.imwrite(os.path.join('./output3/',Path(i).stem + '.png'), img_copy)
    cv2.imwrite(os.path.join('./output2/',Path(i).stem + '.png'), packet_crop)
    cv2.imwrite(os.path.join('./output1/',Path(i).stem + '.png'), label_crop)

    print(f'{i}\t{ind1}_{ind2}\t{img.shape[0:2]}\t{text}', file=f)

f.close()
