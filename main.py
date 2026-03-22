from utils import find_packet,  find_ruler_sift, \
                    find_region, fragments_contours, \
                    save_template, load_template, \
                    filter_features, find_label, \
                    crop_rect, ocr, index1, \
                    index2, date, orient_label

from pathlib import Path
import cv2
import os
import numpy as np
import sys

def dump_cnt(file_name, cnt, label, cls):
    h, w = cv2.imread(file_name).shape[0:2]
    file_name = Path(file_name).stem + '.txt'
    f  = open(file_name, "w")
    for k,c in enumerate(cnt):
        rect = cv2.minAreaRect(c)
        longer = max(rect[1])
        shorter = min(rect[1])
        if (longer / shorter) > 10:
            print(f"longer: {longer}")
            print(f"shorter: {shorter}")
            continue
        box = cv2.boxPoints(rect)
        #print("box:")
        #print(box)
        #print(w, h)
        for i in range(4):
            box[i, 0] = round(box[i,0] / w, 2)
            box[i, 1] = round(box[i,1] / h, 2)

        #print(box)

        print(f"{cls} {box[0,0]:.2f} {box[0,1]:.2f} {box[1,0]:.2f} {box[1,1]:.2f} {box[2,0]:.2f} {box[2,1]:.2f} {box[3,0]:.2f} {box[3,1]:.2f}", file=f)
    for i in range(4):
        label[i, 0] = round(label[i,0] / w, 2)
        label[i, 1] = round(label[i,1] / h, 2)

    print(f"0 {label[0,0]:.2f} {label[0,1]:.2f} {label[1,0]:.2f} {label[1,1]:.2f} {label[2,0]:.2f} {label[2,1]:.2f} {label[3,0]:.2f} {label[3,1]:.2f}", file=f)

    f.close()

#directory = Path("./yolo-teeth/data/images/val/")
#directory = Path("./data2/Зубы/Фото/")
#directory = Path("./data2/Ногтевая пластина/Фото/")
#jpg_files = list(directory.glob("*.jpg")) + list(directory.glob("*.jpeg")) \
#        + list(directory.glob("*.JPG"))


with open("./paths", 'r') as f:
    jpg_files = [line.rstrip() for line in f]

tem = cv2.imread("./ruler_template-v.png")
save_template(tem)

# get features of template with its height width
template_filename = "./template.npz"
kp_t, des_t, h, w = load_template(template_filename)

f = open("res", 'a')



files = jpg_files[:]


failed_label = list()
failed_fragment_contour = list()
failed_second_ruler = list()
failed_ruler = list()
two_hor_rul = list()
not_failed = list()

print(len(files))

for u,i in enumerate(files):
    if u % 50 == 0:
        print(u, file=sys.stderr)
    cls = i.split('/')[1]
    #print(cls)
    img = cv2.imread(i)

    img_copy = img.copy()
    img_copy_label = img.copy()

    mask_packet, packet_rect = find_packet(img)
    packet_crop = crop_rect(img, packet_rect)

    packet_box = np.intp(cv2.boxPoints(packet_rect))
    mask = np.zeros(img.shape[0:2], dtype=np.uint8)


    cv2.drawContours(mask,
                     [packet_box],
                     0,
                     255,
                     -1)


    img_copy_label[mask==0] = np.array([255,255,255])
    label_rect = find_label(img_copy_label)
    if label_rect is None:
        cv2.drawContours(img_copy, [packet_box], 0, (0,255,0), 5)
        cv2.imwrite(os.path.join('./output4/',Path(i).stem + '.png'), img_copy)
        failed_label.append(str(i))
        continue

    label_crop = crop_rect(img, label_rect)
    label = orient_label(label_crop)
    if label is None:
        #print(f'failed to orient {i}')
        cv2.drawContours(img_copy, [packet_box], 0, (0,255,0), 5)
        cv2.drawContours(img_copy, [np.intp(cv2.boxPoints(label_rect))], 0,
                         (0,0,255), 5)
        cv2.imwrite(os.path.join('./output5/',Path(i).stem + '.png'), img_copy)
        failed_label.append(str(i))
        continue

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

    
    




    cnt2_a = cv2.contourArea(cnt2)
    cnt2_ra = np.array(cv2.minAreaRect(cnt2)[1]).prod()
    if (cnt2_a <cv2.contourArea(cnt1) * 0.8) or cnt2_a < cnt2_ra * 0.8: 
        #print(f'failed second ruler {i}')
        failed_second_ruler.append(str(i))
        continue

    if max(cv2.minAreaRect(cnt1)[1]) != 0:
        mes = 255 / max(cv2.minAreaRect(cnt1)[1])
    else:
        #print(f"Failed to find ruler: {i}")
        failed_ruler.append(str(i))
        continue



    final = find_region(img,cnt1, cnt2)
    if final is None:
        two_hor_rul.append(i)
        continue
        
    #ocr work
    #text  = ocr(label,"./api_key", "folder_id", label_box=None)     
    #if text is None:
    #    print(f'OCR failed on file: {i}')
    #    continue

    #d = date(text)
    #ind2 = index2(text)
    #ind1 = index1(text)





    cv2.drawContours(img_copy, [packet_box], 0, (0,255,0), 5)
    cv2.drawContours(img_copy, [cnt1], 0, (255,0,0), 5)
    cv2.drawContours(img_copy, [cnt2], 0, (255,0,0), 5)
    cv2.drawContours(img_copy, [np.intp(cv2.boxPoints(label_rect))], 0, (0,0,255), 5)

    cnt = fragments_contours(final)
    if len(cnt) == 0:
        #print(f'Failed to find fragment contour: {i}')
        failed_fragment_contour.append(str(i))
        continue 

    for k,c in enumerate(cnt):
        rect = cv2.minAreaRect(c)
        longer = max(rect[1])
        shorter = min(rect[1])
        if (longer / shorter) > 10:
            print(f"longer: {longer}")
            print(f"shorter: {shorter}")
            continue
        box = np.intp(cv2.boxPoints(rect))
        width, height = rect[1]
        width = round(width * mes)
        height = round(height * mes)

        x_min = box[:,0].min()-10
        y_min = box[:,1].min()-10


        cv2.imwrite(os.path.join('./output2/',Path(i).stem + f'-{k}' + '.png'),
                    crop_rect(img, rect)
                    )


        cv2.putText(img_copy, f'w:{width} h:{height}--{k}', (x_min, y_min),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2)

        cv2.drawContours(img_copy, [c], 0, (128,0,128),5)
        cv2.drawContours(img_copy, [box], 0, (0,204,255),5)



    cv2.imwrite(os.path.join('./output/',Path(i).stem + '.png'), img_copy)
    cv2.imwrite(os.path.join('./output1/',Path(i).stem + '.png'), label)
    cv2.imwrite(os.path.join('./output3/',Path(i).stem + '.png'), final)

    not_failed.append(i)

    dump_cnt(i, cnt, cv2.boxPoints(label_rect), cls=cls)

    #print(f'{i}\t{ind1}_{ind2}\t{img.shape[0:2]}\t{text}', file=f)

f.close()

print(f"Failed to find label {len(failed_label)}:\n {failed_label}\n")
print(f"Failed to find contour fragment {len(failed_fragment_contour)}:\n {failed_fragment_contour}\n")
print(f"Failed to find second ruler {len(failed_second_ruler)}:\n {failed_second_ruler}\n")
print(f"Failed to find ruler for mes {len(failed_ruler)}:\n {failed_ruler}\n")
print(f"Two rulers oriented the same way {len(two_hor_rul)}:\n {two_hor_rul}")
print(f"Did not fail {len(not_failed)}:\n {not_failed}")
