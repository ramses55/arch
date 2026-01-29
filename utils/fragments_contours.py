import cv2
import numpy as np

def fragments_contours(image: np.ndarray,
                       min_area: float = 0.001
                       ) -> list:

    '''
        Function takes as input image of fragments with whitish background



        Args:
           image (np.ndarray) : Input image 
           min_area (int) : fraction of full image area which can be considered
           as fragment



        Returns:
           new_cnt (list) : List of contours of fragments
    '''


    if not isinstance(image, np.ndarray):
        raise TypeError("image must be np.ndarray")



    if not isinstance(min_area, float):
        raise TypeError("min_area must be float")

    if (min_area > 1) or (min_area <= 0):
        raise TypeError("min_area must be float between 0 and 1")

    blured = cv2.blur(image, (21,21))
    image_hsv = cv2.cvtColor(blured, cv2.COLOR_BGR2HSV) 
    
    #finds mask of fragments
    mask = cv2.inRange(image_hsv, lowerb=(0, 0, 135),  upperb=(179,50,255))
    mask = cv2.bitwise_not(mask)
    
    
    
    cnt, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    new_image = image.copy()
    cv2.drawContours(new_image, cnt, -1, (0,255,0), 10)
    
    
    part_image_area = image.shape[0]*image.shape[1] * min_area
    areas = list()
    new_cnt = list()
    for i in cnt:
        area = cv2.contourArea(i)
        if area > part_image_area:
            new_cnt.append(i)
            areas.append(cv2.contourArea(i))
    
    return new_cnt
