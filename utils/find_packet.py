import cv2
import numpy as np



def crop_rect(image: np.ndarray,
              rect: tuple[tuple[float,float],tuple[float,float], float]
              ) -> np.ndarray:

    '''

        This function crops oriented bounding boxes produced
        by cv2.minAreaRect



        Args:
           image (np.ndarray) : Input image 
           rect (tuple) : Output of cv2.minAreaRect



        Returns:
           image (np.ndarray) : Cropped rectanlge
    '''

    if not isinstance(image, np.ndarray):
        raise TypeError("image must be np.ndarray")


    if not isinstance(rect, tuple):

        raise TypeError("rect must be output tuple")


    center, _, angle = rect
    height, width = image.shape[:2] 

    
    M = cv2.getRotationMatrix2D(center, angle,1)

    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_width = int((height * sin) + (width * cos))
    new_height = int((height * cos) + (width * sin))

    M[0, 2] += (new_width / 2) - center[0]
    M[1, 2] += (new_height / 2) - center[1]


    rotated = cv2.warpAffine(image, M, (new_width, new_height)) 
    
    old_box = np.intp(cv2.boxPoints(rect))
    new_box = np.hstack((old_box, np.ones((4,1)))) @ M.T
    new_box = np.intp(new_box)
    x_min = new_box[:,0].min()
    x_max = new_box[:,0].max()
    y_min = new_box[:,1].min()
    y_max = new_box[:,1].max()

    cropped = rotated[y_min:y_max, x_min:x_max]

    return cropped





def find_packet(image: np.ndarray,
                LOWER_BOUND: tuple[int,int,int] = (93,70,100),
                UPPER_BOUND: tuple[int,int,int] = (120,255,200)
                ) -> tuple[np.ndarray, np.ndarray]:
    '''
        
        This function finds brownish packet in image. Firstly, it finds mask
        of brownish objects in image by color (hsv, inRange). Secondly, it
        finds countour with largest area and fits orientated rectangle to it
        using minAreaRect.


        Args:
            image (np.ndarray) : input image in BGR format
            LOWER_BOUND (tuple[int,int,int]) : lower bound for color
            UPPER_BOUND (tuple[int,int,int]) : upper bound for color


        Returns:
            tuple[np.ndarray, np.ndarray]:
                A tuple containing:
                - mask (np.ndarray): mask of the packet
                - rect (tuple): rectanlge contour of packet, output of
                  minAreaRect


    '''



    if not isinstance(image, np.ndarray):
        raise TypeError("image must be np.ndarray")

    #LOWER_BOUND = (93,70,100)
    #UPPER_BOUND = (120,255,200)
    
    #im = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB)
    im = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    im_hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    
    
    blured = cv2.blur(im_hsv, (21,21))
    packet_mask = cv2.inRange(blured,
                              LOWER_BOUND, UPPER_BOUND
                              )

    packet_mask = cv2.morphologyEx(packet_mask,
                                   cv2.MORPH_CLOSE,
                                   (5,5)
                                   )

    packet_mask = cv2.morphologyEx(packet_mask,
                                   cv2.MORPH_OPEN,
                                   (5,5)
                                   )



    
    
    
    contours, _ = cv2.findContours(packet_mask,
                                   cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE
                                   )
    
    
    mask = np.zeros(im.shape[0:2], dtype=np.uint8)
    
    
    #find contour with largest area
    areas = []
    for i in contours:
        areas.append(cv2.contourArea(i))
    areas = np.array(areas)
    m = areas.argmax()

    cv2.drawContours(mask, contours, m, 255, -1)
    
    
    rect = cv2.minAreaRect(contours[areas.argmax()])
    
    return mask, rect



