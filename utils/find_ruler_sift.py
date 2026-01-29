import numpy as np
import cv2


def find_ruler_sift(image: np.ndarray,
                    template: np.ndarray,
                    ) -> np.ndarray:

    '''
    This function finds ruler in the image using template of the ruler and 
    features extracted using SIFT



    Args:
       image (np.ndarray) : Input image 
       template (np.ndarray) : Template image of ruler
    
    
    
    Returns:
       dst (np.ndarray) : Contour of ruler (rectangle)
    '''



    MIN_MATCH_COUNT = 100

    feat = cv2.SIFT_create()
    
    
    # find the keypoints and descriptors with SIFT
    kp1, des1 = feat.detectAndCompute(template,None)
    kp2, des2 = feat.detectAndCompute(image,None)


    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1,des2,k=2)
    
    
    
    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < n.distance:
            good.append(m)
    
    if len(good)>MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        #matchesMask = mask.ravel().tolist()
        h,w = template.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,M)
    else:
        print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
        #matchesMask = None
        return None
    
    dst = np.int32(dst.reshape(-1,2))
    return dst 
