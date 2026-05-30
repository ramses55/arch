import numpy as np
import cv2


def find_ruler_akaze(image: np.ndarray,
               template: np.ndarray,
               ) -> np.ndarray:


    MIN_MATCH_COUNT = 100
    
    
    
    # Initiate SIFT detector
    #feat = cv2.SIFT_create(
    #        nfeatures=1200,
    #        contrastThreshold=0.06,
    #        edgeThreshold=10,
    #        sigma=1.6
    #        )    

    feat = cv2.AKAZE_create()
    
    
    # find the keypoints and descriptors with SIFT
    kp1, des1 = feat.detectAndCompute(template,None)
    kp2, des2 = feat.detectAndCompute(image,None)


    #FLANN_INDEX_KDTREE = 1
    #index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    #search_params = dict(checks = 50)
    #flann = cv2.FlannBasedMatcher(index_params, search_params)
    #matches = flann.knnMatch(des1,des2,k=2)
    
     
    FLANN_INDEX_LSH = 6
    
    index_params = dict(
        algorithm=FLANN_INDEX_LSH,
        table_number=12,      # 10–20
        key_size=20,          # 15–25
        multi_probe_level=2   # 1–2
    )
    
    search_params = dict(checks=50)
    
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    # IMPORTANT: descriptors must be uint8
    des1 = np.uint8(des1)
    des2 = np.uint8(des2)
    
    matches = flann.knnMatch(des1, des2, k=2)
    
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
    
    return np.int32(dst.reshape(-1,2))
