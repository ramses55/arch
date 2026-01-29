import numpy as np
import cv2



def preprocess(gray):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(gray)

def find_ruler(image: np.ndarray,
               template: np.ndarray,
               ) -> np.ndarray:


    MIN_MATCH_COUNT = 10
    
    
    # Initiate SIFT detector
    feat = cv2.ORB_create(
        nfeatures=5000,        # High feature count
        scaleFactor=1.15,      # Finer scale pyramid (SIFT-like)
        nlevels=12,            # More scale levels
        edgeThreshold=15,      # Detect closer to borders
        firstLevel=0,
        WTA_K=2,
        scoreType=cv2.ORB_HARRIS_SCORE,  # More stable than FAST score
        patchSize=31,
        fastThreshold=7        # Sensitive detector (important)
    )

    template = preprocess(template)
    image = preprocess(image)


    kp1, des1 = feat.detectAndCompute(template,None)
    kp2, des2 = feat.detectAndCompute(image,None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    # Match descriptors using KNN
    matches = bf.knnMatch(des1, des2, k=2)
 
    
    
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
