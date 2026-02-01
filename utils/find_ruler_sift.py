import numpy as np
import cv2


def filter_features(mask, kp, des):
    '''
        This function filters keypoints and descriptors:
        new keypoints and new descriptors will have no points
        inside provided mask


    Args:
       mask (np.ndarray) : mask
       kp (list(cv2.KeyPoint)): keypoints
       des (np.ndarray): descriptors
    
    
    
    Returns:
       kp_f (list(cv2.KeyPoint)) : filtered keypoints
       des_f (np.ndarray): filtered descriptors

    '''


    pts = np.array([a.pt for a in kp]).astype(int)
    keep = mask[pts[:,1], pts[:,0]] == 0
    
    
    kp_f = [k for k,m in zip(kp,keep) if m]
    des_f = des[keep]

    return kp_f, des_f





def save_template(template: np.ndarray)-> bool:
    sift = cv2.SIFT_create()
    kp, des = sift.detectAndCompute(template, None)
    
    kp_array = np.array([
        (k.pt[0], k.pt[1], k.size, k.angle, k.response, k.octave, k.class_id)
        for k in kp
    ], dtype=np.float32)

    shape = np.array(template.shape)

    np.savez("template.npz", keypoints=kp_array, descriptors=des, shape=shape)



def load_template(filename: str)->tuple:
    data = np.load(filename)
    kp = data["keypoints"]
    des = data["descriptors"]
    shape = data["shape"]

    kp = [
        cv2.KeyPoint(
            x=p[0], y=p[1], size=p[2],
            angle=p[3], response=p[4],
            octave=int(p[5]), class_id=int(p[6])
        )
        for p in kp
    ]
    h = shape[0]
    w = shape[1]

    return kp, des, h, w



def find_ruler_sift(kp: list,
                    des: np.ndarray,
                    kp_t: list,
                    des_t: np.ndarray,
                    h: int,
                    w: int,
                    template: np.ndarray = None,
                    ) -> np.ndarray:

    '''
    This function finds ruler in the image using template of the ruler and 
    features extracted using SIFT



    Args:
       kp (list(cv2.KeyPoint)): precomputed keypoints of image
       des (np.ndarray): precomputed descriptors of image
       kp_t (list(cv2.KeyPoint)): precomputed keypoints of template
       des_t (np.ndarray): precomputed descriptors of template
       h (int): height of the template image
       w (int): width of the template image
       template (np.ndarray) : Template image of ruler
    
    
    
    Returns:
       dst (np.ndarray) : Contour of ruler (close to rectangle)
    '''




    #feat = cv2.SIFT_create()
    #
    #
    #if template != None:
    #    kp_t, des_t = feat.detectAndCompute(template,None)





    # find the keypoints and descriptors of image with SIFT
    #kp, des = feat.detectAndCompute(image,None)

    MIN_MATCH_COUNT = 100

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des_t,des,k=2)
    
    
    
    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < n.distance:
            good.append(m)
    
    if len(good)>MIN_MATCH_COUNT:
        src_pts = np.float32([ kp_t[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        #matchesMask = mask.ravel().tolist()
        if template !=None:
            h,w = template.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,M)
    else:
        print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
        #matchesMask = None
        return None
    
    dst = np.int32(dst.reshape(-1,2))
    return dst 
