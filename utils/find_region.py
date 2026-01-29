import numpy as np

def hor(cnt):
    '''
    Checks if found ruler is horizontally placed
    '''

    std_x = cnt[:,0].std()
    std_y = cnt[:,1].std()

    return std_x>std_y

def top(img, cnt):
    '''
    Checks if horizontal ruler is at the top
    '''

    height, width = img.shape[0:2]


    return cnt[:,1].mean()<height//2


def left(img, cnt):
    '''
    Checks if vertical ruler is at the left side
    '''
    height, width = img.shape[0:2]


    return cnt[:,0].mean()<width//2



def ruler_pos(img, cnt):
    '''
        Gets position of ruler hor/ver top/bottom left/right
    '''


    if hor(cnt):
        a = "horizontal"
        if top(img, cnt):
            b = 'top'
        else:
            b = 'bottom'

    else:
        a = 'vertical'
        if left(img, cnt):
            b = 'left'
        else:
            b = 'right'

    return (a,b, cnt)


def find_region(img: np.ndarray,
                cnt1: np.ndarray,
                cnt2: np.ndarray
                ) -> np.ndarray:

    '''
    This function fills with white area not in between to rulers and returns
    this image


        Args:
           img (np.ndarray) : Input image 
           cnt1 (np.ndarray) : Contour of ruler 1
           cnt2 (np.ndarray) : Contour of ruler 2



        Returns:
           img (np.ndarray) : Image with area outside rulers filled with white
    '''


    pos1 = ruler_pos(img, cnt1)
    pos2 = ruler_pos(img, cnt2)

    if pos1[0] == 'horizontal':
        posh = pos1
        posv = pos2
    else:
        posh = pos2
        posv = pos1

    x_min = max(posh[2][:,0].min(), 0)
    x_max = min(posh[2][:,0].max(), img.shape[1])

    y_min = max(posv[2][:,1].min(), 0)
    y_max = min(posv[2][:,1].max(), img.shape[0])

    mask = np.ones_like(img, dtype=bool)

    mask[y_min:y_max,x_min:x_max] = False

    
    img[mask] = 255

    return img
