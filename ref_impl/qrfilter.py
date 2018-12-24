'''
Finding ROIs
------------------
1. sliding window 
2. filters to pre filtering 
3. using CNN


'''
import numpy as np 
import cv2 as cv 

def qr_filter(pp_image):
    # 1. sliding window to get the pic croped 
    # using 64 x 64 with stripe 48
    # in fact we have to do the crop things, when its not the muliplier of 64, the remains are ignored
    stride = 48 
    sq_size = 64
    nrow, ncol, nch = pp_image.shape 
    print(nrow, ncol, nch)
    snrow = nrow - sq_size + 1
    sncol = ncol - sq_size + 1
    row = 0 
    crops = []
    while row < snrow:
        #
        col = 0
        while col < sncol:
            crops.append(pp_image[row:row+sq_size, col:col+sq_size])
            col += stride 
        # after 
        row += stride

    # 2. using small kernel to pre-filter out some parts


    # 3. feed into CNN as a whole group 
    return []



if __name__ == "__main__":
    img = cv.imread("./resources/sel_bin.jpg")
    qr_filter(img) 
