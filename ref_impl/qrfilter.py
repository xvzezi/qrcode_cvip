'''
Finding ROIs
------------------
1. sliding window 
2. filters to pre filtering 
3. using CNN


'''
import numpy as np 
import cv2 as cv 

# global vars
stride = 48 
sq_size = 64

def pre_filter(crop_img):
    # 1. threshold 
    edge_threshold = sq_size * 1.5 
    # 2. calculating 
    row = 1
    edge_amount = 0
    while row < 63:
        col = 1
        while col < 63:
            if crop_img[row, col] != 0:
                col += 1
                continue
            ver_sum = int(crop_img[row, col - 1]) + crop_img[row, col] + crop_img[row, col + 1]
            if ver_sum == 255:
                edge_amount += 1
            hor_sum = int(crop_img[row - 1, col]) + crop_img[row, col] + crop_img[row + 1, col]
            if hor_sum == 255:
                edge_amount += 1
            col += 1
        row += 1
    return edge_amount > edge_threshold

def qr_filter(pp_image):
    # 1. sliding window to get the pic croped 
    # using 64 x 64 with stripe 48
    # in fact we have to do the crop things, when its not the muliplier of 64, the remains are ignored

    nrow, ncol = pp_image.shape 
    print(nrow, ncol)
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
    pre_f_crops = []
    for crop_img in crops:
        if pre_filter(crop_img):
            pre_f_crops.append(crop_img)
    
    print(len(crops), len(pre_f_crops))

    # 3. feed into CNN as a whole group # ignored 
    return pre_f_crops



if __name__ == "__main__":
    img = cv.imread("./resources/sel_bin.jpg", 0)
    qr_filter(img) 
