'''
Pre process of the input image 
-------------
1. Gray Scaling

2. binarized

'''
import numpy as np 
import cv2 as cv 

###############################
# output method
def pre_process(input_image):
    # grey scaling according to the formula 
    input_b, input_g, input_r = cv.split(input_image)
    grey = input_r * 0.299 + input_g * 0.587 + input_b * 0.114
    grey = grey.astype(np.uint8)
    nrow, ncol = grey.shape 
    
    # thresholding 
    dp = 0.85 
    radius = 32
    inc = cv.integral(grey)
    
    for i in range(nrow):
        row = grey[i]
        top_i = i - radius 
        bot_i = i + radius + 1
        if top_i < 0:
            top_i = 0
        if bot_i > nrow:
            bot_i = nrow 
        inc_top_row = inc[top_i]
        inc_bot_row = inc[bot_i]
        height = bot_i - top_i
        for j in range(ncol):
            left = j - radius
            right = j + radius + 1
            if left < 0:
                left = 0
            if right > ncol:
                right = ncol 
            sq_sum = inc_bot_row[right] - inc_bot_row[left] - inc_top_row[right] + inc_top_row[left]
            width = right - left 
            pixel_amnt = height * width 
            sq_aver = sq_sum / pixel_amnt 
            thres = sq_aver * dp
            if row[j] < thres:
                row[j] = 0
            else:
                row[j] = 255
    
    # grey = cv.adaptiveThreshold(grey, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 65, 10)
    # cv.imshow('bin', grey)
    # cv.waitKey(0)
    return grey

if __name__ == "__main__":
    import time 
    path = './resources/raw/'
    output_path = './resources/tmp/3rd'
    
    _acc = 0
    for f in range(1, 51):
        file_path_in = path + str(f) +'.jpg'
        img = cv.imread(file_path_in)
        _st = time.time() 
        grey = pre_process(img)
        _acc += time.time() - _st 
        print(f)
    
    print('time:', _acc)