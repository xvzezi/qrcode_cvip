'''
	// 1. do the decolor option 
	Mat channels[3];
	split(img, channels);	// split the channels

							// -- // 1.2 calculate the diff
	Mat diff[3];
	absdiff(channels[0], channels[1], diff[0]);
	absdiff(channels[0], channels[2], diff[1]);
	absdiff(channels[2], channels[1], diff[2]);
	Mat whole_diff = diff[0] + diff[1] + diff[2];
	whole_diff = whole_diff < color_variation_threshold;
	Mat decolor = channels[0] & whole_diff;

	// 2. edge getter
	Laplacian(decolor, decolor, decolor.depth(), 3);
	decolor &= whole_diff;
	GaussianBlur(decolor, decolor, Size(3, 3), 0);
	decolor &= whole_diff;
	decolor = decolor > 64;

'''
import numpy as np 
import cv2 as cv 

def pre_decolor(input_image):
	b, g, r = cv.split(input_image)
	diff0 = cv.absdiff(b, g)
	diff1 = cv.absdiff(b, r)
	diff2 = cv.absdiff(g, r)
	whole_diff = diff0 + diff1 + diff2 
	idx = whole_diff < 50
	whole_diff[idx] = 255
	whole_diff[~idx] = 0
	decolor = g & whole_diff

	return decolor, whole_diff



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

def black_area_checker(row, col, acc_wd):
	acc_sum = acc_wd[row+sq_size, col+sq_size] - acc_wd[row, col+sq_size] - acc_wd[row+sq_size, col] + acc_wd[row, col]
	acc_aver = acc_sum / sq_size / sq_size
	return acc_aver >= 100

def filter_with_dark(pp_image, whole_diff):
    # 1. sliding window to get the pic croped 
    # using 64 x 64 with stripe 48
    # in fact we have to do the crop things, when its not the muliplier of 64, the remains are ignored

	nrow, ncol = pp_image.shape 
	acc_whole_diff = cv.integral(whole_diff)
	snrow = nrow - sq_size + 1
	sncol = ncol - sq_size + 1
	row = 0 
	crops = []
	while row < snrow:
        #
		col = 0
		while col < sncol:
			if black_area_checker(row, col, acc_whole_diff):
				crops.append(pp_image[row:row+sq_size, col:col+sq_size])
			col += stride 
		# after 
		row += stride
	
	# 2. using small kernel to pre-filter out some parts
	pre_f_crops = []
	for crop_img in crops:
		if pre_filter(crop_img):
			pre_f_crops.append(crop_img)
    
	# print(len(crops), len(pre_f_crops))	

    # 3. feed into CNN as a whole group # ignored 
	return pre_f_crops


if __name__ == "__main__":
	# path = './resources/cap.jpg'
	# img = cv.imread(path) 
	# decolor, count = pre_decolor(img) 
	# decolor = cv.adaptiveThreshold(decolor, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 65, 10)
	# crops = filter_with_dark(decolor, count) 
	# print(len(crops))
	# cv.imshow('sd', decolor) 
	# cv.waitKey(0)

	import time 
	path = './resources/raw/'
	path2 = './resources/tmp/'

	for f in range(1, 51):
		file_path_in = path + str(f) +'.jpg'
		file_path_out = path2 + str(f) +'.jpg'

		img = cv.imread(file_path_in)
		decolor, count = pre_decolor(img)
		decolor = cv.adaptiveThreshold(decolor, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 65, 10)
		crops = filter_with_dark(decolor, count)
		print(len(crops))
		
    
	