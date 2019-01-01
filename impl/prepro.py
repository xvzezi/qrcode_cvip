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

def pre_decolor(input_image, thresh=50):
	b, g, r = cv.split(input_image)
	diff0 = cv.absdiff(b, g)
	diff1 = cv.absdiff(b, r)
	diff2 = cv.absdiff(g, r)
	whole_diff = diff0 + diff1 + diff2 
	idx = whole_diff < thresh
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

def pre_filter5(crop_img):
	# 1. threshold 
	edge_threshold = sq_size * 1.5 
	# 2. calculating 
	row = 2
	edge_amount = 0
	while row < 62:
		col = 2
		while col < 62:
			if crop_img[row, col] != 0:
				col += 1
				continue
			ver_sum1 = int(crop_img[row, col - 1]) + crop_img[row, col - 2] 
			ver_sum2 = int(crop_img[row, col + 1]) + crop_img[row, col + 2]
			if (ver_sum1 == 510 or ver_sum2 == 510) and ((ver_sum2 + ver_sum1) == 510):
				edge_amount += 1
			hor_sum1 = int(crop_img[row - 1, col]) + crop_img[row - 2, col] 
			hor_sum2 = int(crop_img[row + 2, col]) + crop_img[row + 1, col]
			if (hor_sum1 == 510 or hor_sum2 == 510) and ((hor_sum1 + hor_sum2) == 510):
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
		if pre_filter5(crop_img):
			pre_f_crops.append(crop_img)
    
	# print(len(crops), len(pre_f_crops))	

    # 3. feed into CNN as a whole group # ignored 
	return pre_f_crops

def edge_checker_by_integral(row, col, acc_wd):
	acc_sum = acc_wd[row+sq_size, col+sq_size] - acc_wd[row, col+sq_size] - acc_wd[row+sq_size, col] + acc_wd[row, col]
	acc_aver = acc_sum / 255 
	return acc_aver > (sq_size * 1.5)

def filter_with_integral(pp_image, whole_diff, acc_sub):
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
			if black_area_checker(row, col, acc_whole_diff) and edge_checker_by_integral(row, col, acc_sub):
				crops.append(pp_image[row:row+sq_size, col:col+sq_size])
			col += stride 
		# after 
		row += stride

    # 3. feed into CNN as a whole group # ignored 
	return crops



#################################################
# FIP part
def templateMatch(pattern, start, maxIndividualVariance = 0.7, averTotalVariance = 0.23):
	# basic static information 
	template = [1, 1, 3, 1, 1]
	templateLength = 7

	# begin matching
	patternLength = sum(pattern)
	if patternLength < templateLength * 2:
		# unit length of pattern must above 2
		return False
	
	unitWidth = patternLength / templateLength
	maxIndividualVariance *= unitWidth
	totalVariance = 0
	for i in range(5):
		# check each part
		stdLength = template[i] * unitWidth
		variance = abs(stdLength - pattern[(i+start)%5])
		if variance > maxIndividualVariance:
			return False
		totalVariance += variance
		
	# print("reach!!")
	if (totalVariance/patternLength) > averTotalVariance:
		return False
	
	return True 

def pattern_finder_in_crops(binMap):
	# new_crops = np.zeros(pp_image.shape, dtype=np.int8)

	# settings
	dark = 0
	light = 255

	# data
	coords = []
	height = binMap.shape[0]
	width = binMap.shape[1]

	# do it recursively 
	for i in range(height):
		pattern = [0, 0, 0, 0, 0]
		ptr = 0
		curColor = binMap[i][0]
		amount = 0
		for j in range(width):
			if binMap[i][j] == curColor:
				amount += 1
			else:
				pattern[ptr] = amount 
				ptr = (ptr+1)%5
				curColor = binMap[i][j]
				amount = 1 
				if curColor != dark:
					# the last color is dark
					if templateMatch(pattern, ptr):
						length = sum(pattern) // 2
						coords.append((i, j - length, length))
		# after check
		if curColor == dark:
			if templateMatch(pattern, ptr):
				length = sum(pattern) // 2
				coords.append((i, width - length - 1, length))
		
	# return the results 
	return coords 

	# return fip_pts_map

def lineCheck(xbMap, candidate):
	i = candidate[1]
	j = candidate[0]
	radius = int(candidate[2] * 1.2)

	f_h = xbMap.shape[0]
	f_w = xbMap.shape[1]
	
	left_border = j - radius 
	right_border = j + radius 

	if left_border < 0:
		left_border = 0
	if right_border > f_w:
		right_border = f_w

	acpt_range = (right_border - left_border) * 0.9 + left_border
	
	start = -1
	dark = 0
	pattern = []
	counter = 0
	curColor = 0
	for p in range(left_border, right_border):
		if start == -1:
			if xbMap[i][p] > 0:
				continue
			else:
				start = 0
		if curColor == xbMap[i][p]:
			counter += 1
		else:
			pattern.append(counter)
			counter = 1
			curColor = xbMap[i][p]
			if len(pattern) == 5:
				# begin to check 
				if p < acpt_range:
					# if i == 472:
					#     print("x")
					return False, 0
				else:
					return templateMatch(pattern, 0, averTotalVariance=1), 1
					# return True, 1
	return False, 2

def verticalCheck(binMap, candidates):
	xbMap = binMap.T
	checked = [] 
	acpt_false = 0
	match_false = 0
	other_false = 0
	total_radius = 0
	for candidate in candidates:
		c, code = lineCheck(xbMap, candidate)
		if code == 0:
			acpt_false += 1
		if code == 2:
			other_false += 1
		if c:
			checked.append(candidate)
			total_radius += candidate[2]
		elif code == 1:
			match_false += 1
	
	if len(checked) == 0:
		return checked

	aver_radius = total_radius // len(checked) 
	s_checked = []
	r_thresh = aver_radius * 0.3 
	for candidate in candidates:
		diff = abs(aver_radius - candidate[2])
		if diff < r_thresh:
			s_checked.append(candidate)

	print("acpt {}, match {}, other {}, checked {}, magic {}".format(acpt_false, match_false, other_false, len(checked), len(s_checked)))
	return s_checked

def mergePts(candidates):
	res = []
	for can in candidates:
		checker = True
		for r in res:
			dis = abs(r[0] - can[0]) + abs(r[1] - can[1]) 
			mean_r = (r[2] + can[2]) / 2 * 0.3 
			if dis < mean_r:
				checker = False 
				break 
		if checker:
			res.append(can)

	return res 

if __name__ == "__main__":
	path = './resources/cap.jpg'
	img = cv.imread(path) 
	for t in range(1, 256):
		decolor, diff = pre_decolor(img, thresh=t) 
		left = np.sum(diff == 255)
		print(left)
	
	# decolor, count = pre_decolor(img) 
	# decolor = cv.adaptiveThreshold(decolor, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 65, 10)
	# crops = filter_with_dark(decolor, count) 
	# print(len(crops))
	# cv.imshow('sd', decolor) 
	# cv.waitKey(0)

	# import time 
	# path = './resources/raw/'
	# path2 = './resources/tmp2/'
	# output_path = './resources/1.txt'
	# ofs = open(output_path, 'w')

	# for f in range(1, 51):
	# 	file_path_in = path + str(f) +'.jpg'
	# 	file_path_out = path2 + str(f) +'.jpg'

	# 	img = cv.imread(file_path_in)
	# 	decolor, count = pre_decolor(img)
	# 	decolor = cv.adaptiveThreshold(decolor, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 65, 10)
	# 	# ck = pattern_finder_in_crops(decolor) 
	# 	# ck = verticalCheck(decolor, ck)
	# 	# ck = mergePts(ck)
	# 	# for i in ck:
	# 	# 	cv.circle(decolor, (i[1], i[0]), i[2], 128, thickness=3)
	# 	# cv.imshow(' ', decolor)
	# 	# cv.waitKey(0)
	# 	# print(len(ck))
	# 	# exit(0)
	# 	_st = time.time()
	# 	# sub_img = cv.Laplacian(decolor, ddepth=cv.CV_8UC1, ksize=5)
	# 	# sub_img[sub_img > 0] = 255
	# 	# acc_sub = cv.integral(sub_img)
	# 	# crops = filter_with_integral(decolor, count, acc_sub)
	# 	k = pattern_finder_in_crops(decolor) 
	# 	ck = verticalCheck(decolor, k)
	# 	ck = mergePts(ck)
	# 	_acc = time.time() - _st 
	# 	ofs.write('{index}\t{ROI}\t{time}\n'.format_map({
	# 		'index':f,
	# 		'ROI':len(ck),
	# 		'time':_acc
	# 	}))
	# 	for i in ck:
	# 		cv.circle(decolor, (i[1], i[0]), i[2], 128, thickness=1)
	# 	cv.imwrite(file_path_out, decolor) 
	# ofs.close()
		
    
	