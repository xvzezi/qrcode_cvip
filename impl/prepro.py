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