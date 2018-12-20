'''
Paper: 
--------------
Method of localization of an QR Code under complex environment:

        'Input Image'       ________
        |
        'Gray Scaling'          Pre-processing
        |
        'Binarized'         ________
            |
            \/
        'Sliding Window'    ________
        |
        'Pre-filter'            Find good ROIs
        |
        'CNN-filter'        ________
            |
            \/
        'Connect Components'________
        |
        'FIP finder'            Find the QR Code
        |
        'Give out candidates'_______

'''

from prepro import pre_process
from qrfilter import qr_filter

def DetectQRCode(input_img):
    # 1. pre-processing
    pp_img = pre_process(input_img)

    # 2. Find good ROIs
    roi_img = qr_filter(pp_img)

    # 3. Find the QRCode within it 
    # this part will using an existing method to do.
    # in order to demonstrate the effect this method 
    # has, we will hava a comparison onto the precision
    # and detection time as an experiment. 
    # it will be done out of this module, for now 
    # considering the c++ opencv4.0's detection method

    '''
        Return: An image crop of the origin
    '''
    return roi_img