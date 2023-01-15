import numpy as np
import cv2

def translate_image(image, shiftx, shifty):
    h,w = image.shape[:2]
    translation_matrix = np.float32([ [1,0,70], [0,1,110] ])   
    translated = cv2.warpAffine(img, translation_matrix, (w, h))
    return translated

def rotate_image(image, angle):
    h,w = image.shape[:2]
    cX,cY = (w//2,h//2)
    M = cv2.getRotationMatrix2D((cX,cY),angle,1)
    rotated = cv2.warpAffine(image,M , (w,h),flags=cv2.INTER_LINEAR)
    return rotated