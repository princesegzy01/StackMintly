import argparse
import cv2
import numpy as np
import pytesseract
import json
from pprint import pprint
import sys
#import image_slicer
from PIL import Image
import os

import image_slicify
import confusion_return




img = "1000naira.jpeg"

im = cv2.imread(img)

im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

im_blur = cv2.GaussianBlur(im_gray,(5,5),0)
(thresh, im_bw) = cv2.threshold(im_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


result = pytesseract.image_to_string(im_bw)
print("Binarize", result)


#im_color = cv2.cvtColor(np.array(color_image), cv2.COLOR_RGB2BGR)

edged = cv2.Canny(im_bw, 10, 250)

#cv2.imshow("original", im_bw)

# Find contours in the image
_,ctrs, _  = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#dispaly rectangle around contours found on the image
rects = [cv2.boundingRect(ctr) for ctr in ctrs]

# For each rectangular region, calculate HOG features and predict
# the digit using Linear SVM.

print("Lenght of rectangles detected in the countors is  L ", len(rects) ) 

for rect in rects:
    # Draw the rectangles
    cv2.rectangle(im_bw, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3) 
    # Make the rectangular region around the digit
    leng = int(rect[3] * 1.6)
    pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
    pt2 = int(rect[0] + rect[2] // 2 - leng // 2)
    roi = im_bw[pt1:pt1+leng, pt2:pt2+leng]


cv2.imwrite("im_bw.png", im_bw)

'''
idx = 0
for c in ctrs:
    x,y,w,h = cv2.boundingRect(c)
    if w>50 and h>50:
        idx+=1
        new_img= edged[y:y+h,x:x+w]
        cv2.imwrite(str(idx) + '.png', new_img)
        print("reach here")
#cv2.imshow("im",image)
'''

    




