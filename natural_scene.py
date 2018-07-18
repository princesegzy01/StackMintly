import cv2
from matplotlib import pyplot as plt
import sys

img = cv2.imread("1000naira.jpeg")

#cv2.imshow("xxx", img)


im_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

im_blur = cv2.GaussianBlur(im_gray,(5,5),0)
(thresh, im_bw) = cv2.threshold(im_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

im_bw = cv2.Canny(im_bw, 10, 250)

_,ctrs, _  = cv2.findContours(im_bw.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#dispaly rectangle around contours found on the image
rects = [cv2.boundingRect(ctr) for ctr in ctrs]

print("Lenght of rectangles detected in the countors is : ", len(rects) ) 

for i, rect in enumerate(rects):
    # Draw the rectangles
    cv2.rectangle(im_blur, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3) 
    # Make the rectangular region around the digit
    leng = int(rect[3] * 1.6)
    pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
    pt2 = int(rect[0] + rect[2] // 2 - leng // 2)
    roi = im_bw[pt1:pt1+leng, pt2:pt2+leng]
#cv2.imwrite("im_bw.png", im_bw)





cv2.imshow("lao.png", im_blur)
cv2.waitKey(0)
