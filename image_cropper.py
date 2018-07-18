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
from tqdm import tqdm

import image_slicify
import confusion_return



response_list = []

import os
dirname = 'test'
#os.mkdir(dirname)


for iteration in tqdm(range(20, 100, 5)) : 

    tiles = image_slicify.slice('500naira.jpeg', iteration, save = False)

    for i, tile in tqdm(enumerate(tiles)) :

        #print(tile)
        #print(tile.image.show())
        # convert to  grayscale
        img = cv2.cvtColor(np.array(tile.image), cv2.COLOR_RGB2BGR)

        img_new_height, img_new_width, _  = img.shape
        img = cv2.resize(img,(10 * img_new_width, 10 * img_new_height), interpolation = cv2.INTER_LINEAR)

        im_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        response = confusion_return.MultiImageOperation(im_gray, i, iteration)

        if len(response) > 0 :
            for res in tqdm(response) : 
                response_list.append(response)
        

    if len(response_list) == 0 :
        print ("Cropping returns empty serial") 
        sys.exit(0)

sys.exit(0)

from sklearn.externals import joblib
from skimage.feature import hog

#Find contours in the image

for k,im in enumerate(response_list) :
    
    #print("yyyy", k)
    #print(im["sn"])

    im_bw = im["img"]

    #print("Length of tiles is {}".format(len(tiles)))
    color_image = tiles[im["sn"]].image
    im_color = cv2.cvtColor(np.array(color_image), cv2.COLOR_RGB2BGR)

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
        # Resize the image
        #roi = cv2.resize(roi, (28, 28))
        #roi = cv2.dilate(roi, (3, 3))
        # Calculate the HOG features
        #roi_hog_fd = hog(roi, block_norm='L2-Hys', orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualize=False)
        #nbr = clf.predict(np.array([roi_hog_fd], 'float64'))
        #cv2.putText(im, str(int(nbr[0])), (rect[0], rect[1]),cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)
    
        #cv2.imshow("Resulting Image with Rectangular ROIs", im_color)
    
    cv2.imwrite("_" + str(im["sn"]) +  "_" + im["type"] + "_" + "im_color.png", im_bw)

''' 
        idx = 0

        for c in ctrs:
            x,y,w,h = cv2.boundingRect(c)
            if w>50 and h>50:
                idx+=1
                new_img = im_bw[y:y+h,x:x+w]
                cv2.imwrite(str(idx) + "_" + str(im["sn"]) +  "_" + im["type"] + "_" +  '.png', new_img)
                #print("reach here")
                #cv2.imshow( str(im["sn"]) + "_" + im["type"],new_img)
        '''
        








'''
image = cv2.imread("test/12.png")
#gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
edged = cv2.Canny(image, 10, 250)

cv2.imwrite("edged.png", edged)

(_,cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


idx = 0
for c in cnts:
    x,y,w,h = cv2.boundingRect(c)
    if w>50 and h>50:
        idx+=1
        new_img=image[y:y+h,x:x+w]
        cv2.imwrite(str(idx) + '.png', new_img)
        print("reach here")
cv2.imshow("im",image)

cv2.imwrite("f_image.png", image)
'''

'''
for i, c in enumerate(cnts):
    area = cv2.contourArea(c)
    if area > 100:
        cv2.drawContours(image, cnts, i, (255, 0, 0), 3)
cv2.imwrite('Photos/output3.jpg', image)
'''


sys.exit(0)

img = "500naira.jpeg"


with open('rules.json') as f:
    data = json.load(f)

#pprint(data)

ml_output = "1000NGN"
#pprint(data[ml_output])


#sys.exit(0)
#r = x , y , width , height
#r is the location of x and y with higth
#and width of the serial box


color_to_remove = data[ml_output]["color_to_remove"]
serial_length = data[ml_output]["serial_length"]
expected_dimension = data[ml_output]["expected_dimension"]
serial_id_location = data[ml_output]["serial_id_location"]
r = data[ml_output]["r"]

#sys.exit(0)



im = cv2.imread("1000naira.jpeg")



img_height, img_width, _ = im.shape



if abs(int(expected_dimension['height']) - int(img_height)) > 10  :
    print("Image height does not conform to the expected height")

if abs(int(expected_dimension['width']) - int(img_width)) > 10  :
    print("Image width does not conform to the expected width")    



print("Done and dusted")

img = im[r[0]:r[0]+r[2], r[1]:r[1]+r[3]]


cv2.imwrite('raw extracted.png', img)


img_new_height, img_new_width, _  = img.shape
img = cv2.resize(img,(10 * img_new_width, 10 * img_new_height), interpolation = cv2.INTER_LINEAR)

cv2.imwrite('increased extracted.png', img)


#img_extracted = cv2.imread('increased extracted.png')
#img_extracted[np.where((im == [0,0,0]).all(axis = 2))] = [0,33,166]
#cv2.imwrite('output.png', img_extracted)

im_gray = cv2.imread('increased extracted.png')
im_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imshow("xxx", im_gray)

(thresh, im_bw) = cv2.threshold(im_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
#(thresh, im_bw) = cv2.threshold(im_gray, 128, 255, cv2.THRESH_BINARY_INV)


cv2.imwrite("im_bw.png", im_bw)

#print (r)
#r = cv2.boundingRect()
#print("Great")

#result = pytesseract.image_to_string(Image.open("500naira_gray1_big.png"))

## Py Tesseract Result
print("")
print("starting Tesseract Result ===================================================")
print("")

result = pytesseract.image_to_string(im_bw)
print("Binarize", result)

if serial_id_location == "front":
    serial =  result[-serial_length : ]
else:
    serial = result[: -serial_length ]


print("Full serial number is ", result)
print("serial number is ", serial)

print("")
print("Ending Tesseract Result ===================================================")
print("")


#use HOG


#USE MNIST

#cv2.waitKey(0)
#cv2.destroyAllWindows()