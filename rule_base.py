import argparse
import cv2
import numpy as np
#import pytesseract
import json
from pprint import pprint
import sys
#import image_slicer
from PIL import Image
import os
from keras.models import load_model
from keras.preprocessing import image


from sklearn.externals import joblib
from skimage.feature import hog
from sklearn.svm import LinearSVC
import time

import currency_detector
import serial_number_predictor
import utils


#Sort Contours on the basis of their x-axis coordinates in ascending order
def sort_contours(contours):
    # construct the list of bounding boxes and sort them from top to bottom
    boundingBoxes = [cv2.boundingRect(c) for c in contours]
    (contours, boundingBoxes) = zip(*sorted(zip(contours, boundingBoxes), key=lambda b: b[1][0], reverse=False))
    # return the list of sorted contours
    return contours

# ============================================================================



def predictSingleImage(ml_output, img) :
    # ml_output = "1000NGN"
    # img = "1000naira.jpeg"

    with open('rules.json') as f:
        data = json.load(f)

    #pprint(data)


    #pprint(data[ml_output])



    color_to_remove = data[ml_output]["color_to_remove"]
    serial_length = data[ml_output]["serial_length"]
    expected_dimension = data[ml_output]["expected_dimension"]
    serial_id_location = data[ml_output]["serial_id_location"]
    r = data[ml_output]["r"]


    if img[-4:] != "jpeg":
        pass


    #print(img)
    im = cv2.imread(img)


    img_height, img_width, _ = im.shape



    if abs(int(expected_dimension['height']) - int(img_height)) > 10  :
        print("Image height does not conform to the expected height")

    if abs(int(expected_dimension['width']) - int(img_width)) > 10  :
        print("Image width does not conform to the expected width")    

    #print("Done and dusted")



    img = im[r[0]:r[0]+r[2], r[1]:r[1]+r[3]]


    #cv2.imwrite('raw extracted.png', img)


    img_new_height, img_new_width, _  = img.shape
    try:
        img = cv2.resize(img,(10 * img_new_width, 10 * img_new_height), interpolation = cv2.INTER_LINEAR)
    except:
        pass

    #img_extracted = cv2.imread('increased extracted.png')
    #img_extracted[np.where((im == [0,0,0]).all(axis = 2))] = [0,33,166]
    #cv2.imwrite('output.png', img_extracted)

    im_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


    im_blur = cv2.GaussianBlur(im_gray,(5,5),0)

    (thresh, im_bw) = cv2.threshold(im_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)



    cv2.imwrite(os.path.join("result_directly" , "im_bw.png"), im_bw)

    kernel = np.ones((5,5), np.uint8)

    img_erosion = cv2.erode(im_bw, kernel, iterations=1)
    img_dilation = cv2.dilate(im_bw, kernel, iterations=1)

    #cv2.imwrite(os.path.join("result_directly" , str(time.time())+"_img_erosion.png"), img_erosion)
    #cv2.imwrite(os.path.join("result_directly" , str(time.time())+"_img_dilation.png"), img_dilation)




    #result = pytesseract.image_to_string(im_bw)
    #result= "x"
    #print("Binarize", result)



    #im_color = cv2.cvtColor(np.array(color_image), cv2.COLOR_RGB2BGR)

    edged = cv2.Canny(im_bw, 10, 250)
    #cv2.imshow("edged", edged)


    #cv2.imshow("original", im_bw)

    # Find contours in the image
    _,ctrs, hierarchy  = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    #hierarchy = hierarchy[0]  # get the actual inner list of hierarchy descriptions

    # Grab only the innermost child components
    #inner_contours = [c[0] for c in zip(ctrs, hierarchy) if c[1][3] > 0]




    sorted_contours = sort_contours(ctrs)

    #dispaly rectangle around contours found on the image
    #rects = [cv2.boundingRect(ctr) for ctr in ctrs]
    rects = [cv2.boundingRect(ctr) for ctr in sorted_contours]

    # For each rectangular region, calculate HOG features and predict
    # the digit using Linear SVM.

    #print("Lenght of rectangles detected in the countors is  L ", len(rects) ) 


    #Load the classifier
    #clf = joblib.load('digits_cls.pkl')

    counter = 0
    roi_list = []

    '''
    for k, rect in enumerate(rects):
        (x,y,w,h) = rect

        
        if h < 80 :
            #print("countours is less than 100")
            continue
        
        roi = edged[x:x+w, y:y+h]
        roi = img[x:x+60, y:y+90]

        im_height, im_width = roi.shape [:2]
        
        
        #if im_height * im_width == 0:
        #    print("empty, Just continue")
        #    continue

        

        #print(str(counter), "ROI Dimension : ============================================")
        #print("width : ", im_width)
        #print("height : ", im_height)




        #roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
        #roi = cv2.dilate(roi, (3, 3))
        # Calculate the HOG features
        #roi_hog_fd = hog(roi, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualize=False, block_norm='L2-Hys')
        #nbr = clf.predict(np.array([roi_hog_fd], 'float64'))

        #cv2.putText(img, str(int(nbr[0])), (rect[0], rect[1]),cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)

        #cv2.rectangle(img, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)  
        cv2.putText(img, str(counter), (rect[0], rect[1]),cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)
        
        #print("NBR is  : " +  str(counter), nbr)

        counter += 1
        roi_list.append(roi)


        #cv2.imwrite(str(counter) + "_im_roi.png", roi)

        file_name = str(counter) + ".png"

        cv2.imshow(str(counter) + "_img", roi)

        #path = 'D:/OpenCV/Scripts/Images'
        #cv2.imwrite(os.path.join("result" , file_name), roi)



    print("Counter is  : ", counter)
    cv2.imshow("xx", img)
    cv2.waitKey(0)
    '''


    '''
    #from skimage.io import imread
    from skimage.filters import threshold_otsu
    from skimage.transform import resize
    from matplotlib import pyplot as plt

    fig = plt.figure(figsize=(12, 12))

    print(roi_list)
    for i, roi in enumerate(roi_list):
        ax = fig.add_subplot(10, 10, i + 1, xticks=[], yticks=[])
        try:
            ax.imshow(roi)  
        except ValueError:
            pass
        #if 'predicted_char' in what_to_plot:
            #ax.text(-6, 8, str(what_to_plot['predicted_char'][i]), fontsize=22, color='red')
    plt.suptitle("Total Objects Detected", fontsize=20)
    plt.show() 
    '''

    '''
    for i, rect in enumerate(rects):
        # Draw the rectangles
        cv2.rectangle(img, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3) 
        # Make the rectangular region around the digit
        leng = int(rect[3] * 1.6)
        pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
        pt2 = int(rect[0] + rect[2] // 2 - leng // 2)
        roi = img[pt1:pt1+leng, pt2:pt2+leng]

        im_height, im_width,_ = roi.shape

        # Resize the image
        roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
        roi = cv2.dilate(roi, (3, 3))
        # Calculate the HOG features
        roi_hog_fd = hog(roi, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualize=False, block_norm='L2-Hys')
        nbr = clf.predict(np.array([roi_hog_fd], 'float64'))
        cv2.putText(img, str(int(nbr[0])), (rect[0], rect[1]),cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)

        #cv2.imwrite(str(i) + "_im_roi.png", roi)
        #cv2.imwrite(str(i) + "_im_roi.png", roi.image)
    cv2.imwrite("im_bw.png", img)

    cv2.waitKey(0)
    '''

    
    train_data = "train_data.ell"
    _, serial_data_file_name, = utils.returnTrainDataFIleName(train_data)
    classifier = load_model(serial_data_file_name)

    result = []
    #To be considered for Use
    idx = 0
    for c in sorted_contours:
        x,y,w,h = cv2.boundingRect(c)

        if h < 80 :
            #print("countours height is less than 80")
            continue

        if w < 50 :
            #print("countours width is less than 50")
            continue

        if w>50 and h>70:
            idx+=1
            #new_img = im_bw[y:y+h,x:x+w]
            new_img = img_dilation[y:y+h,x:x+w]
            
            
            #cv2.imwrite(str(idx) + '.png', new_img)
            
            #filename = str(idx) + '.png'
            #filename = str(time.time()) + "_" + str(idx) + '.png'
            
            roi = cv2.resize(new_img, (28, 28), interpolation=cv2.INTER_AREA)
            
            band_3 = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)

            #cv2.imwrite(os.path.join("result_chops", filename), roi)
            #print(band_3.shape)
            
            # cv2.imshow("xxx", band_3)
            # cv2.waitKey(0)
            # continue

            className = serial_number_predictor.customDigitPredictor(classifier, band_3)
            result.append(str(className))

            #roi = cv2.dilate(roi, (3, 3))
            # Calculate the HOG features
            #roi_hog_fd = hog(roi, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualize=False, block_norm='L2-Hys')
            #nbr = clf.predict(np.array([roi_hog_fd], 'float64'))

            #cv2.imshow(str(idx) + "_im",new_img)

    final_result = "".join(result)

    return final_result