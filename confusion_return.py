import cv2
import pytesseract
import os
from tqdm import tqdm


def MultiImageOperation(image_gray, slice_num, iteratable_num) :
    
    #list to store all the result of pytesseract
    result_output_list = []

    #directory to store chopped tiles
    dirname = 'test'

    image_type = ["Normal","GussiaBlur"]

    for blur_type in tqdm(image_type) :
        
        if blur_type == "Normal" : 
            im_gray = image_gray
        
        if blur_type == "GussiaBlur" : 
            #Gussian Blur Image
            im_gray = cv2.GaussianBlur(im_gray,(5,5),0)

        thresh_list = ["THRESH_BINARY","THRESH_BINARY_INV","THRESH_TRUNC","THRESH_TOZERO","THRESH_TOZERO_INV","THRESH_BINARY+THRESH_OTSU","THRESH_BINARY_INV+THRESH_OTSU","THRESH_TRUNC+THRESH_OTSU","THRESH_TOZERO+THRESH_OTSU","THRESH_TOZERO_INV+THRESH_OTSU",]
        
        for l_threshold in tqdm(thresh_list) :
            
            if l_threshold == "THRESH_BINARY" :
                (thresh, im_bw) = cv2.threshold(im_gray, 128, 255, cv2.THRESH_BINARY)

            if l_threshold == "THRESH_BINARY_INV" : 
                (thresh, im_bw) = cv2.threshold(im_gray, 128, 255, cv2.THRESH_BINARY_INV)
            
            if l_threshold == "THRESH_TRUNC" :
                (thresh, im_bw) = cv2.threshold(im_gray, 128, 255, cv2.THRESH_TRUNC)
    
            if l_threshold == "THRESH_TOZERO" :
                (thresh, im_bw) = cv2.threshold(im_gray, 128, 255, cv2.THRESH_TOZERO)
                
            if l_threshold == "THRESH_TOZERO_INV" :
                (thresh, im_bw) = cv2.threshold(im_gray, 128, 255, cv2.THRESH_TOZERO_INV)
    
            if l_threshold == "THRESH_BINARY+THRESH_OTSU" :
                (thresh, im_bw) = cv2.threshold(im_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                
            if l_threshold == "THRESH_BINARY_INV+THRESH_OTSU" :
                (thresh, im_bw) = cv2.threshold(im_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
            if l_threshold == "THRESH_TRUNC+THRESH_OTSU" :
                (thresh, im_bw) = cv2.threshold(im_gray, 0, 255, cv2.THRESH_TRUNC + cv2.THRESH_OTSU)

            if l_threshold == "THRESH_TOZERO+THRESH_OTSU" :
                (thresh, im_bw) = cv2.threshold(im_gray, 0, 255, cv2.THRESH_TOZERO + cv2.THRESH_OTSU)
    
            if l_threshold == "THRESH_TOZERO_INV+THRESH_OTSU" :
                (thresh, im_bw) = cv2.threshold(im_gray, 0, 255, cv2.THRESH_TOZERO_INV + cv2.THRESH_OTSU)
    
            file_name = str(iteratable_num) + "_" + str(slice_num) +  "-" + blur_type + "_" + l_threshold + ".png"
            result = pytesseract.image_to_string(im_bw)

            #Get total number of digits in result
            #because most serial contains more digits and less letters
            total_numbers = sum(c.isdigit() for c in result)

            #get all the result string
            total_result_string = result.encode('utf-8')

            #get the length of the whole result
            total_result_length = len(total_result_string)

            if total_numbers >= 5 and total_result_length <= 10 :  
                #Save the image if digit is found
                cv2.imwrite(os.path.join(dirname, file_name), im_bw)
                data = { "sn" : slice_num , "img" : im_bw, "file_name" : file_name, "len_numbers" : total_numbers,  "result" : result, " image_type : ": blur_type, " l_threshold : " : l_threshold} 
                result_output_list.append(data)
                print( "Iteration : ", iteratable_num,  " Slice num : " , slice_num , " Result :  ", result, " image_type : ", image_type, " l_threshold : ", l_threshold)

    return result_output_list


