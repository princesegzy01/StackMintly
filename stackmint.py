## Train data
## Test Data

#get the image
#pass it to  currency detector
#chuncks
#predicting

import zipfile
import sys

# train_data = "train_data.ell"

# if train_data[-3:] != "ell" :
#     print("Supply a valid ell file as train data")

# if zipfile.is_zipfile(train_data) == False :
#     print("Invalid Train Data supplied")

# zf = zipfile.ZipFile('train_data.ell')

# currency_data = zf.read("currency_detector.h5")
# serial_data = zf.read("serial_number_detector.h5")

# print(serial_data)
# sys.exit(0)


from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import os
import sys



def OneHotConverterResult(oneHotData):


    data = []
    for dir in sorted(os.listdir("dataset/training_set")):
        data.append(dir)
    
    values = array(data)

    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
 
    # binary encode
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

    inverted = label_encoder.inverse_transform([argmax(oneHotData)])
    return inverted





import rule_base
import currency_detector
from keras.preprocessing import image


image_path = "1000naira.jpeg"

image = image.load_img( image_path , target_size = (128, 128))

#predict the currency and get ml output
currency = currency_detector.predictCurrency(image)

#convert result from onehot to categorical data
currency_value = OneHotConverterResult(currency)
currency_value = currency_value[0]

# chunk and predict serial of the note passed
serial = rule_base.predictSingleImage(currency_value, image_path)
print(serial)


