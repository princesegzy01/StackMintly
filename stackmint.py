## Train data
## Test Data

#get the image
#pass it to  currency detector
#chuncks
#predicting

import zipfile
import sys
import utils


train_data = "train_data.ell"
isValidData = utils.houseWarming(train_data)

#sys.exit(0)
import os

import rule_base
import currency_detector
from keras.preprocessing import image


image_path = "1000naira.jpeg"

image = image.load_img( image_path , target_size = (128, 128))

#predict the currency and get ml output
currency = currency_detector.predictCurrency(image)

#convert result from onehot to categorical data
currency_value = utils.OneHotConverterResult(currency, "dataset/training_set")
currency_value = currency_value[0]

# chunk and predict serial of the note passed
serial = rule_base.predictSingleImage(currency_value, image_path)
print(serial)


