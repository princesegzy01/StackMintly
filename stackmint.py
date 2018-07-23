

import zipfile
import sys
import utils



import argparse

parser = argparse.ArgumentParser(add_help=True, description="Stackmint is command line application that uses machine learning algorithm to predict currency and output other related results in json format ")
parser.add_argument("train_file", help="Supply the .ell train file")
parser.add_argument("image_file", help="Path to the image you want to predict")
parser.add_argument('-v','--version', action='version', version='%(prog)s 1.0')

args = parser.parse_args()

#print(args.train_file)
#print(args.image_file)


#train_data = "train_data.ell"
train_data = args.train_file
isValidData = utils.houseWarming(train_data)

#sys.exit(0)
import os

import rule_base
import currency_detector
from keras.preprocessing import image


#image_path = "1000naira.jpeg"

image_path = args.image_file

image = image.load_img( image_path , target_size = (128, 128))

#predict the currency and get ml output
currency = currency_detector.predictCurrency(train_data, image)

#convert result from onehot to categorical data
currency_value = utils.OneHotConverterResult(currency, "dataset/training_set")
currency_value = currency_value[0]

# chunk and predict serial of the note passed
serial = rule_base.predictSingleImage(currency_value, image_path)
print(serial)


