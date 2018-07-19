import zipfile
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from numpy import array
from numpy import argmax
import sys
import os


def returnTrainDataFIleName(archive_file):
    archive = zipfile.ZipFile(archive_file)
    return archive.filelist[0].filename, archive.filelist[1].filename 


def houseWarming(train_data_file_name):
    
    #Check if file ends with .ell i.e ellcrys
    if train_data_file_name[-3:] != "ell" :
        print("Supply a valid ell file as train data")
        sys.exit(0)

    #Check if file is a valid zipfile
    if zipfile.is_zipfile(train_data_file_name) == False :
        print("Invalid Train Data supplied")
        sys.exit(0)

    #Read the zipfile
    archive = zipfile.ZipFile(train_data_file_name)

    #Rea the content and check the size of empty files
    for f in archive.infolist(): 
        if f.file_size == 0 :
            print("Train Data returns one or more empty file, please re-download")
            sys.exit(0)

    #Content of archive must always equals to 2
    if len(archive.namelist()) != 2 :
        print("Trained data tampered, Please redownload")
        sys.exit(0)
    
    return True


def OneHotConverterResult(oneHotData, training_set_dir):
    data = []
    for dir in sorted(os.listdir(training_set_dir)):
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





