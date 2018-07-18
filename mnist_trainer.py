#MNINST IMPORTATION
from sklearn.externals import joblib
from sklearn import datasets
from skimage.feature import hog
from sklearn.svm import LinearSVC
import numpy as np
from tqdm import tqdm
import sys

'''
from shutil import copyfileobj
from six.moves import urllib
from sklearn.datasets.base import get_data_home
import os

def fetch_mnist(data_home=None):
    mnist_alternative_url = "https://github.com/amplab/datascience-sp14/raw/master/lab7/mldata/mnist-original.mat"
    data_home = get_data_home(data_home=data_home)
    data_home = os.path.join(data_home, 'mldata')
    if not os.path.exists(data_home):
        os.makedirs(data_home)
    mnist_save_path = os.path.join(data_home, "mnist-original.mat")
    if not os.path.exists(mnist_save_path):
        mnist_url = urllib.request.urlopen(mnist_alternative_url)
        with open(mnist_save_path, "wb") as matlab_file:
            copyfileobj(mnist_url, matlab_file)

fetch_mnist()
'''

dataset = datasets.fetch_mldata("MNIST Original")

#Store features and label in numpy array
features = np.array(dataset.data, 'int16')
labels = np.array(dataset.target, 'int16')

#Calculate the HOG features of each image in a database
#and save them in another numpy array
list_hog_fd = []

for feature in tqdm(features) :
    dg = hog(feature.reshape((28,28)), orientations=9, pixels_per_cell=(14,14), cells_per_block=(1,1), visualise=False)
    list_hog_fd.append(dg)

hog_features = np.array(list_hog_fd, 'float64')

#Creating the classifier object
clf = LinearSVC()

#perform the training
clf.fit(hog_features, labels)


#save the training result
joblib.dump(clf, "digits_cls.pkl", compress=3)

print("Done training mnist")
print(labels)