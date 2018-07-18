from keras.datasets import mnist
import matplotlib.pyplot as plt
import tensorflow as tf

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.utils import np_utils
import numpy 
import cv2
import sys



from keras.preprocessing import image



img = cv2.imread("result_chops/4.png")
img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
img_blur = cv2.GaussianBlur(img_gray,(5,5),0)
img_resize = cv2.resize(img_blur,(28, 28)).flatten()


img = img_resize/255

features = numpy.array([img])
#cv2.waitKey(0)
#sys.exit(0)

def baseArtificialNeuralNetwork(number_pixel, num_classes):
    
    #Create a model
    clf = Sequential()

    #Add input Layer
    clf.add(Dense(number_pixel, input_dim = 784 , activation='relu' ))

    #Add output layer
    clf.add(Dense(units = num_classes, activation='softmax'))

    #Compile Model
    clf.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    #Return the model
    return clf


def ConvolutionalNeuralNetwork(input_matrix):
    cnn = Sequential()

    cnn.add(Conv2D(32,  (3,3), input_shape = (1, 28,28), activation = 'relu'))
    cnn.add(MaxPooling2D(pool_size=(2,2)))
    cnn.add(Flatten())

    cnn.add(Dense(units= 128, activation='relu'))

    cnn.add(Dense(units= 50, activation='relu'))

    cnn.add(Dense(units= 10, activation='softmax'))

    cnn.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return cnn


#create a seed so we can reproduce the result
seed = 9
numpy.random.seed(seed)

#Load / Download mnist dataset 
(X_train, y_train ), (X_test , y_test) = mnist.load_data()


#flatten images to ove dimension vector of 28 x 28 = 784
number_pixel = X_train.shape[1] * X_train.shape[2]

X_train = X_train.reshape(X_train.shape[0], number_pixel).astype('float32')
X_test = X_test.reshape(X_test.shape[0], number_pixel).astype('float32')


#We need to normalize our greyscale pixel value of  0 and 255 to fit between 0 and 1
X_train =  X_train / 255
X_test = X_test / 255


#convert the one hot encoding to categorical output using keras np.utils

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

num_classes = y_test.shape[1]

#create the model object
model =  baseArtificialNeuralNetwork(number_pixel, num_classes)

#fit the model to the train data
model.fit(X_train, y_train, epochs=10, batch_size= 200, verbose= 2)

#evaluate the model on the test data
scores = model.evaluate(X_test, y_test, verbose=0)



print("Baseline Error: %.2f%%" % (100-scores[1]*100))

print("Scores ", scores)



#print(numpy.asarray(img_resize).reshape((784,)))

#pred = model.predict(numpy.asarray(img_resize).reshape((784,)) )


#model.predict()

#result = model.predict_classes(image_data)
#print("result is :", pred )

#print(X_test[0].shape)
#print(image_data.shape)


probs = model.predict(features)

pred_class = model.predict_classes(features)


prediction = probs.argmax(axis=0)

print("Probs : ", probs)
print("prediction : ", prediction)
print("prediction_class : ", pred_class)



print("CNN =========================================")

img_resize = cv2.resize(img_blur,(28, 28))
img = img_resize/255

features = numpy.array([img])

(X_train, y_train), (X_test, y_test) = mnist.load_data()

# reshape to be [samples][pixels][width][height]
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32')

X_train = X_train / 255
X_test = X_test / 255



y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]


cn = ConvolutionalNeuralNetwork(features)
cn.fit(X_train, y_train, validation_data=(X_test, y_test) ,  batch_size=200, epochs=10)
scores = cn.evaluate(X_test, y_test)
print("Baseline Error: %.2f%%" % (100-scores[1]*100))

