from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers.core import Activation
import time
from keras.models import load_model
from keras.preprocessing import image
import numpy as np



def trainCurrencyDetector():
    #Initializing the CNN
    clf = Sequential()

    #Step 1 : Add the first convolutional layer
    clf.add(Conv2D(256, (3,3), input_shape= (128,128,3), activation = 'relu'))
    #Step 2 : Pooling using Max pooling
    clf.add(MaxPooling2D( pool_size = (2,2)))
    
    #Step 1 : Add the first convolutional layer
    clf.add(Conv2D(128, (3,3), input_shape= (128,128,3), activation = 'relu'))
    #Step 2 : Pooling using Max pooling
    clf.add(MaxPooling2D( pool_size = (2,2)))

    #Step 1 : Add the first convolutional layer
    clf.add(Conv2D(64, (3,3), input_shape= (128,128,3), activation = 'relu'))
    #Step 2 : Pooling using Max pooling
    clf.add(MaxPooling2D( pool_size = (2,2)))


    # Flatten our images to vector
    clf.add(Flatten())

    #Fully connected hidden layer
    clf.add(Dense(units = 256, activation = 'relu'))

    #Fully connected hidden layer
    clf.add(Dense(units = 128, activation = 'relu'))

    #Fully connected hidden layer
    clf.add(Dense(units = 64, activation = 'relu'))

    #Our ouput layer
    clf.add(Dense(units = 3, activation = 'softmax'))
    #classifier.add(Dense(units = 5))
    #classifier.add(Activation("softmax"))

    #compile out cnn
    clf.compile(optimizer='adam', loss = 'categorical_crossentropy' , metrics = ['accuracy'])

    #fit our model to training set


    #using imageDataGenerator to preprocess our data
    from keras.preprocessing.image import ImageDataGenerator


    train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory('dataset/training_set',target_size=(128, 128), batch_size=32, class_mode='categorical')
    test_generator = test_datagen.flow_from_directory('dataset/test_set', target_size=(128, 128), batch_size=32, class_mode='categorical')

    clf.fit_generator(train_generator, steps_per_epoch=300, epochs=5, validation_data=test_generator, validation_steps=100)
    #classifier.fit_generator(train_generator, steps_per_epoch=90, epochs=3)
    
    clf.save("currency_detector.h5")

    print("Successfully Trained Creency Detector")
    

#trainCurrencyDetector()
#print("Done Training Currency")

def predictCurrency(test_image):
    
    clf = load_model("currency_detector.h5")

    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    className = clf.predict(test_image)

    return className

