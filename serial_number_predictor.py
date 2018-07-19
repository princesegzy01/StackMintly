from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers.core import Activation
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import zipfile

import time
import sys
import os


def trainConvNet():
    
    #Initializing the CNN
    classifier = Sequential()

    #Step 1 : Add the first convolutional layer
    classifier.add(Conv2D(32, (3,3), input_shape= (28,28,3), activation = 'relu'))

    #Step 2 : Pooling using Max pooling
    classifier.add(MaxPooling2D( pool_size = (2,2)))


    # Flatten our images to vector
    classifier.add(Flatten())

    #Fully connected hidden layer
    classifier.add(Dense(units = 256, activation = 'relu'))

    #Our ouput layer
    classifier.add(Dense(units = 10, activation = 'softmax'))
    #classifier.add(Dense(units = 5))
    #classifier.add(Activation("softmax"))

    #compile out cnn
    classifier.compile(optimizer='adam', loss = 'categorical_crossentropy' , metrics = ['accuracy'])

    #fit our model to training set


    #using imageDataGenerator to preprocess our data
    from keras.preprocessing.image import ImageDataGenerator


    train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory('segunOcr/training',target_size=(28, 28), batch_size=32, class_mode='categorical')
    test_generator = test_datagen.flow_from_directory('segunOcr/testing', target_size=(28, 28), batch_size=32, class_mode='categorical')

    classifier.fit_generator(train_generator, steps_per_epoch=5000, epochs=5, validation_data=test_generator, validation_steps=2000)
    #classifier.fit_generator(train_generator, steps_per_epoch=3000, epochs=3)


    classifier.save("serial_number_detector.h5")
    print("Done Training saving")

#Traina and save the model
#trainConvNet()
#sys.exit(0)

#classifier = load_model("model_result.h5")

#print("Model Loaded")

#t1 = time.time()


#result = []

#for img in sorted(os.listdir("result_chops")):
    #test_image = image.load_img('result_chops/6.png', target_size = (28, 28))
   
def customDigitPredictor(classifier, test_image):
    
    #if img[-3:] != "png":
    #    continue
    
    #img_path = os.path.join("result_chops", img)
    #test_image = image.load_img( img_path , target_size = (28, 28))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    #result = classifier.predict(test_image)

   

    #print("Train generator indices : ", train_generator.class_indices)

    #print(classifier.predict(test_image))
    className = classifier.predict_classes(test_image)
    #print(className[0])
    #result.append(img + " : " + str(className[0]))
    #result.append(str(className[0]))

    return str(className[0])

