from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers.core import Activation
import time
import sys




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

train_generator = train_datagen.flow_from_directory('mnist/training',target_size=(28, 28), batch_size=32, class_mode='categorical')
test_generator = test_datagen.flow_from_directory('mnist/testing', target_size=(28, 28), batch_size=32, class_mode='categorical')

classifier.fit_generator(train_generator, steps_per_epoch=200, epochs=5, validation_data=test_generator, validation_steps=5)
#classifier.fit_generator(train_generator, steps_per_epoch=90, epochs=3)



print("Reach end of file")


t1 = time.time()

import numpy as np
from keras.preprocessing import image
# test_image = image.load_img('mnist/testing/5/53.png', target_size = (28, 28))
test_image = image.load_img('result_chops/2.png', target_size = (28, 28))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
#result = classifier.predict(test_image)

t2 = time.time()

print("Train generator indices : ", train_generator.class_indices)

print(classifier.predict(test_image))
print(classifier.predict_classes(test_image))