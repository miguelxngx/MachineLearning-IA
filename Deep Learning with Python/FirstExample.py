from keras.datasets import mnist

#train_images and train_labels are the data that the model will learn from
#test_image and test_labels are going to test the model
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

#The images are encoded as Numpy arrays and the labels are an array of
#digists, ranging from 0 to 9. The images and labels have a one-to-one 
#correspondence

#trainning data
print("Trainning data")
print(train_images.shape)
print(len(train_labels))
print(train_labels)
print("")

#test data
print("Test data")
print(test_images.shape)
print(len(test_labels))
print(test_labels)
print("")

#2.2 The network architecture
from keras import models
from keras import layers

network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,))) #use relu on middle layers
network.add(layers.Dense(10, activation='softmax')) #use softmax in last layer because of probability

#2.3 The compilation step
network.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

#2.4 Preparing the image data
#This will preprocess the data by reshaping it into the shape the the network
#expects and scling it so that all values are in the [0, 1] interval.
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') /255

#2.5 Preparing the labels
from keras.utils import to_categorical

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

#We're now ready to train the network, wich in Keras is done via call to
#the network's fit method - we fit the model to its training data:
network.fit(train_images, train_labels, epochs=5, batch_size=128)

#Two quantities are displayed during training: the loss of the network
#over the training data, and the accuracy of the network over the
#trainning data.
test_loss, test_acc = network.evaluate(test_images, test_labels)
print('test_acc: ', test_acc)

#The test-set accuracy turns out to be a bit lower than the trainning
#accuracy. This gap between trainning accuracy and test accuracy is an
#example of overfitting

import numpy as np
import matplotlib.pyplot as plt
#let us test the result
idx = np.random.randint(10000)

plt.imshow(test_images[idx])
print("The predicted number is: ", np.argmax(network.predict(test_images[idx].reshape(28 * 28))))