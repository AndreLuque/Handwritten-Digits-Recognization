import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2 as cv

mnist = tf.keras.datasets.mnist #data set with 70000 images of handwriten digits in 28X28 pixel resolution
(x_train, y_train), (x_test, y_test) = mnist.load_data() #load data gives us to tuples with 60000 images and 10000, our train and test data

x_train = tf.keras.utils.normalize(x_train)
x_test = tf.keras.utils.normalize(x_test)
#the images are stored as numpy arrays with the info about the pixels, we normalize the values to make the computing easier. we only do this for x values because Y is 0-9

model = tf.keras.models.Sequential() #this is a linear type of model that is defined layer by layer
model.add(tf.keras.layers.Flatten(input_shape=(28,28))) #the input shape is 28x28 we flatten it and change it to 784x1
model.add(tf.keras.layers.Dense(units=128, activation='relu')) #dense is the basic layer, which is connected to all the neurons of neighnouring layers
model.add(tf.keras.layers.Dense(units=128, activation='relu')) #in Dense there are two inputs, units = number of neurons, activation = which activation function to use(relu is rectified linear unit, f(x) = max(0, x))
model.add(tf.keras.layers.Dense(units=10, activation=tf.nn.softmax)) #for our ouput layer we use ten neurons since there are 10 possible digits. softmax is an activation that makes our end ersults add up to 1

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#adam and sparse_categorical_crossentropy are cmore complicated, doesnt specify explanation. But we know that they are very popuar choices

model.fit(x_train, y_train, epochs = 5) #epochs is how many times we run the algo with the data
loss, accuracy = model.evaluate(x_test, y_test)
print(f'Loss: {loss}') #summation of errors
print(f'Accuracy: {accuracy}') #percentage of correctly classified data


#we take a png 28x28 and predict the number that it is 
image = cv.imread('image.png')[:,:,0]
image = np.invert(np.array([image]))

prediction = model.predict(image)
print("Prediction: {}".format(np.argmax(prediction)))
plt.imshow(image[0])
plt.show()
