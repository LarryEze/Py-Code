''' Image Processing With Neural Networks '''

'''
Introducing convolutional neural networks
- CNNs are powerful algorithms for processing images.
- They are used for identifying the objects in an images.

Images as data
- Images contain data

import matplotlib.pyplot as plt

data = plt.imread('stop_sign.jpg')
plt.imshow(data)
plt.show()

data.shape -> in

(2832, 4256, 3) -> out 

data[1000, 1500] -> in

array([0.73333333, 0.07843137, 0.14509804]) -> out # Result has high intensity in the red channel

data[250, 3500] -> in

array([0.25882353, 0.43921569, 0.77254902]) -> out # Result has high intensity in the blue channel

Modifying image data
data[:, :, 1] = 0
data[:, :, 2] = 0
plt.imshow(data)
plt.show()

Changing an image
data[200:1200, 200:1200, :] = [0, 1, 0]
plt.imshow(data)
plt.show()

Black and white images
- High intensity numbers represents parts of the image that are brighter
- Low intensity numbers represents parts of the image that are darker

tshirt[10:20, 15:25] = 1
plt.imshow(tshirt)
plt.show()
'''

# Import matplotlib

# Load the image
data = plt.imread('bricks.png')

# Display the image
plt.imshow(data)
plt.show()


# Set the red channel in this part (10 by 10 pixels) of the image to 1
data[:10, :10, 0] = 1

# Set the green channel in this part (10 by 10 pixels) of the image to 0
data[:10, :10, 1] = 0

# Set the blue channel in this part (10 by 10 pixels) of the image to 0
data[:10, :10, 2] = 0

# Visualize the result
plt.imshow(data)
plt.show()


'''
Classifying images
Representing class data: one-hot encoding
labels = ['shoe', 'dress', 'shoe', 't-shirt', 'shoe', 't-shirt', 'shoe', 'dress']

array([ [0., 0., 1.], <- shoe
        [0., 1., 0.], <- dress
        [0., 0., 1.], <- shoe
        [1., 0., 0.], <- t-shirt
        [0., 0., 1.], <- shoe
        [1., 0., 0.], <- t-shirt
        [0., 0., 1.], <- shoe
        [0., 1., 0.]]) <-dress 

One-hot encoding
categories = np.array(['t-shirt', 'dress', 'shoe'])
n_categories = 3
ohe_labels = np.zeros((len(labels), n_categories))
for ii in range(len(labels)):
    jj = np.where(categories == labels[ii])
    ohe_labels[ii, jj] = 1

One-hot encoding: testing predictions
test -> in

array([ [0., 0., 1.], 
        [0., 1., 0.], 
        [0., 0., 1.], 
        [0., 1., 0.], 
        [0., 0., 1.], 
        [0., 0., 1.], 
        [0., 0., 1.], 
        [0., 1., 0.]    ]) -> out

prediction -> in

array([ [0., 0., 1.], 
        [0., 1., 0.], 
        [0., 0., 1.], 
        [1., 0., 0.],   <- incorrect
        [0., 0., 1.], 
        [1., 0., 0.],   <- incorrect
        [0., 0., 1.], 
        [0., 1., 0.]    ]) -> out

(test * prediction).sum() -> in

6.0 -> out
'''

# The number of image categories
n_categories = 3

# The unique values of categories in the data
categories = np.array(["shirt", "dress", "shoe"])

# Initialize ohe_labels as all zeros
ohe_labels = np.zeros((len(labels), n_categories))

# Loop over the labels
for ii in range(len(labels)):
    # Find the location of this label in the categories variable
    jj = np.where(labels[ii] == categories)
    # Set the corresponding zero to one
    ohe_labels[ii, jj] = 1


# Calculate the number of correct predictions
number_correct = (test_labels * predictions).sum()
print(number_correct)

# Calculate the proportion of correct predictions
proportion_correct = (test_labels * predictions).sum() / len(predictions)
print(proportion_correct)


'''
Classification with Keras
Keras for image classification
from keras.models import Sequential

model = Sequential()

from keras.layers import Dense

train_data.shape -> in

(50, 28, 28, 1) -> out

model.add(Dense(10, activation='relu', input_shape=(784, )))
model.add(Dense(10, activation='relu'))
model.add(Dense(3, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

train_data = train_data.reshape((50, 784))

model.fit(train_data, train_labels, validation_split=0.2, epochs=3)

test_data = test_data.reshape((10, 784))
model.evaluate(test_data, test_labels) -> in

10/10 [===================] - 0s 335us/step
[1.0191701650619507, 0.4000000059604645] -> out
'''

# Imports components from Keras

# Initializes a sequential model
model = Sequential()

# First layer
model.add(Dense(10, activation='relu', input_shape=(784,)))

# Second layer
model.add(Dense(10, activation='relu'))

# Output layer
model.add(Dense(3, activation='softmax'))


# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])


# Reshape the data to two-dimensional array
train_data = train_data.reshape((50, 784))

# Fit the model
model.fit(train_data, train_labels, validation_split=0.2, epochs=3)


# Reshape test data
test_data = test_data.reshape(10, 784)

# Evaluate the model
model.evaluate(test_data, test_labels)


''' Using Convolutions '''

'''
Convolutions
Using correlations in images
- Natural images contain spatial correlations
- For example, pixels along a contour or edge
- How can we use these correlations?

What is a convolution?
array = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

kernel = np.array([-1, 1])

conv = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])
conv[0] = (kernel * array[0:2]).sum()
conv[1] = (kernel * array[1:3]).sum()
conv[2] = (kernel * array[2:4]).sum()
conv[3] = (kernel * array[3:5]).sum()
conv[4] = (kernel * array[4:6]).sum()
conv[5] = (kernel * array[5:7]).sum()
conv[6] = (kernel * array[6:8]).sum()
conv[7] = (kernel * array[7:9]).sum()
conv[8] = (kernel * array[8:20]).sum()

for ii in range(8):
    conv[ii] = (kernel * array[ii:ii+2]).sum()
    conv -> in

array([0, 0, 0, 0, 1, 0, 0, 0, 0]) -> out

Convolution in one dimension
array = np.array([0, 0, 1, 1, 0, 0, 1, 1, 0, 0])
kernel = np.array([-1, 1])
conv = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])
for ii in range(8):
    conv[ii] = (kernel * array[ii:ii+2]).sum()
    conv -> in

array([0, 1, 0, -1, 0, 1, 0, -1, 0]) -> out

Two-dimensional convolution
kernel = np.array([[-1, 1], [-1, 1]])
conv = np.zeros((27, 27))
for ii in range(27):
    for jj in range(27):
        window = image[ii:ii+2, jj:jj+2]
        conv[ii, jj] = np.sum(window * kernel)
'''

array = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
kernel = np.array([1, -1, 0])
conv = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

# Output array
for ii in range(8):
    conv[ii] = (kernel * array[ii:ii+3]).sum()

# Print conv
print(conv)


kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
result = np.zeros(im.shape)

# Output array
for ii in range(im.shape[0] - 3):
    for jj in range(im.shape[1] - 3):
        result[ii, jj] = (im[ii:ii+3, jj:jj+3] * kernel).sum()

# Print result
print(result)


# Define a kernel that finds a vertical line in images:
kernel = np.array([[-1, 1, -1],
                   [-1, 1, -1],
                   [-1, 1, -1]])

# Define a kernel that finds horizontal lines in images.
kernel = np.array([[-1, -1, -1],
                   [1, 1, 1],
                   [-1, -1, -1]])

# Define a kernel that finds a light spot surrounded by dark pixels.
kernel = np.array([[-1, -1, -1],
                   [-1, 1, -1],
                   [-1, -1, -1]])

# Define a kernel that finds a dark spot surrounded by bright pixels.
kernel = np.array([[1, 1, 1],
                   [1, -1, 1],
                   [1, 1, 1]])


'''
Implementing image convolutions in Keras
Keras Convolution layer
- A dense layer has one weight for each pixel in the image
- A convolution layer has only one weight for each pixel in the kernel

from keras.layers import Conv2D

Conv2D(10, kernel_size=3, activation='relu')

Integrating convolution layers into a network
- Flatten() serves as a connector between convolution and densely connected layers
- The output of a convolution is also referred to as a 'feature map'

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten

img_rows = 28
img_cols = 28
model = Sequential()
model.add(Conv2D(10, kernel_size=3, activation='relu', input_shape=(img_rows, img_cols, 1)))
model.add(Flatten())
model.add(Dense(3, activation='softmax'))

Fitting a CNN
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
train_data.shape -> in

(50, 28, 28, 1) -> out

model.fit(train_data, train_labels, validation_split=0.2, epochs=3)

model.evaluate(test_data, test_labels, epochs=3)
'''

# Import the necessary components from Keras

# Initialize the model object
model = Sequential()

# Add a convolutional layer
model.add(Conv2D(10, kernel_size=3, activation='relu',
          input_shape=(img_rows, img_cols, 1)))

# Flatten the output of the convolutional layer
model.add(Flatten())
# Add an output layer for the 3 categories
model.add(Dense(3, activation='softmax'))


# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])

# Fit the model on a training set
model.fit(train_data, train_labels,
          validation_split=0.2, epochs=3, batch_size=10)


# Evaluate the model on separate test data
model.evaluate(test_data, test_labels, batch_size=10)


'''
Tweaking your convolutions
Convolution with zero padding
Zero padding in keras
- padding argument is used to implement zero padding
* if the value is 'valid', no zero padding is added and it is its default value.
* if the value is 'same', zero padding will be applied to the input to this layer, so that the output of the convolution has the same size as the input into the convolution.

model.add(Conv2D(10, kernel_size=3, activation='relu', input_shape=(img_rows, img_cols, 1)), padding='same')

Strides
- It is the size of the step taken with the kernel between input pixels.
- strides default is 1
* i.e the kernel slides along the image and is multiplied and summed with each pixel location
- If the stride is set to more than 1, the kernel jumps in steps of that number of pixels.

model.add(Conv2D(10, kernel_size=3, activation='relu', input_shape=(img_rows, img_cols, 1)), strides=1)

Calculating the size of the output
O = (( I - K + 2P) / S ) + 1

where,
* I = size of the input
* K = size of the kernel
* P = size of the zero padding
* S = strides

e.g 
28 = (( 28 - 3 + 2) / 1 ) + 1

10 = (( 28 - 3 + 2) / 3 ) + 1

Dilated convolution
- It is useful where you need to aggregate information across multiple scales.
- dilation_rate argument is used to set the distance between subsequent pixels.

model.add(Conv2D(10, kernel_size=3, activation='relu', input_shape=(img_rows, img_cols, 1)), dilation_rate=2)
'''

# Initialize the model
model = Sequential()

# Add the convolutional layer
model.add(Conv2D(10, kernel_size=3, activation='relu',
          input_shape=(img_rows, img_cols, 1), padding='same'))

# Feed into output layer
model.add(Flatten())
model.add(Dense(3, activation='softmax'))


# Initialize the model
model = Sequential()

# Add the convolutional layer
model.add(Conv2D(10, kernel_size=3, activation='relu',
          input_shape=(img_rows, img_cols, 1), strides=2))

# Feed into output layer
model.add(Flatten())
model.add(Dense(3, activation='softmax'))


''' Going Deeper '''

'''
Going Deeper
- The use of artificial neural networks is sometimes also called 'deep learning'
- Networks with more convolution layers are called "deep" networks, and they may have more power to fit complex data, because of their ability to create hierarchical representations of the data that they fit.

model = Sequential()
model.add(Conv2D(10, kernel_size=2, activation='relu', input_shape=(img_rows, img_cols, 1), padding='equal'))

# Second convolutional layer
model.add(Conv2D(10, kernel_size=2, activation='relu'))
model.add(Flatten())
model.add(Dense(3, activation='softmax'))

- Having multiple layers of convolutions in the network allows the networ to gradually build up representations of objects in the images from simple features to more complex features and up to sensitivity to distinct categories of objects.

How deep should the neetwork be?
- Depth comes at a computational cost
- May require more training data
'''


model = Sequential()

# Add a convolutional layer (15 units)
model.add(Conv2D(15, kernel_size=2, activation='relu',
          input_shape=(img_rows, img_cols, 1)))

# Add another convolutional layer (5 units)
model.add(Conv2D(5, kernel_size=2, activation='relu'))

# Flatten and feed to output layer
model.add(Flatten())
model.add(Dense(3, activation='softmax'))


# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])

# Fit the model to training data
model.fit(train_data, train_labels,
          validation_split=0.2, epochs=3, batch_size=10)

# Evaluate the model on test data
model.evaluate(test_data, test_labels, batch_size=10)


'''
How many parameters?
model = Sequential()

model.add(Dense(10, activation='relu', input_shape=(784, )))

model.add(Dense(10, activation='relu'))

model.add(Dense(3, activation='softmax'))

# Call the summary method
model.summary()

Counting parameters
model.add(Dense(10, activation='relu', input_shape=(784, )))
parameters = 784 * 10 + 10
parameters = 7850

model.add(Dense(10, activation='relu'))
parameters = 10 * 10 + 10
parameters = 110

model.add(Dense(3, activation='softmax'))
parameters = 10 * 3 + 10
parameters = 33

Total parameters = 7850 + 110 + 33
Total parameters = 7993

The number of parameters in a CNN
model = Sequential()

model.add(Conv2D(10, kernel_size=3, activation='relu', input_shape=(28, 28, 1), padding='same'))
parameters = 9 * 10 + 10
parameters = 100

model.add(Conv2D(10, kernel_size=3, activation='relu'))
parameters = 10 * 9 * 10 + 10
parameters = 910

model.add(Flatten())
parameters = 0

model.add(Dense(3, activation='softmax'))
parameters = 7840 * 3 + 3
parameters = 23523

model.summary()
Total parameters = 100 + 910 + 0 + 23523
Total parameters = 24533

Increasing the number of units in ech layer
model = Sequential()

model.add(Dense(5, activation='relu', input_shape=(784, ), padding='same'))
parameters = 784 * 5 + 5
parameters =3925

model.add(Dense(15, activation='relu', padding='same'))
parameters = 5 * 15 + 15
parameters = 90

model.add(Dense(3, activation='softmax'))
parameters = 15 * 3 + 3
parameters = 48

Total parameters = 3925 + 90 + 48
Total parameters = 4063

- Convolution networks have more expressive power, so they require less parameters.
* but reading out these more expensive representations then requires many more parameters on the output side.
'''

# CNN model
model = Sequential()

model.add(Conv2D(10, kernel_size=2, activation='relu', input_shape=(28, 28, 1)))
model.add(Conv2D(10, kernel_size=2, activation='relu'))

model.add(Flatten())

model.add(Dense(3, activation='softmax'))

# Summarize the model
model.summary()


'''
Pooling operations
- It is used to summarize the output of convolutional layers in concise manner.

Implementing max pooling
result = np.zeros((im.shape[0]//2, im.shape[1]//2))

result[0, 0] = np.max(im[0:2, 0:2])
result[0, 1] = np.max(im[0:2, 2:4])
result[0, 2] = np.max(im[0:2, 4:6])
. . .
result[1, 0] = np.max(im[2:4, 0:2])
result[1, 1] = np.max(im[2:4, 2:4])
. . .  

for ii in range(result.shape[0]):
    for jj in range(result.shape[1]):
        result[ii, jj] = np.max(im[ii*2:ii*2+2, jj*2:jj*2+2])

Max pooling in Keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPool2D

model = Sequential()
model.add(Conv2D(5, kernel_size=3, activation='relu', input_shape=(img_rows, img_cols, 1)))
model.add(MaxPool2D(2))
model.add(Conv2D(15, kernel_size=3, activation='relu', input_shape=(img_rows, img_cols, 1)))
model.add(MaxPool2D(2))
model.add(Flatten())
model.add(Dense(3, activation='softmax'))

model.summary()

Convolution => Convolution => Flatten => Dense
Convolution => Max pooling => Convolution => Flatten => Dense
'''

# Result placeholder
result = np.zeros((im.shape[0]//2, im.shape[1]//2))

# Pooling operation
for ii in range(result.shape[0]):
    for jj in range(result.shape[1]):
        result[ii, jj] = np.max(im[ii*2:ii*2+2, jj*2:jj*2+2])


# Add a convolutional layer
model.add(Conv2D(15, kernel_size=2, activation='relu',
          input_shape=(img_rows, img_cols, 1)))

# Add a pooling operation
model.add(MaxPool2D(2))

# Add another convolutional layer
model.add(Conv2D(5, kernel_size=2, activation='relu',
          input_shape=(img_rows, img_cols, 1)))

# Flatten and feed to output layer
model.add(Flatten())
model.add(Dense(3, activation='softmax'))
model.summary()


# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])

# Fit to training data
model.fit(train_data, train_labels,
          validation_split=0.2, epochs=3, batch_size=10)

# Evaluate on test data
model.evaluate(test_data, test_labels, batch_size=10)


''' Understanding and Improving Deep Convolutional Networks '''

'''
Tracking learning
training = mode.fit(train_data, train_labels, epochs=3, validation_split=0.2)

import matplotlib.pyplot as plt

plt.plot(training.history['loss'])
plt.plot(training.history['val_loss'])
plt.show()

Storing the optimal parameters
from keras.callbacks import ModelCheckpoint

# This checkpoint object will store the model parameters in the file 'weights.hdf5'
checkpoint = ModelCheckpoint('weights.hdf5', monitor='val_loss', save_best_only=True)

# Store in a list to be used during training
callbacks_list = [checkpoint]

# Fit the model on a training set, using the checkpoint as a callback
model.fit(train_data, train_labels, validation_split=0.2, epochs=3, callbacks=callbacks_list)

Loading stored parameters
model.load_weights('weights.hdf5')
model.predict_classes(test_data) -> in

array([2, 2, 1, 2, 0, 1, 0, 1, 2, 0]) -> out
'''


# Train the model and store the training object
training = model.fit(train_data, train_labels,
                     validation_split=0.2, epochs=3, batch_size=10)

# Extract the history from the training object
history = training.history

# Plot the training loss
plt.plot(history['loss'])
# Plot the validation loss
plt.plot(history['val_loss'])

# Show the figure
plt.show()


# Load the weights from file
model.load_weights('weights.hdf5')

# Predict from the first three images in the test data
model.predict(test_data[:3])


'''
Regularization
Dropout
- In each learning step:
* we choose a random subset of the units in a layer 
* Ignore it in the forward pass
* And in the back-propagation of error

Dropout in Keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout

model = Sequential()

model.add(Conv2D(5, kernel_size=3, activation='relu', input_shape=(img_rows, img_cols, 1)))
model.add(Dropout(0.25))
model.add(Conv2D(15, kernel_size=3, activation='relu'))
model.add(Flatten())
model.add(Dense(3, activation='softmax'))

Batch normalization
- This operation takes the output of a particular layer, and rescales it so that it always has zero (0) mean and standard deviation of 1 in every batch of training.
- The algorithm solves the problem where different batches of input might produce wildly different distributions of outputs in any given layer in the network.

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, BatchNormalization

model = Sequential()

model.add(Conv2D(5, kernel_size=3, activation='relu', input_shape=(img_rows, img_cols, 1)))
model.add(BatchNormalization())
model.add(Conv2D(15, kernel_size=3, activation='relu'))
model.add(Flatten())
model.add(Dense(3, activation='softmax'))

Be careful when using them together!
- Sometimes, dropout and batch normalization do not work well together. 
* This is because while dropout slows down learning, making it more incremental and careful
* Batch normalization tends to make learning go faster. 
- Their effects together may in fact counter each other, and networks sometimes perform worse when both of these methods are used together than they would if neither were used.
* It is called 'the disharmony of batch normalization and dropout'
'''

# Add a convolutional layer
model.add(Conv2D(15, kernel_size=2, activation='relu',
          input_shape=(img_rows, img_cols, 1)))

# Add a dropout layer
model.add(Dropout(0.2))

# Add another convolutional layer
model.add(Conv2D(5, kernel_size=2, activation='relu'))

# Flatten and feed to output layer
model.add(Flatten())
model.add(Dense(3, activation='softmax'))


# Add a convolutional layer
model.add(Conv2D(15, kernel_size=2, activation='relu',
          input_shape=(img_rows, img_cols, 1)))

# Add batch normalization layer
model.add(BatchNormalization())

# Add another convolutional layer
model.add(Conv2D(5, kernel_size=2, activation='relu'))

# Flatten and feed to output layer
model.add(Flatten())
model.add(Dense(3, activation='softmax'))


'''
Interpreting the model
Selecting layers
model.layers -> in

[       <keras.layers.convolutional.Conv2D at 0x109f10c18>,
        <keras.layers.convolutional.Conv2D at 0x109ec5ba8>,
        <keras.layers.coore.Flatten at 0x1221ffcc0>,
        <keras.layers.core.Dense at 0x1221ffef0>        ] -> out

Getting model weights
conv1 = model.layers[0]
weights1 = conv1.get_weights()
len(weights1) -> in

2 -> out

kernels1 = weights1[0]
kernels1.shape -> in

(3, 3, 1, 5) -> out

kernel1_1 = kernels1[:, :, 0. 0]
kernel1_1.shape -> in

(3, 3) -> out

Visualizing the kernel
plt.imshow(kernel1_1)

Visualizing the kernel responses
test_image = test_data[3, :, :, 0]
plt.imshow(test_image)

filtered_image = convolution(test_image, kernel1_1)
plt.imshow(filtered_image)

test_image = test_data[4:, :, :, 1]
plt.imshow(test_image)

filtered_image = convolution(test_image, kernel1_1)
plt.imshow(filtered_image)

kernel1_2 = kernels[:, :, 0, 1]
filtered_image = convolution(test_image, kernel1_2)
plt.imshow(filtered_image)
'''

# Load the weights into the model
model.load_weights('weights.hdf5')

# Get the first convolutional layer from the model
c1 = model.layers[0]

# Get the weights of the first convolutional layer
weights1 = c1.get_weights()

# Pull out the first channel of the first kernel in the first layer
kernel = weights1[0][..., 0, 0]
print(kernel)


# Convolve with the fourth image in test_data
out = convolution(test_data[3, :, :, 0], kernel)

# Visualize the result
plt.imshow(out)
plt.show()


'''
Next steps
Residual networks
- These include connections that skip over several layers, and they are called residual networks because the network will use this skipped connection to compute a difference between the input of a stack of layers and their output.

Transfer learning
- In this approach an already trained network is adapted to a new task.

Fully convolutional networks
- They take an image as input and produce another image as output
- For example, these networks can be used to find the part of an image that contains a particular kind of object, doing segmentation rather than classification.

Generative adversarial networks
- These complex architectures can be used to train a network to create new images that didn't exist before.
'''
