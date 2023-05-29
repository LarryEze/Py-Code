''' Introduction to TensorFlow '''

'''
Constants and variables
What is TensorFlow?
- It is an open-source library for graph-based numerical computation.
- It has both low and high level APIs
* It can be used to perform addition, multiplication, and differentiation
* It can also be used to design and train machine learning models.
- TensorFlow 2.0 brought with it substantial chages
* Eager execution is now enabled by default, which allows users to write simpler and more intuitive code.
* Model building is now centered around the Keras and Estimators high-level APIs.

What is a tensor?
- It is a generalization of vectors and matrices to potentially higher dimensions.
* Or a collection of numbers, which is arranged into a particular shape.

Defining tensors in TensorFlow
import tensorflow as tf

# 0D Tensor
d0 = tf.ones((1, ))

# 1D Tensor
d1 = tf.ones((2, ))

# 2D Tensor
d2 = tf.ones((2, 2))

# 3D Tensor
d3 = tf.ones((2, 2, 2))

# Print the 3D tensor
print(d3.numpy()) -> in

[   [[1. 1.]
    [1. 1.]]

    [[1. 1.]
    [1. 1.]]] -> out

Defining Constants in TensorFlow
- It is the simplest category of tensor in TensorFlow
* It does not change and cannot be trained.
* It can have any dimension

from tensorflow import constant

# Define a 2x3 constant
a = constant(3, shape = [2, 3])

# Define a 2x2 constant
b = constant([1, 2, 3, 4], shape = [2, 2])

Using convenience functions to define constants
Operation           Example
tf.constant()       constant([1, 2, 3])
tf.zeros()          zeros([2, 2])
tf.zeros_like()     zeros_like(input_tensor)
tf.ones()           ones([2, 2])
tf.ones_like()      ones_like(input_tensor)
tf.fill()           fill([3, 3], 7)

- You can use the zeros or ones operations to generate a tensor of arbitrary dimension that is populated entirely with zeros or ones.
- You can use the zeros_like or ones_like operations to populate tensors with zeros or ones, copying the dimension of some input tensor.
- You can use the fill operation to populate a tensor of arbitrary dimension with the same scalar value in each element.

Defining and initializing variables
- Unlike a constant, a variable's value can change during computation.
- The value of a variable is shared, persistent and modifiable, However, its data type and shape are fixed.

import tensorflow as tf

# Define a variable
a0 = tf.Variable([1, 2, 3, 4, 5, 6], dtype=tf.float32)
a1 = tf.Variable([1, 2, 3, 4, 5, 6], dtype=tf.int64)

# Define a constant
b = tf.constant(2, tf.float32)

# Compute their product
c0 = tf.multiply(a0, b)
c1 = a0*b
'''

# Import constant from TensorFlow

# Convert the credit_numpy array into a tensorflow constant
from tensorflow import keras
import  tensorflow as tf
credit_constant = constant(credit_numpy)

# Print constant datatype
print('\n The datatype is:', credit_constant.dtype)

# Print constant shape
print('\n The shape is:', credit_constant.shape)


# Define the 1-dimensional variable A1
A1 = Variable([1, 2, 3, 4])

# Print the variable A1
print('\n A1: ', A1)

# Convert A1 to a numpy array and assign it to B1
B1 = A1.numpy()

# Print B1
print('\n B1: ', B1)


'''
Basic operations
What is a TensorFlow operation?
- TensorFlow has a model of computation that revolves around the use of graphs.
- A tensorFlow graph contains edges and nodes, where the edges are tensors and the nodes are operations.

                    MatMul
            /                 \
        Add                    Add_1
    /      \                 /       \
Const       Const_1     Const_2     Const_3

Applying the addition operator
# Import constant and add from tensorflow
from tensorflow import constant, add

# Define 0-dimensional tensors
A0 = constant([1])
B0 = constant([2])

# Define 1-dimensional tensors
A1 = constant([1, 2])
B1 = constant([3, 4])

# Define 2-dimensional tensors
A2 = constant([[1, 2], [3, 4]])
B2 = constant([[5, 6], [7, 8]])

Applying the addition operator
# Perform tensor addition with add()
C0 = add(A0, B0)
C1 = add(A1, B1)
C2 = add(A2, B2)

NB*: we can perform scalar addition with A0 and B0, vector addition with A1 and B1, and matrix addition with A2 and B2.

Performing tensor addition
- the add() operation performs element-wise addition with two tensors
- Each pair of tensors added must have the same shape.
* Scalar addition: 1 + 2 = 3
* Vector addition: [1, 2] + [3, 4] = [4, 6]
* Matrix addition: [[1, 2], [3, 4]] + [[5, 6], [7, 8]] = [[6, 8], [10, 12]]
- The add() operator is overloaded, which means that we can also perform addition using the plus symbol.

How to perform multiplication in TensorFlow
- For Element-wise multiplication, which is performed with the multiply() operation, 
* The tensors involved must have the same shape.
* e.g [1, 2, 3] and [3, 4, 5] or [1, 2] and [3, 4]
- For Matrix multiplication, you use the matmul() operator.
* The matmul(A, B) operation multiplies A by B
* NB*: Performing matmul(A, B) requires that the number of columns of A equal the number of rows of B.

Applying the multiplication operators
# Import operators from tensorflow
from tensorflow import ones, matmul, multiply

# Define tensors
A0 = ones(1)
A31 = ones([3, 1])
A34 = ones([3, 4])
A43 = ones([4, 3])

- What types of operations are valid?
* multiply(A0, A0), multiply(A31, A31), and multiply(A34, A34)
* matmul(A43, A34), but not matmul(A43, A43)

Summing over tensor dimensions
- The reduce_sum() operator sums over the dimensions of a tensor
* reduce_sum(A) sums over all dimensions of A
* reduce_sum(A, i) sums over dimension i

# Import operations from tensorflow
from tensorflow import ones, reduce_sum

# Define a 2x3x4 tensor of ones
A = ones([2, 3, 4])

# Sum over all dimensions
B = reduce_sum(A)

# Sum over dimensions 0, 1, and 2
B0 = reduce_sum(A, 0)
B1 = reduce_sum(A, 1)
B2 = reduce_sum(A, 2)
'''

# Define tensors A1 and A23 as constants
A1 = constant([1, 2, 3, 4])
A23 = constant([[1, 2, 3], [1, 6, 4]])

# Define B1 and B23 to have the correct shape
B1 = ones_like(A1)
B23 = ones_like(A23)

# Perform element-wise multiplication
C1 = multiply(A1, B1)
C23 = multiply(A23, B23)

# Print the tensors C1 and C23
print('\n C1: {}'.format(C1.numpy()))
print('\n C23: {}'.format(C23.numpy()))


# Define features, params, and bill as constants
features = constant([[2, 24], [2, 26], [2, 57], [1, 37]])
params = constant([[1000], [150]])
bill = constant([[3913], [2682], [8617], [64400]])

# Compute billpred using features and params
billpred = matmul(features, params)

# Compute and print the error
error = bill - billpred
print(error.numpy())


'''
Advanced operations
Oerview of advanced operations
- We have covered basic operations in TensorFlow
* add(), multiply(), matmul(), and reduce_sum()
- In this lesson, we explore advanced operations
* gradient(), reshape(), and random()

Overview of advanced operations
Operation       Use
gradient()      Computes the slope of a function at a point
reshape()       Reshapes a tensor (e.g 10 x 10 to 100 x 1)
random()        Populates tensor with entries drawn from a probability distribution

Finding the Optimum
- In many problems, we will want to find the optimum of a function.
* Minimum: Lowest value of a loss function
* Maximum: Highest value of objective function

- We can do this using the gradient() operation, which tells us the slope of a function at a point.
* Optimum: Find a point where gradient = 0
* Minimum: Change in gradient > 0
* Maximum: Change in gradient < 0

Calculating the gradient
Function with fixed gradient: y = x         # line plot
- The gradient, i.e, the slope at a given point, is constant.
* If we increase x by 1 unit, y also increases by 1 unit.

Function with varying gradient: y = x^2     # U shape plot
- When x is less than 0, y decreases when x increases 
- When x is greater than 0, y increases when x increases 
* Thus, the gradient is initially negative, but becomes positive for x larger than 0. i.e x equals 0 minimizes y.

Gradients in TensorFlow
# Import tensorflow under the alias tf
import tensorflow as tf

# Define x
x = tf.Variable(-1.0)

# Define y within instance of GradientTape
with tf.GradientTape() as tape:
    tape.watch(x)
    y = tf.multiply(x, x)

# Evaluate the gradient of y at x = -1
g = tape.gradient(y, x)
print(g.numpy()) -> in

-2.0 -> out         # it means that y is initially decreasing in x

Images as tensors
How to reshape a grayscale image
# Import tensorflow as alias tf
import tensorflow as tf

# Generate grayscale image
gray = tf.random.uniform([2, 2], maxval=255, dtype='int32')

# Reshape grayscale image
gray = tf.reshape(gray, [2*2, 1])

How to reshape a color image
# Import tensorflow as alias tf
import tensorflow as tf

# Generate color image
color = tf.random.uniform([2, 2, 3], maxval=255, dtype='int32')

# Reshape color image
color = tf.reshape(color, [2*2, 3])
'''

# Reshape the grayscale image tensor into a vector
gray_vector = reshape(gray_tensor, (28*28, 1))

# Reshape the color image tensor into a vector
color_vector = reshape(color_tensor, (28*28*3, 1))


def compute_gradient(x0):
    # Define x as a variable with an initial value of x0
    x = Variable(x0)
    with GradientTape() as tape:
        tape.watch(x)
    # Define y using the multiply operation
        y = multiply(x, x)
    # Return the gradient of y with respect to x
    return tape.gradient(y, x).numpy()


# Compute and print gradients at x = -1, 1, and 0
print(compute_gradient(-1.0))
print(compute_gradient(1.0))
print(compute_gradient(0.0))


# Reshape model from a 1x3 to a 3x1 tensor
model = reshape(model, (1*3, 1))

# Multiply letter by model
output = matmul(letter, model)

# Sum over output and print prediction using the numpy method
prediction = reduce_sum(output)
print(prediction.numpy())


''' Linear models '''

'''
Input data
Importing data for use in TensorFlow
- External Datasets can be imported using tensorflow
* It is useful for managing complex data pipelines.
* It is complicated

- Simpler options to import data include
* Import data using pandas
* Then Convert the data into numpy array
* Use the data in tensorflow without further modification

How to import and convert data
# Import numpy and pandas
import numpy as np
import pandas as pd

# Load data from csv
housing = pd.read_csv('kc_housing.csv')

# Convert to numpy array
housing = np.array(housing)

- We will focus on data stored in csv format in this chapter
- Pandas also has methods for handling data in other formats
* e.g. read_json(), read_html(), read_excel()

Parameters of read_csv()
Parameter               Description                                     Default
filepath_ or _buffer    Accepts a file path or a URL.                   None
sep                     Delimiter between columns.                      ,
delim_whitespace        Boolean for whether to delimit whitespace.      False
encoding                Specifies encoding to be used if any.           None

Using mixed type datasets
date                price   bedrooms    . . .   floors  waterfront  view
20141013T000000     221900  3           . . .   1       0           0
20141209T000000     538000  3           . . .   2       0           0
20150225T000000     180000  2           . . .   1       1           0
20141209T000000     604000  4           . . .   1       0           0
20150218T000000     510000  3           . . .   1       0           2
20140627T000000     257500  3           . . .   2       0           0
20150115T000000     291850  3           . . .   1       0           4
20150415T000000     229500  3           . . .   1       0           0

Setting the data type
# Load KC dataset
housing = pd.read_csv('kc_housing.csv')

# Convert price column to float32
price = np.array(housing['price'], np.float32)

# Convert waterfront column to Boolean
waterfront = np.array(housing['waterfront'], np.bool)

Setting the data type in tensor flow
# Load KC dataset
housing = pd.read_csv('kc_housing.csv')

# Convert price column to float32
price = tf.cast(housing['price'], tf.float32)

# Convert waterfront column to Boolean
waterfront = tf.cast(housing['waterfront'], tf.bool)
'''

# Import pandas under the alias pd

# Assign the path to a string variable named data_path
data_path = 'kc_house_data.csv'

# Load the dataset as a dataframe named housing
housing = pd.read_csv(data_path)

# Print the price column of housing
print(housing['price'])


# Import numpy and tensorflow with their standard aliases

# Use a numpy array to define price as a 32-bit float
price = np.array(housing['price'], np.float32)

# Define waterfront as a Boolean using cast
waterfront = tf.cast(housing['waterfront'], tf.bool)

# Print price and waterfront
print(price)
print(waterfront)


'''
Loss functions
Introduction to loss functions
- Loss functions play a fundamental role in machine learning using tensorflow operation.
* It is used to train models because they tell us how well our model explains the data.
* It is a Measure of model fit i.e it lets us know how to adjust model parameters during the training process.

- A higher loss value -> worse model fit
* Train the model by selecting parameter values that minimize the loss function.
- We can always place a minus sign before the function we want to maximize and and instead minimize it.

Common loss functions in TensorFlow
- TensorFlow has operations for common loss functions
* Mean squared error (MSE)
* Mean absolute error (MAE)
* Huber error

- Loss functions are accessible from tf.keras.losses()
* tf.keras.losses.mse()
* tf.keras.losses.mae()
* tf.keras.losses.Huber()

Why do we care about loss functions?
- The loss functions tells us whether our model predictions are accurate.
- MSE
* Strongly penalizes outliers
* High (gradient) sensitivity near minimum

- MAE
* Scales linearly with size of error
* Low sensitivity near minimum

- Huber
* Similar to MSE near minimum
* Similar to MAE away from minimum

- NB*: For greater sensitivity near the minimum,  you will want to use the MSE or Huber loss.
* To minimize the impact of outliers, you will want to use the MAE or Huber loss.

Defining a loss function
# Import TensorFlow under standard alias
import tensorflow as tf

# Compute the MSE loss
loss = tf.keras.losses.mse(targets, predictions)

# Define a linear reression model
def linear_regression(intercept, slope = slope, features = features):
    return intercept + features*slope

# Define a loss function to compute the MSE
def loss_function(intercept, slope, targets = targets, features = features):
    # Compute the predictions for a linear model
    predictions = linear_regression(intercept, slope)

    # Return the loss
    return tf.keras.losses.mse(targets, predictions)

# Compute the loss for the test data inputs
loss_function(intercept, slope, test_targets, test_features) -> in

10.77 -> out

# Compute the loss for default data inputs
loss_function(intercept, slope) -> in

5.43 -> out
'''

# Import the keras module from tensorflow

# Compute the mean squared error (mse)
loss = keras.losses.mse(price, predictions)

# Print the mean squared error (mse)
print(loss.numpy())

# Import the keras module from tensorflow

# Compute the mean absolute error (mae)
loss = keras.losses.mae(price, predictions)

# Print the mean absolute error (mae)
print(loss.numpy())


# Initialize a variable named scalar
scalar = Variable(1.0, float32)

# Define the model


def model(scalar, features=features):
    return scalar * features

# Define a loss function


def loss_function(scalar, features=features, targets=targets):
    # Compute the predicted values
    predictions = model(scalar, features)

    # Return the mean absolute error loss
    return keras.losses.mae(targets, predictions)


# Evaluate the loss function and print the loss
print(loss_function(scalar).numpy())


'''
Linear regression
What is a linear regression?
- A linear regression model assumes that the relationship between two variables can be captured by a line.
* i.e, two parameters, the line's slope and intercept fully characterize the relationship between the two variables

The linear regression model
- A linear regression model assumes a linear relationship:
* price = intercept + size (variable) * slope + error
- The difference between the predicted price and the actual price is the error, shich can be used to construct a loss function.
- This is an example of a univariate regression
* There is only one feature, size (variable)
- Multiple regression models have more than one feature
* e.g size and location (variables)

Linear regression in TensorFlow
# Define the targets and features
price = np.array(housing['price'], np.float32)
size = np.array(housing['sqft_living'], np.float32)

# Define the intercept and slope
intercept = tf.Variable(0.1, np.float32)
slope = tf.Variable(0.1, np.float32)

# Define a linear regression model
def linear_regression(intercept, slope, features = size):
    return intercept + features*slope

# Compute the predicted values and loss
def loss_function(intercept, slope, targets = price, features = size):
    predictions =  linear_regression(intercept, slope)
    return tf.keras.losses.mse(targets, predictions)

# Define an optimization operation
opt = tf.keras.optimizers.Adam()

# Minimize the loss function and orint the loss
for j in range(1000):
    opt.minimize(lambda: loss_function(intercept, slope), var_list=[intercept, slope])
    print(loss_function(intercept, slope)) -> in

tf.Tensor(10.909373, shape=(), dtype=float32)
. . . 
tf.Tensor(0.15479447, shape=(), dtype=float32) -> out

# Print the trained parameters
print(intercept.numpy(), slope.numpy())
'''

# Define a linear regression model


def linear_regression(intercept, slope, features=size_log):
    return intercept + features*slope

# Set loss_function() to take the variables as arguments


def loss_function(intercept, slope, features=size_log, targets=price_log):
    # Set the predicted values
    predictions = linear_regression(intercept, slope, features)

    # Return the mean squared error loss
    return keras.losses.mse(targets, predictions)


# Compute the loss for different slope and intercept values
print(loss_function(0.1, 0.1).numpy())
print(loss_function(0.1, 0.5).numpy())


# Initialize an Adam optimizer
opt = keras.optimizers.Adam(0.5)

for j in range(100):
    # Apply minimize, pass the loss function, and supply the variables
    opt.minimize(lambda: loss_function(intercept, slope),
                 var_list=[intercept, slope])

    # Print every 10th value of the loss
    if j % 10 == 0:
        print(loss_function(intercept, slope).numpy())

# Plot data and regression line
plot_results(intercept, slope)


# Define the linear regression model
def linear_regression(params, feature1=size_log, feature2=bedrooms):
    return params[0] + feature1*params[1] + feature2*params[2]

# Define the loss function


def loss_function(params, targets=price_log, feature1=size_log, feature2=bedrooms):
    # Set the predicted values
    predictions = linear_regression(params, feature1, feature2)

    # Use the mean absolute error loss
    return keras.losses.mae(targets, predictions)


# Define the optimize operation
opt = keras.optimizers.Adam()

# Perform minimization and print trainable variables
for j in range(10):
    opt.minimize(lambda: loss_function(params), var_list=[params])
    print_results(params)


'''
Batch training
What is batch training?
price       sqft_lot    bedrooms
 221900.0     5650      3
 538000.0     7242      3
 180000.0    10000      2
 604000.0     5000      4
 510000.0     8080      3
1225000.0   101930      4
 257500.0     6819      3
 291850.0     9711      3
 229500.0     7470      3
 323000.0     6560      3
 662500.0     9796      3
 468000.0     6000      2
 310000.0    19901      3
 400000.0     9680      3
 530000.0     4850      5 

price       sqft_lot    bedrooms
 221900.0     5650      3
 538000.0     7242      3
 180000.0    10000      2 -> Batch 1
 604000.0     5000      4
 510000.0     8080      3

1225000.0   101930      4
 257500.0     6819      3
 291850.0     9711      3 -> Batch 2
 229500.0     7470      3
 323000.0     6560      3

 662500.0     9796      3
 468000.0     6000      2
 310000.0    19901      3 -> Batch 3
 400000.0     9680      3
 530000.0     4850      5

- This is the process of dividing a large datset set into batches and training the batches sequentially.
* A single pass over all of the batches is called an Epoch and the process itself is called batch training
* It is quite useful when working with large image datasets.
- Beyond alleviating memory constraints, batch training will also allow you to update model weights and optimizer parameters after each batch
* rather than at the end of the epoch

The chunksize parameter
- pd.read_csv() allows us to load data in batches
* Avoid loading entire dataset
* Chunksize parameter provides batch size

# Import pandas and numpy
import pandas as pd
import numpy as np

# Load data in batches
for batch in pd.read_csv('kc_housing.csv', chunksize=100):
    # Extract price column
    price = np.array(batch['price'], np.float32)

    # Extract size column
    size = np.array(batch['size'], np.float32)

Training a linear model in batches
# Import tensorflow, pandas and numpy
import tensorflow as tf
import pandas as pd
import numpy as np

# Define trainable variables
intercept = tf.Variable(0.1, tf.float32)
slope = tf.Variable(0.1, tf.float32)

# Define the model
def linear_regression(intercept, slope, features):
    return intercept + features*slope

# Compute predicted values and return loss function
def loss_function(intercept, slope, targets, features):
    predictions = linear_regression(intercept, slope, features)
    return tf.keras.losses.mse(targets, predictions)

# Define optimization operation
opt = tf.keras.optimizers.Adam()

# Load data in batches from pandas
for batch in pd.read_csv('kc_housing.csv', chunksize=100):
    # Extract the target and feature columns
    price_batch = np.array(batch['price'], np.float32)
    size_batch = np.array(batch['lot_size'], np.float32)

    # Minimize the loss function
    opt.minimize(lambda: loss_function(intercept, slope, price_batch, size_batch), var_list=[intercept, slope])

# Print parameter values
print(intercept.numpy(), slope.numpy())

- NB*: We did not use the default values for input data. This is becaue our input data was generated in batches during the training process.

Full sample versus batch training
- Full Sample
* One update per epoch
* Accepts dataset without modification
* Limited by memory

- Batch Training
* Multiple updates per epoch
* Requires division od dataset
* No limit on dataset size 

- NB*: high level APIs will not typically load the sample in batches by default, as we have done here.
'''

# Define the intercept and slope
intercept = Variable(10.0, float32)
slope = Variable(0.5, float32)

# Define the model


def linear_regression(intercept, slope, features):
    # Define the predicted values
    return intercept + features*slope

# Define the loss function


def loss_function(intercept, slope, targets, features):
    # Define the predicted values
    predictions = linear_regression(intercept, slope, features)

    # Define the MSE loss
    return keras.losses.mse(targets, predictions)


# Initialize Adam optimizer
opt = keras.optimizers.Adam()

# Load data in batches
for batch in pd.read_csv('kc_house_data.csv', chunksize=100):
    size_batch = np.array(batch['sqft_lot'], np.float32)

    # Extract the price values for the current batch
    price_batch = np.array(batch['price'], np.float32)

    # Complete the loss, fill in the variable list, and minimize
    opt.minimize(lambda: loss_function(intercept, slope,
                 price_batch, size_batch), var_list=[intercept, slope])

# Print trained parameters
print(intercept.numpy(), slope.numpy())


''' Neural Networks '''

'''
Dense layers
- A dense layer applies weights to all nodes from the previous layer. 

The linear regression model
Bill Amount = 3                Married =1
            \ 0.10              / -0.25   # weights
                Default = 0.05

What is a neural network?
Bill Amount = 3                     Married =1
        | 0.10    \ 0.05       / -0.25 | -0.05   # weights
Hidden_1 = 0.05                     Hidden_2 = 0.10
                \ 0.5               / 0.5
                    Default = 0.05

- To get a neural network from a linear regression, you'll add a hidden layer, which consist of nodes
* Each hidden layer node takes inputs, and multiplies them by their respective weights, and sums them together.
-We sum together the outputs of the hidden layers to compute our prediction for default.
- The entire process of generating a prediction is referred to as forward propagation.

Construct neural networks with three types of layers
- Input layer : It consist of our features
- Hidden layer : They take inputs from the previous layer, applies numerical weights to them, sums them together, and then applies an activation function.
- Output layer : It contains our prediction

A simple dense layer
import tensorflow as tf

# Define inputs (features)
inputs = tf.constant([[1, 35]])

# Define weights
weights = tf.Variable([[-0.05], [-0.01]])

# Define the bias
bias = tf.Variable([0.5])

# Multiply inputs (features) by the weights
product = tf.matmul(inputs, weights)

# Define dense layer
dense = tf.keras.activations.sigmoid(product + bias)

- NB*: The bias is not associated with a feature and is analogous to the intercept in a linear regression.


Defining a complete model
import tensorflow as tf

# Define input (features) layer
inputs = tf.constant(data, tf.float32)

# Define first dense layer
dense1 = tf.keras.layers.Dense(10, activation='sigmoid')(inputs)

# Define second dense layer
dense2 = tf.keras.layers.Dense(5, activation='sigmoid')(dense1)

# Define output (predictions) layer
outputs = tf.keras.layers.Dense(1, activation='sigmoid')(dense2)

High-level versus low-level approach
- High-level approach
* High-level API operations
dense = keras.layers.Dense(10, activation='sigmoid')

- Low-level approach
* Linear-algebraic operations
prod = matmul(inputs, weights)
dense = keras.activations.sigmoid(prod)
'''

# From previous step
bias1 = Variable(1.0)
weights1 = Variable(ones((3, 2)))
product1 = matmul(borrower_features, weights1)
dense1 = keras.activations.sigmoid(product1 + bias1)

# Initialize bias2 and weights2
bias2 = Variable(1.0)
weights2 = Variable(ones((2, 1)))

# Perform matrix multiplication of dense1 and weights2
product2 = matmul(dense1, weights2)

# Apply activation to product2 + bias2 and print the prediction
prediction = keras.activations.sigmoid(product2 + bias2)
print('\n prediction: {}'.format(prediction.numpy()[0, 0]))
print('\n actual: 1')


# Compute the product of borrower_features and weights1
products1 = matmul(borrower_features, weights1)

# Apply a sigmoid activation function to products1 + bias1
dense1 = keras.activations.sigmoid(products1 + bias1)

# Print the shapes of borrower_features, weights1, bias1, and dense1
print('\n shape of borrower_features: ', borrower_features.shape)
print('\n shape of weights1: ', weights1.shape)
print('\n shape of bias1: ', bias1.shape)
print('\n shape of dense1: ', dense1.shape)


# Define the first dense layer
dense1 = keras.layers.Dense(7, activation='sigmoid')(borrower_features)

# Define a dense layer with 3 output nodes
dense2 = keras.layers.Dense(3, activation='sigmoid')(dense1)

# Define a dense layer with 1 output node
predictions = keras.layers.Dense(1, activation='sigmoid')(dense2)

# Print the shapes of dense1, dense2, and predictions
print('\n shape of dense1: ', dense1.shape)
print('\n shape of dense2: ', dense2.shape)
print('\n shape of predictions: ', predictions.shape)


'''
Activation functions
What is an Activation functions?
- Components of a typical hidden layer
* Linear : Matrix multiplication
* Nonlinear : Activation function

A simple example
import numpy as np
import tensorflow as tf

# Define example borrower features
young, old = 0.3, 0.6
low_bill, high_bill = 0.1, 0.5

# Apply matrix multiplication step for all feature combinations
young_high = 1.0*young + 2.0* high_bill
young_low = 1.0*young + 2.0* low_bill
old_high = 1.0*old + 2.0* high_bill
old_low = 1.0*old + 2.0* low_bill

# Difference in default predictions for young
print(young_high - young_low) -> in

0.8 -> out

# Difference in default predictions for old
print(old_high - old_low) -> in

0.8 -> out

# Difference in default predictions for young
print(tf.keras.activations.sigmoid(young_high).numpy() - tf.keras.activations.sigmoid(young_low).numpy() ) -> in

0.16337568 -> out

# Difference in default predictions for old
print(tf.keras.activations.sigmoid(old_high).numpy() - tf.keras.activations.sigmoid(old_low).numpy() ) -> in

0.14204389 -> out

The sigmoid activation function
- sigmoid activation function
* It is used primarily in the output layer of binary classification problems.
* Low-level: tf.keras.activations.sigmoid()
* High-level: sigmoid as a parameter to a keras dense layer

The relu activation function
- ReLu activation function
* It is typically used in all layers other than the output layer.
* Low-level: tf.keras.activations.relu()
* High-level: relu as a parameter to a keras dense layer

The softmax activation function
- Softmax activation function
* It is used in the output layer in classification problems with more than two classes.
* Low-level: tf.keras.activations.softmax()
* High-level: softmax as a parameter to a keras dense layer

Activation functions in neural networks
import tensorflow as tf

# Define input layer
inputs = tf.constant(borrower_features, tf.float32)

# Define dense layer 1
dense1 = tf.keras.layers.Dense(16, activation='relu')(inputs)

# Define dense layer 2
dense2 = tf.keras.layers.Dense(8, activation='sigmoid')(dense1)

# Define output layer 
outputs = tf.keras.layers.Dense(4, activation='softmax')(dense2)
'''

# Construct input layer from features
inputs = constant(bill_amounts, float32)

# Define first dense layer
dense1 = keras.layers.Dense(3, activation='relu')(inputs)

# Define second dense layer
dense2 = keras.layers.Dense(2, activation='relu')(dense1)

# Define output layer
outputs = keras.layers.Dense(1, activation='sigmoid')(dense2)

# Print error for first five examples
error = default[:5] - outputs.numpy()[:5]
print(error)


# Construct input layer from borrower features
inputs = constant(borrower_features, float32)

# Define first dense layer
dense1 = keras.layers.Dense(10, activation='sigmoid')(inputs)

# Define second dense layer
dense2 = keras.layers.Dense(8, activation='relu')(dense1)

# Define output layer
outputs = keras.layers.Dense(6, activation='softmax')(dense2)

# Print first five predictions
print(outputs.numpy()[:5])


'''
Optimizers
- This entails finding the set of weights that corresponds to the minimum value of the loss.

The gradient descent optimizer
- Stochastic gradient descent (SGD) optimizer
* tf.keras.optimizers.SGD()
* learning_rate : Typically between 0.5 - 0.001, which will detemine how quickly the model parameters adjust during training.
- The main advantage of SGD is that it is simpler and easier to interpret than more modern optimization algorithms.

The RMS prop optimizer
- Root mean squared (RMS) propagation optimizer.
* tf.keras.optimizers.RMSprop()
* learning_rate
* momentum
* decay
- It has 2 advantages over the SGD
* It applies different learning rates to each feature, which can be useful for high dimensional problems
* It allows you to both build momentum and also allow it to decay

The Adam optimizer
- Adaptive moment (Adam) optimizer
* tf.keras.optimizers.Adam()
* learning_rate
* beta1
- It tends to perform well with default parameter values

A complete example
import tensorflow as tf

# Define the model function
def model(bias, weights, features = borrower_features):
    product = tf.matmul(features, weights)
    return tf.keras.activations.sigmoid(product + bias)

# Compute the predicted values and loss
def loss_function(bias, weights, targets = default, features = borrower_features):
    predictions = model(bias, weights)
    return tf.keras.losses.binary_crossentropy(targets, predictions)

# Minimize the loss function with RMS propagation
opt = tf.keras.optimizers.RMSprop(learning_rate=0.01, momentum=0.9)
opt.minimize(lambda: loss_function(bias, weights), var_list=[bias, weights])
'''

# Initialize x_1 and x_2
x_1 = Variable(6.0, float32)
x_2 = Variable(0.3, float32)

# Define the optimization operation
opt = keras.optimizers.SGD(learning_rate=0.01)

for j in range(100):
    # Perform minimization using the loss function and x_1
    opt.minimize(lambda: loss_function(x_1), var_list=[x_1])
    # Perform minimization using the loss function and x_2
    opt.minimize(lambda: loss_function(x_2), var_list=[x_2])

# Print x_1 and x_2 as numpy arrays
print(x_1.numpy(), x_2.numpy())


# Initialize x_1 and x_2
x_1 = Variable(0.05, float32)
x_2 = Variable(0.05, float32)

# Define the optimization operation for opt_1 and opt_2
opt_1 = keras.optimizers.RMSprop(learning_rate=0.01, momentum=0.99)
opt_2 = keras.optimizers.RMSprop(learning_rate=0.01, momentum=0.00)

for j in range(100):
    opt_1.minimize(lambda: loss_function(x_1), var_list=[x_1])
    # Define the minimization operation for opt_2
    opt_2.minimize(lambda: loss_function(x_2), var_list=[x_2])

# Print x_1 and x_2 as numpy arrays
print(x_1.numpy(), x_2.numpy())


'''
Training a network in TensorFlow
Random initializers
- Often need to initialize thousands of variables
* tf.ones() may perform poorly
* Tedious and difficult to initialize variables individually

- Alternatively, draw initial values from distribution
* Normal distribution
* Uniform distribution

- There are also specialized options, such as
* Glorot initializer which are designed for ML algorithms

Initializing variables in TensorFlow
import tensorflow as tf

# Define 500x500 random normal variable
weights = tf.Variable(tf.random.normal([500, 500]))

# Define 500x500 truncated random normal variable
weights = tf.Variable(tf.random.truncated_normal([500, 500]))

# Define a dense layer with the default initializer
dense = tf.keras.layers.Dense(32, activation='relu')

# Define a dense layer with the zero initializer
dense = tf.keras.layers.Dense(32, activation='relu', kernel_initializer='zeros')

Neural networks and overfitting
- Overfitting is especially problematic for neural networks, which contains many parameters and are quite good at memorization.

Applying dropout
- It is an operation that will randomly drop the weights connected to certain nodes in a layer during the training process
* This will force the network to develop more robust rules for classification, since it cannot rely on any particular nodes being passed to an activation function.
- It will tend to improve out-of-sample performance.

Implementing dropout in a network
import numpy as np
import tensorflow as tf

# Define input data
inputs = np.array(borrower_features, np.float32)

# Define dense layer 1
dense1 = tf.keras.layers.Dense(32, activation='relu')(inputs)

# Define dense layer 2
dense2 = tf.keras.layers.Dense(16, activation='relu')(dense1)

# Apply dropout operation
dropout1 = tf.keras.layers.Dropout(0.25)(dense2)

- The argument 0.25 is used to drop the weights connected to 25% of nodes randomly.

# Define output layer
outputs = tf.keras.layers.Dense(1, activation='sigmoid')(dropout1)
'''

# Define the layer 1 weights
w1 = Variable(random.normal([23, 7]))

# Initialize the layer 1 bias
b1 = Variable(ones([7]))

# Define the layer 2 weights
w2 = Variable(random.normal([7, 1]))

# Define the layer 2 bias
b2 = Variable([0.0])


# Define the model
def model(w1, b1, w2, b2, features=borrower_features):
    # Apply relu activation functions to layer 1
    layer1 = keras.activations.relu(matmul(features, w1) + b1)
    # Apply dropout rate of 0.25
    dropout = keras.layers.Dropout(0.25)(layer1)
    return keras.activations.sigmoid(matmul(dropout, w2) + b2)

# Define the loss function


def loss_function(w1, b1, w2, b2, features=borrower_features, targets=default):
    predictions = model(w1, b1, w2, b2)
    # Pass targets and predictions to the cross entropy loss
    return keras.losses.binary_crossentropy(targets, predictions)


# Train the model
for j in range(100):
    # Complete the optimizer
    opt.minimize(lambda: loss_function(
        w1, b1, w2, b2), var_list=[w1, b1, w2, b2])

# Make predictions with model using test features
model_predictions = model(w1, b1, w2, b2, test_features)

# Construct the confusion matrix
confusion_matrix(test_targets, model_predictions)


''' High Level APIs '''

'''
Defining neural networks with Keras
The sequential API
- This API is simpler and makes strong assumptions about how you will construct your model.
- It assumes :
* Input layer
* Hidden layers
* Output layer
- All these layers are ordered one after the other in a sequence.

Building a sequential model
# Import tensorflow
from tensorflow import keras

# Define a sequential model
model = keras.Sequential()

# Define first hidden layer
model.add(keras.layers.Dense(16, activation='relu', input_shape=(28*28,)))

# Define second hidden layer
model.add(keras.layers.Dense(8, activation='relu'))

# Define output layer
model.add(keras.layers.Dense(4, activation='softmax'))

# Compile the model
model.compile('adam', loss='categorical_crossentropy')

# Summarize the model
print(model.summary())

Using the functional API
- It is used to train 2 models jointly to predict the same target.

# Import tensorflow
import tensorflow as tf

# Define model 1 input layer shape
model1_inputs = tf.keras.Input(shape=(28*28,))

# Define model 2 input layer shape
model2_inputs = tf.keras.Input(shape=(10,))

# Define layer 1 for model 1
model1_layer1 = tf.keras.layers.Dense(12, activation='relu')(model1_inputs)

# Define layer 2 for model 1
model1_layer2 = tf.keras.layers.Dense(4, activation='softmax')(model1_layer1)

# Define layer 1 for model 2
model2_layer1 = tf.keras.layers.Dense(8, activation='relu')(model2_inputs)

# Define layer 2 for model 2
model2_layer2 = tf.keras.layers.Dense(4, activation='softmax')(model2_layer1)

# Merge model 1 and model 2
merged = tf.keras.layers.add([model1_layer2, model2_layer2])

# Define a functional model
model = tf.keras.Model(inputs=[model1_inputs, model2_inputs], outputs=merged)

# Compile the model
model.compile('adam', loss='categorical_crossentropy')
'''

# Define a Keras sequential model
model = keras.Sequential()

# Define the first dense layer
model.add(keras.layers.Dense(16, activation='relu', input_shape=(784,)))

# Define the second dense layer
model.add(keras.layers.Dense(8, activation='relu'))

# Define the output layer
model.add(keras.layers.Dense(4, activation='softmax'))

# Print the model architecture
print(model.summary())


# Define the first dense layer
model.add(keras.layers.Dense(16, activation='sigmoid', input_shape=(784,)))

# Apply dropout to the first layer's output
model.add(keras.layers.Dropout(0.25))

# Define the output layer
model.add(keras.layers.Dense(4, activation='softmax'))

# Compile the model
model.compile('adam', loss='categorical_crossentropy')

# Print a model summary
print(model.summary())


# For model 1, pass the input layer to layer 1 and layer 1 to layer 2
m1_layer1 = keras.layers.Dense(12, activation='sigmoid')(m1_inputs)
m1_layer2 = keras.layers.Dense(4, activation='softmax')(m1_layer1)

# For model 2, pass the input layer to layer 1 and layer 1 to layer 2
m2_layer1 = keras.layers.Dense(12, activation='relu')(m2_inputs)
m2_layer2 = keras.layers.Dense(4, activation='softmax')(m2_layer1)

# Merge model outputs and define a functional model
merged = keras.layers.add([m1_layer2, m2_layer2])
model = keras.Model(inputs=[m1_inputs, m2_inputs], outputs=merged)

# Print a model summary
print(model.summary())


'''
Training and validation with Keras
Overview of training and evaluation
- Load and clean data
- Define model
- Trai and validate model
- Evaluate model

How to train a model
# Import tensorflow
import tensorflow as tf

# Define a sequential model
model = tf.keras.Sequential()

# Define the hidden layer
model.add(tf.keras.layers.Dense(16, activation='relu', input_shape=(784,)))

# Define the output layer
model.add(tf.keras.layers.Dense(4, activation='softmax'))

# Compile the model
model.compile('adam', loss='categorical_crossentropy')

# Train the model
model.fit(image_features, image_labels)

The fit() operation
- Only 2 Required arguments
* features
* labels

- Many optional arguments may include
* batch_size
* epochs
* validation_split

Batch size and epochs
- The number of examples in each batch is the batch size, which is 32 by default.
- The number of times you train on the full set of batches is called the number of epochs
- Using multiple epochs allows the model to revisit the same batches, but with different model weights and possibly optimizers parameters, since they are updated after each batch.

Performing validation
- validation_split parameter divides the dataset into two parts.
* Train set
* Validation set

Performing validation
# Train model with validation split
model.fit(features, labels, epochs=10, validation_split=0.20)

- Selecting aa value of 0.2 will put 20% of the data in the validation set.
- The benefit of using a validation split is that you can see how your model performs on both 
* the data it was trained on, the training set, 
* and the separate dataset it was not trained on, the validation set.

- If the training loss becomes substantially lower than the validation loss, it is an indication that we're overfitting.
* either terminate the training process before that point
* add regularization or dropout

Changing the metric
# Recompile the model with the accuracy metric
model.compile('adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train model with validation split
model.fit(features, labels, epochs=10, validation_split=0.20)

The evaluation() operation
- It is a good idea to split off a test set before you begin to train and validate.
- You can use the evaluate operation to check performance on the test set at the end of the training process.

# Evaluate the test set
model.evaluate(test)

- Since you may tune model parameters in response to validation set performance, using a separate test set will provide you with futher assurance that you have not overfitted.
'''

# Define a sequential model
model = keras.Sequential()

# Define a hidden layer
model.add(keras.layers.Dense(16, activation='relu', input_shape=(784,)))

# Define the output layer
model.add(keras.layers.Dense(4, activation='softmax'))

# Compile the model
model.compile('SGD', loss='categorical_crossentropy')

# Complete the fitting operation
model.fit(sign_language_features, sign_language_labels, epochs=5)


# Define sequential model
model = keras.Sequential()

# Define the first layer
model.add(keras.layers.Dense(32, activation='sigmoid', input_shape=(784,)))

# Add activation function to classifier
model.add(keras.layers.Dense(4, activation='softmax'))

# Set the optimizer, loss function, and metrics
model.compile(optimizer='RMSprop',
              loss='categorical_crossentropy', metrics=['accuracy'])

# Add the number of epochs and the validation split
model.fit(sign_language_features, sign_language_labels,
          epochs=10, validation_split=0.1)


# Define sequential model
model = keras.Sequential()

# Define the first layer
model.add(keras.layers.Dense(1024, activation='relu', input_shape=(784,)))

# Add activation function to classifier
model.add(keras.layers.Dense(4, activation='softmax'))

# Finish the model compilation
model.compile(optimizer=keras.optimizers.Adam(lr=0.001),
              loss='categorical_crossentropy', metrics=['accuracy'])

# Complete the model fit operation
model.fit(sign_language_features, sign_language_labels,
          epochs=50, validation_split=0.5)


# Evaluate the small model using the train data
small_train = small_model.evaluate(train_features, train_labels)

# Evaluate the small model using the test data
small_test = small_model.evaluate(test_features, test_labels)

# Evaluate the large model using the train data
large_train = large_model.evaluate(train_features, train_labels)

# Evaluate the large model using the test data
large_test = large_model.evaluate(test_features, test_labels)

# Print losses
print('\n Small - Train: {}, Test: {}'.format(small_train, small_test))
print('Large - Train: {}, Test: {}'.format(large_train, large_test))


'''
Training models with the Estimators API
What is the Estimators API ?
- It is a high level TensorFlow submodule
High-Level TensorFlow APIs      Estimators
Mid-Level TensorFlow APIs       Layers  Datasets    Metrics
Low-Level TensorFlow APIs       Python

- Relative to the core, lower-level TensorFlow APIs and the high-level keras API, model building in the Estimator API is less flexible
* This is because it enforces a set of best practices by placing restrictions on model architecture and training.
- The upside of using the Estimators API is that it allows for faster deployment.
* Models can be specified, trained, evaluated, and deployed with less code.
- There are many premade models that can be instantiated by setting a handful of model parameters.

Model specification and training
- Define feature columns: It specify the shape and type of your data
- Load and transform your data within a function
* The output of this function will be a dictionary object of features and your labels.
- Define an estimator
- Apply train operation

- NB*: All model objects created through the Estimators API have train, evaluate, and predict operations.

Defining feature columns
# Import tensorflow under its standard alias
import tensorflow as tf

# Define a numeric feature column
size = tf.feature_column.numeric_column('size')

# Define a categorical feature column
rooms = tf.feature_column.categorical_column_with_vocabulary_list('rooms', ['1', '2', '3', '4', '5'])

# Create feature column list
features_list = [size, rooms]

# Define a matrix feature column
features_list = [tf.feature_column.numeric_column('image', shape=(784,))]

Loading and transforming data
# Define input data function
def input_fn():
    # Define feature dictionary
    features = {'size': [1340, 1690, 2720], 'rooms': [1, 3, 4]}
    # Define labels
    labels = [221900, 538000, 180000]
    return features, labels

Define and train a regression estimator
# Define a deep neural network regression
model0 = tf.estimator.DNNRegressor(feature_columns=feature_list, hidden_units=[10, 6, 6, 3])

# Train the regression model
model0.train(input_fn, steps=20)

# Define a deep neural network classifier
model1 = tf.estimator.DNNClassifier(feature_columns=feature_list, hidden_units=[32, 16, 8], n_classes=4)

# Train the classifier model
model1.train(input_fn, steps=20)
'''

# Define feature columns for bedrooms and bathrooms
bedrooms = feature_column.numeric_column("bedrooms")
bathrooms = feature_column.numeric_column("bathrooms")

# Define the list of feature columns
feature_list = [bedrooms, bathrooms]


def input_fn():
    # Define the labels
    labels = np.array(housing['price'])

    # Define the features
    features = {'bedrooms': np.array(
        housing['bedrooms']), 'bathrooms': np.array(housing['bathrooms'])}
    return features, labels


# Define the model and set the number of steps
model = estimator.DNNRegressor(
    feature_columns=feature_list, hidden_units=[2, 2])
model.train(input_fn, steps=1)


# Define the model and set the number of steps
model = estimator.LinearRegressor(feature_columns=feature_list)
model.train(input_fn, steps=2)


'''
TensorFlow extensions
- TensorFlow Hub
* It allows users to import pretrained models that can be used to perform transfer learning.
* It will be particularly useful when you want to train a image classifier with a small number of images, but want to make use of a feature-extractor trained on a much larger set of different images.

- TensorFlow Probability
* One of its benefits is that it provides additional statistical distributions that can be used for random number generation.
* It also enables you to incorporate trainable statistical distributions into your models
* It provides an extended set of optimizers that are commonly used in statistical research.
'''
