''' Basics of deep learning and neural networks '''

'''
Introduction to deep learning
Interactions
- Neural networks are a powerful modeling approach that accounts for interactions really well
- Deep learning is the use of especially powerful neural networks
* They have the ability to capture extremely complex interactions in
** Text
** Images
** Videos
** Audio
** Source code

Build and tune deep learning models using keras
import numpy as np
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

predictors = np.loadtxt('predictors_data.csv', delimiter=',')
n_cols = predictors.shape[1]
model = Sequential()

model.add(Dense(100, activation='relu', input_shape = (n_cols, )))
model.add(Dense(100, activation='relu'))
model.add(Dense(1))

- NB*: The more nodes in the hidden layer, the greater it's ability to capture interactions.
'''


'''
Forward propagation
- Forward Propagation algorithm is the way in which Neural networks use data to make predictions.

Forward propagation
- Multiply - add process
- Dot product
- Forward propagation for one data point at a time
- Output is the prediction for that data point

Forward propagation code
import numpy as np

input_data = np.array([2, 3])
weights = {'node_0': np.array([1, 1]), 'node_1': np.array([-1, 1]), 'output': np.array([2, -1])}
node_0_value = (input_data * weights['node_0']).sum()
node_1_value = (input_data * weights['node_1']).sum()

hidden_layer_values = np.array([node_0_value, node_1_value])
print(hidden_layer_values) -> in

[5, 1] -> out

output = (hidden_layer_values * weights['output']).sum()
print(output) -> in

9 -> out
'''

# Calculate node 0 value: node_0_value
from tensorflow.keras.optimizers import SGD
from sklearn.metrics import mean_squared_error
node_0_value = (input_data * weights['node_0']).sum()

# Calculate node 1 value: node_1_value
node_1_value = (input_data * weights['node_1']).sum()

# Put node values into array: hidden_layer_outputs
hidden_layer_outputs = np.array([node_0_value, node_1_value])

# Calculate output: output
output = (hidden_layer_outputs * weights['output']).sum()

# Print output
print(output)


'''
Activation functions
- It is applied to the hidden layers of neural networks so as to achieve their maximum predictive power.
- It allows the model to capture non-linearities.
- If the relationships in the data aren't straight-line relationships, we will need an activation function that captures non-linearities.
- It is applied to node inputs to produce node output
- For a long time, an s-shaped function called tanh was a popular activation function.
* slope = tanh(input)
- The standard activation function in both industry and research application today is ReLu or Rectified Linear Activation function.
* slope = input if input > 0 
* slope = 0 if input <= 0
- The input is sometimes called the identity function

Activation functions
import numpy as np

input_data = np.array([-1, 2])
weights = {'node_0': np.array([3, 3]), 'node_1': np.array([1, 5]), 'output': np.array([2, -1])}
node_0_input = (input_data * weights['node_0']).sum()
node_0_output = np.tanh(node_0_input)
node_1_input = (input_data * weights['node_1']).sum()
node_1_output = np.tanh(node_1_input)
hidden_layer_outputs = np.array([node_0_output, node_1_output])
output = (hidden_layer_output * weights['output']).sum()
print(output) -> in

1.2382242525694254 -> out
'''


def relu(input):
    '''Define your relu activation function here'''
    # Calculate the value for the output of the relu function: output
    output = max(0, input)

    # Return the value just calculated
    return (output)


# Calculate node 0 value: node_0_output
node_0_input = (input_data * weights['node_0']).sum()
node_0_output = relu(node_0_input)

# Calculate node 1 value: node_1_output
node_1_input = (input_data * weights['node_1']).sum()
node_1_output = relu(node_1_input)

# Put node values into array: hidden_layer_outputs
hidden_layer_outputs = np.array([node_0_output, node_1_output])

# Calculate model output (do not apply relu)
model_output = (hidden_layer_outputs * weights['output']).sum()

# Print model output
print(model_output)


# Define predict_with_network()
def predict_with_network(input_data_row, weights):

    # Calculate node 0 value
    node_0_input = (input_data_row * weights['node_0']).sum()
    node_0_output = relu(node_0_input)

    # Calculate node 1 value
    node_1_input = (input_data_row * weights['node_1']).sum()
    node_1_output = relu(node_1_input)

    # Put node values into array: hidden_layer_outputs
    hidden_layer_outputs = np.array([node_0_output, node_1_output])

    # Calculate model output
    input_to_final_layer = (hidden_layer_outputs * weights['output']).sum()
    model_output = (input_to_final_layer)

    # Return model output
    return (model_output)


# Create empty list to store prediction results
results = []
for input_data_row in input_data:
    # Append prediction to results
    results.append(predict_with_network(input_data_row, weights))

# Print results
print(results)


'''
Deeper networks
- The difference between modern deep learning and the historical neural networks that didn't deliver these amazing results, is the use of models with not just one hidden layer, but with many successive hidden layers.

Representation learning
- Deep networks internally build up representations of the patterns in the data that are useful for making predictions.
- Neural networks partially replace the need for feature engineering, or manually creating better predictive features.
- Deep learning is sometimes called Representation learning, because subsequent layers build increasingly sophisticated representations of the raw data, until we get to a stage where we can make predictions.

Deep learning
- The modeler doesn't need to specify the interactions.
- When training the model, the neural network gets weights that find the relevant patterns to make better predictions
'''


def predict_with_network(input_data):
    # Calculate node 0 in the first hidden layer
    node_0_0_input = (input_data * weights['node_0_0']).sum()
    node_0_0_output = relu(node_0_0_input)

    # Calculate node 1 in the first hidden layer
    node_0_1_input = (input_data * weights['node_0_1']).sum()
    node_0_1_output = relu(node_0_1_input)

    # Put node values into array: hidden_0_outputs
    hidden_0_outputs = np.array([node_0_0_output, node_0_1_output])

    # Calculate node 0 in the second hidden layer
    node_1_0_input = (hidden_0_outputs * weights['node_1_0']).sum()
    node_1_0_output = relu(node_1_0_input)

    # Calculate node 1 in the second hidden layer
    node_1_1_input = (hidden_0_outputs * weights['node_1_1']).sum()
    node_1_1_output = relu(node_1_1_input)

    # Put node values into array: hidden_1_outputs
    hidden_1_outputs = np.array([node_1_0_output, node_1_1_output])

    # Calculate model output: model_output
    model_output = (hidden_1_outputs * weights['output']).sum()

    # Return model_output
    return (model_output)


output = predict_with_network(input_data)
print(output)


''' Optimizing a neural network with backward propagation '''

'''
The need for optimization
Predictions with multiple points
- Making accurate predictions gets harder with more points
- At any set of weights, there are many values of the error
* corresponding to the many points we make predictions for

Loss Function
- It is used to aggregate errors in predictions from many data points into a single number 
- It's a Measure of the model's predictive performance.
- Lower loss function value means a better model

Squared error loss function
- A common loss function for regression tasks is mean-squared error.

Prediction  Actual  Error   Squared Error
10          20      -10     100
8           3       5       25
6           1       5       25 

- Total Squared Error = 150
- Mean Squared Error = 50

Gradient descent steps
- Start at a random point
- Until you are somewhere flat:
* Find the slope
* Take a step downhill
'''

# The data point you will make a prediction for
input_data = np.array([0, 3])

# Sample weights
weights_0 = {'node_0': [2, 1], 'node_1': [1, 2], 'output': [1, 1]}

# The actual target value, used to calculate the error
target_actual = 3

# Make prediction using original weights
model_output_0 = predict_with_network(input_data, weights_0)

# Calculate error: error_0
error_0 = model_output_0 - target_actual

# Create weights that cause the network to make perfect prediction (3): weights_1
weights_1 = {'node_0': [2, 1], 'node_1': [1, 0], 'output': [1, 1]}

# Make prediction using new weights: model_output_1
model_output_1 = predict_with_network(input_data, weights_1)

# Calculate error: error_1
error_1 = model_output_1 - target_actual

# Print error_0 and error_1
print(error_0)
print(error_1)


# Create model_output_0
model_output_0 = []
# Create model_output_1
model_output_1 = []

# Loop over input_data
for row in input_data:
    # Append prediction to model_output_0
    model_output_0.append(predict_with_network(row, weights_0))

    # Append prediction to model_output_1
    model_output_1.append(predict_with_network(row, weights_1))

# Calculate the mean squared error for model_output_0: mse_0
mse_0 = mean_squared_error(target_actuals, model_output_0)

# Calculate the mean squared error for model_output_1: mse_1
mse_1 = mean_squared_error(target_actuals, model_output_1)

# Print mse_0 and mse_1
print("Mean squared error with weights_0: %f" % mse_0)
print("Mean squared error with weights_1: %f" % mse_1)


'''
Gradient descent
- If the slope is positive:
* Going opposite the slope means moving to lower numbers 
* Subtract the slope from the current value
* Too big a step might lead us astray

- Solution: learning rate (multiply the slope by a small number.)
* Update each weight by subtracting learning rate * slope
* Learning rate are frequently around 0.01. This ensures we take small steps, so we reliably move towards the optimal weights.

Slope calculation example
    2
3 ----> 6        Actual Target Value = 10

- To calculate the slope for a weight, need to multiply
* Slope of the loss function with respect to the value at the node we feed into
* The value of the node that feeds into our weight
* Slope of the activation function with respect to the value we feed into

- Slope of the mean-squared loss function with respect to the prediction:
* 2 * (Predicted Value - Actual Value) = 2 * Error
* 2 * (6 - 10)
* 2 * -4

- The value of the node that feeds into our weight
* 2 * error * input_data
* 2 * -4 * 3
* -24

- Slope of the activation function with respect to the value we feed into
* You can leave out this step since there's no activation function.

- If learning rate is 0.01, the new weight would be 
* 2 - 0.01(-24) = 2.24

Code to calculate slopes and update weights
import numpy as np

weights = np.array([1, 2])
input_data = np.array([3, 4])
target = 6
learning_rate = 0.01
preds = (weights * input_data).sum()
error = preds - target
print(error) -> in

5 -> out

gradient = 2 * input_data * error
print(gradient) -> in

array([30, 40]) -> out

weights_updated = weights - learning_rate * gradient
preds_updated = (weights_updated * input_data).sum()
error_updated = preds_updated - target
print(error_updated) -> in

2.5 -> out
'''

# Calculate the predictions: preds
preds = (weights * input_data).sum()

# Calculate the error: error
error = preds - target

# Calculate the slope: slope
slope = 2 * input_data * error

# Print the slope
print(slope)


# Set the learning rate: learning_rate
learning_rate = 0.01

# Calculate the predictions: preds
preds = (weights * input_data).sum()

# Calculate the error: error
error = preds - target

# Calculate the slope: slope
slope = 2 * input_data * error

# Update the weights: weights_updated
weights_updated = weights - learning_rate * slope

# Get updated predictions: preds_updated
preds_updated = (weights_updated * input_data).sum()

# Calculate updated error: error_updated
error_updated = preds_updated - target

# Print the original error
print(error)

# Print the updated error
print(error_updated)


n_updates = 20
mse_hist = []

# Iterate over the number of updates
for i in range(n_updates):
    # Calculate the slope: slope
    slope = get_slope(input_data, target, weights)

    # Update the weights: weights
    weights = weights - 0.01 * slope

    # Calculate mse with new weights: mse
    mse = get_mse(input_data, target, weights)

    # Append the mse to mse_hist
    mse_hist.append(mse)

# Plot the mse history
plt.plot(mse_hist)
plt.xlabel('Iterations')
plt.ylabel('Mean Squared Error')
plt.show()


'''
Backpropagation
- It is used to calculate the slopes to optimize more complex deep learning models
- It takes the error from the output layer and propagates it backward through the hidden layers, towards the input layer.
- It allows gradient descent to update all weights in neural network ( by getting gradients for all weights).
- It comes from chain rule of calculus

Backpropagation process
- It is trying to estimate the slope of the loss function with respect to each weight in our network
- Always Do forward propagation to make a prediction and calculate an error before doing back propagation.
- It goes back one layer at a time

- Gradients for weight is the product of:
* Node value feeding into that weight
* Slope of loss function with respect to the node it feeds into
* Slope of activation function at the node it feeds into

- Need to also keep track of the slopes of the loss function with respect to the node values
* because we use those slopes in our calculations of slopes at weights.

- Slope of node values are the sum of the slopes for all weights that come out of them.
'''


'''
Backpropagation in practice
Calculating slopes associated with any weight
- Gradients for weight is the product of:
* Node value feeding into that weight
* Slope of activation function for the node being fed into
* Slope of loss function with respect to the output node

Backpropagation: Recap
- Start at some random set of weights
- Use forward propagation to make a prediction
- Use backward propagation to calculate the slope of the loss function with respect to each weight
- Multiply that slope by the learning rate, and subtract from the current weights
- Keep going with that cycle until we get to a flat part

Stochastic gradient descent
- It is common to calculate slopes on only a subset of the data ( a Batch), for each update of the weights.
- Use a different batch of data to calculate the next update
- Start over from the beginning once all data is used.
- Each time through the full training data is called an epoch.

- NB*: When slopes are canculated on one batch at a time, rather than on the full data, it  is called stochastic gradient descent
* Rather than, gradient descent, which uses all of the data for each slope calculation.
'''


''' Building deep learning models with keras '''

''' 
Creating a Keras model
Model building steps
- Specify Architecture e.g
* How many layers
* How many nodes in each layer
* What activation function to use in each layer

- Compile the model
* Specify loss function
* Some details about how optimization works

- Fit the model
* Cycle of back-propagation
* Optimization of model weights with the data

- Make Predictions with the model.

Model specification
- Sequential models require that each layer has weights or connections only to the one layer coming directly after it in the network diagram.
- The standard layer type is called a Dense layer
* It is called Dense because all of the nodes in the previous layer connect to all of the nodes in the current layer.

import numpy as np
from tensorflow.keras.layers import Dense
from tensorflow.keras.model import Sequential

predictors = np.loadtxt('predictors_data.csv'), delimiter=',')
n_cols = predictors.shape[1]

model = Sequential()
model.add(Dense(100, activation='relu', input_shape = (n_cols, )))
model.add(Dense(100, activation='relu'))
model.add(Dense(1))
'''

# Import necessary modules

# Save the number of columns in predictors: n_cols
n_cols = predictors.shape[1]

# Set up the model: model
model = Sequential()

# Add the first layer
model.add(Dense(50, activation='relu', input_shape=(n_cols, )))

# Add the second layer
model.add(Dense(32, activation='relu'))

# Add the output layer
model.add(Dense(1))


'''
Compiling and fitting a model
- It sets up the model for optimization.

Why you need to compile your model
- Specify the optimizer
* It controls the learning rate
* Many options and mathematically complex
* 'Adam' is usually a good choice
** It adjusts the learning rate as it does gradient descent, to ensure reasonable values throughout the weight optimization process.

- Loss function
* 'mean_squared_error' is the most common choice for regression problems

n_cols = predictors.shape[1]
model = Sequential()
model.add(Dense(100, activation='relu', input_shape = (n_cols, )))
model.add(Dense(100, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

What is fitting a model
- It is applying backpropagation and gradient descent with the data to update the weights
- Scaling data before fitting can ease optimization
* One common approach is to subtract each feature by that feature mean, and divide it by its standard deviation

n_cols = predictors.shape[1]
model = Sequential()
model.add(Dense(100, activation='relu', input_shape = (n_cols, )))
model.add(Dense(100, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(predictors, target)
'''

# Import necessary modules

# Specify the model
n_cols = predictors.shape[1]
model = Sequential()
model.add(Dense(50, activation='relu', input_shape=(n_cols,)))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Verify that model contains information from compiling
print("Loss function: " + model.loss)


# Import necessary modules

# Specify the model
n_cols = predictors.shape[1]
model = Sequential()
model.add(Dense(50, activation='relu', input_shape=(n_cols,)))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Fit the model
model.fit(predictors, target)


'''
Classification models
Classification
- 'categorical_crossentropy' is the most common loss function for classification
* Similar to log loss: Lower score is better

- Add metrics = ['accuracy'] to compile step for easy-to-understand diagnostics.

- Output layer has separate node for each possible outcome, and uses 'softmax' activation
* The 'softmax' activation function ensures the predictions sum to 1, so they can be interpreted as probabilities.

Quick look at the data
shot_clock  dribbles    touch_time  shot_dis    close_def_dis   shot_result
10.8        2           1.9         7.7         1.3             1
3.4         0           0.8         28.2        6.1             0
0           3           2.7         10.1        0.9             0
10.3        2           1.9         17.2        3.4             0

Transforming to categorical (use one-hot encoding)
shot_result         Outcome 0   Outcome 1
1                   0           1
0                   1           0
0           ->      1           0
0                   1           0

Classification
from tensorflow.keras.utils import to_categorical

data = pd.read_csv('basketball_shot_log.csv')
predictors = data.drop(['shot_result'], axis=1).values
target = to_categorical(data['shot_result'])

model = Sequential()
model.add(Dense(100, activation='relu', input_shape = (n_cols, )))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(2, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(predictors, target)
'''

# Import necessary modules

# Convert the target to categorical: target
target = to_categorical(df.survived)

# Set up the model
model = Sequential()

# Add the first layer
model.add(Dense(32, activation='relu', input_shape=(n_cols, )))

# Add the output layer
model.add(Dense(2, activation='softmax'))

# Compile the model
model.compile(optimizer='sgd', loss='categorical_crossentropy',
              metrics=['accuracy'])

# Fit the model
model.fit(predictors, target)


'''
Using models
- Save a model after training
- Reload the model
- Make predictions with the model

Saving, reloading, and using your Model
from tensorflow.keras.models import load_model

model.save('model_file'h5')
my_model = load_model('model_file.h5')
predictions = my_model.predict(data_to_predict_with)
probability_true = predictions[:, 1]

- Models are saved in a format called hdf5, for which h5 is the common extension.
- load_model() is used to load the file back into memory
- The predictions come in the same format as the prediction target

Verifying model structure
my_model.summary() 
'''

# Specify, compile, and fit the model
model = Sequential()
model.add(Dense(32, activation='relu', input_shape=(n_cols,)))
model.add(Dense(2, activation='softmax'))
model.compile(optimizer='sgd', loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(predictors, target)

# Calculate predictions: predictions
predictions = model.predict(pred_data)

# Calculate predicted probability of survival: predicted_prob_true
predicted_prob_true = predictions[:, 1]

# Print predicted_prob_true
print(predicted_prob_true)


''' Fine-tuning keras models '''

'''
Understanding model optimization
Why optimization is hard
- Simultaneously optimizing 1000s of parameters with complex relationships
- Updates may not improve model meaningfully
- Updates too small (if learning rate is low) or too large (if learning rate is high)

Stochastic gradient descent
def get_new_model(input_shape = input_shape):
    model = Sequential()
    model.add(Dense(100, activation = 'relu', input_shape = input_shape))
    model.add(Dense(100, activation = 'relu'))
    model.add(Dense(100, activation = 'softmax'))
    return(model)

lr_to_test = [0.000001, 0.01, 1]

# Loop over learning rates
for lr in lr_to_test:
    model = get_new_model()
    my_optimizer = SGD(lr = lr)
    model.compile(optimizer = my_optimizer, loss = 'categorical_crossentropy')
    model.fit(predictors, target)

The dying neuron problem
- This occurs when a neuron takes a value less than 0 for all rows of the data.
- Suggest using an activation function whose slope is never exactly zero

Vanishing gradients
- It occurs when many layers have very small slopes (e.g due to being on flat part of tanh curve)
- In deep networks, updates to backprop were close to 0
- Suggest using an activation function that isn't close to flat anywhere

- If the model doesnt train better, changing the activation function may be the solution
'''

# Import the SGD optimizer

# Create list of learning rates: lr_to_test
lr_to_test = [0.000001, 0.01, 1]

# Loop over learning rates
for lr in lr_to_test:
    print('\n\nTesting model with learning rate: %f\n' % lr)

    # Build new model to test, unaffected by previous models
    model = get_new_model()

    # Create SGD optimizer with specified learning rate: my_optimizer
    my_optimizer = SGD(lr=lr)

    # Compile the model
    model.compile(optimizer=my_optimizer, loss='categorical_crossentropy')

    # Fit the model
    model.fit(predictors, target)


'''
Model validation
- Validation data is data that is explicitly held out from training, and used only to test model performance.

Validation in deep learning
- Commonly use validation split rather than cross-validation
- Deep learning is widely used on large datasets
- Single validation score is based on large amount of data and is reliable

Model validation
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(predictors, target, validation_split=0.3)

Early Stopping
- It helps to ensure that the model keeps training while the validation score is improving and then stop training when the validation score isn't improving.

from tensorflow.keras.callbacks import EarlyStopping

early_stopping_monitor = EarlyStopping(patience = 2)
model.fit(predictors, target, validation_split=0.3, epochs=20, callbacks = [early_stopping_monitor])

- By default, keras trains for 10 Epochs
'''

# Save the number of columns in predictors: n_cols
n_cols = predictors.shape[1]
input_shape = (n_cols,)

# Specify the model
model = Sequential()
model.add(Dense(100, activation='relu', input_shape=input_shape))
model.add(Dense(100, activation='relu'))
model.add(Dense(2, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])

# Fit the model
hist = model.fit(predictors, target, validation_split=0.3)


# Import EarlyStopping

# Save the number of columns in predictors: n_cols
n_cols = predictors.shape[1]
input_shape = (n_cols,)

# Specify the model
model = Sequential()
model.add(Dense(100, activation='relu', input_shape=input_shape))
model.add(Dense(100, activation='relu'))
model.add(Dense(2, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])

# Define early_stopping_monitor
early_stopping_monitor = EarlyStopping(patience=2)

# Fit the model
model.fit(predictors, target, epochs=30, validation_split=0.3,
          callbacks=[early_stopping_monitor])


# Define early_stopping_monitor
early_stopping_monitor = EarlyStopping(patience=2)

# Create the new model: model_2
model_2 = Sequential()

# Add the first and second layers
model_2.add(Dense(100, activation='relu', input_shape=input_shape))
model_2.add(Dense(100, activation='relu'))

# Add the output layer
model_2.add(Dense(2, activation='softmax'))

# Compile model_2
model_2.compile(optimizer='adam',
                loss='categorical_crossentropy', metrics=['accuracy'])

# Fit model_1
model_1_training = model_1.fit(predictors, target, epochs=15, validation_split=0.2, callbacks=[
                               early_stopping_monitor], verbose=False)

# Fit model_2
model_2_training = model_2.fit(predictors, target, epochs=15, validation_split=0.2, callbacks=[
                               early_stopping_monitor], verbose=False)

# Create the plot
plt.plot(model_1_training.history['val_loss'],
         'r', model_2_training.history['val_loss'], 'b')
plt.xlabel('Epochs')
plt.ylabel('Validation score')
plt.show()


# The input shape to use in the first hidden layer
input_shape = (n_cols,)

# Create the new model: model_2
model_2 = Sequential()

# Add the first, second, and third hidden layers
model_2.add(Dense(10, activation='relu', input_shape=input_shape))
model_2.add(Dense(10, activation='relu'))
model_2.add(Dense(10, activation='relu'))

# Add the output layer
model_2.add(Dense(2, activation='softmax'))

# Compile model_2
model_2.compile(optimizer='adam',
                loss='categorical_crossentropy', metrics=['accuracy'])

# Fit model 1
model_1_training = model_1.fit(
    predictors, target, epochs=15, validation_split=0.4, verbose=False)

# Fit model 2
model_2_training = model_2.fit(
    predictors, target, epochs=15, validation_split=0.4, verbose=False)

# Create the plot
plt.plot(model_1_training.history['val_loss'],
         'r', model_2_training.history['val_loss'], 'b')
plt.xlabel('Epochs')
plt.ylabel('Validation score')
plt.show()


'''
Thinking about model capacity
Overfitting
- It is the ability of a model to fit oddities in the training data, that are there purely due to happenstance, and that won't apply in a new dataset.

Underfitting
- It is when the model fails to find important predictive patterns in the training data

- Validation score is the ultimater measure of a model's predictive quality.

Model Capacity
- It is a model's ability to capture predictive patterns in the data.

Workflow for optimizing model capacity
- Start with a small network and get the validation score.
- Gradually increase capacity as long as the score keeps improving
- Keep increasing capacity until validation score is no longer improving

Sequential experiments
Hidden Layers   Nodes Per Layer     Mean Squared Error  Next Step
1               100                 5.4                 Increase Capacity
1               250                 4.8                 Increase Capacity
2               250                 4.4                 Increase Capacity
3               250                 4.5                 Decrease Capacity
2               200                 4.3                 Done
'''


'''
Stepping up to images
Recognizing handwritten digits
- MNIST dataset
- 28 x 28 grid flattened to 784 values for each image
- Value in each part of array denotes darkness of that pixel
'''

# Create the model: model
model = Sequential()

# Add the first hidden layer
model.add(Dense(50, activation='relu', input_shape=(784,)))

# Add the second hidden layer
model.add(Dense(50, activation='relu'))

# Add the output layer
model.add(Dense(10, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])

# Fit the model
model.fit(X, y, validation_split=0.3, epochs=10)
