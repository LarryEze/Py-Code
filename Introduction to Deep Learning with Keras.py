''' Introducing Keras '''

'''
What is Keras?
Why use a neural network?
- They are good feature extractors, since they learn the best way to make sense of unstructured data.
* i.e they can perform feature engineering themselves

Machine Learning
Input -> Feature extraction -> Classification -> Output

Deep Learning
Input -> Feature extraction + Classification -> Output

Unstructured data
- It is data that is not easily put into a table e.g sound, videos, images etc.
- It is a type of data where performing feature engineering can be more challenging
* That's why a good idea is to leave the task to neural networks

When to use neural networks?
- When dealing with unstructured data
- Don't need easily interpretable results
- When the problem can benefit from a known architecture

Example: Classify images of cats and dogs
- Images -> Unstructured data
- You don't care about why the network knows it's a cat or a dog
- You can benefit from convolutional neural networks
'''


'''
Your first neural network
- A neural network is a ML algorithm that is fetch with training data through its input layer to then predict a value at its output layer.
* Input Layer -> Hidden Layer -> Output Layer

Parameters
- Each connection from one neuron to another has an associated weight, w.
- Each neuron, except the input layer which just holds the input value, also has an extra weight and we call this the bias weight, b.
- During feed-forward, our input gets transformed by weight multiplications and additions at each layer.
- The output of each neuron can also get transformed by the application of what we call an activation function.

Gradient descent
- Learning in neural networks consists of tuning the weights or parameters to give the desired output.
- One way of achieving this is by using the famous gradient descent algorithm
* and applying weight updates incrementally via a process known as back-propagation.

- NB*: Keras allows models building in two different ways
* Using the Functional API
* Using the Sequential API

The Sequential API
Defining a neural network
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Create a new sequential model
model = Sequential()

# Add an input and dense layer
model.add(Dense(2, input_shape=(3, ), activation='relu'))

# Add a final 1 neuron layer
model.add(Dense(1))

Summarize your model!
model.summary()
'''

# Import the Sequential model and Dense layer

# Create a Sequential model
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.layers import Conv2D, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
model = Sequential()

# Add an input layer and a hidden layer with 10 neurons
model.add(Dense(10, input_shape=(2,), activation="relu"))

# Add a 1-neuron output layer
model.add(Dense(1))

# Summarise your model
model.summary()


# Instantiate a new Sequential model
model = Sequential()

# Add a Dense layer with five neurons and three inputs
model.add(Dense(5, input_shape=(3, ), activation="relu"))

# Add a final Dense layer with one neuron and no activation
model.add(Dense(1))

# Summarize your model
model.summary()


# Instantiate a Sequential model
model = Sequential()

# Build the input and hidden layer
model.add(Dense(3, input_shape=(2, )))

# Add the ouput layer
model.add(Dense(1))


'''
Surviving a meteor strike
Compiling
- A model needs to be compiled before training

# Compiling your previously built model
model.compile(optimizer='adam', loss='mse')

- The optimizer is an algorithm that will be used to update the neural network weights
- Loss function is the function we want to minimize during training.

Training
# Train your model
model.fit(X_train, y_train, epochs=5)

- During an epoch, the entire training data passes through the network and the repective weight updates take place using back-propagation.

Predicting
# Predict on new data
preds = model.predict(X_test)

# Look at the predictions
print(preds)

Evaluating
# Evaluate your results
model.evaluate(X_test, y_test)
'''

# Instantiate a Sequential model
model = Sequential()

# Add a Dense layer with 50 neurons and an input of 1 neuron
model.add(Dense(50, input_shape=(1,), activation='relu'))

# Add two Dense layers with 50 neurons and relu activation
model.add(Dense(50, activation='relu'))
model.add(Dense(50, activation='relu'))

# End your model with a Dense layer and no activation
model.add(Dense(1))


# Compile your model
model.compile(optimizer='adam', loss='mse')

print("Training started..., this can take a while:")

# Fit your model on your data for 30 epochs
model.fit(time_steps, y_positions, epochs=30)

# Evaluate your model
print("Final loss value:", model.evaluate(time_steps, y_positions))


# Predict the twenty minutes orbit
twenty_min_orbit = model.predict(np.arange(-10, 11))

# Plot the twenty minute orbit
plot_orbit(twenty_min_orbit)

# Predict the eighty minute orbit
eighty_min_orbit = model.predict(np.arange(-40, 41))

# Plot the eighty minute orbit
plot_orbit(eighty_min_orbit)


''' Going Deeper '''

'''
Binary classification
When to use Binary classification?
- It is used when to solve problems where you predict whether an observation belongs to one of two possible classes.

Our dataset
co-ordinates        labels
[0.242, 0.038]      1
[0.044, -0.057]     1
[-0.787, -0.076]    0

Pairplots
import seaborn as sns

# Plot a pairplot
sns.pairplot(circles, hue='target')

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Instantiate a sequential model
model = Sequential()

# Add input and hidden layer
model.add(Dense(4, input_shape=(2, ), activation='tanh'))

# Add output layer, use sigmoid
model.add(Dense(1, activation='sigmoid'))

# Compile model
model.compile(optimizer='sgd', loss='binary_crossentropy')

# Train model
model.fit(coordinates, labels, epochs=20)

# Predict with trained model
preds = model.predict(coordinates) 

The sigmoid function
- It is the probability of a pair of cooredinates being in one class or another
* 3 (neuron output) -> sigmoid activation -> 0.95 (transformed output) -> 1 (rounded output)
- The sigmoid activation function squashes the neuron output of the second to last layer to a floating point number between 0 and 1.
'''

# Import seaborn

# Use pairplot and set the hue to be our class column
sns.pairplot(banknotes, hue='class')

# Show the plot
plt.show()

# Describe the data
print('Dataset stats: \n', banknotes.describe())

# Count the number of observations per class
print('Observations per class: \n', banknotes['class'].value_counts())


# Import the sequential model and dense layer

# Create a sequential model
model = Sequential()

# Add a dense layer
model.add(Dense(1, input_shape=(4,), activation='sigmoid'))

# Compile your model
model.compile(loss='binary_crossentropy',
              optimizer='sgd', metrics=['accuracy'])

# Display a summary of your model
model.summary()


# Train your model for 20 epochs
model.fit(X_train, y_train, epochs=20)

# Evaluate your model accuracy on the test set
accuracy = model.evaluate(X_test, y_test)[1]

# Print accuracy
print('Accuracy:', accuracy)


'''
Multi-class classification
Throwing darts
The dataset
xCoord      yCoord      competitor
-0.037673   0.057402    Steve
-0.331021  -0.585035    Susan
-0.123567   0.839730    Susan
-0.086160   0.959787    Michael
-0.902632   0.078753    Michael

The softmax activation
- It is used to make sure, the total sum of probabilities for the output neurons equals 1 (one).
e.g
# Instantiate a sequential model
# . . . 
# Add an input and hidden layer
# . . . 
# Add more hidden layers
# . . . 
# Add your output layer
model.add(Dense(4, activation='softmax')

Categorical cross-entropy / Log loss
- It measures the difference between the predicted probabilities and the true label of the class we should have predicted
e.g
model.compile(optimizer='adam', loss='categorical_crossentropy')

Preparing a dataset
import pandas as pd
from tensorflow.keras.utils import to_categorical

# Load dataset
df = pd.read_csv('data.csv')

# Turn response variable into labeled codes
df.response = pd.Categorical(df.response)
df.response = df.response.cat.codes

# Turn response variable into one-hot response vector
y = to_categorical(df.response)

One-hot encoding
Label Encoding                                  One Hot Encoding
Food Name   Categorical #   Calories            Apple   Chicken     Broccoli    Calories
Apple       1               95                  1       0           0           95 
Chicken     2               231         ->      0       1           0           231
Broccoli    3               50                  0       0           1           50
'''

# Instantiate a sequential model
model = Sequential()

# Add 3 dense layers of 128, 64 and 32 neurons each
model.add(Dense(128, input_shape=(2,), activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))

# Add a dense layer with as many neurons as competitors
model.add(Dense(4, activation='softmax'))

# Compile your model using categorical_crossentropy loss
model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])


# Transform into a categorical variable
darts.competitor = pd.Categorical(darts.competitor)

# Assign a number to each category (label encoding)
darts.competitor = darts.competitor.cat.codes

# Import to_categorical from keras utils module

coordinates = darts.drop(['competitor'], axis=1)
# Use to_categorical on your labels
competitors = to_categorical(darts.competitor)

# Now print the one-hot encoded labels
print('One-hot encoded competitors: \n', competitors)


# Fit your model to the training data for 200 epochs
model.fit(coord_train, competitors_train, epochs=200)

# Evaluate your model accuracy on the test data
accuracy = model.evaluate(coord_test, competitors_test)[1]

# Print accuracy
print('Accuracy:', accuracy)


# Predict on coords_small_test
preds = model.predict(coords_small_test)

# Print preds vs true values
print("{:45} | {}".format('Raw Model Predictions', 'True labels'))
for i, pred in enumerate(preds):
    print("{} | {}".format(pred, competitors_small_test[i]))

# Extract the position of highest probability from each pred vector
preds_chosen = [np.argmax(pred) for pred in preds]

# Print preds vs true values
print("{:10} | {}".format('Rounded Model Predictions', 'True labels'))
for i, pred in enumerate(preds_chosen):
    print("{:25} | {}".format(pred, competitors_small_test[i]))


'''
Multi-label classification
- Multi-class and Multi-label classification both deal with predicting classes, but
* In Multi-label classification, a single input can be assigned to more than one class.
* e.g to tag a serie's genres by its plot summary.
- In Multi-class problems, each individual in the sample will belong to a unique class
- In Multi-label problems, each individual in the sample can have all, none or a subset of the available classes.

The architecture
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Instantiate model
model = Sequential()

# Add input and hidden layers
model.add(Dense(2, input_shape=(1, )))

# Add an output layer fot the 3 classes and sigmoid activation
model.add(Dense(3, activation='sigmoid'))

# Compile the model with binary crossentropy
model.compile(optimizer='adam', loss='binary_crossentropy')

# Train your model, recall validation_split
model.fit(X_train, y_train, epochs=100, validation_split=0.2)

- validation_split is used to print validation loss and accuracy as it trains.
'''

# Instantiate a Sequential model
model = Sequential()

# Add a hidden layer of 64 neurons and a 20 neuron's input
model.add(Dense(64, input_shape=(20, ), activation='relu'))

# Add an output layer of 3 neurons with sigmoid activation
model.add(Dense(3, activation='sigmoid'))

# Compile your model with binary crossentropy loss
model.compile(optimizer='adam', loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()


# Train for 100 epochs using a validation split of 0.2
model.fit(sensors_train, parcels_train, epochs=100, validation_split=0.2)

# Predict on sensors_test and round up the predictions
preds = model.predict(sensors_test)
preds_rounded = np.round(preds)

# Print rounded preds
print('Rounded Predictions: \n', preds_rounded)

# Evaluate your model's accuracy on the test data
accuracy = model.evaluate(sensors_test, parcels_test)[1]

# Print accuracy
print('Accuracy:', accuracy)


'''
Keras callbacks
What is a callback?
- It is a function that is executed after some other function, event, or task has finished.

Callbacks in Keras
- It is a block of code that gets executed after each epoch during training or after the training is finished.
- They are useful to store metrics as the model trains and to make decisions as the training goes by.

A callback you've been missing
# Training a model and saving its history
history = model.fit(X_train, y_train, epochs=100, metrics=['accuracy'])
print(history.history['loss']) -> in

[0.6753975939750672, . . . , 0.3155936544282096] -> out

print(history.history['accuracy']) -> in

[0.7030952412741525, . . . , 0.8604761900220599] -> out

# Training a model and saving its history
history = model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test), metrics=['accuracy'])
print(history.history['val_loss']) -> in

[0.7753975939750672, . . . , 0.4155936544282096] -> out

print(history.history['val_accuracy']) -> in

[0.6030952412741525, . . . , 0.7604761900220599] -> out

History plots
# Plot train vs test accuracy per epoch
plt.figure()

# Use the history metrics
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])

# Make it pretty
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'])
plt.show() 

Early stopping
- It can be used to solve overfitting problems, sice it stops its training when it no longer improves.
* It is extremely useful since deep neural models can take a long time to train and we don't know beforehand how many epochs will be needed.


Early stopping
# Import early stopping from keras callbacks
from tensorflow.keras.callbacks import EarlyStopping

# Instantiate an early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=5)

# Train your model with the callback
model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test), callbacks = [early_stopping])

- Patience is the number of epochs to wait for the model to improve before stopping it's training.
* It is good to avoid low values, that way the model has a chace to improve at a later epoch.

Model checkpoint
- It allows us to save our model as it trains.
* We specify the model filename with a name and the .hdf5 extension
- You can decide what to monitor to determine which model is best with the monitor parameter.
* Default is Validation loss
- Setting the save_best_only parameter to True guarantees that the latest best model according to the quantity monitored will not be overwritten

# Import model checkpoint from keras callbacks
from keras.callbacks import ModelCheckpoint

# Instantiate an early stopping callback
model_save = ModelCheckpoint('best_model.hdf5', save_best_only=True)

# Train your model with the callback
model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test), callbacks = [model_save])
'''

# Train your model and save its history
h_callback = model.fit(X_train, y_train, epochs=25,
                       validation_data=(X_test, y_test))

# Plot train vs test loss during training
plot_loss(h_callback.history['loss'], h_callback.history['val_loss'])

# Plot train vs test accuracy during training
plot_accuracy(h_callback.history['accuracy'],
              h_callback.history['val_accuracy'])


# Import the early stopping callback

# Define a callback to monitor val_accuracy
monitor_val_acc = EarlyStopping(monitor='val_accuracy', patience=5)

# Train your model using the early stopping callback
model.fit(X_train, y_train, epochs=1000, validation_data=(
    X_test, y_test), callbacks=[monitor_val_acc])


# Import the EarlyStopping and ModelCheckpoint callbacks

# Early stop on validation accuracy
monitor_val_acc = EarlyStopping(monitor='val_accuracy', patience=3)

# Save the best model as best_banknote_model.hdf5
model_checkpoint = ModelCheckpoint(
    'best_banknote_model.hdf5', save_best_only=True)

# Fit your model for a stupid amount of epochs
h_callback = model.fit(X_train, y_train, epochs=1000000000000, callbacks=[
                       monitor_val_acc, model_checkpoint], validation_data=(X_test, y_test))


''' Improving Your Model Performance '''

'''
Learning curves
- They provide a lot of information about the model
- Loss tends to decrease as epochs go by
* It's expected since our model is essentially learning to minimize the loss function  

- Accuracy tends to increase as epochs go by
* It's expected since our model makes fewer mistakes as it learns.

- Overfitting is when our model starts learning particularities of our training data which don't generalize well on unseen data
* The early stopping callback is useful to stop our model before it starts overfitting.

- There are many reasons that can lead to unstable learning curves; the chosen 
* optimizer
* learning rate
* batch-size
* network architecture
* weight initialization etc

- Neural networks are well known for surpassing traditional ML techniques as we increase the size of our datasets.
* We can check whether collecting more data would increase a model's generalization and accuracy.

# Store initial model weights
init_weights = model.get_weights()

# Lists for storing accuracies
train_accs = []
tests_accs = []

for train_size in train_sizes:
    # Split a fraction according to train_size
    X_train_frac, _, y_train_frac, _ = train_test_split(X_train, y_train, train_size=train_size)

    # Set model initial weights
    model.set_erights(initial_weights)

    # Fit model on the training set fraction
    model.fit(X_train_frac, y_train_frac, epochs=100, verbose=0, callbacks=[EarlyStopping(monitor='loss', patience=1)])

    # Get the accuracy for this training set fraction
    train_acc = model.evaluate(X_train_frac, y_train_frac, verbose=0)[1]
    train_accs.append(train_acc)

    # Get the accuracy on the whole test set
    test_acc = model.evaluate(X_test, y_test, verbose=0)[1]
    tests_accs.append(test_acc)

    print('Done with size: ', train_size)
'''

# Instantiate a Sequential model
model = Sequential()

# Input and hidden layer with input_shape, 16 neurons, and relu
model.add(Dense(16, input_shape=(8*8, ), activation='relu'))

# Output layer with 10 neurons (one per digit) and softmax
model.add(Dense(10, activation='softmax'))

# Compile your model
model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])

# Test if your model is well assembled by predicting before training
print(model.predict(X_train))


# Train your model for 60 epochs, using X_test and y_test as validation data
h_callback = model.fit(X_train, y_train, epochs=60,
                       validation_data=(X_test, y_test), verbose=0)

# Extract from the h_callback object loss and val_loss to plot the learning curve
plot_loss(h_callback.history['loss'], h_callback.history['val_loss'])


for size in training_sizes:
    # Get a fraction of training data (we only care about the training data)
    X_train_frac, y_train_frac = X_train[:size], y_train[:size]

    # Reset the model to the initial weights and train it on the new training data fraction
    model.set_weights(initial_weights)
    model.fit(X_train_frac, y_train_frac, epochs=50, callbacks=[early_stop])

    # Evaluate and store both: the training data fraction and the complete test set results
    train_accs.append(model.evaluate(X_train_frac, y_train_frac)[1])
    test_accs.append(model.evaluate(X_test, y_test)[1])

# Plot train vs test accuracies
plot_results(train_accs, test_accs)


'''
Activation functions
- It impact learning time
* making the model converge faster or slower and achieving lower or higher accuracy
- It allows to learn more complex function

- 4 very well known activation functions are:
* The sigmoid, whih varies between 0 and 1 for all possible X input values.
* The tanh or Hyperbolic tangent, which is similar to the sigmoid in shape but varies between -1 and 1.
* The ReLu (Rectified Linear Unit) which varies between 0 and infinity 
* The leaky ReLu, which we can look as a smoothed version of ReLu that doesn't sit at 0, allowing negative values for negative inputs.

Effects of activation functions
- Changing the activation functionused in the hidden layer of the model we built for binary classification results in different classification boundaries

Which activation function to use?
- Theres no magic formula
- Based on their different properties
- Depends on the problem
- Depends on the goal to achieve in a given layer
- ReLu are a good first choice
* They train fast ad tend to generalize well to most problems
- Sigmoids are not recommended for deep models (avoid them)
- Tune with experimentation

Comparing activation functions
# set a random seed
np.random.seed(1)

# Return a new model with the given activation
def get_model(act_function):
    model = Sequential()
    model.add(Dense(4, input_shape=(2, ), activation=act_function))
    model.add(Dense(1, activation='sigmoid'))
    return model

# Activation functions to try out
activations = ['relu', 'sigmoid', tanh']

# Dictionary to store results
activation_results = {}
for funct in activations:
    model = get_model(act_function = funct)
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, verbose=0)
    activation_results[funct] = history

import pandas as pd

# Extract val_loss history of each activation function
val_loss_per_funct = {k:v.history['val_loss'] for k, v in activation_results.items()}

# Turn the dictionary into a pandas dataframe
val_loss_curves = pd.DataFrame(val_loss_per_funct)

# Plot the curves
val_loss_curves.plot(title='Loss per Activation function')
'''

# Activation functions to try
activations = ['relu', 'leaky_relu', 'sigmoid', 'tanh']

# Loop over the activation functions
activation_results = {}

for act in activations:
    # Get a new model with the current activation
    model = get_model(act)

    # Fit the model and store the history results
    h_callback = model.fit(X_train, y_train, validation_data=(
        X_test, y_test), epochs=20, verbose=0)
    activation_results[act] = h_callback


# Create a dataframe from val_loss_per_function
val_loss = pd.DataFrame(val_loss_per_function)

# Call plot on the dataframe
val_loss.plot()
plt.show()

# Create a dataframe from val_acc_per_function
val_acc = pd.DataFrame(val_acc_per_function)

# Call plot on the dataframe
val_acc.plot()
plt.show()


'''
Batch size and batch normalization
- A mini-batch is a subset of data samples 

Mini-batches
Advantages
- Networks train faster (more weight updates in same amount of time)
- Less RAM memory required, can train on huge datasets
- Noise can help networks reach a lower error, escaping local minima

Disadvantages
- More iterations need to be run
- Need to be adjusted, we need to find a good batch size

BAtch size in keras
- It uses a default batch size of 32
- Increasing powers of 2 tend to be used.
* as a rule of thumb, you tend to make your batch size bigger, the bigger the dataset.

# Fitting an already built and compiled model
model.fit(X_train, y_train, epochs=100, batch_size=128)

Batch normalization
- Normalization is a common pre-processing step in machine learning algorithms, especially when features have different scales.
- One way to normalize data is to subtract its mean value and divide by the standard deviation.
* i.e ( data -  mean ) / standard deviation
- Model inputs tend to be normalized to avoid problems with activation functions and gradients
- Normalizing neural networks inputs improve our model
* But deeper layers are trained based on previous layer outputs and since weights get updated via gradient descent, consecutive layers no longer benefit from normalization and they need to adapt to previous layers weight changes, finding more trouble to learn their own weights.

- Batch normalization makes sure that, independently of the changes, the inputs to the next layers are normalized.

Batch normalization advantages
- It improves gradient flow
- It allows higher learning rates
- It reduces dependence on weight initializations
- It acts as an unintended form of regularization
- It limits internal covariate shift i.e  a layer's dependence on the previous layer outputs when learning its weights.

Batch normalization in Keras
# Import BatchNormalization from keras layers
from tensorflow.keras.layers import BatchNormalization

# Instatiate a Sequential model
model = Sequential()

# Add an input layer
model.add(Dense(3, input_shape=(2, ), activation='relu'))

# Add batch normalization for the outputs of the layer above
model.add(BatchNormalization())

# Add an output layer
model.add(Dense(1, activation='sigmoid'))
'''

# Get a fresh new model with get_model
model = get_model()

# Train your model for 5 epochs with a batch size of 1
model.fit(X_train, y_train, epochs=5, batch_size=1)
print("\n The accuracy when using a batch of size 1 is: ",
      model.evaluate(X_test, y_test)[1])

model = get_model()

# Fit your model for 5 epochs with a batch of size the training set
model.fit(X_train, y_train, epochs=5, batch_size=X_train.shape[0])
print("\n The accuracy when using the whole training set as batch-size was: ",
      model.evaluate(X_test, y_test)[1])


# Import batch normalization from keras layers

# Build your deep network
batchnorm_model = Sequential()
batchnorm_model.add(Dense(50, input_shape=(
    64,), activation='relu', kernel_initializer='normal'))
batchnorm_model.add(BatchNormalization())
batchnorm_model.add(Dense(50, activation='relu', kernel_initializer='normal'))
batchnorm_model.add(BatchNormalization())
batchnorm_model.add(Dense(50, activation='relu', kernel_initializer='normal'))
batchnorm_model.add(BatchNormalization())
batchnorm_model.add(Dense(10, activation='softmax',
                    kernel_initializer='normal'))

# Compile your model with sgd
batchnorm_model.compile(
    optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])


# Train your standard model, storing its history callback
h1_callback = standard_model.fit(
    X_train, y_train, validation_data=(X_test, y_test), epochs=10, verbose=0)

# Train the batch normalized model you recently built, store its history callback
h2_callback = batchnorm_model.fit(
    X_train, y_train, validation_data=(X_test, y_test), epochs=10, verbose=0)

# Call compare_histories_acc passing in both model histories
compare_histories_acc(h1_callback, h2_callback)


'''
Hyperparameter tuning
Neural network hyperparameters
- Number of layers
- Number of neurons per layer
- Layer order
- Layer activations
- Batch sizes
- Learning rates
- Optimizers
- . . . 

Sklearn recap
# Import RandomizedSearchCV
from sklearn.model_selection import RandomizedSearchCV

# Instantiate your classifier
tree = DecisionTreeClassifier()

# Define a series of parameters to look over
params = {'max_depth': [3, None], 'max_features': range(1, 4), 'min_samples_leaf': range(1, 4)}

# Perform random search with cross validation
tree_cv = RandomizedSearchCV(tree, params, cv=5)
tree_cv.fit(X, y)

# Print the best parameters
print(tree_cv.best_params_) -> in

{'min_samples_leaf': 1, 'max_features': 3, max_depth': 3} -> out

Turn a Keras model into a Sklearn estimator
# Function that creates our keras model
def create_model(optimizer='adam', activation='relu'):
    model = Sequential()
    model.add(Dense(16, input_shape=(2, ), activation=activation))
    model.add(Dense(1, activation='sigmoid))
    model.compile(optimizer=optimizer, loss='binary_crossentropy')
    return model

# Import sklearn wrapper from keras
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

# Create a model as a sklearn estimator
model = KerasClassifier(build_fn = create_model, epochs = 6, batch_size = 16)

Cross-validation
# Import cross_val_score
from sklearn.model_selection import cross_val_score

# Check how your keras model performs with 5 fold crossvalidation
kfold = cross_val_score(model, X, y, cv=5)

# Print the mean accuracy per fold
kfold.mean() -> in

0.913333 -> out

# Print the standard deviation per fold
kfold.std() -> in

0.110754 -> out

Tips for neural networks hyperparameter tuning
- Random search is preferred over grid search
- Don't use many epochs
- Use a smaller sample of the dataset
- Play with batch sizes, activations, optimizers and learning rates

Random search on Keras models
# Define a series of parameters
params = dict(optimizer=['sgd', 'adam'], epochs=3, batch_size=[5, 10, 20], activation=['relu', 'tanh'])

# Create a random search cv object and fit it to the data
random_search = RandomizedSearchCV(model, params_dist=params, cv=3)
random_search_results = random_search.fit(X, y)

# Print results
print('Best: %f using %s'.format(random_search_results.best_score_, random_search_results.best_params_)) -> in

Best: 0.94 using {'optimizer': 'adam', 'epochs': 3, 'batch_size': 10, 'activation': 'relu'} -> out

Tuning other hyperparameters
- nl parameter determines the number of hidden layers
- nn parameter determines the number of neurons in these layers

def create_model(nl=1, nn=256):
    model = Sequential()
    model.add(Dense(16, input_shape=(2, ), activation='relu'))

    # Add as many hidden layers as specified in nl
    for i in range(nl):
        # Layers have nn neurons
        model.add(Dense(nn, activation='relu'))
    # End defining and compiling the model . . .

# Define parameters, named just like in create_model()
params = dict(nl=[1. 2. 9], nn=[128, 256, 1000])

# Repeat the random search . . .

# Print results . . . 
Best: 0.87 using {'nl': 2, 'nn': 128} -> out
'''

# Creates a model given an activation and learning rate


def create_model(learning_rate, activation):
    # Create an Adam optimizer with the given learning rate
    opt = Adam(lr=learning_rate)

    # Create your binary classification model
    model = Sequential()
    model.add(Dense(128, input_shape=(30,), activation=activation))
    model.add(Dense(256, activation=activation))
    model.add(Dense(1, activation='sigmoid'))

    # Compile your model with your optimizer, loss, and metrics
    model.compile(optimizer=opt, loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


# Import KerasClassifier from tensorflow.keras scikit learn wrappers

# Create a KerasClassifier
model = KerasClassifier(build_fn=create_model)

# Define the parameters to try out
params = {'activation': ['relu', 'tanh'], 'batch_size': [
    32, 128, 256], 'epochs': [50, 100, 200], 'learning_rate': [0.1, 0.01, 0.001]}

# Create a randomize search cv object passing in the parameters to try
random_search = RandomizedSearchCV(
    model, param_distributions=params, cv=KFold(3))

# Running random_search.fit(X,y) would start the search,but it takes too long!
show_results()


# Import KerasClassifier from tensorflow.keras wrappers

# Create a KerasClassifier
model = KerasClassifier(build_fn=create_model(
    learning_rate=0.001, activation='relu'), epochs=50, batch_size=128, verbose=0)

# Calculate the accuracy score for each fold
kfolds = cross_val_score(model, X, y, cv=3)

# Print the mean accuracy
print('The mean accuracy was:', kfolds.mean())

# Print the accuracy standard deviation
print('With a standard deviation of:', kfolds.std())


''' Advanced Model Architectures '''

'''
Tensors, layers, and autoencoders
Accessing Keras layers
# Accessing the first layer of a keras model
first_layer = model.layers[0]

# Printing the layer, and its input, output and weights
print(first_layer.input)
print(first_layer.output)
print(first_layer.weights) -> in

<tf.Tensor 'dense_1_input:0' shape=(?, 3) dtype=float32>

<tf.Tensor 'dense_1/Relu:0' shape=(?, 2) dtype=float32>

[   <tf.Variable 'dense_1/kernel:0' shape=(3, 2) dtype=float32_ref>,
    <tf.Variable 'dense_1/bias:0' shape=(2, ) dtype=float32_ref>,
] -> out

What are tensors?
- They are the main data structures used in deep learning, inputs, outputs and transformations in neural networks are all represented using tensors and tensor multiplication.
- It is a multidimensional array of numbers.
- A 2-D tensor is a matrix
- A 3-D tensor is an array of matrices

# Defining a rank 2 tensor (2 dimensions)
T2 = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

# Defining a rank 3 tensor (3 dimensions)
T3 = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15], [16, 17, 18], [19, 20, 21], [22, 23, 24], [25, 26, 27]]

# Import Keras backend
import tensorflow.keras.backend as K

# Get the input and output tensors of a model layer
inp = model.layers[0].input
out = model.layers[0].output

# Function that maps layer inouts to outputs
inp_to_out = K.function([inp], [out])

# We pass an input and get the output we'd get in the first layer
print(inp_to_out([X_train]) -> in

# Outputs of the first layer per sample in X_train
[array([[0.7, 0], . . . , [0.1, 0.3]])] -> out

Autoencoders
- They are models that aim at producing the same inputs as outputs.
* We are effectively making our network learn to compress its inputs into a mall set of neurons.

Autoencoder use cases
- Dimensionality reduction: 
* Smaller dimensional space representation of our inputs
- De-noising data:
* If trained with clean data, irrelevant noise will be filtered out during reconstruction
- Anomaly detection:
* A poor reconstruction will result when the model is fed with unseen inputs.
- . . . 

Building a simple autoencoder
# Instantiate a sequential model
autoencoder = Sequential()

# Add a hidden layer of 4 neurons and an input layer of 100
autoencoder.add(Dense(4, input_shape=(100, ), activation='relu')

# Add an output layer of 100 neurons
autoencoder.add(Dense(100, activation='sigmoid')

# Compile your model with the appropriate loss
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

Breaking it into an encoder
# Building a separate model to encode inputs
encoder = Sequential()
encoder.add(autoencoder.layers[0])

# Predicting returns the four hidden layer neuron outputs
encoder.predict(X_test) -> in

# Four numbers for each observation in X_test
array([10.0234375, 5.833543, 18.90444, 9.20348], . . . ) -> out
'''

# Import tensorflow.keras backend

# Input tensor from the 1st layer of the model
inp = model.layers[0].input

# Output tensor from the 1st layer of the model
out = model.layers[0].output

# Define a function from inputs to outputs
inp_to_out = K.function([inp], [out])

# Print the results of passing X_test through the 1st layer
print(inp_to_out([X_test]))


for i in range(0, 21):
    # Train model for 1 epoch
    h = model.fit(X_train, y_train, batch_size=16, epochs=1, verbose=0)
    if i % 4 == 0:

        # Get the output of the first layer
        layer_output = inp_to_out([X_test])[0]

        # Evaluate model accuracy for this epoch
        test_accuracy = model.evaluate(X_test, y_test)[1]

        # Plot 1st vs 2nd neuron output
        plot()


# Start with a sequential model
autoencoder = Sequential()

# Add a dense layer with input the original image pixels and neurons the encoded representation
autoencoder.add(Dense(32, input_shape=(784, ), activation="relu"))

# Add an output layer with as many neurons as the orginal image pixels
autoencoder.add(Dense(784, activation="sigmoid"))

# Compile your model with adadelta
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

# Summarize your model structure
autoencoder.summary()


# Build your encoder by using the first layer of your autoencoder
encoder = Sequential()
encoder.add(autoencoder.layers[0])

# Encode the noisy images and show the encodings for your favorite number [0-9]
encodings = encoder.predict(X_test_noise)
show_encodings(encodings, number=1)

# Predict on the noisy images with your autoencoder
decoded_imgs = autoencoder.predict(X_test_noise)

# Plot noisy vs decoded images
compare_plot(X_test_noise, decoded_imgs)


'''
Intro to CNNs (Convolutional Neural Networks)
- A convolutional model uses convolutional layers.
- A convolution is a simple mathematical operation that preserves spatial relationships.
- When applied to images, it can detect relevant areas of interest like edges, corners, vertical lines etc.
- It consists of applying a filter, also known as kernel, of a given size.
- The secret sauce of CNNs resides in letting the network itself find the best filter values and to combine them to achieve a given task.
- Convolutional layers perform feature learning, we then flatten the outputs into a unidimensional vector and pass it to fully connected layers that carry out classification.

Images
- They are 3D tensors, i.ehave width, height and depth
- The depth is given by the color channels
- input_shape = (WIDTH, HEIGHT, CHANNELS)
* input_shape = (28, 28, 3)
* Color images have 3 channels RGB (Red Green Blue)

# Import Conv2D layer and Flatten from tensorflow keras layers
from tensorflow.keras.layers import Dense, Conv2D, Flatten

# Instantiate your model as usual
model = Sequential()

# Add a convolutional layer with 32 filters of size 3 x 3
model.add(Conv2D(filters=32, kernel_size=3, input_shape=(28, 28, 1), activation='relu'))

# Add another convolutional layer
model.add(Conv2D(8, kernel_size=3, activation='relu'))

# Flatten the output of the previous layer
model.add(Flatten())

# End this multiclass model with 3 outputs and softmax
model.add(Dense(3, activation='softmax'))

Pre-processing images for ResNet50
# Import image from keras preprocessing
from tensorflow.keras.preprocessing import image

# Import preprocess_input from tensorflow keras applications resnet50
from tensorflow.keras.applications.resnet50 import preprocess_input

# Load the image with the right target size for the model
img = image.load_img(img_path, target_size=(224, 224))

# Turn it into an array
img = image.img_to_array(img)

# Expand the dimensions so that it's understood by our network:
# img.shape turns from (224, 224, 3) into (1, 224, 224, 3)
img = np.expand_dims(img, axis=0)

# Pre-process the img in the same way training images were
img = preprocess_input(img)

# Import ResNet50 and decode_predictions from tensorflow.keras.applications.resnet50
from tensorflow.keras.applications.resnet50 import ResNet50, decode_predictions

# Instantiate a ResNet50 model with imagenet weights
model = ResNet(weights = 'imagenet')

# Predict with ResNet50 on our img
preds = model.predict(img)

# Decode predictions and print it
print('Predicted:', decode_predictions(preds, top=1)[0]) -> in

Predicted: [('n07697313', 'cheeseburger', 0.9868016)] -> out
'''

# Import the Conv2D and Flatten layers and instantiate model
model = Sequential()

# Add a convolutional layer of 32 filters of size 3x3
model.add(Conv2D(32, kernel_size=3, input_shape=(28, 28, 1), activation='relu'))

# Add a convolutional layer of 16 filters of size 3x3
model.add(Conv2D(16, kernel_size=3, activation='relu'))

# Flatten the previous layer output
model.add(Flatten())

# Add as many outputs as classes with softmax activation
model.add(Dense(10, activation='softmax'))


# Obtain a reference to the outputs of the first layer
first_layer_output = model.layers[0].output

# Build a model using the model's input and the first layer output
first_layer_model = Model(
    inputs=model.layers[0].input, outputs=first_layer_output)

# Use this model to predict on X_test
activations = first_layer_model.predict(X_test)

# Plot the activations of first digit of X_test for the 15th filter
axs[0].matshow(activations[0, :, :, 14], cmap='viridis')

# Do the same but for the 18th filter now
axs[1].matshow(activations[0, :, :, 17], cmap='viridis')
plt.show()


# Import image and preprocess_input

# Load the image with the right target size for your model
img = image.load_img(img_path, target_size=(224, 224))

# Turn it into an array
img_array = image.img_to_array(img)

# Expand the dimensions of the image, this is so that it fits the expected model input format
img_expanded = np.expand_dims(img_array, axis=0)

# Pre-process the img in the same way original images were
img_ready = preprocess_input(img_expanded)


# Instantiate a ResNet50 model with 'imagenet' weights
model = ResNet50(weights='imagenet')

# Predict with ResNet50 on your already processed img
preds = model.predict(img_ready)

# Decode the first 3 predictions
print('Predicted:', decode_predictions(preds, top=3)[0])


'''
Intro to LSTMs (Long Short Term Memory networks)
- They are a type of recurrent neural network (RNN)
* A simple RNN is a neural network that can use past predictions in order to infer new ones.
* It allows us to solve problems where there is a dependence on past inputs.

When to use LSTMs?
- It is used for image captioning
- Speech to text
- Text translation
- Document summarization
- Text generation
- Musical composition
- . . .

Example:
How to use LSTMs with text data to predict the next word in a sentence!

* Neural networks can oly deal with numbers, not text.
* We need to transform each unique word into a number. 
- Then these numbers can be used as inputs to an embedding layer
* Embedding layers learn to represent words as vectors of a predetermined size
- These vectors encode meaning and are used by subsequent layers
e.g
this is a sentence -> 42 11 23 1

text = 'Hi this is a small sentence'

# We choose a sequence length
seq_len = 3

# Split text into a list of words
words = text.split() -> in

['Hi', 'this', 'is', 'a', 'small', 'sentence'] -> out

Makes lines
lines = []
for i in range(seq_len, len(words) + 1):
    line = ' '.join(words[i - seq_len: i])
    lines.append(line)

# Import Tokenizer from keras preprocessing text
from tensorflow.keras.preprocessing.text import Tokenizer

# Instantiate Tokenizer
tokenizer = Tokenizer()

# Fit it on the previous lines
tokenizer.fit_on_texts(lines)

# Turn the lines into numeric sequences
sequences = tokenizer.texts_to_sequences(lines) -> in

array([[5, 3, 1], [3, 1, 2], [1, 2, 4], [2, 4, 6]]) -> out

print(tokenizer.index_word) -> in

{1: 'is', 2: 'a', 3: 'this', 4: 'small', 5: 'hi', 6: 'sentence'} -> out

# Import Dense, LSTM and Embedding layers
from tensorflow.keras.layers import Dense, LSTM, Embedding

model = Sequential()

# Vocabulary size
vocab_size = len(tokenizer.index_word) + 1

# Starting with an embedding layer
model.add(Embedding(input_dim=vocab_size, output_dim=8, input_length=2))

# Adding an LSTM layer
model.add(LSTM(8))

# Adding a Dense hidden layer
model.add(Dense(8, activation='relu'))

# Adding an output layer with softmax
model.add(Dense(vocab_size, activation='softmax'))
'''

# Split text into an array of words
words = text.split()

# Make sentences of 4 words each, moving one word at a time
sentences = []
for i in range(4, len(words)):
    sentences.append(' '.join(words[i-4:i]))

# Instantiate a Tokenizer, then fit it on the sentences
tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences)

# Turn sentences into a sequence of numbers
sequences = tokenizer.texts_to_sequences(sentences)
print("Sentences: \n {} \n Sequences: \n {}".format(
    sentences[:5], sequences[:5]))


# Import the Embedding, LSTM and Dense layer

model = Sequential()

# Add an Embedding layer with the right parameters
model.add(Embedding(input_dim=vocab_size, input_length=3, output_dim=8, ))

# Add a 32 unit LSTM layer
model.add(LSTM(32))

# Add a hidden Dense layer of 32 units and an output layer of vocab_size with softmax
model.add(Dense(32, activation='relu'))
model.add(Dense(vocab_size, activation='softmax'))
model.summary()


def predict_text(test_text, model=model):
    if len(test_text.split()) != 3:
        print('Text input should be 3 words!')
        return False

    # Turn the test_text into a sequence of numbers
    test_seq = tokenizer.texts_to_sequences([test_text])
    test_seq = np.array(test_seq)

    # Use the model passed as a parameter to predict the next word
    pred = model.predict(test_seq).argmax(axis=1)[0]

    # Return the word that maps to the prediction
    return tokenizer.index_word[pred]
