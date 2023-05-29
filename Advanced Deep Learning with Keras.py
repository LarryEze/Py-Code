''' The Keras Functional API '''

'''
Keras input and dense layers
Course Datasets: College basketball data, 1989-2017
- Dataset 1: Regular season
* Team ID 1
* Team ID 2
* Home vs Away
* Score Difference ( Team 1 -  Team 2)
* Team 1 score
* Team 2 score
* Won vs Lost

- Datset 2: Tournament games
* Same as Dataset 1
* Also has difference in Seed

Import pandas as pd
games_season = pd.read_csv('datasets/games_season.csv')
games_season.head()     -> in

out[1]:
    season  team_1  team_2  home    score_diff  score_1     score_2     won
0   1985    3745    6664    0       17          81          64          1
1   1985    126     7493    1       7           77          70          1
2   1985    288     3593    1       7           63          56          1
3   1985    1846    9881    1       16          70          54          1
4   1985    2675    10298   1       12          86          74          1   -> out

games_tourney = pd.read_csv('datasets/games_tourney.csv')
games_tourney.head() -> in

out[2]:
    season  team_1  team_2  home    seed_diff   score_diff  score_1     score_2     won
0   1985    288     73      0       -3          -9          41          50          0
1   1985    5929    73      0       4           6           61          55          1
2   1985    9884    73      0       5           -4          59          63          0
3   1985    73      288     0       3           9           50          41          1
4   1985    3920    410     0       1           -9          54          63          0   -> out

Inputs and outputs
Two fundamental parts of a keras model:
- Input layer
- Output layer

Inputs
from tensorflow.keras.layers import Input
input_tensor = Input(shape=(1, ))
print(input_tensor) -> in

KerasTensor(type_spec=TensorSpec(shape=(None, 1), dtype=tf.float32, name='input_1'), name='input_1', description="created by layer 'input_1'") -> out

Outputs
from tensorflow.keras.layers import Dense
output_layer = Dense(1)
print(output_layer) -> in

<keras.layers.core.Dense object at 0x7fa91fee2760> -> out

- Layers are used to construct a deep learning model
- Tensors are used to define the data flow through the model

Connecting inputs to outputs
from tensorflow.keras.layers import Input, Dense
input_tensor = Input(shape=(1, ))
output_layer = Dense(1)
output_tensor = output_layer(input_tensor)
print(output_tensor) -> in

KerasTensor(type_spec=TensorSpec(shape=(None, 1), dtype=tf.float32, name=None), name='dense_1/BiasAdd:0', description="created by layer 'dense_1'") -> out
'''

# Import Input from tensorflow.keras.layers

# Create an input layer of shape 1
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Subtract
from tensorflow.keras.layers import Input, Embedding, Flatten
from numpy import unique
import matplotlib.pyplot as plt
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Input
input_tensor = Input(shape=(1,))


# Load layers

# Input layer
input_tensor = Input(shape=(1,))

# Dense layer
output_layer = Dense(1)

# Connect the dense layer to the input_tensor
output_tensor = output_layer(input_tensor)


# Load layers

# Input layer
input_tensor = Input(shape=(1,))

# Create a dense layer and connect the dense layer to the input_tensor in one step
# Note that we did this in 2 steps in the previous exercise, but are doing it in one step now
output_tensor = Dense(1)(input_tensor)


'''
Build and compile a model
from tensorflow.keras.layers import Input , Dense

input_tensor = Input(shape=(1, ))
output_tensor = Dense(1)(input_tensor)

Keras models
from tensorflow.keras.models import Model

model = Model(input_tensor, output_tensor)

Compile a model
- Must compile the model before fitting it to data
- This step finalizes the model and gets it completely ready for use in fitting and predicting

model.compile(optimizer='adam', loss=mae')

Summarize the model
model.summary()

Plot model using keras
input_tensor = Input(shape=(1, ))
output_layer = Dense(1, name='Predicted-Score-Diff')
output_tensor = output_layer(input_tensor)
model = Model(input_tensor, output_tensor)
plot_model(model, to_file='model.png')

from matplotlib import pyplot as plt
img = plt.imread('model.png')
plt.imshow(img)
plt.show()
'''

# Input/dense/output layers
input_tensor = Input(shape=(1,))
output_tensor = Dense(1)(input_tensor)

# Build the model
model = Model(input_tensor, output_tensor)


# Compile the model
model.compile(optimizer='adam', loss='mean_absolute_error')


# Import the plotting function

# Summarize the model
model.summary()

# Plot the model
plot_model(model, to_file='model.png')

# Display the image
data = plt.imread('model.png')
plt.imshow(data)
plt.show()


'''
Fit and evaluate a model
Basketball Data
Goal: Predict tournament outcomes
Data Available: team ratings from the tournament organizers

import pandas as pd
games_tourney = pd.read_csv('datasets/games_tourney.csv')
games_tourney.head() -> in

out[1]:
    season  team_1  team_2  home    seed_diff   score_diff  score_1     score_2     won
0   1985    288     73      0       -3          -9          41          50          0
1   1985    5929    73      0       4           6           61          55          1
2   1985    9884    73      0       5           -4          59          63          0
3   1985    73      288     0       3           9           50          41          1
4   1985    3920    410     0       1           -9          54          63          0   -> out

Input:
- Seed difference - one number: -15 to +15
- Seed range from 1 - 16
- Highest difference is 16 - 1 = +15
- Lowest differenec is 1 - 16 = -15

- Seed differnce: +15
* Team 1: 16
* Team 2: 1
- Seed differnce: -15
* Team 1: 1
* Team 2: 16

- Score differnce: -9
* Team 1: 41
* Team 2: 50
- Score differnce: 6
* Team 1: 61
* Team 2: 55

Output:
- Score difference - one number: -50 to + 50

Build the model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

input_tensor = Input(shape=(1, ))
output_tensor = Dense(1)(input_tensor)
model = Model(input_tensor, output_tensor)
model.compile(optimizer='adam', loss='mae')

Fit the model
from pandas import read_csv

games = read_csv('datasets/games_tourney.csv')
model.fit(games['seed_diff'], games['score_diff'], batch_size=64, validation_split= 0.2, verbose=True)

- Batch size sets how many rows of data are used for each step of stochastic gradient descent
- Validation split tells Keras to use a holdout set, and return metrics on accuracy using that data.
* It can be useful for validating that the models will perform well on new data.
- When verbose is set to True, Keras prints a log during training.
* It can be useful for debugging

Evaluate the model
- It is done using a new dataset, to make sure the model is predicting as expected

model.evaluate(games['seed_diff'], games['score_diff']) -> in

1000/1000 [====================] -0s 26us/step
out[1]: 9.145421981811523 -> out
'''

# Now fit the model
model.fit(games_tourney_train['seed_diff'], games_tourney_train['score_diff'],
          epochs=1, batch_size=128, validation_split=0.1, verbose=True)


# Load the X variable from the test data
X_test = games_tourney_test['seed_diff']

# Load the y variable from the test data
y_test = games_tourney_test['score_diff']

# Evaluate the model on the test data
print(model.evaluate(X_test, y_test, verbose=False))


''' Two Input Networks Using Categorical Embeddings, Shared Layers, and Merge Layers '''

'''
Category embeddings
- They are an advanced type of layer, only available in deep learning libraries.
- They are extremely useful for dealing with high cardinality categorical data
- they are also useful for dealing with text data, such as in Word2vec models

- Input: integers
- Output: floats
- NB*: Increased dimensionality: output layer flattens back to 2D

Input Layer (integer) -> Embedding Layer (lookup table) -> Output Layer (float)

Inputs
input_tensor = Input(shape=(1, ))

Embedding Layer
from tensorflow.keras.layers import Embedding

input_tensor = Input(shape=(1, ))
n_teams = 10887
embed_layer = Embedding(input_dim = n_teams, input_length = 1, output_dim = 1, name = 'Team-Strength-Lookup')
embed_tensor = embed_layer(input_tensor)

Flattening
from tensorflow.keras.layers import Flatten

flatten_tensor = Flatten()(embed_tensor)

- It is also the output layer for the embedding process.
- It is an advanced layer for deep learning models, and can be used to transform data from multiple dimensions back down to 2-Dimensions.
- It is useful for dealing with:
* time series data
* text data
* images

Put it all together
input_tensor = Input(shape=(1, ))
n_teams = 10887
embed_layer = Embedding(input_dim = n_teams, input_length = 1, output_dim = 1, name = 'Team-Strength-Lookup')
embed_tensor = embed_layer(input_tensor)
flatten_tensor = Flatten()(embed_tensor)
model = Model(input_tensor, flatten_tensor)
'''

# Imports

# Count the unique number of teams
n_teams = unique(games_season['team_1']).shape[0]

# Create an embedding layer
team_lookup = Embedding(input_dim=n_teams, output_dim=1,
                        input_length=1, name='Team-Strength')


# Imports

# Create an input layer for the team ID
teamid_in = Input(shape=(1,))

# Lookup the input in the team strength embedding layer
strength_lookup = team_lookup(teamid_in)

# Flatten the output
strength_lookup_flat = Flatten()(strength_lookup)

# Combine the operations into a single, re-usable model
team_strength_model = Model(
    teamid_in, strength_lookup_flat, name='Team-Strength-Model')


'''
Shared layers
- They are an advanced deep learning concept, and are only possible with the Keras functional API.
- It allows you to define an operation and then apply the exact same operation (with the exact same weights) on different inputs.

input_tensor_1 = Input((1, ))
input_tensor_2 = Input((1, ))

shared_layer = Dense(1)
output_tensor_1 = shared_layer(input_tensor_1)
output_tensor_2 = shared_layer(input_tensor_2)

Sharing multiple layers as a model
input_tensor = Input(shape=(1, ))
n_teams = 10887
embed_layer = Embedding(input_dim = n_teams, input_length = 1, output_dim = 1, name = 'Team-Strength-Lookup')
embed_tensor = embed_layer(input_tensor)
flatten_tensor = Flatten()(embed_tensor)
model = Model(input_tensor, flatten_tensor)

input_tensor_1 = Input((1, ))
input_tensor_2 = Input((1, ))
output_tensor_1 = model(input_tensor_1)
output_tensor_2 = model(input_tensor_2)
'''

# Load the input layer from tensorflow.keras.layers

# Input layer for team 1
team_in_1 = Input((1,), name='Team-1-In')

# Separate input layer for team 2
team_in_2 = Input((1,), name='Team-2-In')


# Lookup team 1 in the team strength model
team_1_strength = team_strength_model(team_in_1)

# Lookup team 2 in the team strength model
team_2_strength = team_strength_model(team_in_2)


'''
Merge layers
- It allows to define advanced, non-sequential network topologies.
* It can give a lot of flexibility to creatively design networks to solve specific problems
- Types of merge layers include:
* Add
* Subtract
* Multiply
* Concatenate

from tensorflow.keras.layers import Input, Add

in_tensor_1 = Input((1, ))
in_tensor_2 = Input((1, ))
out_tensor = Add()([in_tensor_1, in_tensor_2])

in_tensor_3 = Input((1, ))
out_tensor = Add()([in_tensor_1, in_tensor_2, in_tensor_3])

- NB*: All the inputs are required to have the same shape, so they can be combined element-wise except in concatenate.

Create the model
from tensorflow.keras.models import Model

model = Model([in_tensor_1, in_tensor_2], out_tensor)

Compile the model
model.compile(optimizer='adam', loss='mean_absolute_error')
'''

# Import the Subtract layer from tensorflow.keras

# Create a subtract layer using the inputs from the previous exercise
score_diff = Subtract()([team_1_strength, team_2_strength])


# Imports

# Subtraction layer from previous exercise
score_diff = Subtract()([team_1_strength, team_2_strength])

# Create the model
model = Model([team_in_1, team_in_2], score_diff)

# Compile the model
model.compile(optimizer='adam', loss='mean_absolute_error')


'''
Predict from your model
Fit with multiple inputs
model.fit([data_1, data_2], target)

Predict with multiple inputs 
model.predict([np.array([[1]]), np.array([[2]])]) -> in

np.array([[3.]], dtype=float32) -> out

model.predict([np.array([[42]]), np.array([[119]])]) -> in

np.array([[161.]], dtype=float32) -> out

Evaluate with multiple inputs
model.evaluate([np.array([[-1]]), np.array([[-2]])], np.array([[-3]])) -> in

1/1 [===================] -0s 801us/step
Out[21]: 0.0 -> out
'''

# Get the team_1 column from the regular season data
input_1 = games_season['team_1']

# Get the team_2 column from the regular season data
input_2 = games_season['team_2']

# Fit the model to input 1 and 2, using score diff as a target
model.fit([input_1, input_2], games_season['score_diff'], epochs=1,
          batch_size=2048, validation_split=0.1, verbose=True)


# Get team_1 from the tournament data
input_1 = games_tourney['team_1']

# Get team_2 from the tournament data
input_2 = games_tourney['team_2']

# Evaluate the model using these inputs
print(model.evaluate([input_1, input_2],
      games_tourney['score_diff'], verbose=False))


''' Multiple Inputs: 3 Inputs (and Beyond!) '''

'''
Three-input models
Simple model with 3 inputs
from tensorflow.keras.layers import Input, Concatenate, Dense

in_tensor_1 = Input(shape=(1,))
in_tensor_2 = Input(shape=(1,))
in_tensor_3 = Input(shape=(1,))
out_tensor = Concatenate()([in_tensor_1, in_tensor_2, in_tensor_3])
output_tensor = Dense(1)(out_tensor)

from tensorflow.keras.models import Model

model = Model([in_tensor_1, in_tensor_2, in_tensor_3], out_tensor)

Shared layers with 3 inputs
shared_layer = Dense(1)
shared_tensor_1 = shared_layer(in_tensor_1)
shared_tensor_2 = shared_layer(in_tensor_2)
out_tensor = Concatenate()([shared_tensor_1, shared_tensor_2, in_tensor_3])
out_tensor = Dense(1)(out_tensor)

Fitting a 3 input model
from tensorflow.keras.models import Model

model = Model([in_tensor_1, in_tensor_2, in_tensor_3], out_tensor)
model.compile(loss='mae', optimizer='adam')
model.fit([[train['col1'], train['col2'], train['col3']], train_data['target'])
model.evaluate([[test['col1'], test['col2'], test['col3']], test['target'])
'''

# Create an Input for each team
team_in_1 = Input(shape=(1,), name='Team-1-In')
team_in_2 = Input(shape=(1,), name='Team-2-In')

# Create an input for home vs away
home_in = Input(shape=(1,), name='Home-In')

# Lookup the team inputs in the team strength model
team_1_strength = team_strength_model(team_in_1)
team_2_strength = team_strength_model(team_in_2)

# Combine the team strengths with the home input using a Concatenate layer, then add a Dense layer
out = Concatenate()([team_1_strength, team_2_strength, home_in])
out = Dense(1)(out)


# Import the model class

# Make a Model
model = Model([team_in_1, team_in_2, home_in], out)

# Compile the model
model.compile(optimizer='adam', loss='mean_absolute_error')


# Fit the model to the games_season dataset
model.fit([games_season['team_1'], games_season['team_2'], games_season['home']],
          games_season['score_diff'], epochs=1, verbose=True, validation_split=0.1, batch_size=2048)

# Evaluate the model on the games_tourney dataset
print(model.evaluate([games_tourney['team_1'], games_tourney['team_2'],
      games_tourney['home']], games_tourney['score_diff'], verbose=False))


'''
Summarizing and plotting models
Understanding a model summary
- It shows all the layers in the model, as well as how many parameters each layer has.
- Keras models can have non-trainable parameters that are fixed and don not change
- As well as trainable parameters, that are learned from the data when the model is fit.
- Models with more trainable parameters are typically more flexible, and more prone to overfitting
- Models with fewer trainable parameters are typically less flexible, and less likely to overfitting
- A model's trainable parameters are usually in its Dense layers
- Embedding layers often add a very large number of trainable parameters to a model.
* They map integers to floats: each unique value of the embedding input gets a parameter for its output.

Total parameters = No of Dense layer + No of embeddings (if applicable)
'''

# Imports

# Plot the model
plot_model(model, to_file='model.png')

# Display the image
data = plt.imread('model.png')
plt.imshow(data)
plt.show()


'''
Stacking models
- It is a very advanced data science concept
- It is often employed to win popular predictive modeling competitions.

Stacking models requires 2 datasets
from pandas import read_csv

games_season = read_csv('datasets/games_season.csv')
games_season.head() -> in

    team_1  team_2  home    score_diff
0   3745    6664    0       17
1   126     7493    1       7
2   288     3593    1       7
3   1846    9881    1       16
4   2675    10298   1       12 -> out

games_tourney = read_csv('datasets/games_tourney.csv')
games_tourney.head() -> in

    team_1  team_2  home    seed_diff   score_diff
0   288     73      0       -3          -9
1   5929    73      0       4           6
2   9884    73      0       5           -4
3   73      288     0       3           9
4   3920    410     0       1           -9 -> out

Enrich the tournament data
in_data_1 = games_tourney['team_1']
in_data_2 = games_tourney['team_2']
in_data_3 = games_tourney['home']
pred = regular_season_model.predict([in_data_1, in_data_2, in_data_3])

games_tourney['pred'] = pred
games_tourney.head() -> in

    team_1  team_2  home    seed_diff   pred        score_diff
0   288     73      0       -3          0.582556    -9
1   5929    73      0       4           0.707279    6
2   9884    73      0       5           1.364844    -4
3   73      288     0       3           0.699145    9
4   3920    410     0       1           0.833066    -9 -> out

3 input model with pure numeric data
games_tourney[['home', 'seed_diff', 'pred']].head() -> in

    home    seed_diff   pred
0   0       -3          0.582556
1   0       4           0.707279
2   0       5           1.364844
3   0       3           0.699145
4   0       1           0.833066 -> out

3 input model with pure numeric data
from tensorflow.keras.layers import Input, Dense
in_tensor = Input(shape=(3, ))
out_tensor = Dense(1)(in_tensor)

from tensorflow.keras.models import Model
model = Model(in_tensor, out_tensor)
model.compile(optimizer='adam', loss='mae')
train_X = train_data[['home', 'seed_diff', 'pred']]
train_y = train_data['score_diff']
model.fit(train_X, train_y, epochs=10, validation_split=0.1)

test_x = test_data[['home', 'seed_diff', 'pred']]
test_y = test_data['score_diff']
model.evaluate(test_X, test_y) -> in

1066/1066 [===================] - Os 14us/step
9.11321775461451 -> out
'''

# Predict
games_tourney['pred'] = model.predict(
    [games_tourney['team_1'], games_tourney['team_2'], games_tourney['home']])


# Create an input layer with 3 columns
input_tensor = Input((3,))

# Pass it to a Dense layer with 1 unit
output_tensor = Dense(1)(input_tensor)

# Create a model
model = Model(input_tensor, output_tensor)

# Compile the model
model.compile(optimizer='adam', loss='mean_absolute_error')


# Fit the model
model.fit(games_tourney_train[['home', 'seed_diff', 'pred']],
          games_tourney_train['score_diff'], epochs=1, verbose=True)


# Evaluate the model on the games_tourney_test dataset
print(model.evaluate(games_tourney_test[[
      'home', 'seed_diff', 'prediction']], games_tourney_test['score_diff'], verbose=False))


''' Multiple Outputs '''

'''
Two-output models
Simple model with 2 outputs
- The only difference between the 2 output model and the 1 output model is the size of the output layer.

from tensorflow.keras.layers import Input, Concatenate, Dense

input_tensor = Input(shape=(1, ))
output_tensor = Dense(2)(input_tensor)

from tensorflow.keras.models import Model

model = Model(input_tensor, output_tensor)
model.compile(optimizer='adam', loss='mean_absolute_error')

Fitting a model with 2 outputs
games_tourney_train[['seed_diff', 'score_1', 'score_2']].head() -> in

    seed_diff   score_1     score_2
0   -3          41          50
1   4           61          55
2   5           59          63
3   3           50          41
4   1           54          63 -> out

X = games_tourney_train[['seed_diff']]
y = games_tourney_train[['score_1', 'score_2']]
model.fit(X, y, epochs=500)

Inspecting a 2 output model
model.get_weights() -> in

[   array([[ 0.60714734, -0.5988793 ]]. dtype=float32). 
    array([70.39491, 70.39306], dtype=float32)  ] -> out

Evaluationg a model with 2 outputs
X = games_tourney_test[['seed_diff']]
y = games_tourney_test[['score_1', 'score_2']]
model.evaluate(X, y) -> in

11.528035634635021 -> out
'''

# Define the input
input_tensor = Input((2,))

# Define the output
output_tensor = Dense(2)(input_tensor)

# Create a model
model = Model(input_tensor, output_tensor)

# Compile the model
model.compile(optimizer='adam', loss='mean_absolute_error')


# Fit the model
model.fit(games_tourney_train[['seed_diff', 'pred']], games_tourney_train[[
          'score_1', 'score_2']], verbose=True, epochs=100, batch_size=16384)


# Print the model's weights
print(model.get_weights())

# Print the column means of the training data
print(games_tourney_train.mean()[1])


# Evaluate the model on the tournament test data
print(model.evaluate(games_tourney_test[[
      'seed_diff', 'pred']], games_tourney_test[['score_1', 'score_2']], verbose=False))


'''
Single model for classification and regression
Build a simple regressor / classifier
from tensorflow.keras.layers import Input, Dense

input_tensor = Input(shape=(1, ))
output_tensor_reg = Dense(1)(input_tensor)
output_tensor_class = Dense(1, activation='sigmoid')(output_tensor_reg)

Input Layer -> Regression Output -> Classification Output (sigmoid)

Make a regressor / classifier model
from tensorflow.keras.models import Model

model = Model(input_tensor, [output_tensor_reg, output_tensor_class])
model.compile(loss = ['mean_absolute_error', 'binary_crossentropy'], optimizer='adam')

Fit the combination classifier / regressor
X = games_tourney_train[['seed_diff']]
y_reg = games_tourney_train[['score_diff']]
y_class = games_tourney_train[['won']]
model.fit(X, [y_reg, y_class], epochs=100)

Look at the model's weights
model.get_weights() -> in

[   array([[1.2371823]], dtype=float32),
    array([[-0.05451894]], dtype=float32),
    array([[0.13870609]], dtype=float32),
    array([[0.00734114]], dtype=float32)    ] -> out

from scipy.special import expit as sigmoid
print(sigmoid(1 * 0.13870609 + 0.00734114)) -> in

0.5364470465211318 -> out

Evaluate the model on new data
X = games_tourney_test[['seed_diff']]
y_reg = games_tourney_test[['score_diff']]
y_class = games_tourney_test[['won']]
model.evaluate(X, [y_reg, y_class]) -> in

[9.866300069455413, 9.281179495657208, 0.585120575627864] -> out
'''

# Create an input layer with 2 columns
input_tensor = Input((2,))

# Create the first output
output_tensor_1 = Dense(1, activation='linear', use_bias=False)(input_tensor)

# Create the second output (use the first output as input here)
output_tensor_2 = Dense(1, activation='sigmoid',
                        use_bias=False)(output_tensor_1)

# Create a model with 2 outputs
model = Model(input_tensor, [output_tensor_1, output_tensor_2])


# Import the Adam optimizer

# Compile the model with 2 losses and the Adam optimzer with a higher learning rate
model.compile(loss=['mean_absolute_error', 'binary_crossentropy'],
              optimizer=Adam(learning_rate=0.01))

# Fit the model to the tournament training data, with 2 inputs and 2 outputs
model.fit(games_tourney_train[['seed_diff', 'pred']], [games_tourney_train[[
          'score_diff']], games_tourney_train[['won']]], epochs=10, verbose=True, batch_size=16384)


# Print the model weights
print(model.get_weights())

# Print the training data means
print(games_tourney_train.mean()[1])


# Import the sigmoid function from scipy

# Weight from the model
weight = 0.14

# Print the approximate win probability predicted close game
print(sigmoid(1 * weight))

# Print the approximate win probability predicted blowout game
print(sigmoid(10 * weight))


# Evaluate the model on new data
print(model.evaluate(games_tourney_test[['seed_diff', 'pred']], [
      games_tourney_test[['score_diff']], games_tourney_test[['won']]], verbose=False))


'''
Shared layers
- They are useful for making comparisons e.g
* Basketball teams
* Image similarity / retrieval
* Document similarity
** its knowns in the academic literature as Siamese networks

Multiple inputs
- They are useful when you want to process different types of data withing a model

Input Text      ->      Embedding Layer     ->  LSTM Layer \
Input Numerics  --------------------------------------------> Concat Layer -> Output Layer
Input Images -> Convolutional Layer -> Convolutional Layer /

Multiple outputs
- It can do both classification and regression
- In the regression problem, the neural network gets penalized less for random chance

Skip connections
input_tensor = Input((100,))
hidden_tensor = Dense(256, activation='relu')(input_tensor)
hidden_tensor = Dense(256, activation='relu')(hidden_tensor)
hidden_tensor = Dense(256, activation='relu')(hidden_tensor)
output_tensor = Concatenate()([input_tensor, hidden_tensor])
output_tensor = Dense(256, activation='relu')(output_tensor)
* Visualizing the Loss Landscape of Neural Networks
'''
