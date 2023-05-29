''' Hyperparameters and Parameters '''

'''
Introduction & 'Parameters'
The dataset
- The dataset relates to credit card defaults
- It contains variables related to the financial history of some consumers in Taiwan. It has 30,000 users and 24 attributes
- Our modeling target is whether they defaulted on their loan
- It has already been preprocessed and at times we will take smaller samples to demonstarte a concept
- Extra information about the dataset can be found here:
* https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients

Parameters Overview
What is a Parameter?
- They are components of the final model that are learned through the modeling process.

Parameters in Logistic Regression
A simple logistic regression model:
log_reg_clf = LogisticRegression()
log_reg_clf.fit(X_train, y_train)
print(log_reg_clf.coef_) -> in

array([[   -2.88651273e-06,    -8.23168511e-03,     7.50857018e-04,
            3.94375060e-04,     3.79423562e-04,     4.34612046e-04,
            4.37561467e-04,     4.12107102e-04,     -6.41089138e-06,
            -4.39364494e-06,    cont . . .  ]]) -> out

Tidy up the coefficients:
# Get the original variable names
original_variables = list(X_train.columns)

# Zip together the names and coefficients
zipped_together = list(zip(original_variables, log_reg_clf.coef_[0]))
coefs = [list(x) for x in zipped_together]

# Put into a DataFrame with column labels
coefs = pd.DataFrame(coefs, columns=['Variable', 'Coefficient'])

Now sort and print the top 3 coefficients
coefs.sort_values(by=['Coefficient'], axis=0, inplace=True, ascending=False)
print(coefs.head(3))

Variable    Coefficient
PAY_0       0.000751
PAY_5       0.000438
PAY_4       0.000435 -> out

- In the data, the PAY variables relate to how many months people have previously delayed their payments.
* Having a high number of months of delayed payments, makes someone more likely to default the next month.

Where to find Parameters
To find parameters we need:
- To know a bit about the algorithm itself and how it works.
- Consult the Scikit Learn documentation to see where the parameter is stored in the returned object.
* Parameters will be found under the 'Attributes' section, not the 'parameters' section!

Parameters in Random Forest
What about tree based algorithms?
- Random forest has no coefficients, but node decisions ( what feature and what value to split on ).

# A simple random forest estimator
rf_clf = RandomForestClassifier(max_depth = 2)
rf_clf.fit(X_train, y_train)

# Pull out one tree from the forest
chosen_tree = rf_clf.estimators_[7]

Extracting Node Decisions
- We can pull out details of the left, second-from-top node:

# Get the column it split on
split_column = chosen_tree.tree_.feature[1]
split_column_name = X_train.columns[split_column]

# Get the level it split on
split_value = chosen_tree.tree_.threshold[1]
print('This node split on feature {}, at a value of {}'.format(split_column_name, split_value)) -> in

'This node split on feature PAY_0, at a value of 1.5' -> out
'''

# Create a list of original variable names from the training DataFrame
original_variables = list(X_train.columns)

# Extract the coefficients of the logistic regression estimator
model_coefficients = log_reg_clf.coef_[0]

# Create a dataframe of the variables and coefficients & print it out
coefficient_df = pd.DataFrame(
    {"Variable": original_variables, "Coefficient": model_coefficients})
print(coefficient_df)

# Print out the top 3 positive variables
top_three_df = coefficient_df.sort_values(
    by=['Coefficient'], axis=0, ascending=False)[0:3]
print(top_three_df)


# Extract the 7th (index 6) tree from the random forest
chosen_tree = rf_clf.estimators_[6]

# Visualize the graph using the provided image
imgplot = plt.imshow(tree_viz_image)
plt.show()

# Extract the parameters and level of the top (index 0) node
split_column = chosen_tree.tree_.feature[0]
split_column_name = X_train.columns[split_column]
split_value = chosen_tree.tree_.threshold[0]

# Print out the feature and level
print("This node split on feature {}, at a value of {}".format(
    split_column_name, split_value))


'''
Introducing Hyperparameters
What is a hyperparameter
Hyperparameters:
- They are set before the modeling process

Hyperparameters in Random Forest
- Create a simple radom forest estimator and print it out:

rf_clf = RandomForestClassifier()
print(rf_clf) -> in

RandomForestClassifier(n_estimators=100, *, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='sqrt', max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=True, oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False, class_weight=None, ccp_alpha=0.0, max_samples=None) -> out

A single hyperparameter
Take the n_estimators parameter.

Data Type & Default Value:
    n_estimators : integer, optional (default = 10)
Definition:
    The number of trees in the forest.

Setting hyperparameters
- Set some hyperparameters at estimator creation:

rf_clf = RandomForestClassifier(n_estimators=100, criterion='entropy')
print(rf_clf)

RandomForestClassifier(n_estimators=100, criterion='entropy', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='sqrt', max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=True, oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False, class_weight=None, ccp_alpha=0.0, max_samples=None) -> out

Hyperparameters in Logistic Regression
- Find the hyperparameters of a Logistic Regression:

log_reg_clf = LogisticRegression()
print(log_reg_clf) -> in

LogisticRegression(penalty='l2', *, dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver='lbfgs', max_iter=100, multi_class='auto', verbose=0, warm_start=False, n_jobs=None, l1_ratio=None) -> out

Hyperparameter Importance
- Some hyperparameters are more important than others.
- Some will not help model performance:
* These are related to computational decisions or what information to retain for analysis.
- For the random forest classifier:
* n_jobs : It tells how many cores to use and will only speed up modeling time.
* random_state
* verbose : It tells whether to print out information as the modeling occurs
- Not all hyperparameters make sense to 'train'

Random Forest: Important Hyperparameters
- Some umportant hyperparameters:
* n_estimatores (how many trees in the forest) : should be set to a high value, 500 or 1000 or even more is not uncommon
* max_features (how many features to consider when splitting, which is vital to ensure tree diversity) : try different values
* max_depth & min_sample_leaf (important to control overfitting of individual trees)
* criterion : it may have a small impact but it is not generally a primary hyperparameter to consider.

How to find hyperparameters that matter?
- Academic papers
- Blogs and tutorials from trusted sources (Like DataCamp!)
- The Scikit Learn module documentation
- Practical experience
'''

# Print out the old estimator, notice which hyperparameter is badly set
print(rf_clf_old)

# Get confusion matrix & accuracy for the old rf_model
print("Confusion Matrix: \n\n {} \n Accuracy Score: \n\n {}".format(confusion_matrix(
    y_test, rf_old_predictions), accuracy_score(y_test, rf_old_predictions)))

# Create a new random forest classifier with better hyperparamaters
rf_clf_new = RandomForestClassifier(n_estimators=500)

# Fit this to the data and obtain predictions
rf_new_predictions = rf_clf_new.fit(X_train, y_train).predict(X_test)

# Assess the new model (using new predictions!)
print("Confusion Matrix: \n\n", confusion_matrix(y_test, rf_new_predictions))
print("Accuracy Score: \n\n", accuracy_score(y_test, rf_new_predictions))


# Build a knn estimator for each value of n_neighbours
knn_5 = KNeighborsClassifier(n_neighbors=5)
knn_10 = KNeighborsClassifier(n_neighbors=10)
knn_20 = KNeighborsClassifier(n_neighbors=20)

# Fit each to the training data & produce predictions
knn_5_predictions = knn_5.fit(X_train, y_train).predict(X_test)
knn_10_predictions = knn_10.fit(X_train, y_train).predict(X_test)
knn_20_predictions = knn_20.fit(X_train, y_train).predict(X_test)

# Get an accuracy score for each of the models
knn_5_accuracy = accuracy_score(y_test, knn_5_predictions)
knn_10_accuracy = accuracy_score(y_test, knn_10_predictions)
knn_20_accuracy = accuracy_score(y_test, knn_20_predictions)
print("The accuracy of 5, 10, 20 neighbours was {}, {}, {}".format(
    knn_5_accuracy, knn_10_accuracy, knn_20_accuracy))


'''
Setting & Analyzing Hyperparameter Values
- Some hyperparameters are more important than others to begin tuning.
- Which values to try for hyperparameters?
* It is specific to each algorithm and hyperparameter itself
* Some best practice guidelines and tips do exist

Conflicting Hyperparameter Choices
- Be aware of conflicting hyperparameter choices.
* LogisticRegression() conflicting parameter options of 'solver' & 'penalty' that conflict.
** The 'newton-cg', 'sag' and 'lbfgs' solvers support only l2 penalties.

* Some aren't explicit, but wi;; just 'ignore' (from 'ElasticNet' with the 'normalize' hyperparameter):
** This parameter is ignored when fit_intercept is set to False

Silly Hyperparameters Values
- Be aware of setting 'silly' values for different algorithms:
* Randomforest with low number of trees
** Would you consider it a 'forest' with only 2 trees?

* 1 Neighbor in KNN algorithm
** Averaging the 'votes' of one person doesn't sound very robust!

* Increasing a hyperparameter by a very small amount

- Spending time documenting sensible values for hyperparameters is a valuable activity.

Automating Hyperparameter Choice
- In the previous exercise, we built models as: 

knn_5 = KNeighborsClassifier(n_neighbors=5)
knn_10 = KNeighborsClassifier(n_neighbors=10)
knn_20 = KNeighborsClassifier(n_neighbors=20)

- This is quite inefficient. Can we do better?

- Try a for loop to iterate through options:

neighbors_list = [3, 5, 10, 20, 50, 75]
accuracy_list = []

for test_number in neighbors_list:
    model = KNeighborsClassifier(n_neighbors=test_number)
    predictions = model.fit(X_train, y_train).predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    accuracy_list.append(accuracy)

- We can store the results in a DataFrame to view:

results_df = pd.DataFrame({'neighbors': neighbors_list, 'accuracy': accuracy_list})
print(results_df) -> in

Neigbors 3 5 10 20 50 75
Accuracy 0.71 0.7125 0.765 0.7825 0.7825 0.7825 -> out

Learning Curves
- It is a common tool that is used to assist with analyzing the impact of a singular hyperparameter on an end result.
- Let's create a learning curve graph
- We'll test many more values this time

neighbors_list = list(range(5, 500, 5))
accuracy_list = []

for test_number in neighbors_list:
    model = KNeighborsClassifier(n_neighbors=test_number)
    predictions = model.fit(X_train, y_train).predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    accuracy_list.append(accuracy)

results_df = pd.DataFrame({'neighbors': neighbors_list, 'accuracy': accuracy_list})

- We can plot the larger DataFrame:

plt.plot(results_df['neighbors'], results_df['accuracy'])

# Add the labels and title
plt.gca().set(xlabel='n_neighbors', ylabel='Accuracy', title='Accuracy for different n_neighbors')
plt.show()

A handy trick for generating values
- Python's 'range' function does not work for decimal steps.
- A handy trick uses NumPy's np.linspace(start, end, num)
- It create a number of values (num) evenly spread withing an interval (start, end) that you specify.
* e.g print(np.linspace(1, 2, 5)) -> in

[1.   1.25 1.5  1.75 2.  ] -> out
'''

# Set the learning rates & results storage
learning_rates = [0.001, 0.01, 0.05, 0.1, 0.2, 0.5]
results_list = []

# Create the for loop to evaluate model predictions for each learning rate
for learning_rate in learning_rates:
    model = GradientBoostingClassifier(learning_rate=learning_rate)
    predictions = model.fit(X_train, y_train).predict(X_test)
    # Save the learning rate and accuracy score
    results_list.append([learning_rate, accuracy_score(y_test, predictions)])

# Gather everything into a DataFrame
results_df = pd.DataFrame(results_list, columns=['learning_rate', 'accuracy'])
print(results_df)


# Set the learning rates & accuracies list
learn_rates = np.linspace(0.01, 2, num=30)
accuracies = []

# Create the for loop
for learn_rate in learn_rates:
    # Create the model, predictions & save the accuracies as before
    model = GradientBoostingClassifier(learning_rate=learn_rate)
    predictions = model.fit(X_train, y_train).predict(X_test)
    accuracies.append(accuracy_score(y_test, predictions))

# Plot results
plt.plot(learn_rates, accuracies)
plt.gca().set(xlabel='learning_rate', ylabel='Accuracy',
              title='Accuracy for different learning_rates')
plt.show()


''' Grid search '''

'''
Introducing Grid Search
Automating 2 Hyperparameters
- What about testing values of 2 hyperparameters?
- Using a GBM algorithm:
* learn_rate = [0.001, 0.01, 0.05]
* max_depth = [4, 6, 8, 10]
- We could use a (nested) for loop!

- Firstly a model creation function:

def gbm_grid_search(learn_rate, max_depth):
    model = GradientBoostingClassifier(learning_rate=learn_rate, max_depth=max_depth)
    predictions = model.fit(X_train, y_train).predict(X_test)
    return([learn_rate, max_depth, accuracy_score(y_test, predictions)])

- Now we can loop through our lists of hyperparameters and call our function:

results_list = []

for learn_rate in learn_rate_list:
    for max_depth in max_depth_list:
        results_list.append(gbm_grid_search(learn_rate, max_depth))

- We can put these results into a DataFrame as well and print out:

results_df = pd.DataFrame(results_list, columns=['learning_rate', 'max_depth', 'accuracy'])
print(results_df) -> in

learning_rate   max_depth   accuracy
0.001           4           0.75
0.001           6           0.75
0.01            4           0.77
0.01            6           0.76 -> out

How many models?
- There were many more models built by adding more hyperparameters and values.
* The relationship is not linear, it is exponential
* One more value of a hyperparameter is not just one model
* 5 for Hyperparameter 1 and 10 for Hyperparameter 2 is 50 models!
- What about cross-validation?
* 10-fold cross-validation would make 50 x 10 = 500 models!

From 2 to N hyperparameters
- What about adding more hyperparameters?
- We could nest our loop!

# Adjust the list of values to test
learn_rate_list = [0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5]
max_depth_list = [4, 6, 8, 10, 12, 15, 20, 25, 30]
subsample_list = [0.4, 0.6, 0.7, 0.8, 0.9]
max_features_list = ['auto', 'sqrt']

- Ajust our function:

def gbm_grid_search(learn_rate, max_depth, subsample, max_features):
    model = GradientBoostingClassifier(learning_rate=learn_rate, max_depth=max_depth, subsample=subsample, max_features=max_features)
    predictions = model.fit(X_train, y_train).predict(X_test)
    return([learn_rate, max_depth, accuracy_score(y_test, predictions)])

- Adjusting our for loop (nesting):

for learn_rate in learn_rate_list:
    for max_depth in max_depth_list:
        for subsample in subsample_list:
            for max_features in max_features_list:
                results_list.append(gbm_grid_search(learn_rate, max_depth, subsample, max_features))

results_df = pd.DataFrame(results_list, columns=['learning_rate', 'max_depth', 'subsample', 'max_features', 'accuracy'])
print(results_df) -> in

- How many models now?
* 7 x 9 x 5 x 2 = 630 (6,300 if cross-validated!)
- We can't keep nesting forever!
- Plus, what if we wanted:
* Details on training times and scores
* Details on cross-validation scores

Introducing Grid Search
- Let's create a grid:
* Down the left all values of max_depth
* Across the top all values of learning_rate

    0.001           0.01            0.005
4   (4, 0.001)      (4, 0.01)       (4, 0.05)
6   (6, 0.001)      (6, 0.01)       (6, 0.05)
8   (8, 0.001)      (8, 0.01)       (8, 0.05)

- It is the process of running a model for every cell in the grid with the hyperparameters specified
- (4, 0.001) is equivalent to making an estimator like so:
* GradientBoostingClassifier(max_depth=4, learning_rate=0.001)

Grid Search Pros & Cons
- Some advantages of this approach:
* It's programmatic, and saves many lines of code.
* It is guaranteed to find the best model within the grid you specify.
* It is an easy methodology to explain

- Some disadvantages of this approach:
* It is computationally expensive!
* It is 'uninformed'. Results of one model don't help creating the next model.
'''

# Create the function


def gbm_grid_search(learning_rate, max_depth):

    # Create the model
    model = GradientBoostingClassifier(
        learning_rate=learning_rate, max_depth=max_depth)

    # Use the model to make predictions
    predictions = model.fit(X_train, y_train).predict(X_test)

    # Return the hyperparameters and score
    return ([learning_rate, max_depth, accuracy_score(y_test, predictions)])


# Create the relevant lists
results_list = []
learn_rate_list = [0.01, 0.1, 0.5]
max_depth_list = [2, 4, 6]

# Create the for loop
for learn_rate in learn_rate_list:
    for max_depth in max_depth_list:
        results_list.append(gbm_grid_search(learn_rate, max_depth))

# Print the results
print(results_list)


results_list = []
learn_rate_list = [0.01, 0.1, 0.5]
max_depth_list = [2, 4, 6]

# Extend the function input


def gbm_grid_search_extended(learn_rate, max_depth, subsample):

    # Extend the model creation section
    model = GradientBoostingClassifier(
        learning_rate=learn_rate, max_depth=max_depth, subsample=subsample)

    predictions = model.fit(X_train, y_train).predict(X_test)

    # Extend the return part
    return ([learn_rate, max_depth, subsample, accuracy_score(y_test, predictions)])


results_list = []

# Create the new list to test
subsample_list = [0.4, 0.6]

for learn_rate in learn_rate_list:
    for max_depth in max_depth_list:
        # Extend the for loop
        for subsample in subsample_list:
            # Extend the results to include the new hyperparameter
            results_list.append(gbm_grid_search_extended(
                learn_rate, max_depth, subsample))

# Print results
print(results_list)


'''
Grid Search with Scikit Learn
Steps in a Grid Search
- Select an algorithm or estimator to tune the hyperparameters.
- Define which hyperparameters to be tuned
- Define a range of values for each hyperparameter
- Decide a cross-validation scheme
- Define a scoring function to determine which model was the best.
- Include extra useful information or functions

GridSearchCV Object Inputs
- The important inputs are: 
* estimator
* param_grid
* cv
* scoring
* refit
* n_jobs
* return_train_score

GridSearchCV 'estimator'
- The estimator input:
* It is essentially our algorithm
* e.g KNN, Random Forest, GBM, Logistic Regression etc
- Remenber: Only one estimator per GridSearchCV object

GridSearchCV 'param_grid'
- The param_grid input:
* It is how we tell GridSearchCV which hyperparameters and which values to test.
** Rather than a list:
max_depth_list = [2, 4, 6,8]
min_samples_leaf_list = [1, 2, 4, 6]

** This would be a dictionary:
param_grid = {'max_depth': [2, 4, 6, 8], 'min_samples_leaf': [1, 2, 4, 6]}

- The keys must be the hyperparameter names, the values a list of values to test.
- The keys in the param_grid dictionary must be valid hyperparameters else the GridSearch will fail.
e.g
# Incorrect
param_grid = {'C': [0.1, 0.2, 0.5], 'best_choice': [10, 20, 50]} -> in

ValueError: Invalid parameter best_choice for estimator LogisticRegression -> out

GridSearchCV 'cv'
- The cv input:
* It specifies the choice of cross-validation to undertake
* Using a integer undertakes k-fold cross validation where 5 or 10 is usually standard

GridSearchCV 'scoring'
- The scoring input:
* It is used to evaluate the model's performance (choose the best grid square)
* Use your own or Scikit Learn's metrics module

- You can check all the built in scoring functions this way:
from sklearn import metrics

sorted(metrics.SCORERS.keys())

GridSearchCV 'refit'
- The refir input:
* If it is set to true, it means the best hyperparameter combinations are used to undertake a fitting to the training data.
* It allows the GridSearchCV object to be used as an estimator (for prediction)
* It is a very handy option as you don't need to save out the best hyperparameters and train another model!

GridSearchCV 'n_jobs'
- The n_jobs input:
* It assists with parallel execution
* It can effectively split-up the work and have many models being created at the same time.
- Some handy code to check number of cores that can be ran at once:
import os 

print(os.cpu_count())

- Careful using all the cores for modelling if you want to do other work!

GridSearchCV 'return_train_score'
- The return_train_score input:
* It logs statistics about the training runs that were undertaken.
* It can be useful for plotting and understanding (analyzing) test vs training set performance (and hence bias-variance tradeoff) but adds computational expense.
* It does not assist in picking the best model, its only for analysis purposes.

Building a GridSearchCV object
# Create the grid
parameter_grid = {'max_depth': [2, 4, 6, 8], 'min_samples_leaf': [1, 2, 4, 6]}

# Get a base classifier with some set parameters.
rf_class = RandomForestClassifier(criterion='entropy', max_features='auto')

- Putting the pieces together:
grid_rf_class = GridSearchCV(estimator = rf_class, param_grid = parameter_grid, scoring='accuracy', n_jobs=4, cv=10, refit=True, return_train_Score=True)

- Because we set refit = True, we can directly use the object:
# Fit the object to our data
grid_rf_class.fit(X_train, y_train)

# Make predictions
grid_rf_class.predict(X_test)
'''

# Create a Random Forest Classifier with specified criterion
rf_class = RandomForestClassifier(criterion='entropy')

# Create the parameter grid
param_grid = {'max_depth': [2, 4, 8, 15], 'max_features': ['auto', 'sqrt']}

# Create a GridSearchCV object
grid_rf_class = GridSearchCV(estimator=rf_class, param_grid=param_grid,
                             scoring='roc_auc', n_jobs=4, cv=5, refit=True, return_train_score=True)
print(grid_rf_class)


'''
Understanding a grid search output
Analyzing the output
- 3 different groups for the GridSearchCV properties:
- A result log
* cv_results_

- The best results
* best_index_, best_params_ & best_score_

- 'Extra information'
* scorer_, n_splits_ and refit_time_

Accessing object properties
- Properties are accessed using the dot notation
e.g grid_search_object.property
* where property is the actual property you want to retrieve

The .cv_results_ property
- The cv_results_ property:
* It is a dictionary that can be read into pandas DataFrame to explore (print and analyze).

cv_results_df = pd.DataFrame(grid_rf_class.cv_results_)
print(cv_results_df.shape) -> in

(12, 23)

* The 12 rows for the 12 squares in our grid or 12 models we ran.

The .cv_results_ 'time' columns
- The time columns refer to the time it took to fit (and score) the model.
* Remember how we did a 5-fold cross-validation? This ran 5 times and stored the average and standard deviation of the times it took in seconds.

    mean_fit_time   std_fit_time    mean_score_time     std_score_time
0   0.321069        0.007236        0.015008            0.000871
1   0.678216        0.066385        0.034155            0.003767
2   0.939865        0.009502        0.055868            0.004148
3   0.296547        0.006261        0.017990            0.002803
4   0.686065        0.016163        0.040048            0.001304
5   1.097201        0.006327        0.057136            0.004468
6   0.416973        0.085533        0.021157            0.003901
7   0.788864        0.021954        0.042638            0.004802
8   1.198466        0.054694        0.049674            0.006884
9   0.398824        0.027500        0.025307            0.009473
10  0.719588        0.019231        0.035629            0.005712
11  0.847477        0.036584        0.029104            0.005220 

The .cv_results_ 'param_' columns
- The param_ columns store the parameters it tested on that row, one column per parameter

param_max_depth     param_min_samples_leaf  param_n_estimators
10                  1                       100
10                  1                       200
10                  2                       100
10                  2                       200
10                  2                       300

The .cv_results_ 'param' column
- The params column contains dictionary of all parameters:

pd.set_option('display.max_colwidth', -1)
print(cv_results_df.loc[:, 'params'])

                        params
{'max_depth': 10, 'min_samples_leaf': 1, 'n_estimators': 100}
{'max_depth': 10, 'min_samples_leaf': 1, 'n_estimators': 200}
{'max_depth': 10, 'min_samples_leaf': 2, 'n_estimators': 100}
{'max_depth': 10, 'min_samples_leaf': 2, 'n_estimators': 200}
{'max_depth': 10, 'min_samples_leaf': 2, 'n_estimators': 300} 

- set_option is used to ensure we don't truncate the results being printed out.

The .cv_results_ 'test_score' columns
- The test_score columns contain the scores on the test set for each of the cross-folds as well as some summary statistics:

split0_test_score   split1_test_score   . . .   mean_test_score     std_test_score
0.72820401          0.7859811           . . .   0.76010401          0.02995142
0.73539669          0.7963085           . . .   0.76590708          0.02721413
0.72929381          0.78686003          . . .   0.7718143           0.02775648
0.72820401          0.78554164          . . .   0.77044862          0.02794597
0.72885789          0.78795869          . . .   0.77122424          0.03288053

The .cv_results_ 'rank_test_score' column
- The rank column, ordering the mean_test_score from best to worst:

rank_test_score
    9
    4
    1
    3
    2

Extracting the best row
- We can select the best grid search square for analysis from cv_results_ using the rank_test_score column

best_row = cv_results_df[cv_results_df['rank_test_score']== 1]
print(best_row) -> in

mean_fit_time   . . .   params                                                          . . .   mean_test_score     rank_test_score
0.97765441      . . .   {'max_depth': 10, 'min_samples_leaf': 2, 'n_estimators': 200}   . . .   0.7718143           1

The .cv_results_ 'train_score' columns
- The test_score columns are then repeated for the training_scores.
- Some important notes to keep in mind:
* return_train_score must be True to include training scores columns.
- There is no ranking column for the training scores, as we only care about performance on the the test set in each fold.

The best grid square
- Information on the best grid square is neatly summarized in the following 3 properties:
* best_params_ , the dictionary of parameters that gave the best score.
* best_score_ , the actual best score.
* best_index_ , the row in our cv_results_.rank_test_score that was the best.

The best_estimator_ property
- The best_estimator_ property is an estimator built using the best parameters from the gridsearch.
- For us, this is a Random Forest estimator:
type(grid_rf_class.best_estimator_) -> in

sklearn.ensemble.forest.RandomForestClassifier -> out

- We could also directly use this object as an estimator if we want! 
* This is why we set refit = True when creating the grid search, otherwise we would need to refit using the best parameters ourself before using the best estimator.

print(grid_rf_class.best_estimator_) -> in

randomForestClassifier(bootstrap=True, class_weight=None, criterion='entropy', max_depth=10, max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=300, n_jobs=None, oob_score=False, random_state=None, verbose=0, warm_start=False) -> out

Extra information
- Some extra information is available in the following properties:
- scorer_
* What scorer function was used on the held out data. (we set it to AUC)
- n_splits_
* How many cross-validation splits. (We set to 5)
- refit_time_
* The number of seconds used for refitting the best model on the whole dataset.
'''

# Read the cv_results property into a dataframe & print it out
cv_results_df = pd.DataFrame(grid_rf_class.cv_results_)
print(cv_results_df)

# Extract and print the column with a dictionary of hyperparameters used
column = cv_results_df.loc[:, ['params']]
print(column)

# Extract and print the row that had the best mean test score
best_row = cv_results_df[cv_results_df['rank_test_score'] == 1]
print(best_row)


# Print out the ROC_AUC score from the best-performing square
best_score = grid_rf_class.best_score_
print(best_score)

# Create a variable from the row related to the best-performing square
cv_results_df = pd.DataFrame(grid_rf_class.cv_results_)
best_row = cv_results_df.loc[[grid_rf_class.best_index_]]
print(best_row)

# Get the n_estimators parameter from the best-performing square and print
best_n_estimators = grid_rf_class.best_estimator_.get_params()["n_estimators"]
print(best_n_estimators)


# See what type of object the best_estimator_ property is
print(type(grid_rf_class.best_estimator_))

# Create an array of predictions directly using the best_estimator_ property
predictions = grid_rf_class.best_estimator_.predict(X_test)

# Take a look to confirm it worked, this should be an array of 1's and 0's
print(predictions[0:5])

# Now create a confusion matrix
print("Confusion Matrix \n", confusion_matrix(y_test, predictions))

# Get the ROC-AUC score
predictions_proba = grid_rf_class.best_estimator_.predict_proba(X_test)[:, 1]
print("ROC-AUC Score \n", roc_auc_score(y_test, predictions_proba))


''' Random Search '''

'''
Introducing Random Search
- It is very similar to grid search:
* Define an estimator, which hyperparameters to tune and the range of values for each hyperparameter
* Set a c cross-validation scheme and scoring function
- But we instead randomly select grid squares.

Why does this work?
- Bengio and Bergstra (2012):
* This paper shows empirically and theoretically that randomly chosen trials are more efficient for hyper-parameter optimization than trials on a grid.
- Two main reasons:
* Not every hyperparameter is as important
* A little trick of probability

A probability Trick
- If we randomly select hyperparameter combinations uniformly, let's consider the chance fo MISSING every single trial, to show how unlikely that is (5 good model in 100)
* Trial 1 = 0.05 chance of success and (1 - 0.05) of missing
* Trial 2 = (1 - 0.05) x (1 - 0.05) of missing the range
* Trial 1 = (1 - 0.05) x(1 - 0.05) x (1 - 0.05) of missing again

- In fact, with n trials, we have (1 - 0.05)^n chance that every single trial misses that desired spot.

- So how many trials to have a high (95%) chance of getting in that region?
* we have (1 - 0.05)^n chance to miss everything
* So we must have ( 1 - miss everything ) chance to get in there or ( 1 - ( 1 - 0.05 )^n)
* Solving 1 - (1 - 0.05)^100 >= 0.95  gives us n >= 59

- What does that all mean?
* You are unlikely to keep completely missing the 'good area' for a long time when randomly picking new spots
* A grid search may spend lots of time in a 'bad area' as it covers exhaustively.

Some important notes
- Remember:
* The maximum is still only as good as the grid you set!
* Remember to fairly compare this to grid search, you need to have the same modeling 'budget'

Creating a random sample of hyperparameters
- We can create our own random sample of hyperparameter combinations:

# Set some hyperparameter lists
learn_rate_list = np.linspace(0.001, 2, 150)
min_samples_leaf_list = list(range(1, 51))

# Create list of combinations
from itertools import product

combinations_list = [list(x) for x in product(learn_rate_list, min_samples_leaf_list)]

# Select 100 models from our larger set
random_combinations_index = np.random.choice(range(0, len(combinations_list)), 100, replace=False)
combinations_random_chosen = [combinations_list[x] for x in random_combinations_index]

Visualizing a Random Search
- We can also visualize the random search coverage by plotting the hyperparameter choices on an X and Y axis.
'''

# Create a list of values for the learning_rate hyperparameter
learn_rate_list = list(np.linspace(0.01, 1.5, 200))

# Create a list of values for the min_samples_leaf hyperparameter
min_samples_list = list(range(10, 41))

# Combination list
combinations_list = [list(x)
                     for x in product(learn_rate_list, min_samples_list)]

# Sample hyperparameter combinations for a random search.
random_combinations_index = np.random.choice(
    range(0, len(combinations_list)), 250, replace=False)
combinations_random_chosen = [combinations_list[x]
                              for x in random_combinations_index]

# Print the result
print(combinations_random_chosen)


# Create lists for criterion and max_features
criterion_list = ['gini', 'entropy']
max_feature_list = ["auto", "sqrt", "log2", None]

# Create a list of values for the max_depth hyperparameter
max_depth_list = list(range(3, 56))

# Combination list
combinations_list = [list(x) for x in product(
    criterion_list, max_feature_list, max_depth_list)]

# Sample hyperparameter combinations for a random search
combinations_random_chosen = random.sample(combinations_list, 150)

# Print the result
print(combinations_random_chosen)


# Confirm how many hyperparameter combinations & print
number_combs = len(combinations_list)
print(number_combs)

# Sample and visualise specified combinations
for x in [50, 500, 1500]:
    sample_and_visualize_hyperparameters(x)

# Sample all the hyperparameter combinations & visualise
print(sample_and_visualize_hyperparameters(number_combs))


'''
Random Search in Scikit Learn
Comparing to GridSearchCV
- We don't need to reinvent the wheel. Let's recall the steps for a Grid Search:
* Decide an algorithm / estimator
* Defining which hyperparameters we will tune
* Defining a range of values for each hyperparameter
* Setting a cross-validation scheme; and
* Define a score function
* Include extra useful information or functions

- There is only one difference:
* Decide how many samples (hyperparameter combinations) to take to build models and then undertake this sampling before we model.

Comparing Scikit Learn Modules
- The modules are similar too:

GridSearchCV:
sklearn.model_selection.GridSearchCV(estimator, param_grid, scoring=None, fit_params=None, n_jobs=None, refit=True, cv='warn', verbose=0, pre_dispatch='2*n_jobs', error_score='raise-deprecating', return_train_score='warn')

RandomizedSearchCV:
sklearn.model_selection.RandomizedSearchCV(estimator, param_distributions, n_iter=10, scoring=None, fit_params=None, n_jobs=None, refit=True, cv='warn', verbose=0, pre_dispatch='2*n_jobs', random_state=None, error_score='raise-deprecating', return_train_score='warn')

Key differences
- Two key differences:
* n_iter which is the number of samples for the random search to take from the grid. In the previous example we did 300
* param_distributions is slightly different from param_grid, allowing optional ability to set a distribution for sampling.
** The default is all combinations have equal chance to be chosen.

Build a RandomizedSearchCV object
- Now we can build a random search object just like the grid search, but with our small change:

# Set up the sample space
learn_rate_list = np.linspace(0.001, 2, 150)
min_samples_leaf_list = list(range(1, 51))

# Create the grid
parameter_grid = {'learning_rate': learn_rate_list, 'min_samples_leaf': min_samples_leaf_list}

# Define how many samples
number_models = 10

- Now we can build the object

# Create a random search object
random_GBM_class = RandomizedSearchCV(estimator = GradientBoostingClassifier(), param_distributions = parameter_grid, n_iter = number_models, scoring = 'accuracy', n_jobs = 4, cv = 10, refit = True, return_train_Score = True)

# Fit the object to our data
random_GBM_class.fit(X_train, y_train)

Analyze the output
- The output is exactly the same as in the GridSearchCV!
- How do we see what hyperparameter values were chosen?
* The cv_results_ dictionary (in the relevant param_ columns)!

- Extract the lists:
rand_x = list(random_GBM_class.cv_results_['param_learning_rate'])
rand_y = list(random_GBM_class.cv_results_['param_min_samples_leaf'])

- Build our visualization:
# Make sure we set the limits of Y and X appropriately
x_lims = [np.min(learn_rate_list), np.max(learn_rate_list)]
y_lims = [np.min(min_samples_leaf_list), np.max(min_samples_leaf_list)]

# Plot grid results
plt.scatter(rand_y, rand_x, c=['blue']*10)
plt.gca().set(xlabel='learn_rate', ylabel='min_samples_leaf', title='Random Search Hyperparameters')
plt.show()
'''

# Create the parameter grid
param_grid = {'learning_rate': np.linspace(
    0.1, 2, 150), 'min_samples_leaf': list(range(20, 65))}

# Create a random search object
random_GBM_class = RandomizedSearchCV(estimator=GradientBoostingClassifier(
), param_distributions=param_grid, n_iter=10, scoring='accuracy', n_jobs=4, cv=5, refit=True, return_train_score=True)

# Fit to the training data
random_GBM_class.fit(X_train, y_train)

# Print the values used for both hyperparameters
print(random_GBM_class.cv_results_['param_learning_rate'])
print(random_GBM_class.cv_results_['param_min_samples_leaf'])


# Create the parameter grid
param_grid = {'max_depth': list(range(5, 26)), 'max_features': [
    'auto', 'sqrt']}

# Create a random search object
random_rf_class = RandomizedSearchCV(estimator=RandomForestClassifier(
    n_estimators=80), param_distributions=param_grid, n_iter=5, scoring='roc_auc', n_jobs=4, cv=3, refit=True, return_train_score=True)

# Fit to the training data
random_rf_class.fit(X_train, y_train)

# Print the values used for both hyperparameters
print(random_rf_class.cv_results_['param_max_depth'])
print(random_rf_class.cv_results_['param_max_features'])


'''
Comparing Grid and Random Search
What's the same?
- Similarities between Random and Grid Search?
* Both are automated ways of tuning different hyperparameters
* For both, you set the grid to sample from (which hyperparameters and values for each)
** Remember to think carefully about your grid!
* For both, you set a cross-validation scheme and scoring function

What's differnet?
- Grid Search:
* Exhaustively tries all combinations within the sample space
* No Sampling methodology
* More computationally expensive
* Guaranteed to find the best score in the sample space

- Random Search:
* Randomly selects a subset of combinations within the sample space ( that you must specify)
* Can select a sampling methodology (other than 'uniform' which is the default) 
* Less computationally expensive
* Not guaranteed to find the best score in the sample space (but likely to find a good one faster)

Which should I use?
- So which one should I use?
* What are my consideratios?
- How much data do you have?
* More data means random search may be a better option.
- How many hyperparameters and values do you want to tune?
* More of these means random search may be a better option
- How much resources do you have? (Time, computing power)
* Less resources means random search may be a better option.

- If you wish to view the visualize_search() function definition, you can run this code:
import inspect

print(inspect.getsource(visualize_search))
'''

# Sample grid coordinates
grid_combinations_chosen = combinations_list[0:300]

# Create a list of sample indexes
sample_indexes = list(range(0, len(combinations_list)))

# Randomly sample 300 indexes
random_indexes = np.random.choice(sample_indexes, 300, replace=False)

# Use indexes to create random sample
random_combinations_chosen = [combinations_list[index]
                              for index in random_indexes]

# Call the function to produce the visualization
visualize_search(grid_combinations_chosen, random_combinations_chosen)


''' Informed Search '''

'''
Informed Search: Coarse to Fine
- So far everything (Grid and Random Search) we have done has been uninformed search:
* Uninformed search: Where each iteration of hyperparameter tuning does not learn from the previouss iterations.
- This is what allows us to parallelize our work. Though this doesn,t sound very efficient?

Informed vs Uninformed
The process so far:
- Uninformed
All Models + Measure -> Pick Best

- Informed
    One Model
    /      \
Learn <- Measure

Coarse to Fine Tuning
- It is a basic informed search methodology:
- It start out with a rough, random approach and iteratively refine the search.

- The process is:
* Random search
* Find promising areas
* Grid search in the smaller area
* Continue until optimal score is obtained
- You could substitute (3) with further random searches before the grid search.

Why Coarse to Fine?
- Coarse to fine tuning has some advantages:
- It utilizes the advantages of grid and random search.
* Wide search to begin with
* Deeper search once you know where agood spot is likelyto be
- Better spending of time and computational efforts mean you can iterate quicker
* No need to waste time on search spaces that are not giving good results!
- NB*: This isn't informed on one model but batches

Undertaking Coarse to Fine
- Let's take an example with the following hyperparameter ranges:
* Max_depth_list between 1 and 65
* min_sample_list between 3 and 17
* learn_rate_list, 150 values between 0.01 and 150

- How many possible models do we have?
combinations_list = [ list(x) for x in product(max_depth_list, min_sample_list, learn_rate_list) ]
print(len(combinations_list)) -> in

134400 -> out

Visualizing Coarse to Fine
Top results:
max_depth   min_samples_leaf    learn_rate      accuracy
10          7                   0.01            96
19          7                   0.023355705     96
30          6                   1.038389262     93
27          7                   1.11852349      91
16          7                   0.597651007     91


The next steps
- What we know from iteration one:
* max_depth between 8 and 30
* learn_rate less than 1.3
* min_samples_leaf perhaps less than 8

- Where to next? Another random or grid search with what we know!
- NB*: This was only bivariate analysis.

- If you wish to view the visualize_first() (or the visualize_second()) function definition, you can run this code:
import inspect

print(inspect.getsource(visualize_first))
print(inspect.getsource(visualize_second))
'''

# Confirm the size of the combinations_list
print(len(combinations_list))

# Sort the results_df by accuracy and print the top 10 rows
print(results_df.sort_values(by='accuracy', ascending=False).head(10))

# Confirm which hyperparameters were used in this search
print(results_df.columns)

# Call visualize_hyperparameter() with each hyperparameter in turn
visualize_hyperparameter('max_depth')
visualize_hyperparameter('min_samples_leaf')
visualize_hyperparameter('learn_rate')


# Use the provided function to visualize the first results
results_df['max_depth']
results_df['learn_rate']
visualize_first()

# Use the provided function to visualize the first results
# visualize_first()

# Create some combinations lists & combine:
max_depth_list = list(range(1, 21))
learn_rate_list = np.linspace(0.001, 1, 50)

# Call the function to visualize the second results
visualize_second()


'''
Informed Search: Bayesian Statistics
Bayes Introduction
- Bayes Rule:
* A statistical method of using new evidence to iteratively update our beliefs about some outcome
- Intuitively fits with the idea of informed search. etting better as we get more evidence.

- Bayes Rule has the formula:
P(A | B) = ( P(B | A) P(A) ) / P(B)

- LHS (Left hand side) =  The probability of A, given B has occured. B is some new evidence.
* This is known as the 'posterior'

- RHS (Right hand side) is how we calculate this.
* P(A) is the 'prior'. The initial hypothesis about the event. It is different to P(A | B). The P(A | B) is the probability given new evidence.
* P(B) is the 'marginal likelihood' and it is the probability of observing this new evidence
* P(B | A) is the 'likelihood' which is the probability of observing the evidence, given the event we care about.

- This all may be quite confusing, but let's use a common example of a medical diagnosis to demonstrate.

Bayes in Medicine
A medical example:
- 5% of people in the general population have a certain disease
* p(D)

- 10% of people are predisposed (i.e because of their genetics, they are more likely to get this condition)
* P(Pre)

- 20% of people with the disease are predisposed
* P(Pre | D)

What is the probability that any persson has the disease?
P(D)  = 5 / 100
P(D) = 0.05
- This is simply our prior as we have no evidence.

- What is the probability that a predisposed person has the disease?
P(D | Pre) = ( P(Pre | D) P(D) ) / P(Pre)

P(D | Pre) = ( 0.2 * 0.05 ) / 0.1
P(D | Pre) = 0.1

Bayes in Hyperparameter Tuning
- We can apply this logic to hyperparameter tuning:
* Pick a hyperparameter combination
* Build a model
* Get new evidence (the score of the model)
* Update our beliefs and chose better hyperparameters next round

- Bayesian hyperparameter tuning is very new but quite popular for larger and more complex hyperparameter tuning tasks as they work well to find optimal hyperparameter combinations in these situations.

Bayesian Hyperparameter Tuning with Hyperopt
- Introducing the Hyperopt package
- To undertake bayesian hyperparameter tuning, we need to:
* Set the Domain: Our Grid (with a bit of a twist)
* Set the optimization algorithm (use default TPE)
* Objective function to minimize: we will use 1-Accuracy

Hyperopt: Set the Domain (grid)
- There are many options to set the grid:
* Simple numbers
* Choose from a list
* Distribution of values
- Hyperopt does not use point values on the grid but instead each point represents probabilities for each hyperparameter value.
- We will do a simple uniform distribution but there are many more if you check the documentation.

The Domain
- Set up the grid:

space = {'max_depth': hp.quniform('max_depth', 2, 10, 2), 'min_samples_leaf': hp.quniform('min_samples_leaf', 2, 8, 2), 'learning_rate': hp.uniform('learning_rate', 0.01, 1, 55)}

- quniform means uniform but quantized (or binned) by the specified third number.

The objective function
- The objective function runs the algorithm:

def objective(params):
    params = {'max_depth': int(params['max_depth']), 'min_samples_leaf': int(params['min_samples_leaf']), 'learning_rate': params['learning_rate']}
    gbm_clf = GradientBoostingClassifier(n_estimators=500, **params)
    best_score = cross_val_score(gbm_clf, X_train, y_train, scoring='accuracy', cv=10, n_jobs=4).mean()
    loss = 1 - best_score
    write_results(best_score, params, iteration)
    return loss

Run the algorithm
- Run the algorithm:

best_result = fmin(fn=objective, space=space, max_evals=500, rstate=np.random.default_rng(42), algo=tpe.suggest)
'''

# These are the probabilities we know: 7% (0.07) of people are likely to close their account next month, 15% (0.15) of people with accounts are unhappy with your product (you don't know who though!), 35% (0.35) of people who are likely to close their account are unhappy with your product
# Assign probabilities to variables
p_unhappy = 0.15
p_unhappy_close = 0.35

# Probabiliy someone will close
p_close = 0.07

# Probability unhappy person will close
p_close_unhappy = (0.35 * 0.07) / 0.15
print(p_close)


# Set up space dictionary with specified hyperparameters
space = {'max_depth': hp.quniform(
    'max_depth', 2, 10, 2), 'learning_rate': hp.uniform('learning_rate', 0.001, 0.9)}

# Set up objective function


def objective(params):
    params = {'max_depth': int(
        params['max_depth']), 'learning_rate': params['learning_rate']}
    gbm_clf = GradientBoostingClassifier(n_estimators=100, **params)
    best_score = cross_val_score(
        gbm_clf, X_train, y_train, scoring='accuracy', cv=2, n_jobs=4).mean()
    loss = 1 - best_score
    return loss


# Run the algorithm
best = fmin(fn=objective, space=space, max_evals=20,
            rstate=np.random.default_rng(42), algo=tpe.suggest)
print(best)


'''
Informed Search: Genetic Algorithms
A lesson on genetics
- In genetic evolution in the real world, we have the following process:
* There are many creatures existing ('offspring')
* The strongest creatures survive and pair off
* There is osme 'crossover' as they form offspring
* There are random mutations to some of the offspring
** These mutations sometimes help give some offspring an advantage
* Go back to (1)!

Genetics in Machine Learning
- We can apply the same idea to hyperparameter tuning:
* We can create some models (that have hyperparameter settings)
* We can pick the best (by our scoring function)
** These are the ones that 'survive'
* We can create new models that are similar to the best ones
* We add in some randomness so we don't reach a local optimum
* Repeat until we are happy!

Why does this work well?
- This is an informed search that has a number of advantages:
* It allows us to learn fromprevious iterations, just like bayesian hyperparameter tuning.
* It has the additional advantage of some randomness
* (the package we'll use) takes care of many tedious aspects of machine learning.

Introducing TPOT
- It is a useful library for genetic hyperparameter tuning
* Consider TPOT your Data Science Assistant. TPOT is a Python Automated Machine Learning tool that optimizes machine learning pipelines using genetic programming.
- Pipelines not only include the model (or multiple models) but also work on features and other aspects of the process. Plus it returns the Python code of the pipeline for you!

TPOT components
- The key arguments to a TPOT classifier are:
* generations: number of Iterations to run training for
* population_size: The number of models to keep after each iteration
* offspring_size: Number of models to produce in each iteration
* mutation_rate: The proportion of pipelines to apply randomness to (between 0 and 1)
* crossover_rate: The proportion of pipelines to breed each iteration
* scoring: The function to determine the best models
* cv: Cross-validation strategy to use.
* verbosity: This parameter will print out the process as it goes

A simple example
from tpot import TPOTClassifier

tpot = TPOTClassifier(generations=3, population_size=5, verbosity=2, offspring_size=10, scoring='accuracy', cv=5)
tpot.fit(X_train, y_train)
print(tpot.score(X_test, y_test))

- We will keep default values for 'mutation_rate' and 'crossover_rate' as they are best left to the default without deeper knowledge on genetic programming.
- Notice: No algorithm-specific hyperparameters?
'''

# Assign the values outlined to the inputs
number_generations = 3
population_size = 4
offspring_size = 3
scoring_function = 'accuracy'

# Create the tpot classifier
tpot_clf = TPOTClassifier(generations=number_generations, population_size=population_size,
                          offspring_size=offspring_size, scoring=scoring_function, verbosity=2, random_state=2, cv=2)

# Fit the classifier to the training data
tpot_clf.fit(X_train, y_train)

# Score on the test set
print(tpot_clf.score(X_test, y_test))


# Create the tpot classifier
tpot_clf = TPOTClassifier(generations=2, population_size=4, offspring_size=3,
                          scoring='accuracy', cv=2, verbosity=2, random_state=42)

# Fit the classifier to the training data
tpot_clf.fit(X_train, y_train)

# Score on the test set
print(tpot_clf.score(X_test, y_test))


# Create the tpot classifier
tpot_clf = TPOTClassifier(generations=2, population_size=4, offspring_size=3,
                          scoring='accuracy', cv=2, verbosity=2, random_state=122)

# Fit the classifier to the training data
tpot_clf.fit(X_train, y_train)

# Score on the test set
print(tpot_clf.score(X_test, y_test))


# Create the tpot classifier
tpot_clf = TPOTClassifier(generations=2, population_size=4, offspring_size=3,
                          scoring='accuracy', cv=2, verbosity=2, random_state=99)

# Fit the classifier to the training data
tpot_clf.fit(X_train, y_train)

# Score on the test set
print(tpot_clf.score(X_test, y_test))


'''
Hyperparameters vs Parameters
- Hyperparameters are components of the model that you set. They are not learned during the modeling process.
- Parameters are not set by you. The algorithm will discover these for you
'''
