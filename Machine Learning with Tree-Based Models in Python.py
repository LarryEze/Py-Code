''' Classification and Regression Trees '''

'''
Decision tree for classification
Classification-tree
- Given a labeled dataset, a classification tree learns a sequence of if-else questions about individual features in order to infer the labels.
- Objective: infer class labels.
- In contrast to linear models, trees are able to capture non-linear relationships between features and labels
- In addition, trees don't require the features to be on the same scale (example: Standardization, ...)

Breast Cancer Dataset
Decision- tree Diagram 
                                        o 
                            [ Concave points_mean ] <= 0.051
                            /                      \
                        True                      False
                        /                             \
                        o                               o
                    [radius_mean] <= 14.98 [radius_mean] <= 11.345 
            /             \                               /                   \  
        True            False                           True                False
        /                   \                           /                       \
        o                   o                           o                       o
257 benign,             9 benign,                   4 benign,               15 benign,
7 malignant             11 malignant                0 malignant             152 malignant  
Predict -> benign       Predict -> malignant        Predict -> benign       Predict -> malignant

- The maximum number of branches separating the top from an extreme-end is known as the maximum depth which is equal to 2 here.

Classification-tree in scikit-learn
# Import DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
# Import train_test_split
from sklearn.model_selection import train_test_split
# Import accuracy_score
from sklearn.metrics import accuracy_score

# Split the dataset into 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=1)
# Instantiate dt
dt = DecisionTreeClassifier(max_depth=2, random_state=1)
# Fit dt to the training set
dt.fit(X_train, y_train)
# Predict the test set labels
y_pred = dt.predict(X_test)
# Evaluate the test-set accuracy
accuracy_score(y_test, y_pred) -> in

0.90350877192982459 -> out

- Set stratify to y so the train and test sets have the same proportion of class labels as the unsplit dataset.

Decision Regions
- Decision region: region in the feature space where all instances are assigned to one class label.
- Decision Boundary: surface separating different decision regions.
- The decision region of a linear-classifier boundary is a straight-line, while a classification-tree produces rectangular decision-regions in the feature-space. and it happens because at each split made by the tree, only one feature is involved.*
'''

# Import DecisionTreeClassifier from sklearn.tree
from sklearn.tree import DecisionTreeClassifier

# Instantiate a DecisionTreeClassifier 'dt' with a maximum depth of 6
dt = DecisionTreeClassifier(max_depth=6, random_state=SEED)

# Fit dt to the training set
dt.fit(X_train, y_train)

# Predict test set labels
y_pred = dt.predict(X_test)
print(y_pred[0:5])


# Import accuracy_score
from sklearn.metrics import accuracy_score

# Predict test set labels
y_pred = dt.predict(X_test)

# Compute test set accuracy  
acc = accuracy_score(y_pred, y_test)
print("Test set accuracy: {:.2f}".format(acc))


# Import LogisticRegression from sklearn.linear_model
from sklearn.linear_model import  LogisticRegression

# Instatiate logreg
logreg = LogisticRegression(random_state=1)

# Fit logreg to the training set
logreg.fit(X_train, y_train)

# Define a list called clfs containing the two classifiers logreg and dt
clfs = [logreg, dt]

# Review the decision regions of the two classifiers
plot_labeled_decision_regions(X_test, y_test, clfs)


'''
Classification tree Learning
Building Blocks of a Decision-Tree
- Decision-tree: Data structure consisting of a hierarchy of individual units called nodes
- Node: It is a point that involves either a question or a prediction.
* Three kinds of nodes:
** Root: It is the node at which the decision-tree starts growing. It has no parent node and involves a question that gives rise to 2 children nodes through two branches.
** Interval node: It is a node that has a parent and also involves a question that gives rise to 2 children nodes
** Leaf: It is a node that has no children and it has one parent node and involves no questions (It is where a prediction is made).

Information Gain (IG)
IG( f, sp ) = I(parent) - ( N(left) / N x I(left) + N(right) / N x I(right) )

f : feature
sp : Split-point

- Criteria to measure the impurity of a node I(node):
* gini index
* entropy

Classificaation-Tree Learning
- When an unconstrained tree is trained, the nodes are grown recursively i.e a node exists based on the state of its predecessors.
- At a non-leaf node, the data is split based o:
* feature f and split-point sp to maximize IG(node)
- If the IG(ode) = 0, declare the node a leaf
- If the maximum depth of a tree is constrained to 2 for example, all nodes having a depth of 2 will be declared leafs even if the IG obtained by splitting such nodes is not zero (0)

# Import DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
# Import train_test_split
from sklearn.model_selection import train_test_split
# Import accuracy_score
from sklearn.metrics import accuracy_score

# Split the dataset into 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=1)
# Instantiate dt, set 'criterion' to 'gini'
dt = DecisionTreeClassifier(criterion='gini', random_state=1)

# Fit dt to the training set
dt.fit(X_train, y_train)
# Predict the test set labels
y_pred = dt.predict(X_test)
# Evaluate the test-set accuracy
accuracy_score(y_test, y_pred) -> in

0.92105263157894735 -> out

NB: The gini index is slightly faster to compute and is the default criterion used in the DecisionTreeClassifier model of scikit-learn
'''

# Import DecisionTreeClassifier from sklearn.tree
from sklearn.tree import DecisionTreeClassifier

# Instantiate dt_entropy, set 'entropy' as the information criterion
dt_entropy = DecisionTreeClassifier(max_depth=8, criterion='entropy', random_state=1)

# Fit dt_entropy to the training set
dt_entropy.fit(X_train, y_train)


# Import accuracy_score from sklearn.metrics
from sklearn.metrics import accuracy_score

# Use dt_entropy to predict test set labels
y_pred= dt_entropy.predict(X_test)

# Evaluate accuracy_entropy
accuracy_entropy = accuracy_score(y_test, y_pred)

# Print accuracy_entropy
print(f'Accuracy achieved by using entropy: {accuracy_entropy:.3f}')

# Print accuracy_gini
print(f'Accuracy achieved by using the gini index: {accuracy_gini:.3f}')


'''
Decision tree for regression
Auto-mpg Dataset

    mpg     displ   hp  weight  accel   origin  size
0  18.0     250.0   88    3139   14.5       US  15.0
1   9.0     304.0  193    4732   18.5       US  20.0
2  36.1      91.0   60    1800   16.4     Asia  10.0
3  18.5     250.0   98    3525   19.0       US  15.0
4  34.3      97.0   78    2188   15.8   Europe  10.0
5  32.9     119.0  100    2615   14.8     Asia  10.0

Regression-Tree in scikit-learn
# Import DecisionTreeRegressor
from sklearn.tree import DecisionTreeRegressor
# Import train_test_split
from sklearn.model_selection import train_test_split
# Import mean_squared_error as MSE
from sklearn.metrics import mean_squared_error as MSE

# Split the dataset into 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)
# Instantiate a DecisionTreeRegressor 'dt'
dt = DecisionTreeRegressor(max_depth=4, min_samples_leaf=0.1, random_state=3)

- (min_samples_leaf = 0.1) imposes a stopping condition in which each leaf has to contain at least 10% of the training data

# Fit dt to the training set
dt.fit(X_train, y_train)
# Predict the test set labels
y_pred = dt.predict(X_test)
# Compute test-set MSE
mse_dt = MSE(y_test, y_pred) 
# Compute test-set RMSE
rmse_dt = mse_dt**(1/2)
# Print rmse_dt
print(rmse_dt) -> in

5.1023068889 -> out

- The regression tree shows a greater flexibility and is able to capture the non-linearity, though not fully
'''

# Import DecisionTreeRegressor from sklearn.tree
from sklearn.tree import DecisionTreeRegressor

# Instantiate dt
dt = DecisionTreeRegressor(max_depth=8, min_samples_leaf=0.13, random_state=3)

# Fit dt to the training set
dt.fit(X_train, y_train)


# Import mean_squared_error from sklearn.metrics as MSE
from sklearn.metrics import mean_squared_error as MSE

# Compute y_pred
y_pred = dt.predict(X_test)

# Compute mse_dt
mse_dt = MSE(y_pred, y_test)

# Compute rmse_dt
rmse_dt = mse_dt ** (1/2)

# Print rmse_dt
print("Test set RMSE of dt: {:.2f}".format(rmse_dt))


# Predict test set labels 
y_pred_lr = lr.predict(X_test)

# Compute mse_lr
mse_lr = MSE(y_pred_lr, y_test)

# Compute rmse_lr
rmse_lr = mse_lr ** (1/2)

# Print rmse_lr
print('Linear Regression test set RMSE: {:.2f}'.format(rmse_lr))

# Print rmse_dt
print('Regression Tree test set RMSE: {:.2f}'.format(rmse_dt))


''' The Bias-Variance Tradeoff '''

'''
Generalization Error
Supervised Learning - Under the Hood
* In supervised learning, you make the assumption that there's a mapping between features and labels.
i.e y = f(x), f is unknown

Goals of Supervised Learning
- Find a model fhat that best approximates f
- Fhat can be Logistic Regression, Decision Tree, Neural Network ...
- Discard noise as much as possible
- End goal: fhat should achieve a low predictive error on unseen datasets

Difficulties in Approximating f
- Overfitting: It's when fhat fits the noise in the training set
- Underfitting: It's when fhat is not flexible enough to approximate f

Overfitting
- When a model overfits the training set, its predictive power on unseen datasets is pretty low.
- The model memorized the noise present in the training set
- Such model achieves a low training set error and a high test set error.

Underfitting
- The training set error is roughly equal to the test set error, However, both errors are relatively high.

Generalization Error
- It tells you how much it generalizes on unseen data.
- It can be decomposed into 3 terms: bias, variance and irreducible error
- Generalization Error of fhat = bias ** 2 + variance + irreducible error
* The irreducible error is the error contribution of noise.

Bias
- Bias: Its an error term that tells you, on average, how much fhat and f are different
- High bias models lead to underfitting

Variance
- Variance: it tells you how much fhat is inconsistent over different training sets.
- High variance models lead to overfitting

Model Complexity
- Model Complexity: It sets the flexibility to approximate the true function f.
- Example: Increasing the maximum-tree-depth increases the, increases the complexity of a decision tree
- When the model complexity increases, the variance increases while the bias decreases
- When the model complexity decreases, the variance decreases while the bias increases

Bias-Variance Tradeoff
- Bias-Variance Tradeoff: its when as bias increases, variance decreases and vice versa
'''


'''
Diagnose bias and variance problems
Estimating the Generalization Error
- How do we estmate the generalization error of a mode?
- This cannot be done directly because:
* f is unknown
* usually we only have one dataset
- we don't have access to the error term due to noise (noise is unpredictable)

Solution:
- First split the data to training and test sets.
- fit fhat to the training set
- evaluate the error of fhat on the unseen test set
- The generalization error of fhat is roughly approximated by fhat error on the test set 

Better Model Evaluation with Cross-Validation
- test set should be kept untouched until one is confident about fhats performance.
- It should only be used to evaluate fhat's final performance or error.
- Evaluating fhat's performance on the training set may produce an optimistic estimation of the error because fhat was already exposed to the training set when it was fit
- To obtain a reliable estimate of fhat's performance, we should use a technique called cross-validation or CV                                                                                                                                
- Solution -> Cross-Validation (CV):
* K-Fold CV
* Hold-Out CV

K-Fold CV
CV error = mean of the obatained errors.

CV error = ( E1 + ... + E10 ) / 10

Diagnose Variance Problems
- If fhat suffers from high variance: CV error of fhat > training set error of fhat
- Fhat is said to overfit the training set. To remedy overfitting:
* Decrease model complexity,
* for example: decrease max depth, increase min samples per leaf, ...
* Gather more data to train fhat

Diagnose Bias Problems
- If fhat suffers from high bias: CV error of fhat is roughly equal to the training error of fhat but much greater than the desired error
- fhat is said to underfit the training set. To remedy underfitting:
* Increase model complexity
* for example: increase max depth, decrease min samples per leaf, ..
* Gather more relevant features for the problem

K-Fold CV in sklearn on Auto Dataset
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE
from sklearn.model_selection import cross_val_score

# Set seed for reproducibility
SEED = 123
# Split data into 70% train and 30% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=SEED)
# Instantiate decision tree regressor and assign it to 'dt'
dt = DecisionTreeRegressor(max_depth=4, min_samples_leaf=0.14, random_state=SEED)
# Evaluate the list of MSE obtained by 10-fold CV
# Set n_jobs to -1 in order to exploit all CPU cores in computation
MSE_CV = - cross_val_score(dt, X_train, y_train, cv=10, scoring='neg_mean_squared_error', n_jobs=-1)

# Fit dt to the training set
dt.fit(X_train, y_train)
# Predict the labels of training set
y_predict_train = dt.predict(X_train)
# Predict the labels of test set
y_predict_test = dt.predict(X_test)
# CV MSE
print('CV MSE: {:.2f}'.format(MSE_CV.mean())) -> in

CV MSE: 20.51 -> out

# Training set MSE
print('Train MSE: {:.2f}'.format(MSE(y_train, y_predict_train))) -> in

Train MSE: 15.30 -> out

# Test set MSE
print('Test MSE: {:.2f}'.format(MSE(y_test, y_predict_test))) -> in

Test MSE: 20.92 -> out

- Given that the training set error is smaller than the CV-error, we can deduce that dt overfits the training set and that it suffers from high variance.

-Note that in scikit-learn, the MSE of a model can be computed as follows:
    MSE_model = mean_squared_error(y_true, y_predicted)
'''

# Import train_test_split from sklearn.model_selection
from sklearn.model_selection import train_test_split 

# Set SEED for reproducibility
SEED = 1

# Split the data into 70% train and 30% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=SEED)

# Instantiate a DecisionTreeRegressor dt
dt = DecisionTreeRegressor(max_depth=4, min_samples_leaf=0.26, random_state=SEED)


# Compute the array containing the 10-folds CV MSEs
MSE_CV_scores = - cross_val_score(dt, X_train, y_train, cv=10, scoring='neg_mean_squared_error', n_jobs=-1)

# Compute the 10-folds CV RMSE
RMSE_CV = (MSE_CV_scores.mean())**(1/2)

# Print RMSE_CV
print('CV RMSE: {:.2f}'.format(RMSE_CV))


# Import mean_squared_error from sklearn.metrics as MSE
from sklearn.metrics import mean_squared_error as MSE 

# Fit dt to the training set
dt.fit(X_train, y_train)

# Predict the labels of the training set
y_pred_train = dt.predict(X_train)

# Evaluate the training set RMSE of dt
RMSE_train = (MSE(y_train, y_pred_train))**(0.5)

# Print RMSE_train
print('Train RMSE: {:.2f}'.format(RMSE_train))


'''
Ensemble Learning
Advantages of CARTs (Classification and Regression Trees)
- They are simple to understand
- Their output is easy to interpret
- They are easy to use
- Gives Flexibility: Its the ability to describe nonlinear dependencies between features and labels
- Preprocessing: It doesn't need a lot of features preprocessing to train (i.e no need to standardize or normalize features, ...)

Limitations of CARTs
- Classification: can only produce orthogonal decision boundaries
- They are also very sensitive to small variations in the training set
- High variance: unconstrained CARTs may overfit the training set
- A Solution that takes advantage of the flexibility of CARTs while reducing their tendency to memorize noise is ensemble learning

Ensemble Learning
- As a first step, different models are trained on the same datset
- Let each model make its own predictions
- Meta-model: It aggregates the predictions of individual models and outputs a final prediction
- Final prediction: It is more robust and less prone to errors than each individual model
- Best results: It is obtained when the models are skillful but in different ways

Ensemble Learning: A Visual Explanation

                                Final ensemble prediction
                                            |
                                        Meta-model
                        |                |              |         |
Predictions             P1               P2             P3        P4
                        |                |              |         |
                Decision Tree   Logistic Regression   KNN       Other...
                        \                |              |         /
Training                 \               |              |        /        
                                            Traing set

Ensemble Learning in Practice: Voting Classifier
- Binary classification task
- N classifiers make predictions: P1, P2, ..., P(n) wit P(i) = 0 or 1
- Meta-model prediction: It outputs the final prediction by hard voting.

Hard Voting
                                        1
                                        |
                                Voting Classifier
                        |               |               |
Predictions             1               0               1
                        |               |               |      
                Decision Tree   Logistic Regression     KNN
                        \               |               /
                                    New data point

Voting Classifier in sklearn (Breast-Cancer dataset)
# Import functions to compute accuracy and split data
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Import models, including VotingClassifier meta-model
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.ensemble import VotingClassifier

# Set seed for reproducibility
SEED = 1

# Split data into 70% train and 30% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3, random_state= SEED)
# Instantiate individual classifiers
lr = LogisticRegression(random_state=SEED)
knn = KNN()
dt = DecisionTreeClassifier(random_state=SEED)
# Define a list called classifier that contains the tuples (classifier_name, classifier)
classifiers = [('Logistic Regression', lr), ('K Nearest Neighbours', knn), ('Classification Tree', dt)]

# Iterate over the defined list of tuples containing the classifiers
for clf_name, clf in classifiers:
    # fit clf to the training set
    clf.fit(X_train, y_train)

    # Predict the labels of the test set
    y_pred = clf.predict(X_test)

    # Evaluate the accuracy of clf on the test set
    print('{:s} : {:.3f}'.format(clf_name, accuracy_score(y_test, y_pred))) -> in

Logistic Regression: 0.947
K Nearest Neighbours: 0.930
Classification Tree: 0.930 -> out

# Instantiate a VotingClassifier 'vc'
vc = VotingClassifier(estimators = classifiers)

# Fit 'vc' to the training set and predict test set labels
vc.fit(X_train, y_train)
y_pred = vc.predict(X_test)

# Evaluate the test-set accuracy of 'vc'
print('Voting Classifier: {.3f}'.format(accuracy_score(y_test, y_pred))) -> in

Voting Classifier: 0.953 -> out
'''

# Set seed for reproducibility
SEED=1

# Instantiate lr
lr = LogisticRegression(random_state=SEED)

# Instantiate knn
knn = KNN(n_neighbors=27)

# Instantiate dt
dt = DecisionTreeClassifier(min_samples_leaf=0.13, random_state=SEED)

# Define the list classifiers
classifiers = [('Logistic Regression', lr), ('K Nearest Neighbours', knn), ('Classification Tree', dt)]


# Iterate over the pre-defined list of classifiers
for clf_name, clf in classifiers:    

    # Fit clf to the training set
    clf.fit(X_train, y_train)    

    # Predict y_pred
    y_pred = clf.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_pred, y_test) 

    # Evaluate clf's accuracy on the test set
    print('{:s} : {:.3f}'.format(clf_name, accuracy))


# Import VotingClassifier from sklearn.ensemble
from sklearn.ensemble import VotingClassifier 

# Instantiate a VotingClassifier vc
vc = VotingClassifier(estimators=classifiers)     

# Fit vc to the training set
vc.fit(X_train, y_train)   

# Evaluate the test set predictions
y_pred = vc.predict(X_test)

# Calculate accuracy score
accuracy = accuracy_score(y_pred, y_test)
print('Voting Classifier: {:.3f}'.format(accuracy))


''' Bagging and Random Forests '''

'''
Bagging
Ensemble Methods
- Voting classifier:
* Its an ensemble of models that are fit to the same training set
* Its done using different algorithms
* Final predictions were obtained by majority voting
- Bagging
* Its an ensemble of models that is formed using the same training algorithm
* The models are not trained on the entire training set but instead, each model is trained on a different subset of the data
* Bagging: Bootstrap Aggregation
* It uses a technique known as the bootstrap
* It has the effect of reducing the variance of individual models in the ensemble

Bagging: Prediction
                    Final Prediction
                            |
                        Bagging
Predictions         |           |
                Model 1 .... Model N
                    \         /
                    New instance

Bagging: Classification & Regression
- Classification:
* The final prediction is obtained by majority voting
* The corresponding classifier in scikit-learn is BaggingClassifier
- Regression:
* The final prediction is the average of the predictions made by the individual models forming the ensemble
* The corresponding classifier in scikit-learn is BaggingRegressor

Bagging Classifier in sklearn (Breast-Cancer dataset)
# Import models and utility functions
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Set seed for reproducibility
SEED = 1

# Split data into 70% train and 30% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=SEED)
# Instantiate a classification tree 'dt'
dt = DecisionTreeClassifier(max_depth=4, min_samples_leaf=0.16, random_state=SEED)
# Instantiate a BaggingClassifier 'bc'
bc = BaggingClassifier(base_estimator=dt, n_estimators=300, n_jobs=-1) 
# Fit 'bc' to the training set
bc.fit(X_train, y_train)
# Predict test set labels
y_pred = bc.predict(X_test)

# Evaluate and print test-set accuracy
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy of Bagging Classifier: {:.3f}'.format(accuracy)) -> in

Accuracy of Bagging Classifier: 0.936 -> out
'''

# Import DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier 

# Import BaggingClassifier
from sklearn.ensemble import BaggingClassifier 

# Instantiate dt
dt = DecisionTreeClassifier(random_state=1)

# Instantiate bc
bc = BaggingClassifier(base_estimator=dt, n_estimators=50, random_state=1)


# Fit bc to the training set
bc.fit(X_train, y_train)

# Predict test set labels
y_pred = bc.predict(X_test)

# Evaluate acc_test
acc_test = accuracy_score(y_pred, y_test)
print('Test set accuracy of bc: {:.2f}'.format(acc_test)) 


'''
Out of Bag Evaluation
Bagging
- some instances may be sampled several times for one model
- other instances may not be sampled at all

Out Of Bag (OOB) instances
- On average, for each model, 63% of the training instances are sampled.
- The remaining 37% constitute the OOB instances
- OOB instances are used to estimate the performance of the ensemble without the need for Cross-Validation.

OOB Evaluation

                                OOB score = ( OOB1 + ... + OOB(n) ) / N
                                    /                               \
                                OOB1       ...                      OOB(n)
                                    |      ...       |
                Model 1                            Model N
                    |                |               |                  |
                Train             Evaluate         Train             Evaluate  
                    |                |               |                  |
            Bootstrap Samples   OOB Samples   Bootstrap Samples     OOB Samples
                    \                 \             /                   /
                                        Training set

OOB Evaluation in sklearn (Breast Cancer Dataset)
# Import models and split utility function
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Set seed for reproducibility
SEED = 1

# Split data into 70% train and 30% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3, stratify= y, random_state= SEED)
# Instantiate a classification-tree 'dt'
dt = DecisionTreeClassifier(max_depth=4, min_samples_leaf=0.16, random_state=SEED)
# Instantiate a BaggingClassifier 'bc'; set oob_score = True
bc = BaggingClassifier(base_estimator=dt, n_estimators=300, oob_score=True, n_jobs=-1)

- oob_score is set to True in order to evaluate the OOB-accuracy of bc after training
- In scikit learn, the oob_score corresponds to the accuracy for classifiers and the r-squared score for regressors 

# Fit 'bc' to the training set
bc.fit(X_train, y_train)
# Predict the test set labels
y_pred = bc.predict(X_test)
# Evaluate test set accuracy
test_accuracy = accuracy_score(y_test, y_pred)
# Extract the OOB accuracy from 'bc'
oob_accuracy = bc.oob_score_

# Print test set accuracy
print('Test set accuracy: {:.3f}'.format(test_accuracy)) -> in

Test set accuracy: 0.936 -> out

# Print OOB accuracy
print('OOB accuracy: {:.3f}'.format(oob_accuracy)) -> in

OOB accuracy: 0.925 -> out

- These results highlight how OOB-evaluation can be an efficient technique to obtain a performance estimate of a bagged-ensemble on unseen data without performing cross-validation.
'''

# Import DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier

# Import BaggingClassifier
from sklearn.ensemble import BaggingClassifier 

# Instantiate dt
dt = DecisionTreeClassifier(min_samples_leaf=8, random_state=1)

# Instantiate bc
bc = BaggingClassifier(base_estimator=dt, n_estimators=50, oob_score=True, random_state=1)


# Fit bc to the training set 
bc.fit(X_train, y_train)

# Predict test set labels
y_pred = bc.predict(X_test)

# Evaluate test set accuracy
acc_test = accuracy_score(y_pred, y_test)

# Evaluate OOB accuracy
acc_oob = bc.oob_score_

# Print acc_test and acc_oob
print('Test set accuracy: {:.3f}, OOB accuracy: {:.3f}'.format(acc_test, acc_oob))


'''
Random Forests (RF)
Bagging
- Base estimator: It can be any model including a Decision Tree, Logistic Regression or even neural network
- Each estimator is trained on a distinct bootstrap sample drawn from the training set 
- Estimators use all available features for training and prediction

Further Diversity with random Forests
- Base estimator: It uses a Decision tree
- Each estimator is trained on a different bootstrap sample having the same size as the training set
- RF introduces further randomization in the training of individual trees
- When each tree is trained, only d features can be sampled at each node without replacement
( d < total number of features)
- The node is then split using the sampled feature that maximizes information gain
- In scikit-learn d defaults to the square-root of the number of features

Random Forests: Prediction

                            Final Prediction
                                    |
                            Random Forest
Predictions             |                   |
                Decision Tree 1 . . . Decision Tree K
                        \                   / 
                            New instance

Random Forest: Classification & Regression
- Classification:
* The final prediction is made by majority voting
* The corresponding scikit-learn class is RandomForestClassifier
- Regression:
* The final prediction is the average of all the labels predicted by the base estimators
* The corresponding scikit-learn class is RandomForestRegressor
- In general, Random Forests achieves a lower variance than individual trees.

Random Forest Regressor in sklearn (auto dataset)
# Basic imports
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE
# Set seed for reproducibility
SEED = 1

# Split dataset into 70% train and 30% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3, random_state= SEED)
# Instantiate a random forests regressor 'rf' with 400 estimators
rf = RandomForestRegressor(n_estimators= 400, min_samples_leaf= 0.12, random_state= SEED)
# Fit 'rf' to the training set
rf.fit(X_train, y_train)
# Predict the test set labels 'y_pred'
y_pred = rf.predict(X_test)

# Evaluate the test set RMSE
rmse_test = MSE(y_test, y_pred) ** (1/2)
# Print the test set RMSE
print('Test set RMSE of rf: {:.2f}'.format(rmse_test)) -> in

Test set RMSE of rf: 3.98 -> out

Feature Importance
- Tree-based methods: enables measuring the importance of each feature in prediction.
- In sklearn:
* it is measured by how much the tree nodes use a particular feature (weighted average) to reduce impurity
- The importance of a feature is expressed as a percentage indicating the weight of that feature in training and prediction.
- The features importances can be accessed using the attribute feature_importance_

Feature Importance in sklearn
import pandas as pd
import matplotlib.pyplot as plt

# Create a pd.Series of features importances
importances_rf = pd.Series(rf.feature_importances_, index = X.columns)

# Sort importances_rf 
sorted_importances_rf = importances_rf.sort_values()

# Make a horizontal bar plot
sorted_importances_rf.plot(kind='barh', color='lightgreen'); plt.show()
'''

# Import RandomForestRegressor
from sklearn.ensemble import RandomForestRegressor 

# Instantiate rf
rf = RandomForestRegressor(n_estimators=25, random_state=2)

# Fit rf to the training set    
rf.fit(X_train, y_train) 


# Import mean_squared_error as MSE
from sklearn.metrics import mean_squared_error as MSE

# Predict the test set labels
y_pred = rf.predict(X_test)

# Evaluate the test set RMSE
rmse_test = MSE(y_test, y_pred) ** (1 / 2)

# Print rmse_test
print('Test set RMSE of rf: {:.2f}'.format(rmse_test))


# Create a pd.Series of features importances
importances = pd.Series(data=rf.feature_importances_, index= X_train.columns)

# Sort importances
importances_sorted = importances.sort_values()

# Draw a horizontal barplot of importances_sorted
importances_sorted.plot(kind='barh', color='lightgreen')
plt.title('Features Importances')
plt.show()


''' Boosting '''

'''
Adaboost
Boosting
- Boosting: It refers to an ensemble method in which many predictors are trained and each predictor learns from the errors of its predecessor (i.e combining several weak learners to form a strong learner).
- Weak learner: It is a Model doing slightly better than random guessing
* Examples of weak learner: a Decision tree with a maximum-depth of one (aka Decision stump)

- In boosting, an ensemble of predictors are trained sequentially and each predictor tries to correct the errors made by its predecessor
- Most popular boosting methods:
* AdaBoost,
* Gradient Boosting

Adaboost
- Adaboost stands for Adaptive Boosting
- Each predictor pays more attetion to the instances wrongly predicted by its predecessor
- It is achieved by constantly changing the weights of training instances
- each predictor is assigned a coefficient alpha
- alpha weighs its contribution in the ensemble's final prediction
- alpha depends on the predictor's training error

Learning Rate
- It is an important parameter used in training and its represented by eta
- Eta is a number between 0 and 1
    Learning rate: 0 < eta <=1
- It is used to shrink the coefficient alpha of a trained predictor
- A smaller value of eta should be compensated by a greater number of estimators

AdaBoost: Prediction
- Classification:
* Each predictor predicts the label of the new instance and the ensemble's prediction is obtained by weighted majority voting
* In sklearn: AdaBoostClassifier
- Regression:
* Each predictor predicts the label of the new instance and the ensemble's prediction is obtained by performing weighted average
* In sklearn: AdaBoostRegressor
- Its important to note that individual predictors need not to be CARTs, However they are used most of the time in boosting because of their high variance.

AdaBoost Classification in sklearn (Breast Cancer dataset)
# Import models and utility functions
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

# Set seed for reproducibility
SEED = 1

# Split data into 70% train and 30% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3, stratify= y, random_state= SEED)

# Instantiate a classification-tree 'dt'
dt = DecisionTreeClassifier(max_depth=1, random_state=SEED)

# Instantiate an AdaBoost classifier 'adb_clf'
adb_clf = AdaBoostClassifier(base_estimator=dt, n_estimator=100)

# Fit 'adb_clf' to the training set
adb_clf.fit(X_train, y_train)

# Predict the test set probabilities of positive class
y_pred_proba = adb_clf.predict(X_test)[:, 1] 

# Evaluate test-set roc_auc_score
adb_clf_roc_auc_score = roc_auc_score(y_test, y_pred_proba)
# Print adb_clf_roc_auc_score
print('ROC AUC score: {:.2f}'.format(adb_clf_roc_auc_score)) -> in

ROC AUC score: 0.99 -> out
'''

# Import DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier

# Import AdaBoostClassifier
from sklearn.ensemble import AdaBoostClassifier 

# Instantiate dt
dt = DecisionTreeClassifier(max_depth=2, random_state=1)

# Instantiate ada
ada = AdaBoostClassifier(base_estimator=dt, n_estimators=180, random_state=1)


# Fit ada to the training set
ada.fit(X_train, y_train)

# Compute the probabilities of obtaining the positive class
y_pred_proba = ada.predict_proba(X_test)[:, 1]


# Import roc_auc_score
from sklearn.metrics import roc_auc_score 

# Evaluate test-set roc_auc_score
ada_roc_auc = roc_auc_score(y_test, y_pred_proba)

# Print roc_auc_score
print('ROC AUC score: {:.2f}'.format(ada_roc_auc))


'''
Gradient Boosting (GB)
- it is a popular boosting algorithm that has a proven track record of winning many machine learning competitions.

Gradient Boosted Trees
- Each predictor in the ensemble corrects its predecessor's error (i.e sequential correction of predecessor's errors).
- In contrast to AdaBoost, the weights of the training instances are not tweaked
- Instead, each predictor is trained using the residual errors of its predecessor as labels
- Gradient Boosted Trees: a CART is used as a base learner

Shrinkage
- It is an important parameter used in training gradient boosted trees
- It refers to the fact that the prediction of each tree in the ensemble is shrinked after it is multiplied by a learning rate eta which is a number between 0 and 1
- A smaller value of eta should be compensated by a greater number of estimators in order for the ensemble to reach a certain performance.

Gradient Boosted Trees: Prediction
- Regression:
* y(pred) = y1 + eta*r1 + . . . + eta*r(n)
* In sklearn: GradientBoostingRegressor
- Classification:
* In sklearn: GradientBoostingClassifier

Gradient Boosting in sklearn (auto dataset)
# Import models and utility functions
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE

# Set seed for reproducibility
SEED = 1

# Split data into 70% train and 30% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3, random_state= SEED)

# Instantiate a GradientBoostingRegressor 'gbt'
gbt = GradientBoostingRegressor(n_estimators=300, max_depth=1, random_state=SEED)

# Fit 'gbt' to the training set
gbt.fit(X_train, y_train)

# Predict the test set labels
y_pred = gbt.predict(X_test)

# Evaluate the test-set RMSE
rmse_test = MSE(y_test, y_pred) ** (1/2)

# Print the test set RMSE
print('Test set RMSE: {:.2f}'.format(rmse_test)) -> in

Test set RMSE: 4.01 -> out
'''

# Import GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingRegressor 

# Instantiate gb
gb = GradientBoostingRegressor(max_depth= 4, n_estimators= 200, random_state=2)


# Fit gb to the training set
gb.fit(X_train, y_train)

# Predict test set labels
y_pred = gb.predict(X_test)


# Import mean_squared_error as MSE
from sklearn.metrics import mean_squared_error as MSE

# Compute MSE
mse_test = MSE(y_test, y_pred) 

# Compute RMSE
rmse_test = mse_test ** (1/2)

# Print RMSE
print('Test set RMSE of gb: {:.3f}'.format(rmse_test))


'''
Stochastic Gradient Boosting (SGB)

Gradient Boosting: Cons
- GB involves an exhaustive search procedure
- Each tree (CART) is trained to find the best split points and features
- This procedure may lead to CARTs using the same split points and maybe the same features

Stochastic Gradient Boosting (SGB)
- Each CART is trained on a random subset of rows of the training data.
- The sampled instances (40%-80% of the training set) are sampled without replacement
- At the level of each node, features are sampled without replacement when choosing the best split-points.
- Result: It creates further diversity in the ensemble
- Effect: It adds more variance to the ensemble of trees

Stochastic Gradient Boosting in sklearn (auto dataset)
# Import models and utility functions
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE

# Set seed for reproducibility
SEED = 1

# Split data into 70% train and 30% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3, random_state= SEED)

# Instantiate a stochastic GradientBoostingRegressor 'sgbt'
sgbt = GradientBoostingRegressor(max_depth=1, subsample=0.8, max_features=0.2, n_estimators=300, random_state=SEED)

- subsample = 0.8 allows for each tree to sample 80% of the data for training
- max_features=0.2 allows each tree to use 20% of available features to perform the best split 

# Fit 'gbt' to the training set
sgbt.fit(X_train, y_train)

# Predict the test set labels
y_pred = sgbt.predict(X_test)

# Evaluate the test-set RMSE
rmse_test = MSE(y_test, y_pred) ** (1/2)

# Print the test set RMSE
print('Test set RMSE: {:.2f}'.format(rmse_test)) -> in

Test set RMSE: 3.95 -> out
'''

# Import GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingRegressor

# Instantiate sgbr
sgbr = GradientBoostingRegressor(max_depth=4, subsample=0.9, max_features=0.75, n_estimators=200, random_state=2)


# Fit sgbr to the training set
sgbr.fit(X_train, y_train)

# Predict test set labels
y_pred = sgbr.predict(X_test)


# Import mean_squared_error as MSE
from sklearn.metrics import mean_squared_error as MSE 

# Compute test set MSE
mse_test = MSE(y_test, y_pred)

# Compute test set RMSE
rmse_test = mse_test ** (1/2)

# Print rmse_test
print('Test set RMSE of sgbr: {:.3f}'.format(rmse_test))


''' Model Tuning '''

'''
Tuning a CART's Hyperparameters
- To obtain a better performance, the hyperparameters of a machine learning should be tuned.

Hyperparameters
Machine learning model:
-Parameters: Its learned from data through training
* CART examples: split-point of a node, split-feature of a node,
- Hyperparameters: Its not learned from data; they should be set prior to training
* CART example: max_depth, min_sample_leaf, splitting criterion

What is hyperparameter tuning?
- Problem: It consists of searching for the set of optimal hyperparameters for the learning algorithm
- Solution: It involves finding the set of optimal hyperparameters yielding an optimal model
- Optimal model: This yields an optimal score
- score: This function measures the agreement between true labels and a model's predictions.
* In sklearn defaults to accuracy (classification) and R^2 (regression)
- Cross validation is used to estimate the generalization performance.

Why tune hyperparameters?
- In sklearn, a model default hyperparameters are not optimal for all problems
- Hyperparameters should be tuned to obtain the best model performance.

Approaches to hyperparameter tuning
- Grid Search
- Random Search
- Bayesian Optimization
- Genetic Algorithms
* ...

Grid search cross validation
- First, manually set a grid of discrete hyperparameter values
- Set a metric for scoring model performance
- Search exhaustively through the grid
- For each set of hyperparameters, evaluate each model's CV score
- The optimal hyperparameters are those of the model achieving the best CV score
- NB: Grid-search suffers from the curse of dimensionality i.e the bigger the grid, the longer it takes to find the solution.

Grid search cross validation: example
- Hyperparameters grids:
* max_depth = {2, 3, 4}
* min_samples_leaf = {0.05, 0.1}
- Hyperparameter space = { (2, 0.05), (2, 0.1), (3, 0.05), ... }
- CV scores = { score(2, 0.05), ... }
- Optimal hyperparameters = set of hyperparameters corresponding to the model achieving the best CV score

Inspecting the hyperparameters of a CART in sklearn
# Import DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier

# Set seed to 1 for reproducibility
SEED = 1

# Instantiate a DecisionTreeClassifier 'dt'
dt = DecisionTreeClassifier(random_state=SEED)

# Print out 'dt's hyperparameters
print(dt.get_params()) -> in

{   'ccp_alpha': 0.0, 
    'class_weight': None, 
    'criterion': 'gini', 
    'max_depth': None, 
    'max_features': None, 
    'max_leaf_nodes': None, 
    'min_impurity_decrease': 0.0, 
    'min_impurity_split': None, 
    'min_samples_leaf': 1, 
    'min_samples_split': 2, 
    'min_weight_fraction_leaf': 0.0,
    'presort': False, 
    'random_state': 1, 
    'splitter': 'best'  } -> out

NB: max_features is the number of features to consider when looking for the best split.
* When it is a float, it is interpreted as a percentage

# Import GridSearchCV
from sklearn.model_selection import GridSearchCV
# Define the grid of hyperparameters 'params_dt'
params_dt = { 'max_depth': [3, 4, 5, 6], 'min_samples_leaf': [0.04, 0.06, 0.08], 'max_features': [0.2, 0.4, 0.6, 0.8] }
# Instantiate a 10-fold CV grid search object 'grid_dt'
grid_dt = GridSearchCV(estimator=dt, param_grid=params_dt, scoring='accuracy', cv=10, n_jobs=-1)
# Fit 'grid_dt' to the training data
grid_dt.fit(X_train, y_train)

Extracting the best hyperparameters
# Extract best hyperparameters from'grid_dt'
best_hyperparams = grid_dt.best_params_
print('Best hyperparameters:\n', best_hyperparams) -> in

Best hyperparameters:
{'max_depth': 3, 'max_features': 0.4, 'min_samples_leaf': 0.06} -> out

# Extract best CV score from'grid_dt'
best_CV_score = grid_dt.best_score_
print('Best CV accuracy:'.format(best_CV_score)) -> in

Best CV accuracy: 0.938 -> out

Extracting the best estimator
# Extract best model from 'grid_dt'
best_model = grid_dt.best_estimator_

# Evaluate test set accuracy
test_acc = best_model.score(X_test, y_test)

# Print test set accuracy
print('Test set accuracy of best model: {:.3f}'format(test_acc)) -> in

Test set accuracy of best model: 0.947 -> out
'''

# Define params_dt
params_dt = {'max_depth': [2, 3, 4], 'min_samples_leaf': [0.12, 0.14, 0.16, 0.18]}


# Import GridSearchCV
from sklearn.model_selection import GridSearchCV 

# Instantiate grid_dt
grid_dt = GridSearchCV(estimator=dt, param_grid=params_dt, scoring='roc_auc', cv=5, n_jobs=-1)


# Import roc_auc_score from sklearn.metrics
from sklearn.metrics import roc_auc_score 

# Extract the best estimator
best_model = grid_dt.best_estimator_

# Predict the test set probabilities of the positive class
y_pred_proba = grid_dt.predict_proba(X_test)[:, 1]

# Compute test_roc_auc
test_roc_auc = roc_auc_score(y_test, y_pred_proba)

# Print test_roc_auc
print('Test set ROC AUC score: {:.3f}'.format(test_roc_auc))


'''
Tuning a RF's Hyperparameters
Random Forests Hyperparameters
- CART hyperparameters
- number of estimators
- bootstrap
- ...

Tuning is expensive
Hyperparameter tuning:
- It is computationally expensive.
- It may sometimes lead only to very slight improvement of the model's performance
* It is desired to weigh the impact of tuning on the pipeline of the data analysis project as a whole in order to understand if it is worth pursuing

Inspecting RF hyperparameters in sklearn
# Import RandomForestRegressor
from sklearn.ensemble import RandomForestRegressor

# Set seed for reproducibility
SEED = 1

# Instantiate a random forest regressor 'rf'
rf = RandomForestRegressor(random_state=SEED)

# Inspect 'rf' s hyperparameters
rf.get_params() -> in

{   'bootstrap': True, 
    'criterion': 'mse',
    'class_weight': None, 
    'max_depth': None, 
    'max_features': None, 
    'max_leaf_nodes': None, 
    'min_impurity_decrease': 0.0, 
    'min_impurity_split': None, 
    'min_samples_leaf': 1, 
    'min_samples_split': 2, 
    'min_weight_fraction_leaf': 0.0,
    'n_estimators': 10,
    'n_jobs': -1,
    'oob_score': False,
    'random_state': 1, 
    'verbose': 0,
    'warm_start': False } -> out

# Basic import
from sklearn.metrics import mean_squared_error as MSE
from sklearn.model_selection import GridSearchCV
# Define the grid of hyperparameters 'params_rf'
params_rf = { 'n_estimators': [300, 400, 500], 'max_depth': [4, 6, 8], 'min_samples_leaf': [0.1, 0.2], 'max_features': ['log2', 'sqrt'] }
# Instantiate 'grid_rf'
grid_rf = GridSearchCV(estimator=rf, param_grid=params_rf, scoring='neg_mean_squared_error', cv=3, verbose=1, n_jobs=-1)

- NB: verbose controls verbosity; the higher its value, the more messages are printed during fitting.

# Fit 'grid_rf' to the training set
grid_rf.fit(X_train, y_train) -> in

Fitting 3 folds for each of 36 candidates, totalling 108 fits
[Parallel(n_jobs=-1)]: Done 42 tasks | elapsed: 10.0s
[Parallel(n_jobs=-1)]: Done 108 tasks | elapsed: 24.3s finished
RandomForestRegressor(bootstrap= True, criterion= 'mse',  max_depth= 4, max_features= 'log2', max_leaf_nodes= None, min_impurity_decrease= 0.0, min_impurity_split= None, min_samples_leaf= 0.1, min_samples_split= 2, min_weight_fraction_leaf= 0.0, n_estimators= 400, n_jobs= 1, oob_score= False, random_state= 1, verbose= 0, warm_start= False) -> out

Extracting the best hyperparameters
# Extract best hyperparameters from'grid_rf'
best_hyperparams = grid_rf.best_params_

print('Best hyperparameters:\n', best_hyperparams) -> in

Best hyperparameters:
{'max_depth': 4, 'max_features': 'log2', 'min_samples_leaf': 0.1, 'n_estimators': 400} -> out

Evaluating the best model performance
# Extract best model from 'grid_rf'
best_model = grid_rf.best_estimator_

# Predict the test set labels
y_pred = best_model.predict(X_test)

# Evaluate the test set RMSE
rmse_test = MSE(y_test, y_pred) ** (1/2)

# Print the test set RMSe
print('Test set RMSE of rf: {:.2f}'format(rmse_test)) -> in

Test set RMSE of rf: 3.89 -> out
'''

# Define the dictionary 'params_rf'
params_rf = {'n_estimators': [100, 350, 500], 'max_features': ['log2', 'auto', 'sqrt'], 'min_samples_leaf': [2, 10, 30]}


# Import GridSearchCV
from sklearn.model_selection import GridSearchCV 

# Instantiate grid_rf
grid_rf = GridSearchCV(estimator=rf, param_grid=params_rf, scoring='neg_mean_squared_error', cv=3, verbose=1, n_jobs=-1)


# Import mean_squared_error from sklearn.metrics as MSE 
from sklearn.metrics import mean_squared_error as MSE 

# Extract the best estimator
best_model = grid_rf.best_estimator_

# Predict test set labels
y_pred = best_model.predict(X_test)

# Compute rmse_test
rmse_test = MSE(y_test, y_pred) ** (1/2)

# Print rmse_test
print('Test RMSE of best model: {:.3f}'.format(rmse_test)) 
