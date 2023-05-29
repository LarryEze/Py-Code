''' Applying logistic regression and SVM '''

'''
scikit-learn refresher
Fitting and predicting
import sklearn.datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

newsgroups = sklearn.datasets.fetch_20newsgroups_vectorized()

X, y = newsgroup.data, newsgroups.target
X.shape -> in

(11314, 130107) -> out

y.shape -> in

(11314, ) -> out

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X, y)
y_pred = knn.predict(X)

Model evaluation
knn.score(X, y) -> in

0.99991 -> out

X_train, X_test, y_train, y_test = train_test_split(X, y)
knn.fit(X_train, y_train)
knn.score(X_test, y_test) -> in

0.66242 -> out
'''


# Create and fit the model
from sklearn.svm import SVC, LinearSVC
from sklearn import datasets
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

# Predict on the test features, print the results
pred = knn.predict(X_test)[0]
print("Prediction for test example 0:", pred)


'''
Applying logistic regression and SVM
Using LogisticRegression
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(X_train, y_train)
lr.predict(X_test)
lr.score(X_test, y_test)

LogisticRegression example
import sklearn.datasets
wine = sklearn.datasets.load_wine()

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(wine.data, wine.target)
lr.score(wine.data, wine.target) -> in

0.966 -> out

lr.predict_proba(wine.data[:1]) -> in

array([[9.966e-01, 2.740e-03, 6.787e-04]]) -> out

Using LinearSVC (Support Vector CLassifier)
LinearSVC works the same way: 

import sklearn.datasets
wine = sklearn.datasets.load_wine()

from sklearn.svm import LinearSVC
svm = LinearSVC()
svm.fit(wine.data, wine.target)
svm.score(wine.data, wine.target) -> in

0.955 -> out

Using SVC (Support Vector CLassifier)
import sklearn.datasets
wine = sklearn.datasets.load_wine()

from sklearn.svm import SVC
svm = SVC()
svm.fit(wine.data, wine.target)
svm.score(wine.data, wine.target) -> in

0.708 -> out

- NB*: a Hyperparameter is a choice about the model you make before fitting to the data, and often controls the complexity of the model.
- Model complexity review:
* Underfitting: model is too simple, low training accuracy
* Overfitting: model is too complex, low test accuracy
'''

digits = datasets.load_digits()
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target)

# Apply logistic regression and print scores
lr = LogisticRegression()
lr.fit(X_train, y_train)
print(lr.score(X_train, y_train))
print(lr.score(X_test, y_test))

# Apply SVM and print scores
svm = SVC()
svm.fit(X_train, y_train)
print(svm.score(X_train, y_train))
print(svm.score(X_test, y_test))


# Instantiate logistic regression and train
lr = LogisticRegression()
lr.fit(X, y)

# Predict sentiment for a glowing review
review1 = "LOVED IT! This movie was amazing. Top 10 this year."
review1_features = get_features(review1)
print("Review:", review1)
print("Probability of positive review:",
      lr.predict_proba(review1_features)[0, 1])

# Predict sentiment for a poor review
review2 = "Total junk! I'll never watch a film by that director again, no matter how good the reviews."
review2_features = get_features(review2)
print("Review:", review2)
print("Probability of positive review:",
      lr.predict_proba(review2_features)[0, 1])


'''
Linear classifiers
Linear decision boundaries
- A decision boundary tells us what class our classifier will predict for any value of x.

Definitions
Vocabulary:
- Classification: Its supervised learning to predict categories
- Regression: its supervised learning to predict a continuous value
- Decision boundary: The surface separating different predicted classes
- Linear classifier: A classifier that learns linear decision boundaries
* e.g., Logistic regression, linear SVM
- Linearly separable: A data set that can be perfectly explained by linear classifier
'''


# Define the classifiers
classifiers = [LogisticRegression(), LinearSVC(), SVC(),
               KNeighborsClassifier()]

# Fit the classifiers
for c in classifiers:
    c.fit(X, y)

# Plot the classifiers
plot_4_classifiers(X, y, classifiers)
plt.show()


''' Loss functions '''

'''
Linear classifiers: the coefficients
Dot Products
x = np.arange(3)
x -> in

array([0, 1, 2]) -> out

y = np.arange(3, 6)
y -> in

array([3, 4, 5]) -> out

x * y -> in

array([0, 4, 10]) -> out 

np.sum(x * y) -> in

14 -> out

x @ y -> in

14 -> out

- X @ y is called the dot product of x and y, and is written x.y

Linear classifier prediction
- raw model output = coefficients.features + intercept
- Linear classifier prediction: compute raw model output, check the sign
* if positive, predict one class
* if negative, predict the other class
- This is the same for logistic regression and linear SVM
* fit is different but predict is the same
* The differences in 'fit' relates to loss functions.

How LogisticRegression makes predictions
lr = LogisticRegression()
lr.fit(X, y)
lr.predict(X)[10] -> in

0 -> out

lr.predict(X)[20] -> in 

1 -> out

lr.coef_ @ X[10] + lr.intercept_ # raw model output -> in

array([-33.78572166]) -> out

lr.coef_ @ X[20] + lr.intercept_ # raw model output -> in

array([0.08050621]) -> out

-NB*: The values of the coefficients and intercept determine the boundary (coefficients determine the slope of the boundary and the intercept shifts it.)
'''


'''
What is a loss function?
Least squares: the squared loss
- scikit-learn's LinearRegression minimizes a loss:
* It minimizes the sum of squares of the errors made on your training set.
Error = True target value - Predicted target value
- Minimization is with respect to coefficients or parameters of the model.
- The loss function is a penalty score that tells us how well (or, to be precise, how poorly) the model is doing on the training data.
- The fit function is a running code that minimizes the loss
- NB*: the model.score() in scikit-learn isn't necessarily the loss function.
- The loss is used to fit te model on the data, and the score is used to see how well we're doing.

Classification errors: the 0 - 1 loss 
- The squared error from LinearRegression is not appropriate for classification problems, because our y-values are categories, not numbers
- For classification, a natural quantity to think about is the number of errors we've made
* This is the 0 - 1 loss: It's 0 for a correct prediction and 1 for an incorrect prediction
- This loss is hard to minimize!

Minimizing a loss
from scipy.optimize import minimize

( function y = x^2
- The second argument is our initial guess )

minimize(np.square, 0).x -> in

array([0.]) -> out

minimize(np.square, 2).x -> in

array([-1.88846401e-08]) -> out
'''

# The squared error, summed over training examples


def my_loss(w):
    s = 0
    for i in range(y.size):
        # Get the true and predicted target values for example 'i'
        y_i_true = y[i]
        y_i_pred = w@X[i]
        s = s + (y_i_true - y_i_pred)**2
    return s


# Returns the w that makes my_loss(w) smallest
w_fit = minimize(my_loss, X[0]).x
print(w_fit)

# Compare with scikit-learn's LinearRegression coefficients
lr = LinearRegression(fit_intercept=False).fit(X, y)
print(lr.coef_)


'''
Loss function diagrams
- loss diagram can be drawn from least squares linear regression.
* In linear regression, the raw model output is the prediction 
* The loss is higher as the prediction is further away from the true target value

* 0 - 1 loss diagram : looks like a step
* Linear regression loss diagram : looks like a U-shape curve
* Logistic loss diagram : looks like a downward sloping curve
* Hinge loss diagram: looks like a logistic loss but the slopes are straight
'''

# Mathematical functions for logistic and hinge losses


def log_loss(raw_model_output):
    return np.log(1+np.exp(-raw_model_output))


def hinge_loss(raw_model_output):
    return np.maximum(0, 1-raw_model_output)


# Create a grid of values and plot
grid = np.linspace(-2, 2, 1000)
plt.plot(grid, log_loss(grid), label='logistic')
plt.plot(grid, hinge_loss(grid), label='hinge')
plt.legend()
plt.show()


# The logistic loss, summed over training examples
def my_loss(w):
    s = 0
    for i in range(y.size):
        raw_model_output = w@X[i]
        s = s + log_loss(raw_model_output * y[i])
    return s


# Returns the w that makes my_loss(w) smallest
w_fit = minimize(my_loss, X[0]).x
print(w_fit)

# Compare with scikit-learn's LogisticRegression
lr = LogisticRegression(fit_intercept=False, C=1000000).fit(X, y)
print(lr.coef_)


''' Logistic regression '''

'''
Logistic regression and regularization
Regularized logistic regression
- Regularization combats overfitting by making the model coefficients smaller
- In scikit-learn, the hyperparameter 'C' is the inverse of the regularization strength
* i.e larger C = Less regularization and Smaller C = more regularization

How does regularization affect training accuracy?
lr_weak_reg = LogisticRegressio(C=100)
lr_strong_reg = LogisticRegressio(C=0.01)

lr_weak_reg.fit(X_train, y_train)
lr_strong_reg.fit(X_train, y_train)

lr_weak_reg.score(X_train, y_train)
lr_strong_reg.score(X_train, y_train) -> in

1.0
0.92 -> out

regularized loss = original loss + large coefficient penalty

- more regularization: lower training accuracy

How does regularization affect test accuracy?
lr_weak_reg.score(X_test, y_test) -> in

0.86 -> out

lr_strong_reg.score(X_test, y_test) -> in 

0.88 -> out

- more regularization: (almost always) higher test accuracy

L1 vs L2 regularization
- Lasso = linear regression with L1 regularization
- Ridge = linear regression with L2 regularization
- For other models like logistic regression, we just say L1, L2, etc.
* They both help reduce overfitting, and L1 also performs feature selection

lr_L1 = LogisticRegression(solver='liblinear', penalty='l1')
lr_L2 = LogisticRegression() # penalty = 'l2' by default

lr_L1.fit(X_train, y_train)
lr_L2.fit(X_train, y_train)

plt.plot(lr_L1.coef_.flatten())
plt.plot(lr_L2.coef_.flatten())

-NB*: the solver argument controls the optimization method used to find the coefficients, and needs to be set because the default solver is not compatible with L1 regularization
'''

# Train and validaton errors initialized as empty list
train_errs = list()
valid_errs = list()

# Loop over values of C_value
for C_value in [0.001, 0.01, 0.1, 1, 10, 100, 1000]:
    # Create LogisticRegression object and fit
    lr = LogisticRegression(C=C_value)
    lr.fit(X_train, y_train)

    # Evaluate error rates and append to lists
    train_errs.append(1.0 - lr.score(X_train, y_train))
    valid_errs.append(1.0 - lr.score(X_valid, y_valid))

# Plot results
plt.semilogx(C_values, train_errs, C_values, valid_errs)
plt.legend(("train", "validation"))
plt.show()


# Specify L1 regularization
lr = LogisticRegression(solver='liblinear', penalty='l1')

# Instantiate the GridSearchCV object and run the search
searcher = GridSearchCV(lr, {'C': [0.001, 0.01, 0.1, 1, 10]})
searcher.fit(X_train, y_train)

# Report the best parameters
print("Best CV params", searcher.best_params_)

# Find the number of nonzero coefficients (selected features)
best_lr = searcher.best_estimator_
coefs = best_lr.coef_
print("Total number of features:", coefs.size)
print("Number of selected features:", np.count_nonzero(coefs))


# Get the indices of the sorted cofficients
inds_ascending = np.argsort(lr.coef_.flatten())
inds_descending = inds_ascending[::-1]

# Print the most positive words
print("Most positive words: ", end="")
for i in range(5):
    print('excellent', end=", ")
print("\n")

# Print most negative words
print("Most negative words: ", end="")
for i in range(5):
    print('lame', end=", ")
print("\n")


'''
Logistic regression and probabilities
Logistic regression probabilities
e.g
without regularization : (C = 10e8)
- model coefficients : [[1.55 1.57]]
- model intercept : [-0.64]

with regularization : (C = 1)
- model coefficients : [[0.45 0.64]]
- model intercept : [-0.26]

- The effect of regularizatin is that the probabilities are closer to 0.5
* i.e smaller coefficients = less confident predictions
- The ratio of the coefficients gives us the slope of the line, and the magnitude of the coefficients gives us our confidence level
- Regularization doen not only affects the confidence, but also the orientation of the boundary.

How are these probabilities computed?
- Logistic regression predictions: they come from the sign of the raw model output
- Logistic regression probabilities: they 'squash' the raw model output to be between 0 and 1 and the sigmoid function takes care of that
* i.e when the raw model output is 0, the probability is o.5 and it means, we're on the boundary
* when the raw model output is positive, we would have predicted the positive class and the probability approaches 1
* when the raw model output is negative, we would have predicted the negative class and the probability approaches 0
'''

# Set the regularization strength
model = LogisticRegression(C=1)

# Fit and plot
model.fit(X, y)
plot_classifier(X, y, model, proba=True)

# Predict probabilities on training points
prob = model.predict_proba(X)
print("Maximum predicted probability", prob.max())


# Set the regularization strength
model = LogisticRegression(C=0.1)

# Fit and plot
model.fit(X, y)
plot_classifier(X, y, model, proba=True)

# Predict probabilities on training points
prob = model.predict_proba(X)
print("Maximum predicted probability", np.max(prob))


lr = LogisticRegression()
lr.fit(X, y)

# Get predicted probabilities
proba = lr.predict_proba(X)

# Sort the example indices by their maximum probability
proba_inds = np.argsort(np.max(proba, axis=1))

# Show the most confident (least ambiguous) digit
show_digit(proba_inds[-1], lr)

# Show the least confident (most ambiguous) digit
show_digit(proba_inds[0], lr)


'''
Multi-class logistic regression
- 2 popular approaches to multi-class classification
* The first is to train a series of binary classifiers for each class (One -vs-rest)
The other way is to modify the loss function so that it directly tries to optimize accuracy on the multi_class problem(aka Multinomial, softmax or cross-entropy loss)

Combining binary classifiers with one-vs-rest
lr0.fit(X, y==0)
lr1.fit(X, y==1)
lr2.fit(X, y==2)

# get raw model output
lr0.decision_function(X)[0] -> in

6.124 -> out

lr1.decision_function(X)[0] -> in

-5.429 -> out

lr2.decision_function(X)[0] -> in

-7.532 -> out

- decision_function, in scikit-learn gives the largest raw model output
- to fit a logistic regression model on the originak multi-class data set, set the multi_class parameter to 'ovr'

lr = LogisticRegression(multi_class='ovr')
lr.fit(X, y)
lr.predict(X)[0] -> in

0 -> out

One-vs-rest
- fit a binary classifier for each class
- predict with all, take largest output
- pro: simple, modular
- con: not directly optimizing accuracy
- common for SVMs as well
- can produce probabilities

Multinomial or softmax
- fit a single classifier for all classes
- prediction directly outputs best class
- con: more complicated, new code
- pro: tackle the problem directly
- possible for SVMs, but less common
- can produce probabilities
- its the standard in the field of neural networks

Model coefficients for multi-class
lr_ovr = LogisticRegression(multi_class='ovr')
lr_ovr.fit(X, y)
lr_ovr.coef_.shape -> in

(3, 13) -> out

lr_ovr.intercept_.shape -> in

(3, ) -> out 

lr_mn = LogisticRegression(multi_class='multinomial')
lr_mn.fit(X, y)
lr_mn.coef_.shape -> in

(3, 13) -> out

lr_mn.intercept_.shape -> in

(3, ) -> out
'''

# Fit one-vs-rest logistic regression classifier
lr_ovr = LogisticRegression(multi_class='ovr')
lr_ovr.fit(X_train, y_train)

print("OVR training accuracy:", lr_ovr.score(X_train, y_train))
print("OVR test accuracy    :", lr_ovr.score(X_test, y_test))

# Fit softmax classifier
lr_mn = LogisticRegression(multi_class='multinomial')
lr_mn.fit(X_train, y_train)

print("Softmax training accuracy:", lr_mn.score(X_train, y_train))
print("Softmax test accuracy    :", lr_mn.score(X_test, y_test))


# Print training accuracies
print("Softmax     training accuracy:", lr_mn.score(X_train, y_train))
print("One-vs-rest training accuracy:", lr_ovr.score(X_train, y_train))

# Create the binary classifier (class 1 vs. rest)
lr_class_1 = LogisticRegression(C=100)
lr_class_1.fit(X_train, y_train == 1)

# Plot the binary classifier (class 1 vs. rest)
plot_classifier(X_train, y_train == 1, lr_class_1)


# We'll use SVC instead of LinearSVC from now on

# Create/plot the binary classifier (class 1 vs. rest)
svm_class_1 = SVC()
svm_class_1.fit(X_train, y_train == 1)
plot_classifier(X_train, y_train == 1, svm_class_1)


''' Support Vector Machines '''

'''
What is an SVM?
- Linear classifiers (so far)
* They are trained using the hinge loss and L2 regularization.
- The logistic and hinge losses look fairly similar, A key difference is in the 'flat' part of the hinge loss, which occurs when the raw model output is greater than 1 (meaning you predicted an example correctly beyond some margin of error).

Support vectors
- They are defined as a training examples that are not in the flat part of the loss diagram.
* They include incorrectly classified examples, as well as correctly classified examples that are close to the boundary.
- How close is considered close enough is controlled by the regularization strength.
- If an example is not a support vector, removing it has no effect on the model, because its loss was already zero
- Having a small number of support vectors makes kernel SVMs really fast

Max-margin viewpoint
- If the regularization strength is not too large, the SVM maximizes the 'margin' for linearly separable datasets
- Margin: It is the distance from the boundary to the closest points
'''

# Train a linear SVM
svm = SVC(kernel="linear")
svm.fit(X, y)
plot_classifier(X, y, svm, lims=(11, 15, 0, 6))

# Make a new data set keeping only the support vectors
print("Number of original examples", len(X))
print("Number of support vectors", len(svm.support_))
X_small = X[svm.support_]
y_small = y[svm.support_]

# Train a new SVM using only the support vectors
svm_small = SVC(kernel="linear")
svm_small.fit(X, y)
plot_classifier(X_small, y_small, svm_small, lims=(11, 15, 0, 6))


'''
Kernel SVMs
Transforming your features
- fitting a linear model in a transformed spaces corresponds to fitting a non-linear model in the original space.

Kernels SVMs
- Kernels and Kernel SVMs implement feature transformations in a computationally efficient way.

from sklearn.svm import SVC

svm = SVC(gamma = 1) # default is kernel = 'rbf'

- the default behavior is called an RBF (Radial Basis Function) kernel.
- the gamma hyperparameter controls the smoothness of the boundary.
* i.e smaller gamma leads to smoother boundaries
- NB*: with the right hyperparameters, RBF SVMs are capable of perfectly separating almost any data set.
- larger gamma leads to more complex boundaries and can lead to overfitting
'''

# Instantiate an RBF SVM
svm = SVC()

# Instantiate the GridSearchCV object and run the search
parameters = {'gamma': [0.00001, 0.0001, 0.001, 0.01, 0.1]}
searcher = GridSearchCV(svm, parameters)
searcher.fit(X, y)

# Report the best parameters
print("Best CV params", searcher.best_params_)


# Instantiate an RBF SVM
svm = SVC()

# Instantiate the GridSearchCV object and run the search
parameters = {'C': [0.1, 1, 10], 'gamma': [0.00001, 0.0001, 0.001, 0.01, 0.1]}
searcher = GridSearchCV(svm, parameters)
searcher.fit(X_train, y_train)

# Report the best parameters and the corresponding score
print("Best CV params", searcher.best_params_)
print("Best CV accuracy", searcher.best_score_)

# Report the test accuracy using these best parameters
print("Test accuracy of best grid search hypers:",
      searcher.score(X_test, y_test))


'''
Comparing logistic regression and SVM (and beyond)
Logistic regression
- Is a linear classifier
- Can use with kernels, but slow
- Outputs meaningful probabilities
- Can be extended to multi-class
- All data points affect fit
- L2 or L1 regularization

Support vector machine (SVM)
- Is a linear classifier
- Can use with kernels, and fast
- Does not naturally output probabilities
- Can be extended to multi-class
- Only 'support vectors' affect fit
- Conventionally just L2 regularization

Use in scikit-learn
Logistic regression in sklearn:
* linear_model.LogisticRegression
key hyperparameters in sklearn:
* C (inverse regularization strength)
* penalty (type of regularization)
* multi_class (type of multi-class)

SVM in sklearn:
* svm.LinearSVC and svm.SVC
key hyperparameters in sklearn:
* C (inverse regularization strength)
* kernel (type of kernel)
* gamma (inverse RBF smoothness)

SGDClassifier (Stochastic Gradient Descent)
- SGDClassifier: scales well to large datasets

from sklearn.linear_model import SGDClassifier

logreg = SGDClassifier(loss = 'log_loss')

linsvm = SGDClassifier(loss = 'hinge')

- To switch between the logistic regression and a linear SVM, one only has to set the loss hyperparameter of the SGDClassifier i.e the model is the same but the loss changes

- SGDClassifier hyperparameter is called alpha instead of C (i.e bigger alpha = more regularization). 
- alpha is the inverse of C
- One advantage of SGDClassifier is that it's very fast compared to LogisticRegression or LinearSVC
'''

# We set random_state=0 for reproducibility
linear_classifier = SGDClassifier(random_state=0)

# Instantiate the GridSearchCV object and run the search
parameters = {'alpha': [0.00001, 0.0001, 0.001,
                        0.01, 0.1, 1], 'loss': ['hinge', 'log_loss']}
searcher = GridSearchCV(linear_classifier, parameters, cv=10)
searcher.fit(X_train, y_train)

# Report the best parameters and the corresponding score
print("Best CV params", searcher.best_params_)
print("Best CV accuracy", searcher.best_score_)
print("Test accuracy of best grid search hypers:",
      searcher.score(X_test, y_test))
