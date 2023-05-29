''' Clustering for dataset exploration '''

'''
Unsupervised Learning
- Unsupervised learning finds patterns in data e.g., 
* Clustering customers by their purchases
* Compressing the data using purchase patterns (dimension reduction)

Supervised vs unsupervised learning
- Supervised learning finds patterns for a prediction task
* e.g., Classify tumors as benign or cancerous (Labels)
- Unsupervised learning finds patterns in data
* ... but without a specific prediction task in mind

Iris dataset
- Measurements of many iris plants
- Three species of iris:
* setosa
* versicolor
* virginica
- Petal length, Petal width, Sepal length, Sepal width (the features of the dataset)

Arrays, features & samples
- 2D NumPy array
- Columns are measurements (the features)
- Rows represent iris plants (the samples)

Iris data is 4-dimensional
- Iris samples are points in 4 dimentional space
- Dimension = number of features
- Dimension too high to visualize!
* ... but unsupervised learning gives insight

k-means clustering
- Finds clusters of samples
- Number of clusters must be specified
- Implemented in sklearn ('scikit-learn')

print(samples) -> in

[   [5.  3.3 1.4 0.2]
    [5.  3.5 1.3 0.3]
    ...
    [7.2 3.2 6.  1.8]   ] -> out

from sklearn.cluster import KMeans
model = KMeans(n_clusters = 3)
model.fit(samples) -> in

KMeans(n_clusters=3) -> out

labels = model.predict(samples)
print(labels) -> in

[0 0 1 1 0 1 2 1 0 1 ...] -> out

Cluster labels for new samples
- New samples can be assigned to existing clusters
- k-means remembers the mean of each cluster (the 'centroids')
- Finds the nearest centroid to each new sample

print(new_samples) -> in

[   [ 5.7 4.4 1.5 0.4]
    [ 6.5 3.  5.5 1.8]
    [ 5.8 2.7 5.1 1.9]  ] -> out

new_labels = model.predict(new_samples)
print(new_labels) -> in

[0 2 1] -> out

Scatter plots
- Scatter plot of sepal length vs petal length
- Each point represents an iris sample
- Color points by cluster labels
- PyPlot (matplotlib.pyplot)

import matplotlib.pyplot as plt
xs = samples[:, 0]
ys = samples[:, 2]
plt.scatter(xs, ys, c=labels)
plt.show()
'''

# Import KMeans

# Create a KMeans instance with 3 clusters: model
from sklearn.preprocessingimport import Normalizer, MaxAbsScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from scipy.stats import pearsonr
from scipy.cluster.hierarchy import fcluster
from sklearn.preprocessing import normalize
from sklearn.preprocessing import Normalizer
model = KMeans(n_clusters= 3)

# Fit model to points
model.fit(points)

# Determine the cluster labels of new_points: labels
labels = model.predict(new_points)

# Print cluster labels of new_points
print(labels)


# Import pyplot

# Assign the columns of new_points: xs and ys
xs = new_points[:, 0]
ys = new_points[:, 1]

# Make a scatter plot of xs and ys, using labels to define the colors
plt.scatter(xs, ys, c=labels, alpha=0.5)

# Assign the cluster centers: centroids
centroids = model.cluster_centers_

# Assign the columns of centroids: centroids_x, centroids_y
centroids_x = centroids[:, 0]
centroids_y = centroids[:, 1]

# Make a scatter plot of centroids_x and centroids_y
plt.scatter(centroids_x, centroids_y, marker='D', s=50)
plt.show()


'''
Evaluating a clustering
- Can check correspondence with e.g. iris species
* ... but what if there are no species to check against?
- Measure quality of a clustering
- Informs cloice of how many clusters to look for

Iris: clusters vs species
- k-means found 3 clusters amongst the iris samples
- Do the clusters correspond to the species?

species setosa versicolor virginica
labels
0            0          2        36
1           50          0         0
2            0         48        14

Cross tabulation with pandas
- Clusters vs species is a 'cross-tabulation'
- Use the pandas library
- Given the species of each sample as a list species

print(species) -> in

['setosa', 'setosa', 'versicolor', 'virginica', ...] -> out

Aligning labels and species
import pandas as pd
df = pd.DataFrame({'labels': labels, 'species': species})
print(df) -> in

    labels    species
0      1       setosa
1      1       setosa
2      2   versicolor
3      2    virginica
4      1       setosa
... -> out

Crosstab of labels and species
ct = pd.crosstab(df['labels'], df['species'])
print(ct) -> in

species setosa versicolor virginica
labels
0            0          2        36
1           50          0         0
2            0         48        14 -> out

Measuring clustering quality
- Using only samples and their cluster labels
- A good clustering has tight clusters
- Samples in each cluster bunched together

Inertia measures clustering quality
- Measures how spread out the clusters are (Lower is better)
- Distance from each sample to centroid of its cluster
- After .fit(), available as attribute .inertia_
- k-means attempts to minimize the inertia when choosing clusters

from sklearn.cluster import KMeans

model = KMeans(n_clusters = 3)
model.fit(samples)
print(model.inertia_) -> in

78.9408414261 -> out

The number of clusters
- Clusterings of the iris dataset with different numbers of clusters
- More clusters means lower inertia
- What is the best nummber of clusters ?

How many clusters to choose?
- A god clustering has tight clusters (so low inertia)
* ... but not too many clusters!
- Choose an 'elbow' in the inertia plot
* Where inertia begins to decrease more slowly
- e.g., for iris dataset, 3 is a good choice
'''

ks = range(1, 6)
inertias = []

for k in ks:
    # Create a KMeans instance with k clusters: model
    model = KMeans(n_clusters=k)

    # Fit model to samples
    model.fit(samples)

    # Append the inertia to the list of inertias
    inertias.append(model.inertia_)

# Plot ks vs inertias
plt.plot(ks, inertias, '-o')
plt.xlabel('number of clusters, k')
plt.ylabel('inertia')
plt.xticks(ks)
plt.show()


# Create a KMeans model with 3 clusters: model
model = KMeans(n_clusters=3)

# Use fit_predict to fit model and obtain cluster labels: labels
labels = model.fit_predict(samples)

# Create a DataFrame with labels and varieties as columns: df
df = pd.DataFrame({'labels': labels, 'varieties': varieties})

# Create crosstab: ct
ct = pd.crosstab(df['labels'], df['varieties'])

# Display ct
print(ct)


'''
Transforming features for better clusterings
Piedmont wines dataset
- 178 samples from 3 distinct varieties of red wine: Barolo, Grignolino and Barbera
- Features measure chemical composition e.g. alcohol content
* Visual properties like 'color intensity'

Clustering the wines
from sklearn.cluster import KMeans
model = KMeans(n_clusters = 3)
labels = model.fit_predict(samples)

Cluster vs varieties
df = pd.DataFrame({'labels': labels, 'varieties': varieties})
ct = pd.crosstab(df['labels'], df['varieties'])
print(ct) -> in

varieties Barbera Barolo Grignolino
labels
0              29     13         20
1               0     46          1
2              19      0         50 -> out

Feature variances
- The wine features have very different variances!
- Variance of a feature measures spread of its values

feature         variance
alcohol             0.65
malic_acid          1.24
...
od280               0.50
proline         99166.71

StandardScaler
- In kmeans: feature variance = fature influence
- StandardScaler transforms each feature to have mean 0 and variance 1
- Features are said to be 'standardized'

sklearn StandardScaler
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(samples)
StandardScaler(copy=True, with_mean=True, with_std=True)
samples_scaled = scaler.transform(samples)

Similar methods
- StandardScaler and KMeans have similar methods
- Use .fit() / .transform() with StandardScaler
- Use .fit() / .predict() with KMeans

StandardScaler, then KMeans
- Need to perform two steps: StandardScaler, then KMeans
- Use sklearn pipeline to combine multiple steps
- Data flows from one step into the next

Pipelines combine multiple steps
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
scaler = StandardScaler()
kmeans = KMeans(n_clusters = 3)

from sklearn.pipeline import make_pipeline
pipeline = make_pipeline(scaler, kmeans)
pipeline.fit(samples) -> in

Pipeline(steps=...) -> out

labels = pipeline.predict(samples)

Feature standardization improves clustering
with feature standardization:

varieties Barbera Barolo Grignolino
labels
0               0     59          3
1              48      0          3
2               0      0         65

without feature standardization was very bad:

varieties Barbera Barolo Grignolino
labels
0              29     13         20
1               0     46          1
2              19      0         50 -> out

sklearn preprocessing steps
- StandardScaler is a 'preprocessing' step
- MaxAbsScaler and Normalizer are other examples
'''

# Perform the necessary imports

# Create scaler: scaler
scaler = StandardScaler()

# Create KMeans instance: kmeans
kmeans = KMeans(n_clusters=4)

# Create pipeline: pipeline
pipeline = make_pipeline(scaler, kmeans)


# Import pandas

# Fit the pipeline to samples
pipeline.fit(samples)

# Calculate the cluster labels: labels
labels = pipeline.predict(samples)

# Create a DataFrame with labels and species as columns: df
df = pd.DataFrame({'labels': labels, 'species': species})

# Create crosstab: ct
ct = pd.crosstab(df['labels'], df['species'])

# Display ct
print(ct)


# Import Normalizer

# Create a normalizer: normalizer
normalizer = Normalizer()

# Create a KMeans model with 10 clusters: kmeans
kmeans = KMeans(n_clusters=10)

# Make a pipeline chaining normalizer and kmeans: pipeline
pipeline = make_pipeline(normalizer, kmeans)

# Fit pipeline to the daily price movements
pipeline.fit(movements)


# Import pandas

# Predict the cluster labels: labels
labels = pipeline.predict(movements)

# Create a DataFrame aligning labels and companies: df
df = pd.DataFrame({'labels': labels, 'companies': companies})

# Display df sorted by cluster label
print(df.sort_values('labels'))


''' Visualization with hierarchical clustering and t-SNE '''

'''
Visualizing hierarchies
Visualizations communicate insight
- 't-SNE': Creates a 2D map of a dataset and conveys useful information about the proximity of the samples to one another
- 'Hierarchical clustering'

A hierarchy of groups
- Groups of living things can form a hierarchy
- Clusters are contained in one another

        animals
        /     \
    mammals   reptiles                    
    /    \     /     \
humans  apes snakes lizards

Eurovision scoring dataset
- Countries gave scores to songs performed at the Eurovision 2016
- 2D array of scores
- Rows are countries, columns are songs

Hierarchical clustering
- Every country begins in a separate cluster
- At each step, the two closest clusters are merged
- Continue until all countries are in a single cluster
- This is 'agglomerative' hierarchical clustering
- 'Divisive clustering' -> other way round

The dendrogram of a hierarchical clustering
- Read from the bottom up
- Vertical lines represent clusters

Hierarchical clustering with SciPy
- Given samples (the array of scores), and country_names (the list of country names)

import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
mergings = linkage(samples, method='complete')
dendrogram(mergings, labels=country_names, leaf_rotation=90, leaf_font_size=6)
plt.show() 
'''

# Perform the necessary imports

# Calculate the linkage: mergings
mergings = linkage(samples, method='complete')

# Plot the dendrogram, using varieties as labels
dendrogram(mergings, labels=varieties, leaf_rotation=90, leaf_font_size=6)
plt.show()


# Import normalize

# Normalize the movements: normalized_movements
normalized_movements = normalize(movements)

# Calculate the linkage: mergings
mergings = linkage(normalized_movements, method='complete')

# Plot the dendrogram
dendrogram(mergings, labels=companies, leaf_rotation=90, leaf_font_size=6)
plt.show()


'''
Cluster labels in hierarchical clustering
- Hierarchical clustering is not only a visualization tool!
- Cluster labels at any intermediate stage can be recovered
- Cluster labels can be used for e.g. cross-tabulations

Intermediate clusterings & height on dendrogram
- e.g. at height 15:
* Bulgaria, Cyprus, Greece are one cluster
* Russia and Moldova are another
* Armenia in a cluster on its own

Dendrograms show cluster distances
- Height on dendrogram = distance between merging clusters
- e.g. clusters with only Cyprus and Greece had distance approx. 6
* This new cluster distance approx. 12 from cluster with only Bulgaria 

Intermediate clusterings & height on dendrogram
- Height on dendrogram specifies max. distance between merging clusters 
- Don't merge clusters further apart than this (e.g. 15)

Distance between clusters
- Defined by a 'linkage method'
- In 'complete' linkage: distance between clusters is max. distance between their samples
- Specified via method parameter, e.g. linkage(samples, method='complete')
- Different linkage method, different hierarchical clustering! 

Extracting cluster labels
- Use the fcluster() function
- Returns a NumPy array of cluster labels

Extracting cluster labels using fcluster
from scipy.cluster.hierarchy import linkage
mergings = linkage(samples, method='complete')

from scipy.cluster.hierarchy import fcluster 
labels = fcluster(mergings, 15, criterion='distance')
print(labels) -> in

[ 9 8 11 20 2 1 17 14 ... ] -> out

Aligning cluster labels with country names
Given a list of strings country_names:

import pandas as pd
pairs = pd.DataFrame({'labels': labels, 'countries': country_names})
print(pairs.sort_values('labels')) -> in

    countries labels
5     Belarus      1
40    Ukraine      1
...
36      Spain      5
8    Bulgaria      6
19     Greece      6
10     Cyprus      6
28    Moldova      7
... -> out

NB: SciPy cluster labels satart at 1, not at 0 like they do in scikit-learn 
'''

# Perform the necessary imports

# Calculate the linkage: mergings
mergings = linkage(samples, method='single')

# Plot the dendrogram
dendrogram(mergings, labels=country_names, leaf_rotation=90, leaf_font_size=6)
plt.show()


# Perform the necessary imports

# Use fcluster to extract labels: labels
labels = fcluster(mergings, 6, criterion='distance')

# Create a DataFrame with labels and varieties as columns: df
df = pd.DataFrame({'labels': labels, 'varieties': varieties})

# Create crosstab: ct
ct = pd.crosstab(df['labels'], df['varieties'])

# Display ct
print(ct)


'''
t-SNE for 2-dimensional maps
t-SNE = 't-distributed stochastic neighbor embedding'
- Maps samples to 2D pace (or 3D)
- Map approximately preserves nearness of samples
- Great for inspecting datasets

t-SNE on the iris dataset3
- Iris dataset has 4 measurements, so samples are 4-dimensional
- t-SNE maps samples to 2D space
- t-SNE didn't know that there were different species
* ... yet the species mostly separate

Interpreting t-SNE scatter plots
- 'versicolor' and 'virginica' are harder to distinguish from one another
- Consistent with k-means inertia plot: could argue for 2 clusters, or for 3

t-SNE in sklearn
- 2D NumPy array of samples

print(samples) -> in

[   [5.  3.3 1.4 0.2]
    [5.  3.5 1.3 0.3]
    [4.9 2.4 3.3 1. ]
    [6.3 2.8 5.1 1.5]
    ...
    [4.9 3.1 1.5 0.1]   ] -> out

- List of species giving species of labels as number (0, 1, or 2)

print(species) -> in

[0, 0, 1, 2, ..., 0] -> out

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
model = TSNE(learning_rate = 100)
transformed = model.fit_transform(samples)
xs = transformed[:, 0]
ys = transformed[:, 1]
plt.scatter(xs, ys, c=species)
plt.show()

t-SNE has only fit_transform()
- Has only a .fit_transform() method
* Simultaneously fits the model and transforms the data
- Has no separate .fit() or .transform() methods
- Can't extend a t-SNE map to include new data samples
- Instead, must start over each time!

t-SNE learning rate 
- Choose learning rate for the dataset
- Wrong choice: points appear bunched together in the scatter plot
- Try values between 50 and 200

Different every time
- t-SNE features are different every time
- e.g. Piedmont wines, 3 runs, 3 different scatter plots!
* ... however: The wine varieties (= colors) have same position relative to one another
'''

# Import TSNE

# Create a TSNE instance: model
model = TSNE(learning_rate=200)

# Apply fit_transform to samples: tsne_features
tsne_features = model.fit_transform(samples)

# Select the 0th feature: xs
xs = tsne_features[:, 0]

# Select the 1st feature: ys
ys = tsne_features[:, 1]

# Scatter plot, coloring by variety_numbers
plt.scatter(xs, ys, c=variety_numbers)
plt.show()


# Import TSNE

# Create a TSNE instance: model
model = TSNE(learning_rate=50)

# Apply fit_transform to normalized_movements: tsne_features
tsne_features = model.fit_transform(normalized_movements)

# Select the 0th feature: xs
xs = tsne_features[:, 0]

# Select the 1th feature: ys
ys = tsne_features[:, 1]

# Scatter plot
plt.scatter(xs, ys, alpha=0.5)

# Annotate the points
for x, y, company in zip(xs, ys, companies):
    plt.annotate(company, (x, y), fontsize=5, alpha=0.75)
plt.show()


''' Decorrelating your data and dimension reduction '''

'''
Visualizing the PCA transformation
Dimension reduction
- It finds patterns in data, and uses these patterns to re-express it in a compressed form.
* More efficient storage and computation
- The most important function of dimension reduction is to reduce a dataset to its 'bare bone'. i.e remove less-informative 'noise' features
* ... which cause problems for prediction tasks, e.g. classification, regression

Principal Component Analysis
- PCA = 'Principal Component Analysis'
- Fundamental dimension reduction technique
- First step 'decorrelation'
- Second step reduces dimension

PCA aligns data with axes
- Rotates data samples to be aligned with axes
- Shifts data samples so they have mean 0

PCA follows the fit/transform pattern
- PCA is a scikit-learn component like KMeans or StandardScaler
- .fit() learns the transformation from a given data
- .transform() applies the learned transformation
* .transform() can also be applied to new data

Using scikit-learn PCA
- samples = array of two features (total_phenols & od280)

[   [ 2.8  3.92]
    ...
    [ 2.05 1.6 ]   ]

from sklearn.decomposition import PCA 
model = PCA()
model.fit(samples) -> in

PCA() -> out

transformed = model.transform(samples)

print(transformed) -> in

[   [ 1.32771994e+00  4.51396070e-01]
    [ 8.32496068e-01  2.33099664e-01]
    ...
    [-9.33526935e-01 -4.60559297e-01] ] -> out

PCA features
- Rows of transformed correspond to samples
- Columns of transformed are the 'PCA features'
- Row gives PCA feature values of corresponding sample

PCA features are not correlated
- Features of dataset are often correlated, e.g. total_phenols and od280
- PCA aligns the data with axes
- Resulting PCA features are not linearly correlated ('decorrelation')

Pearson correlation
- Measures linear correlation of features
- Value between -1 and 1
- Value of 0 means no linear correlation

Principal components
- 'Principal components' = directions of variance
- PCA aligns principal components with the axes
- After a PCA model has been fit, the principal components are available as .components_ attribute of PCA object
- Each row defines each principal component (displacement from mean)

print(model.components_) -> in

[   [ 0.64116665 0.76740167]
    [-0.76740167 0.64116665]   ] -> out
'''

# Perform the necessary imports

# Assign the 0th column of grains: width
width = grains[:, 0]

# Assign the 1st column of grains: length
length = grains[:, 1]

# Scatter plot width vs length
plt.scatter(width, length)
plt.axis('equal')
plt.show()

# Calculate the Pearson correlation
correlation, pvalue = pearsonr(width, length)

# Display the correlation
print(correlation)


# Import PCA

# Create PCA instance: model
model = PCA()

# Apply the fit_transform method of model to grains: pca_features
pca_features = model.fit_transform(grains)

# Assign 0th column of pca_features: xs
xs = pca_features[:, 0]

# Assign 1st column of pca_features: ys
ys = pca_features[:, 1]

# Scatter plot xs vs ys
plt.scatter(xs, ys)
plt.axis('equal')
plt.show()

# Calculate the Pearson correlation of xs and ys
correlation, pvalue = pearsonr(xs, ys)

# Display the correlation
print(correlation)


'''
Intrinsic dimension
Intrinsic dimension of a flight path
- 2 features: longitude and latitude at points along a flight path
- Dataset appears to be 2-dimensional
* But can approximate using one feature: displacement along flight path
* Is intrinsically 1-dimensional

Intrinsic dimension
- Intrinsic dimension = number of features needed to approximate the dataset
- Essential idea behind dimension reduction
- What is the most compact epresentation of the samples?
- Can be detected with PCA

Versicolor dataset
- 'versicolor', one of the iris species
- Only 3 features: sepal length, sepal width, and petal width
- Samples are points in 3D space

Versicolor dataset has intrinsic dimension 2
- Samples lie close to a flat 2-dimensional sheet
* So can be approximated using 2 features

PCA identifies intrinsic dimension
- Scatter plots work only if samples have 2 or 3 features
- PCA identifies intrinsic dimension when samples have any number of features
- Intrinsic dimension = number of PCA features with significant variance

Variance and intrinsic dimension
- Intrinsic dimension is number of PCA features with significant variance
- In our example: the first 2 PCA features
- So intrinsic dimension is 2

Plotting the variances of PCA features
- samples = array of versicolor samples

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
pca = PCA()
pca.fit(samples) -> in

PCA() -> out

features = range(pca.n_components_)

plt.bar(features, pca.explained_variance_)
plt.xticks(features)
plt.ylabel('variance')
plt.xlabel('PCA feature')
plt.show()

Intrinsic dimension can be ambiguous
- Intrinsic dimension is an idealization 
* ... there is not always one correct answer!
- Piedmont wines: could argue for 2, or for 3, or more
'''

# Make a scatter plot of the untransformed points
plt.scatter(grains[:, 0], grains[:, 1])

# Create a PCA instance: model
model = PCA()

# Fit model to points
model.fit(grains)

# Get the mean of the grain samples: mean
mean = model.mean_

# Get the first principal component: first_pc
first_pc = model.components_[0, :]

# Plot first_pc as an arrow, starting at mean
plt.arrow(mean[0], mean[1], first_pc[0], first_pc[1], color='red', width=0.01)

# Keep axes on same scale
plt.axis('equal')
plt.show()


# Perform the necessary imports

# Create scaler: scaler
scaler = StandardScaler()

# Create a PCA instance: pca
pca = PCA()

# Create pipeline: pipeline
pipeline = make_pipeline(scaler, pca)

# Fit the pipeline to 'samples'
pipeline.fit(samples)

# Plot the explained variances
features = range(pca.n_components_)
plt.bar(features, pca.explained_variance_)
plt.xlabel('PCA feature')
plt.ylabel('variance')
plt.xticks(features)
plt.show()


'''
Dimension reduction with PCA
Dimension reduction
- Represents same data, using less features
- Important part of machine-learning pipelines
- Can be performed using PCA

Dimension reduction with PCA
- PCA features are in decreasing order of variance
- Assumes the low variance features are 'noise'
* ... and high variance features are informative

Dimension reduction with PCA
- Specify how many features to keep
* e.g. PCA(n_components = 2)
* Keeps the first 2 PCA features
- Intrinsic dimension is a good choice

Dimension reduction of iris dataset
- samples = array of iris measurements (4 features)
- species = list of iris species numbers

from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
pca.fit(samples)

PCA(n_components=2) -> in

transformed = pca.transform(samples)
print(transformed.shape) -> in

(150, 2) -> out

Iris dataset in 2 dimensions
import matplotlib.pyplot as plt
xs = transformed[:, 0]
ys = transformed[:, 1]
plt.scatter(xs, ys, c=species)
plt.show()

- PCA has reduced the dimension to 2
- Retained the 2 PCA features with highest variance
- Important information preserved: species remain distinct

Dimension reduction with PCA
- Discards low variance PCA features
- Assumes the high variance features are informative
- Assumption typically holds in practice (e.g. for iris)

Word frequency arrays
- rows represents documents, columns represent words
- Entries measure presence of each word in each document
* ... measure using 'tf - idf'

            aardvark    apple ... zebra
document0          0,    0.1, ...    0.
document1
.            word frquencies ('tf-idf')
.
.

Sparse arrays and csr_matrix
- 'Sparse': most entries are zero
- Can use scipy.sparse.csr_matrix instead of NumPy array
- csr_matrix remembers only the non-zero entries (saves space!)

TruncatedSVD and csr_matrix
- scikit-learn PCA doesn't support csr_matrix
- Use scikit-learn TruncatedSVD instead
* Performs same transformation

from sklearn.decomposition import TruncatedSVD
model = TruncatedSVD(n_components = 3)
model.fit(documents) # documents is csr_matrix
transformed = model.transform(documents)
'''

# Import PCA

# Create a PCA model with 2 components: pca
pca = PCA(n_components=2)

# Fit the PCA instance to the scaled samples
pca.fit(scaled_samples)

# Transform the scaled samples: pca_features
pca_features = pca.transform(scaled_samples)

# Print the shape of pca_features
print(pca_features.shape)


# Import TfidfVectorizer

# Create a TfidfVectorizer: tfidf
tfidf = TfidfVectorizer()

# Apply fit_transform to document: csr_mat
csr_mat = tfidf.fit_transform(documents)

# Print result of toarray() method
print(csr_mat.toarray())

# Get the words: words
words = tfidf.get_feature_names()

# Print words
print(words)


# Perform the necessary imports

# Create a TruncatedSVD instance: svd
svd = TruncatedSVD(n_components=50)

# Create a KMeans instance: kmeans
kmeans = KMeans(n_clusters=6)

# Create a pipeline: pipeline
pipeline = make_pipeline(svd, kmeans)


# Import pandas

# Fit the pipeline to articles
pipeline.fit(articles)

# Calculate the cluster labels: labels
labels = pipeline.predict(articles)

# Create a DataFrame aligning labels and titles: df
df = pd.DataFrame({'label': labels, 'article': titles})

# Display df sorted by cluster label
print(df.sort_values('label'))


''' Discovering interpretable features '''

'''
Non-negative matrix factorization (NMF)
- NMF = 'non-negative matrix factorization'
- Its a Dimension reduction technique
- NMF models are interpretable (unlike PCA)
- Its easy to interpret means easy to explain!
- However, all sample features must be non-negative (>= 0)

Interpretable parts
- NMF achieves its interpretability by decomposing samples as sums of their parts
- NMF expresses documents as combinations of topics (or 'themes')
- NMF expresses images as combinations of patterns

Using scikit-learn NMF
- Follows .fit() / .transform() pattern
- Must specify the number of components e.g NMF(n_components = 2)
- Works with NumPy arrays and with csr_matrix

Example word-frequency array
- Word frequency array, 4 words, many documents
- Measure presence of words in each document using 'tf-idf'
* 'tf' = frequency of word in document
* 'idf' reduces influence of frequent words like 'the'

            course datacamp potato the
document0     0.2,     0.3,   0.0, 0.1
document1     0.0,     0.0,   0.4, 0.1
...                        ...

Example usage of NMF
- samples is the word-frequency array

from sklearn.decomposition import NMF
model = NMF(n_components = 2)
model.fit(samples) -> in

NMF(n_components=2) -> out

nmf_features = model.transform(samples)

- NMF feature values are non-negative
- Can be used to reconstruct the samples
* ... combine feature values with components

print(nmf_features) -> in

[   [ 0.   0.2  ]
    [ 0.19 0.   ]
    ...
    [ 0.15 0.12 ]   ] -> out

- NMF has components
* ... just like PCA has principal components
- Dimension of components = dimension of samples
- Entries are non-negative

print(model.components_) -> in

[   [ 0.01 0.   2.13 0.54]
    [ 0.99 1.47 0.   0.5  ] ] -> out

Reconstruction of a sample
print(samples[i, :]) -> in

[ 0.12 0.18 0.32 0.14] -> out

print(nmf_features[i, :]) -> in

[ 0.15 0.12] -> out

- Multiply components by feature values, and add up
- It can also be expressed as a product of matrices
* This is the 'Matrix Factorization' in 'NMF'

NMF fits to non-negative data only 
e.g
- Word frequencies in each document
- Images encoded as arrays
- Audio spectrograms
- Purchase histories on e-commerce sites
... and many more!
'''

# Import NMF

# Create an NMF instance: model
model = NMF(n_components=6)

# Fit the model to articles
model.fit(articles)

# Transform the articles: nmf_features
nmf_features = model.transform(articles)

# Print the NMF features
print(nmf_features.round(2))


# Import pandas

# Create a pandas DataFrame: df
df = pd.DataFrame(nmf_features, index=titles)

# Print the row for 'Anne Hathaway'
print(df.loc['Anne Hathaway'])

# Print the row for 'Denzel Washington'
print(df.loc['Denzel Washington'])


'''
NMF learns interpretable parts
Example: NMF learns interpretable parts
- Word-frequency array articles (tf-idf)
- 20,000 scientific articles (rows)
- 800 words (columns)

            aardvark arch . . . zebra
article0
article1
.                      articles
.
.

Applying NMF to the articles
print(articles.shape) -> in

(20000, 800) -> out

from sklearn.decomposition import NMF
nmf = NMF(n_components = 10)
nmf.fit(articles) -> in

NMF(n_components=10) -> out

print(nmf.components_.shape) -> in

(10, 800) -> out

NMF components
- For documents:
* NMF components represent topics
* NMF features combine topics into documents
- For images, NMF components are parts of images

Grayscale images
- 'Grayscale' image = no colors, only shades of gray
- It measure pixel brightness
- It is represented with value between 0 and 1 (where 0 is totally black and 1 is totally white)
- The image can be represented as 2-dimensional array of numbers

Grayscale image example
- An 8 x 8 grayscale image of the moon, written as an array

[   [ 0.  0.  0.  0.  0.  0.  0.  0. ]
    [ 0.  0.  0.  0.7 0.8 0.  0.  0. ]
    [ 0.  0.  0.8 0.8 0.9 1.  0.  0. ]
    [ 0.  0.7 0.9 0.9 1.  1.  1.  0. ]
    [ 0.  0.8 0.9 1.  1.  1.  1.  0. ]
    [ 0.  0.  0.9 1.  1.  1.  0.  0. ]
    [ 0.  0.  0.  0.9 1.  0.  0.  0. ]
    [ 0.  0.  0.  0.  0.  0.  0.  0. ] ]

Grayscale images as flat arrays
- Enumerate the entries
* Row-by-row
* From left to righ, top to bottom
e.g
[   [ 0. 1. 0.5]
    [ 1. 0. 1. ]   ]   ->  [ 0. 1. 0.5  1. 0. 1.  ]

Encoding a collection of images
- Collection of images of the same size
* Encode as 2D array
* Each row corresponds to an image
* Each column corresponds to a pixel
* ... can apply NMF!

Visualizing samples
print(sample) -> in

[ 0. 1. 0.5  1. 0. 1.  ] -> out

bitmap = sample.reshape((2, 3))
print(bitmap) -> in

[   [ 0. 1. 0.5]
    [ 1. 0. 1. ]    ] -> out

from matplotlib import pyplot as plt
plt.imshow(bitmap, cmap='gray', interpolation='nearest')
plt.show()
'''

# Import pandas

# Create a DataFrame: components_df
components_df = pd.DataFrame(model.components_, columns=words)

# Print the shape of the DataFrame
print(components_df.shape)

# Select row 3: component
component = components_df.iloc[3]

# Print result of nlargest
print(component.nlargest())


# Import pyplot

# Select the 0th row: digit
digit = samples[0]

# Print digit
print(digit)

# Reshape digit to a 13x8 array: bitmap
bitmap = digit.reshape((13, 8))

# Print bitmap
print(bitmap)

# Use plt.imshow to display bitmap
plt.imshow(bitmap, cmap='gray', interpolation='nearest')
plt.colorbar()
plt.show()


def show_as_image(sample):
    bitmap = sample.reshape((13, 8))
    plt.figure()
    plt.imshow(bitmap, cmap='gray', interpolation='nearest')
    plt.colorbar()
    plt.show()


# Import NMF

# Create an NMF model: model
model = NMF(n_components=7)

# Apply fit_transform to samples: features
features = model.fit_transform(samples)

# Call show_as_image on each component
for component in model.components_:
    show_as_image(component)

# Select the 0th row of features: digit_features
digit_features = features[0]

# Print digit_features
print(digit_features)


# Import PCA

# Create a PCA instance: model
model = PCA(n_components=7)

# Apply fit_transform to samples: features
features = model.fit_transform(samples)

# Call show_as_image on each component
for component in model.components_:
    show_as_image(component)


'''
Building recommender systems using NMF
Finding similar articles
- Assuming you're an Engineer at a large online newspaper
- Task: Recommend articles similar to article being read by customer
- Similar articles should have similar topics

Strategy
- Apply NMF to the word-frequency array
- NMF feature values decribe the topics
* ... so similar documents have similar NMF feature values
- Compare NMF feature values?

Apply NMF to the word-frequency array
- articles is a word frequency array

from sklearn.decomposition import NMF
nmf = NMF(n_components = 6)
nmf_features = nmf.fit_transform(articles)

Versions of articles
- Different versions of the same document have same topic proportions
* ... exact feature values may be different!
* e.g because one version uses many meaningless words
* But all versions lie on the same line through the origin

Cosine similarity
- Uses the angle between the lines
- Higher values means more similar
- Maximum value is 1, when angle is 0 degrees

Calculating the cosine similarities
from sklearn.preprocessing import normalize
norm_features = normalize(nmf_features)
# if has index 23
current_article =  norm_features[23, :]
similarities = norm_features.dot(current_article)
print(similarities) ->  in

[ 0.7150569 0.26349967 ..., 0.20323616 0.05047817] -> out

DataFrames and labels
- Label similarities with the article titles, using a DataFrame
- Titles given as a list: titles

import pandas as pd
norm_features = normalize(nmf_features)
df = pd.DataFrame(norm_features, index=titles)
current_article = df.loc['Dog bites man']
similarities = df.dot(current_article)
print(similarities.nlargest()) -> in

Dog bites man                       1.000000
Hound mauls cat                     0.979946
Pets go wild!                       0.979708
Dachshunds are dangerous            0.949641
Our streets are no longer safe      0.900474
dtype: float64 -> out
'''

# Perform the necessary imports

# Normalize the NMF features: norm_features
norm_features = normalize(nmf_features)

# Create a DataFrame: df
df = pd.DataFrame(norm_features, index=titles)

# Select the row corresponding to 'Cristiano Ronaldo': article
article = df.loc['Cristiano Ronaldo']

# Compute the dot products: similarities
similarities = df.dot(article)

# Display those with the largest cosine similarity
print(similarities.nlargest())


# Perform the necessary imports

# Create a MaxAbsScaler: scaler
scaler = MaxAbsScaler()

# Create an NMF model: nmf
nmf = NMF(n_components=20)

# Create a Normalizer: normalizer
normalizer = Normalizer()

# Create a pipeline: pipeline
pipeline = make_pipeline(scaler, nmf, normalizer)

# Apply fit_transform to artists: norm_features
norm_features = pipeline.fit_transform(artists)


# Import pandas

# Create a DataFrame: df
df = pd.DataFrame(norm_features, index=artist_names)

# Select row of 'Bruce Springsteen': artist
artist = df.loc['Bruce Springsteen']

# Compute cosine similarities: similarities
similarities = df.dot(artist)

# Display those with highest cosine similarity
print(similarities.nlargest())
