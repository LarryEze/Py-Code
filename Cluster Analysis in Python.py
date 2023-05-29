''' Introduction to Clustering '''

'''
Unsupervised learning: basics
Everyday example: Google news
- How does Google News classify articles?
* Unsupervised Learning Algorithm: 
** Clustering
* Match frequent terms in articles to find similarity

Labeled and unlabeled data
- UnLabeled data are data with no other characteristics available to distinguish the data points.
e.g
point 1: (1, 2)
point 2: (2, 2)
point 3: (3, 1)

- Labeled data are data with a group associated with it before hand to distinguish the data points.
e.g
point 1: (1, 2), Label: Danger Zone
point 2: (2, 2), Label: Normal Zone
point 3: (3, 1), Label: Normal Zone

What is unsupervised learning?
- Its an umbrella term for a group of machine learning algorithms that are used to find patterns in data.
- Data for this algorithms is not labeled, classified or characterized prior to running the algorithm.
- The objective of the algorithm is to find and explain (interpret) inherent structures within the data.
- Common unsupervised learning algorithms:
* Clustering: used to group similar data points together
* Neural networks
* Anomaly detection

What is clustering?
- The process of grouping items with similar characteristics
- The groups are formed such that items in a single group are closer to each other in terms of some characteristics as compared to items in other clusters
- Example: distance between points on a 2D plane

Plotting data for clustering - Pokemon sightings
from matplotlib import pyplot as plt

x_coordinates = [80, 93, 86, 98, 86, 9, 15, 3, 10, 20, 44, 56, 49, 62, 44]
y_coordinates = [87, 96, 95, 92, 92, 57, 49, 47, 59, 55, 25, 2, 10, 24, 10]

plt.scatter(x_coordinates, y_coordinates)
plt.show()
'''

# Import plotting class from matplotlib library

# Create a scatter plot
from matplotlib import image as img
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.cluster.hierarchy import linkage, fcluster
plt.scatter(x, y)

# Display the scatter plot
plt.show()


'''
Basics of cluster analysis
what is cluster?
- A group of items with similar characteristics
e.g
* Google News: articles where similar words and word associations appear together
* Customer segments based on their spending habits

Clustering algorithms
- Hierarchical clustering
- K means clustering
- Other clustering algorithms: Density Based Scan technique (DBSCAN), Gaussian models / methods 

Hierarchical clustering in SciPy
from scipy.cluster.hierarchy import linage, fcluster
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd

x_coordinates = [80.1, 93.1, 86.6, 98.5, 86.4, 9.5, 15.2, 3.4, 10.4, 20.3, 44.2, 56.8, 49.2, 62.5, 44.0]
y_coordinates = [87.2, 96.1, 95.6, 92.4, 92.4, 57.7, 49.4, 47.3, 59.1, 55.5, 25.6, 2.1, 10.9, 24.1, 10.3]

df = pd.DataFrame({'x_coordinate': x_coordinates, 'y_coordinate': y_coordinates})

Z = linkage(df, 'ward')
df['cluster_labels'] = fcluster(Z, 3, criterion='maxclust')

sns.scatterplot(x='x_coordinate', y='y_coordinate', hue='cluster_labels', data=df)
plt.show()

K-means clustering in SciPy
from scipy.cluster.vq import kmeans, vq
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd

import random
random.seed((1000, 2000))

x_coordinates = [80.1, 93.1, 86.6, 98.5, 86.4, 9.5, 15.2, 3.4, 10.4, 20.3, 44.2, 56.8, 49.2, 62.5, 44.0]
y_coordinates = [87.2, 96.1, 95.6, 92.4, 92.4, 57.7, 49.4, 47.3, 59.1, 55.5, 25.6, 2.1, 10.9, 24.1, 10.3]

df = pd.DataFrame({'x_coordinate': x_coordinates, 'y_coordinate': y_coordinates})

centroids = kmeans(df, 3)
df['cluster_labels'], _ = vq(df, centroids)

sns.scatterplot(x='x_coordinate', y='y_coordinate', hue='cluster_labels', data=df)
plt.show()
'''

# Import linkage and fcluster functions

# Use the linkage() function to compute distance
Z = linkage(df, 'ward')

# Generate cluster labels
df['cluster_labels'] = fcluster(Z, 2, criterion='maxclust')

# Plot the points with seaborn
sns.scatterplot(x='x', y='y', hue='cluster_labels', data=df)
plt.show()


# Import kmeans and vq functions

# Compute cluster centers
centroids, _ = kmeans(df, 2)

# Assign cluster labels
df['cluster_labels'], _ = vq(df, centroids)

# Plot the points with seaborn
sns.scatterplot(x='x', y='y', hue='cluster_labels', data=df)
plt.show()


'''
Data preparation for cluster analysis
why do we need to prepare data for clustering?
- Variables have incomparable units (e.g product dimensions in cm vs  price in $)
- Variables with same units have vastly different scales and variances (e.g expenditures on cereals vs travel)
- Data in raw form may lead to bias in clustering
- Clusters may be heavily dependent on one variable
- Solution: normalization of individual variables

Normalization of data
- It is a process y which we rescale the values of a variable with respect to standard deviation of the data (1).

x_new = x / std_dev(x)

from scipy.cluster.vq import whiten

data = [5, 1, 3, 3, 2, 3, 3, 8, 1, 2, 2, 3, 5]

scaled_data = whiten(data)
print(scaled_data) -> in

[2.73, 0.55, 1.64, 1.64, 1.09, 1.64, 1.64, 4.36, 0.55, 1.09, 1.09, 1.64, 2.73] -> out

Illustration: normalization of data
# Import plotting library
from matplotlib import pyplot as plt

# Initialize original, scaled data
plt.plot(data, label='original')
plt.plot(scaled_data, label='scaled')

# Show legend and display plot
plt.legend()
plt.show()
'''

# Import the whiten function

goals_for = [4, 3, 2, 3, 1, 1, 2, 0, 1, 4]

# Use the whiten() function to standardize the data
scaled_data = whiten(goals_for)
print(scaled_data)


# Plot original data
plt.plot(goals_for, label='original')

# Plot scaled data
plt.plot(scaled_data, label='scaled')

# Show the legend in the plot
plt.legend()

# Display the plot
plt.show()


# Prepare data
rate_cuts = [0.0025, 0.001, -0.0005, -0.001, -
             0.0005, 0.0025, -0.001, -0.0015, -0.001, 0.0005]

# Use the whiten() function to standardize the data
scaled_data = whiten(rate_cuts)

# Plot original data
plt.plot(rate_cuts, label='original')

# Plot scaled data
plt.plot(scaled_data, label='scaled')

plt.legend()
plt.show()


# Scale wage and value
fifa['scaled_wage'] = whiten(fifa['eur_wage'])
fifa['scaled_value'] = whiten(fifa['eur_value'])

# Plot the two columns in a scatter plot
fifa.plot(x='scaled_wage', y='scaled_value', kind='scatter')
plt.show()

# Check mean and standard deviation of scaled values
print(fifa[['scaled_wage', 'scaled_value']].describe())


''' Hierarchical Clustering '''

'''
Basics of hierarchical clustering
Creating a distance matrix using linkage
scipy.cluster.hierarchy.linkage(observations, method='single', metric='euclidean', optimal_ordering=False)

- method: It is how to calculate proximity between two clusters
- metric: It decides the distance between two objects (Euclidean distance is a straight line distance between two points on a 2D plane)
- optimal_ordering: it is an optinal argument that changes the order of linkage matrix

which method should use?
- 'single': It decides the proximity of clusters based on their 2 closest objects
- 'complete': It decides the proximity of clusters centers based on their two farthest ojects
- 'average': it decides cluster proximities based on the arithmetic mean of all objects
- 'centroid': it decides cluster proximities based on the geometric mean of all objects
- 'median': it decides cluster proximities based on the median of all objects
- 'ward': it decides cluster proximities based on the sum of squares

Create cluster labels with fcluster
scipy.cluster.hierarchy.fcluster(distance_matrix, num_clusters, criterion)

- distance_matrix: output of linkage() method 
- num_clusters: number of clusters
- criterion: how to decide thresholds to form clusters

Final thougts on selecting a method
- There is no one right method that you can  apply to all problems
- Need to carefully understad the distribution of data that you are going to handle to decide which method is right for your case
'''

# Import the fcluster and linkage functions

# Use the linkage() function
distance_matrix = linkage(
    comic_con[['x_scaled', 'y_scaled']], method='ward', metric='euclidean')

# Assign cluster labels
comic_con['cluster_labels'] = fcluster(
    distance_matrix, 2, criterion='maxclust')

# Plot clusters
sns.scatterplot(x='x_scaled', y='y_scaled',
                hue='cluster_labels', data=comic_con)
plt.show()


# Import the fcluster and linkage functions

# Use the linkage() function
distance_matrix = linkage(
    comic_con[['x_scaled', 'y_scaled']], method='single', metric='euclidean')

# Assign cluster labels
comic_con['cluster_labels'] = fcluster(distance_matrix, 2, 'maxclust')

# Plot clusters
sns.scatterplot(x='x_scaled', y='y_scaled',
                hue='cluster_labels', data=comic_con)
plt.show()


# Import the fcluster and linkage functions

# Use the linkage() function
distance_matrix = linkage(
    comic_con[['x_scaled', 'y_scaled']], method='complete', metric='euclidean')

# Assign cluster labels
comic_con['cluster_labels'] = fcluster(distance_matrix, 2, 'maxclust')

# Plot clusters
sns.scatterplot(x='x_scaled', y='y_scaled',
                hue='cluster_labels', data=comic_con)
plt.show()


'''
Visualize clusters
Why visulaize clusters?
- One can quickly make sense of the clusters formed
- Can serve as an additional step in validation of clusters formed
- You may also spot trends in data

An introduction to seaborn
- seaborn: It is a data visualization library in Python that is based on matplotlib
- It has better default plotting themes, which can be easily and intuitively modified
- It hs functions for quick visualizations in the context of data analytics
- Use case for clustering: hue parameter for plots

Visualize clusters with matplotlib
from matplotlib import pyplot as plt

df = pd.DataFrame({'x': [2, 3, 5, 6, 2], 'y': [1, 1, 5, 5, 2], 'labels': ['A', 'A', 'B', 'B', 'A']})
colors = {'A': 'red', 'B': 'blue'}
df.plot.scatter(x='x', y='y', c=df['labels'].apply(lambda x: colors[x]))
plt.show()

Visualize clusters with matplotlib
from matplotlib import pyplot as plt
import seaborn as sns

df = pd.DataFrame({'x': [2, 3, 5, 6, 2], 'y': [1, 1, 5, 5, 2], 'labels': ['A', 'A', 'B', 'B', 'A']})
sns.scatterplot(x='x', y='y', hue- 'labels', data=df)
plt.show()
'''

# Import the pyplot class

# Define a colors dictionary for clusters
colors = {1: 'red', 2: 'blue'}

# Plot a scatter plot
comic_con.plot.scatter(x='x_scaled', y='y_scaled',
                       c=comic_con['cluster_labels'].apply(lambda x: colors[x]))
plt.show()


# Import the seaborn module

# Plot a scatter plot using seaborn
sns.scatterplot(x='x_scaled', y='y_scaled',
                hue='cluster_labels', data=comic_con)
plt.show()


'''
How many clusters?
Introduction to dendrograms
- Strategy till now - decide clusters on visual inspection
- Dendrograms help in showing progressions as clusters are merged
- A dendrogram is a branching diagram that demonstrates how each cluster is composed by branching out into its child nodes

Create a dendrogram in SciPy
from scipy.cluster.hierarchy import dendrogram

Z = linkage(df[['x_whiten', 'y_whiten']], method='ward', metric='euclidean')
dn = dendrogram(Z)
plt.show()
'''

# Import the dendrogram function

# Create a dendrogram
dn = dendrogram(
    linkage(comic_con[['x_scaled', 'y_scaled']], method='ward', metric='euclidean'))

# Display the dendogram
plt.show()


'''
Limitations of hierarchical clustering
Measuring speed in hierarchical clustering
- timeit module
- Measure the speed of .linkage() method
- Use randomly generated points
- Run various iterations to extrapolate

Use of timeit module
from scipy.cluster.hierarchy import linkage
import pandas as pd
import random
import timeit

points = 100
df = pd.DataFrame({'x': random.sample(range(0, points), points), 'y': random.sample(range(0, points), points)})
%timeit linkage(df[['x', 'y']], method = 'ward', metric = 'euclidean') -> in

Comparison of runtime of linkage method
- Increasing runtime with data points
- Quadratic increase of runtime
- Not feasible for large datasets
'''

# Fit the data into a hierarchical clustering algorithm
distance_matrix = linkage(
    fifa[['scaled_sliding_tackle', 'scaled_aggression']], 'ward')

# Assign cluster labels to each row of data
fifa['cluster_labels'] = fcluster(distance_matrix, 3, criterion='maxclust')

# Display cluster centers of each cluster
print(fifa[['scaled_sliding_tackle', 'scaled_aggression',
      'cluster_labels']].groupby('cluster_labels').mean())

# Create a scatter plot through seaborn
sns.scatterplot(x='scaled_sliding_tackle', y='scaled_aggression',
                hue='cluster_labels', data=fifa)
plt.show()


''' K-Means Clustering '''

'''
Basics of k-means clustering
Why k-means clustering?
- Its a critical drawback of hierachical clustering: runtime
- K means runs significantly faster on large datasets

Step 1: Generate cluster centers
kmeans(obs, k_or_guess, iter, thresh, check_finite)

- obs: It is the list of observations, which have been standardized through the whiten method
- k_or_guess: It is the number of clusters
- iter: It is the number of iterations of the algorithm to perform (default: 20)
- thres: It is its threshold (default: 1e-05)
- check_finite: It is a boolean value indicating if a check needs to be performed on the data for the presence of infinite or NaN values (default: True).

- It returns two objects: Cluster centers, Distortion
- The cluster centers is also known as the code book
- The distortion is calculated as the sum of square of distances between the data points and cluster centers

Step 2: Generate cluster labels
vq(obs, code_book, check_finite=True)

- obs: It is the list of observations, which have been standardized through the whiten method
- code_book: cluster centers
- check_finite: It is a boolean value indicating if a check needs to be performed on the data for the presence of infinite or NaN values (default: True).

- It returns two objects: a list of Cluster labels, a list of Distortion

A note on distortions
- Kmeans returns a single value of distortions based on the overall data
- vq returns a list of distortions, one for each data point.
- The mean of the list of distortions from the vq method should approximately equal the distortion value of the kmeans method if the same list of observations is passed.

Running k-means
# Import kmeans and vq functions
from scipy.cluster.vq import kmeans, vq

# Generate cluster centers and labels
cluster_centers, _ = kmeans(df[['scaled_x', 'scaled_y']], 3)
df['cluster_labels'], _ = vq(df[['scaled_x', 'scaled_y']], cluster_centers)

# Plot clusters
sns.scatterplot(x='scaled_x', y='scaled_y', hue='cluster_laels', data=df)
plt.show()
'''

# Import the kmeans and vq functions

# Generate cluster centers
cluster_centers, distortion = kmeans(comic_con[['x_scaled', 'y_scaled']], 2)

# Assign cluster labels
comic_con['cluster_labels'], distortion_list = vq(
    comic_con[['x_scaled', 'y_scaled']], cluster_centers)

# Plot clusters
sns.scatterplot(x='x_scaled', y='y_scaled',
                hue='cluster_labels', data=comic_con)
plt.show()


'''
How many clusters?
How to find the right k?
- There is no absolute method (right way) of finding right number of clusters (k) in k-means clustering dataset
- Elbow plot method

Distortions revisited
- It is  the sum of the squares of distances between data points and its cluster center
- It has an inverse relationship with the number of clusters i.e distortion decreases with increasing number of clusters
-  It becomes zero when the number of clusters equals the number of points
- Elbow plot: it is a line plot between the number of clusters and their corresponding distortions

Elbow method
- It is a line plot between the number of clusters and their corresponding distortions
- The ideal point is one beyond which the distortion decreases relatively less on increasing the number of clusters

Elbow method in Python
# Declaring variables for use
distortions = []

num_clusters = range(2, 7)

# Populating distortions for various clusters
for i in num_clusters:
    centroids, distortion = kmeans(df[['scaled_x', 'scaled_y']], i)
    distortions.append(distortion)

# Plotting elbow plot data
elbow_plot_data = pd.DataFrame({'num_clusters': num_clusters, 'distortions': distortions})

sns.lineplot(x='num_clusters', y='distortions', data= elbow_plot_data)
plt.show()

Final thoughts on using the elbow method
- It only gives an indication of optimal k (numbers of clusters)
- It does not pinpoint how many k (numbers of clusters)
- Other methods to find k: Average silhouette and gap statistic methods
'''

distortions = []
num_clusters = range(1, 7)

# Create a list of distortions from the kmeans function
for i in num_clusters:
    cluster_centers, distortion = kmeans(
        comic_con[['x_scaled', 'y_scaled']], i)
    distortions.append(distortion)

# Create a DataFrame with two lists - num_clusters, distortions
elbow_plot = pd.DataFrame(
    {'num_clusters': num_clusters, 'distortions': distortions})

# Creat a line plot of num_clusters and distortions
sns.lineplot(x='num_clusters', y='distortions', data=elbow_plot)
plt.xticks(num_clusters)
plt.show()


distortions = []
num_clusters = range(2, 7)

# Create a list of distortions from the kmeans function
for i in num_clusters:
    cluster_centers, distortion = kmeans(
        uniform_data[['x_scaled', 'y_scaled']], i)
    distortions.append(distortion)

# Create a DataFrame with two lists - number of clusters and distortions
elbow_plot = pd.DataFrame(
    {'num_clusters': num_clusters, 'distortions': distortions})

# Creat a line plot of num_clusters and distortions
sns.lineplot(x='num_clusters', y='distortions', data=elbow_plot)
plt.xticks(num_clusters)
plt.show()


'''
Limitations of k-means clustering
- Procedures to find the right _K_ (number of clusters)?
- The impact of seeds on clustering
- The formation of equal sized clusters

Impact of seeds
Initialize a random seed

from numpy import random
random.seed(12)

- The effect of seeds is only seen when the data to be clustered is fairly uniform, i.e if the data has distinct clusters before clustering is performed, the effect of seeds will not result in any changes in the formation of resulting clusters


Final thoughts
- Each technique has its pros and cons
- Consider your data size and patterns before deciding on algorithm
- Clustering is the exploratory phase of analysis
'''

# Import random class

# Initialize seed
random.seed(0)

# Run kmeans clustering
cluster_centers, distortion = kmeans(comic_con[['x_scaled', 'y_scaled']], 2)
comic_con['cluster_labels'], distortion_list = vq(
    comic_con[['x_scaled', 'y_scaled']], cluster_centers)

# Plot the scatterplot
sns.scatterplot(x='x_scaled', y='y_scaled',
                hue='cluster_labels', data=comic_con)
plt.show()


# Import random class

# Initialize seed
random.seed([1, 2, 1000])

# Run kmeans clustering
cluster_centers, distortion = kmeans(comic_con[['x_scaled', 'y_scaled']], 2)
comic_con['cluster_labels'], distortion_list = vq(
    comic_con[['x_scaled', 'y_scaled']], cluster_centers)

# Plot the scatterplot
sns.scatterplot(x='x_scaled', y='y_scaled',
                hue='cluster_labels', data=comic_con)
plt.show()


# Import the kmeans and vq functions

# Generate cluster centers
cluster_centers, distortion = kmeans(mouse[['x_scaled', 'y_scaled']], 3)

# Assign cluster labels
mouse['cluster_labels'], distortion_list = vq(
    mouse[['x_scaled', 'y_scaled']], cluster_centers)

# Plot clusters
sns.scatterplot(x='x_scaled', y='y_scaled', hue='cluster_labels', data=mouse)
plt.show()


# Set up a random seed in numpy
random.seed([1000, 2000])

# Fit the data into a k-means algorithm
cluster_centers, _ = kmeans(fifa[['scaled_def', 'scaled_phy']], 3)

# Assign cluster labels
fifa['cluster_labels'], _ = vq(
    fifa[['scaled_def', 'scaled_phy']], cluster_centers)

# Display cluster centers
print(fifa[['scaled_def', 'scaled_phy', 'cluster_labels']
           ].groupby('cluster_labels').mean())

# Create a scatter plot through seaborn
sns.scatterplot(x='scaled_def', y='scaled_phy',
                hue='cluster_labels', data=fifa)
plt.show()


''' Clustering in Real World '''

'''
Dominant colors in images
- All images consist of pixels 
- Each pixel has 3 values: Red, Green and Blue
- Pixel color: combination of these RGB values
- Perform k-means on standardized RGB values to find cluster centers
- Uses: Identifying features in satelite images

Tools to find dominant colors
- Convert image to pixels: matplotlib.image.imread
- Display colors of cluster centers: matplotlib.pyplot.imshow

Convert image to RGB matrix
import matplotlib.image as img
image = img.imread('sea.jpg')
image.shape -> in

(475, 764, 3) -> out

r = []
g = []
b = []

for row in image:
    for pixel in row:
        # A pixel contains RGB values
        temp_r, temp_g, temp_b = pixel
        r.append(temp_r)
        g.append(temp_g)
        b.append(temp_b)

DataFrame with RGB values
pixels = pd.DataFrame({'red': r, 'blue': b, 'green': g})

Create an elbow plot
distortions = []
num_clusters = range(1, 11)

# Create a list of distortions from the kmeans method
for i in num_clusters:
    cluster_centers, _ = kmeans(pixels[['scaled_red', 'scaled_blue', scaled_green']], i)
    distortions.append(distortion)

# Create a DataFrame with 2 lists - number of clusters and distortions
elbow_plot = pd.DataFrame({'num_clusters': num_clusters, 'distortions': distortions})

# Create a line plot of num_clusters and distortions
sns.lineplot(x='num_clusters', y='distortions', data=elbow_plot)
plt.xticks(num_clusters)
plt.show()

Find dominant colors
cluster_centers, _ = kmeans(pixels[['scaled_red', scaled_green', 'scaled_blue']], 2)

colors = []

# Find Stadard Deviations
r_std, g_std, b_std = pixels[['red', 'green', 'blue']].std()

# Scale actual RGB values in range of 0-1
for cluster_center in cluster_centers:
    scled_r, scaled_g, scaled_b = cluster_center
    colors.append((scaled_r * r_std / 255, scaled_g * g_std / 255, scaled_b * b_std / 255))

Display dominant colors
# Dimensions: 2 x 3 ( N x 3 matrix)
print(colors) -> in

[(0.08192923122023911, 0.34205845943857993, 0.2824002984155429), (0.893281510956742, 0.899818770315129, 0.8979114272960784)] -> out

# Dimensions: 1 x 2 x 3 (1 xN x 3 matrix)
plt.imshow([colors])
plt.show()
'''

# Import image class of matplotlib

# Read batman image and print dimensions
batman_image = img.imread('batman.jpg')
print(batman_image.shape)

# Store RGB values of all pixels in lists r, g and b
for pixel in batman_image:
    for temp_r, temp_g, temp_b in pixel:
        r.append(temp_r)
        g.append(temp_g)
        b.append(temp_b)


distortions = []
num_clusters = range(1, 7)

# Create a list of distortions from the kmeans function
for i in num_clusters:
    cluster_centers, distortion = kmeans(
        batman_df[['scaled_red', 'scaled_blue', 'scaled_green']], i)
    distortions.append(distortion)

# Create a DataFrame with two lists, num_clusters and distortions
elbow_plot = pd.DataFrame(
    {'num_clusters': num_clusters, 'distortions': distortions})

# Create a line plot of num_clusters and distortions
sns.lineplot(x='num_clusters', y='distortions', data=elbow_plot)
plt.xticks(num_clusters)
plt.show()


# Get standard deviations of each color
r_std, g_std, b_std = batman_df[['red', 'green', 'blue']].std()

for cluster_center in cluster_centers:
    scaled_r, scaled_g, scaled_b = cluster_center
    # Convert each standardized value to scaled value
    colors.append((
        scaled_r * r_std / 255,
        scaled_g * g_std / 255,
        scaled_b * b_std / 255
    ))

# Display colors of cluster centers
plt.imshow([colors])
plt.show()


'''
Document clustering
Document clustering: concepts
- Clean data before processing and remove items such as punctuation, emoticons and words such as 'the, is, are'.
- Determine the importance of the terms in a document (in TF-IDF matrix, or a weighted statistic)
- Cluster the TF-IDF matrix
- Find top terms, documents in each cluster

Clean and tokenize data
- Convert text into smaller parts called tokens, clean data for processing

from nltk.tokenize import word_tokenize
import re

def remove_noise(text, stop_words = []):
    tokens = word_tokenize(text)
    cleaned_tokens = []
    for token in tokens:
        token = re.sub('[A-Za-z0-9]+', '', token)
        if len(token) > 1 and token.lower() not in stope_words:
            # Get lowercase
            cleaned_tokens.append(token.lower())
    return cleaned _tokens

remove_noise('It is lovely weather we are having. I hope the weather continues.') -> in

['lovely', 'weather', 'hope'. 'weather', 'continues'] -> out

Document term matrix and sparse matrices
- Document term matrix formed
- Most elements in matrix are zeros
- Sparse matrix is created ( contains terms which have non zero elements

TD-IDF (Term Frequencey - Inverse Document Frequency)
- A weighted measure: evaluate how important a word is to a document in a collection

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=50, min_df=0.2, tokenizer=remove_noise)
tfidf_matrix = tfidf_vectorizer.fit_transform(data)

Clustering with sparse matrix
- kmeans() in SciPy doesn not support sparse matrices
- Use .todense() to convert to a matrix

cluster_centers, distortion = kmeans(tfidf_matrix.todense(), num_clusters)

Top terms per cluster
- Cluster centers: its a list of tfidf weights, which signifies the importance of each term in the matrix.
- Each value in the cluster center is its importance
- Create a dictionary and print top terms

terms = tfidf_vectorizer.get_feature_names_out()

for i in range(num_clusters):
    center_terms = dict(zip(terms, list(cluster_centers[i])))
    sorted_terms = sorted(center_terms, key=center_terms.get, reverse=True)
    print(sorted_terms[:3]) -> in

['room', 'hotel', 'staff']

['bad', 'location', 'breakfast'] -> out

More considerations
- Can modify the remove_noise() method to filter hyperlinks, or replace emoticons with text.
- Can Normalize every word to its base form.(e.g run, ran and running -> run)
- .todense() method may not work with large datasets, and may need to consider an implementation of k-means that works with sparse matrices
'''

# Import TfidfVectorizer class from sklearn

# Initialize TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(
    max_df=0.75, max_features=50, min_df=0.1, tokenizer=remove_noise)

# Use the .fit_transform() method on the list plots
tfidf_matrix = tfidf_vectorizer.fit_transform(plots)


num_clusters = 2

# Generate cluster centers through the kmeans function
cluster_centers, distortion = kmeans(tfidf_matrix.todense(), num_clusters)

# Generate terms from the tfidf_vectorizer object
terms = tfidf_vectorizer.get_feature_names_out()

for i in range(num_clusters):
    # Sort the terms and print top 3 terms
    center_terms = dict(zip(terms, list(cluster_centers[i])))
    sorted_terms = sorted(center_terms, key=center_terms.get, reverse=True)
    print(sorted_terms[:3])


'''
Clustering with multiple features 
- This step assumes that you have created the elbow plot, performed the clustering process and generated cluster labels.

Basic checks
# Cluster centers
print(fifa.groupby('cluster_labels')[['scaled_heading_accuracy', 'scaled_volleys', 'scaled_finishing']].mean()

# Cluster sizes
print(fifa.groupby('cluster_labels')['ID'].count())

Visualizations
- Visualize cluster centers
- Visualize other variables stacked against eachother
e.g
# Plot cluster centers
fifa.groupby('cluster_labels')[scaled_features].mean().plot(kind = 'bar')
plt.show()

Top items in clusters
# Get the name column of top 5 players in each cluster
for cluster in fifa['cluster_labels'] == cluster]['name'].values[:5])

Feature reduction
- Popular tools include:
* Factor analysis
- Multidimensional scaling
'''

# Print the size of the clusters
print(fifa.groupby('cluster_labels')['ID'].count())

# Print the mean value of wages in each cluster
print(fifa.groupby('cluster_labels')['eur_wage'].mean())


# Create centroids with kmeans for 2 clusters
cluster_centers, _ = kmeans(fifa[scaled_features], 2)

# Assign cluster labels and print cluster centers
fifa['cluster_labels'], _ = vq(fifa[scaled_features], cluster_centers)
print(fifa.groupby('cluster_labels')[scaled_features].mean())

# Plot cluster centers to visualize clusters
fifa.groupby('cluster_labels')[scaled_features].mean().plot(
    legend=True, kind='bar')
plt.show()

# Get the name column of first 5 players in each cluster
for cluster in fifa['cluster_labels'].unique():
    print(cluster, fifa[fifa['cluster_labels'] == cluster]['name'].values[:5])
