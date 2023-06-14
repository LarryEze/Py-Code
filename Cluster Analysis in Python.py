# import necessary packages
from numpy import random
import matplotlib.pyplot as plt
import matplotlib.image as img
import seaborn as sns
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage, dendrogram
from scipy.cluster.vq import kmeans, vq, whiten
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
import re


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

x = [9, 6, 2, 3, 1, 7, 1, 6, 1, 7, 23, 26, 25, 23, 21, 23, 23, 20, 30, 23]

y = [8, 4, 10, 6, 0, 4, 10, 10, 6, 1, 29, 25, 30, 29, 29, 30, 25, 27, 26, 30]

# Import plotting class from matplotlib library

# Create a scatter plot
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

x_list = list(map(float, x))
y_list = list(map(float, y))


df = pd.DataFrame({'x': x_list, 'y': y_list})
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


fifa = pd.read_csv('Cluster Analysis in Python/fifa_18_sample_data.csv')

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
- optimal_ordering: it is an optional argument that changes the order of linkage matrix

which method should use?
- 'single': It decides the proximity of clusters based on their 2 closest objects
- 'complete': It decides the proximity of clusters centers based on their two farthest objects
- 'average': it decides cluster proximities based on the arithmetic mean of all objects
- 'centroid': it decides cluster proximities based on the geometric mean of all objects
- 'median': it decides cluster proximities based on the median of all objects
- 'ward': it decides cluster proximities based on the sum of squares

Create cluster labels with fcluster
scipy.cluster.hierarchy.fcluster(distance_matrix, num_clusters, criterion)

- distance_matrix: output of linkage() method 
- num_clusters: number of clusters
- criterion: how to decide thresholds to form clusters

Final thoughts on selecting a method
- There is no one right method that you can  apply to all problems
- Need to carefully understand the distribution of data that you are going to handle to decide which method is right for your case
'''

x_coordinate = [17, 20, 35, 14, 37, 33, 14, 30, 35, 17, 11, 21, 13, 10, 81, 84,
                87, 83, 90, 97, 94, 88, 89, 93, 92, 82, 81, 92, 91, 22, 23, 25, 25, 27, 17, 17]
y_coordinate = [4,   6,   0,   0,   4,   3,   1,   6,   5,   4,   6,  10,   8, 10,  97,  94,  99,
                95,  95,  97,  99,  99,  94,  99,  90,  98, 100,  93,  98,  15,  10,   0,  10,   7,  17,  15]
x_scaled = [0.50934905, 0.59923418, 1.04865981, 0.41946392, 1.10858323, 0.98873639, 0.41946392, 0.89885127, 1.04865981, 0.50934905, 0.3295788, 0.62919589, 0.38950222, 0.29961709, 2.42689842, 2.51678354, 2.60666867, 2.48682183,
            2.6965538, 2.90628576, 2.81640063, 2.63663038, 2.66659209, 2.78643892, 2.75647721, 2.45686013, 2.42689842, 2.75647721, 2.72651551, 0.65915759, 0.6891193, 0.74904272, 0.74904272, 0.80896614, 0.50934905, 0.50934905]
y_scaled = [0.09000985, 0.13501477, 0., 0., 0.09000985, 0.06750738, 0.02250246, 0.13501477, 0.11251231, 0.09000985, 0.13501477, 0.22502461, 0.18001969, 0.22502461, 2.18273875, 2.11523137, 2.22774367, 2.13773383,
            2.13773383, 2.18273875, 2.22774367, 2.22774367, 2.11523137, 2.22774367, 2.02522152, 2.20524121, 2.25024613, 2.0927289, 2.20524121, 0.33753692, 0.22502461, 0., 0.22502461, 0.15751723, 0.38254184, 0.33753692]

comic_con = pd.DataFrame({'x_coordinate': x_coordinate,
                         'y_coordinate': y_coordinate, 'x_scaled': x_scaled, 'y_scaled': y_scaled})
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
Why visualize clusters?
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

# Display the dendrogram
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

fifa = pd.read_csv('Cluster Analysis in Python/fifa_18_dataset.csv')

# Scale sliding_tackle and aggression
fifa['scaled_sliding_tackle'] = whiten(fifa['sliding_tackle'])
fifa['scaled_aggression'] = whiten(fifa['aggression'])

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
- Its a critical drawback of hierarchical clustering: runtime
- K means runs significantly faster on large datasets

Step 1: Generate cluster centers
kmeans(obs, k_or_guess, iter, thresh, check_finite)

- obs: It is the list of observations, which have been standardized through the whiten method
- k_or_guess: It is the number of clusters
- iter: It is the number of iterations of the algorithm to perform (default: 20)
- thresh: It is its threshold (default: 1e-05)
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
sns.scatterplot(x='scaled_x', y='scaled_y', hue='cluster_labels', data=df)
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

# Create a line plot of num_clusters and distortions
sns.lineplot(x='num_clusters', y='distortions', data=elbow_plot)
plt.xticks(num_clusters)
plt.show()


x_coordinate = [39, 42, 58, 43, 13, 32, 60, 13, 26, 27, 29, 51, 14,
                50, 62, 59, 50, 62, 65, 17, 25, 45, 55, 48, 42, 58, 68, 58, 37, 55]
y_coordinate = [3, 7, 3, 3, 6, 5, 3, 4, 0, 9, 6, 3, 0,
                7, 4, 1, 3, 0, 2, 5, 9, 5, 8, 6, 3, 1, 4, 2, 8, 7]
x_scaled = [2.37619911, 2.55898365, 3.53383457, 2.61991184, 0.79206637, 1.94970183, 3.65569093, 0.79206637, 1.58413274, 1.64506092, 1.76691728, 3.10733729, 0.85299455, 3.04640911,
            3.7775473, 3.59476275, 3.04640911, 3.7775473, 3.96033185, 1.0357791, 1.52320456, 2.7417682, 3.35105002, 2.92455275, 2.55898365, 3.53383457, 4.14311639, 3.53383457, 2.25434274, 3.35105002]
y_scaled = [1.15223748, 2.68855411, 1.15223748, 1.15223748, 2.30447496, 1.9203958, 1.15223748, 1.53631664, 0., 3.45671243, 2.30447496, 1.15223748, 0., 2.68855411, 1.53631664,
            0.38407916, 1.15223748, 0., 0.76815832, 1.9203958, 3.45671243, 1.9203958, 3.07263327, 2.30447496, 1.15223748, 0.38407916, 1.53631664, 0.76815832, 3.07263327, 2.68855411]

uniform_data = pd.DataFrame({'x_coordinate': x_coordinate,
                            'y_coordinate': y_coordinate, 'x_scaled': x_scaled, 'y_scaled': y_scaled})

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

# Create a line plot of num_clusters and distortions
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


x_coordinate = [33.87552801, 38.20874789, 35.74058788, 32.54696343, 62.06314583, 53.76968389, 32.49035022, 55.32520176, 68.10254176, 53.28312355, 67.71803005, 32.84550285, 61.78083409, 62.07246733, 37.2423105, 48.39279681, 35.48978845, 65.60854566, 38.02888394, 65.49367503, 41.60372439, 48.41398475, 41.8196699, 51.7691827, 45.17995059, 38.02218597, 56.37185206, 52.05250382, 44.24373638, 40.3328119, 53.5965755, 45.87265296, 54.0000117, 52.6886879, 53.25422918, 63.54212627, 65.38884629, 49.73312042, 48.58080531, 48.03000662, 44.6671251, 65.30935896, 40.58584807, 43.93533778, 32.87374964, 43.17638646, 61.44735547, 47.90212636, 47.57275054, 50.65221647, 45.44652101, 48.17757171, 50.11893028, 37.33082581, 53.28605845, 32.60601743, 46.59545712, 54.91413224, 61.18758173, 62.256238, 47.80710345, 65.44265167, 36.17446485, 60.90339818, 44.23811688, 40.34232189, 57.19910238, 36.08806305, 35.56663867, 42.41968177, 59.65269146, 32.62462929, 55.80216992, 65.4991867, 45.54670964, 61.00536902, 58.19939104, 63.11931771, 55.31708313, 45.60947216,
                45.63141645, 46.74818575, 68.28352429, 61.96714675, 48.96104619, 45.37856388, 50.84116864, 52.56595621, 39.13493121, 48.99176373, 58.05200858, 41.37122591, 64.35140055, 35.02226785, 48.60847435, 42.64864932, 57.58495149, 49.08107021, 51.42659241, 62.31551719, 26.84075282, 26.9519412, 32.56213982, 27.10884541, 20.77972888, 34.64864231, 27.3347144, 38.63528806, 27.36278088, 22.25173068, 23.56181791, 28.68340206, 33.84077613, 28.22316026, 38.08583938, 29.88562784, 28.40850494, 31.72078699, 25.2107398, 28.67688927, 32.24862075, 26.38752714, 28.69901004, 26.07549743, 34.28710983, 24.02682404, 30.83187787, 32.37571559, 22.76230802, 35.25122351, 70.18239123, 79.58988482, 65.72716493, 68.483399, 76.41358468, 79.26874523, 64.84182259, 71.11774102, 78.34020408, 63.61846311, 67.53968321, 69.15087805, 67.7921338, 76.87140431, 62.00437158, 67.49125961, 71.90747913, 65.27576084, 77.58529515, 79.66066418, 77.3246352, 68.5790018, 65.97649575, 71.55575552, 69.83671683, 70.49432188, 72.62743438, 67.66970193, 63.44559189, 75.45783601]
y_coordinate = [44.89342095, 41.11632695, 57.41800588, 57.21808249, 47.19694448, 35.95195917, 42.02185361, 52.54832221, 52.31212456, 31.63628689, 47.60524807, 54.23496141, 60.8249876, 65.23558661, 48.23757709, 67.52456486, 54.10211414, 51.8513512, 39.82817721, 44.28386899, 60.21348801, 45.35513786, 35.81066079, 56.95992181, 61.05673695, 44.3455702, 49.53371078, 57.76601798, 41.7427565, 52.07742897, 55.58701182, 43.81629667, 61.46690781, 44.37759236, 44.72275113, 47.98225455, 47.73690086, 46.04982378, 56.78708413, 56.60380297, 61.71604212, 56.52781557, 45.81317068, 49.10776882, 41.14505632, 64.84595737, 62.46952021, 58.29912471, 48.53320807, 48.79467674, 40.65821118, 30.46658829, 65.18053982, 41.7300068, 69.24321762, 56.00106984, 39.63738121, 38.40458709, 39.63915235, 38.73957078, 36.00996935, 45.62729042, 57.17020665, 51.32130045, 40.89880525, 41.15484687, 58.21497749, 43.45728537, 42.72274718, 39.27124659, 42.66172394, 57.40714755, 34.25546896, 61.29461186, 30.51392786, 65.03769516, 63.37885691, 62.92620573, 35.88019397,
                52.89656063, 37.25902081, 31.67028002, 47.16555533, 50.88239704, 41.7797539, 49.38557057, 52.58865855, 40.89034886, 65.91171887, 46.57669291, 63.55568215, 45.4501224, 43.70526533, 46.58949707, 38.91660456, 55.46764508, 67.89462117, 36.02362381, 59.98536063, 54.70049854, 82.72356822, 77.94443633, 75.40050307, 68.91634077, 72.8427495, 74.71747016, 70.61496975, 76.12679438, 80.60280865, 70.19696262, 68.23133123, 83.83579387, 83.97371743, 70.71122299, 72.22425104, 79.61715462, 74.81720151, 81.28549433, 79.83947613, 82.5450522, 72.41216494, 74.6039565, 68.84261456, 73.2893676, 80.51600557, 82.63965729, 66.38754732, 73.35737436, 72.53759838, 73.71960775, 78.16166533, 75.99349931, 67.23738628, 82.50606009, 77.72556254, 73.68057343, 83.33967753, 84.67897398, 71.78110972, 68.16591019, 83.87925655, 74.22367682, 78.15268345, 74.21207007, 75.13106436, 77.71960453, 71.08256275, 74.66344797, 80.3555998, 73.34017066, 70.12614603, 77.46797551, 77.65232497, 76.70948616, 73.94355101, 71.19864361, 76.43418687, 70.62890953, 78.5744389, 68.44893502]
x_scaled = [2.20945827, 2.4920832, 2.33110278, 2.12280551, 4.04793487, 3.5070117, 2.11911304, 3.608467, 4.44184145, 3.47527685, 4.4167625, 2.1422771, 4.0295217, 4.04854284, 2.4290494, 3.15631583, 2.31474493, 4.27917593, 2.48035196, 4.27168374, 2.71351322, 3.15769777, 2.7275978, 3.37653332, 2.94676486, 2.4799151, 3.6767325, 3.39501233, 2.88570231, 2.63062069, 3.49572107, 2.99194488, 3.52203433, 3.43650606, 3.47339227, 4.14439818, 4.26484651, 3.24373555, 3.16857828, 3.13265362, 2.91331693, 4.25966212, 2.64712443, 2.86558768, 2.14411944, 2.81608671, 4.00777127, 3.12431291, 3.10283008, 3.30368161, 2.96415134, 3.14227824, 3.26889916, 2.43482262, 3.47546827, 2.12665718, 3.03908822, 3.58165587, 3.99082809, 4.06052889, 3.11811524, 4.26835585, 2.35940147, 3.97229283, 2.88533579, 2.63124096, 3.73068812, 2.3537661, 2.31975732, 2.76673228, 3.89071818, 2.1278711, 3.63957622, 4.27204323, 2.97068593, 3.97894366, 3.79592979, 4.1168214, 3.60793748, 2.97477948,
            2.97621075, 3.04904962, 4.45364564, 4.04167354, 3.19337867, 2.95971899, 3.3160056, 3.42850115, 2.55249151, 3.19538216, 3.78631709, 2.69834901, 4.19718135, 2.28425191, 3.17038293, 2.78166619, 3.7558543, 3.20120698, 3.35418861, 4.06439525, 1.75063023, 1.75788225, 2.12379536, 1.76811599, 1.35531301, 2.25988299, 1.78284781, 2.51990336, 1.78467838, 1.45132116, 1.53676877, 1.87081305, 2.20719166, 1.84079477, 2.48406675, 1.94922564, 1.85288348, 2.06891994, 1.64431615, 1.87038827, 2.10334676, 1.72106956, 1.87183105, 1.7007181, 2.23630281, 1.56709779, 2.01094276, 2.11163624, 1.48462246, 2.29918505, 4.57749514, 5.19107863, 4.28691262, 4.46668204, 4.9839113, 5.17013299, 4.22916807, 4.63850132, 5.1095709, 4.14937709, 4.40513021, 4.51021692, 4.42159576, 5.0137716, 4.04410145, 4.40197189, 4.69001028, 4.25747076, 5.06033359, 5.19569506, 5.04333261, 4.47291753, 4.30317468, 4.66706987, 4.55494928, 4.59784015, 4.73696781, 4.4136104, 4.13810193, 4.92157466]
y_scaled = [2.97752364, 2.72701061, 3.80820766, 3.79494789, 3.13030317, 2.38448766, 2.78706901, 3.48522942, 3.46956378, 2.09825382, 3.15738361, 3.59709454, 4.03417326, 4.32670304, 3.19932237, 4.47851787, 3.58828354, 3.43900332, 2.64157501, 2.93709554, 3.99361599, 3.00814668, 2.37511614, 3.77782557, 4.04954387, 2.94118783, 3.2852875, 3.83128931, 2.7685581, 3.45399777, 3.68676831, 2.90608415, 4.07674815, 2.94331168, 2.96620408, 3.18238829, 3.16611538, 3.05422122, 3.76636223, 3.75420624, 4.0932718, 3.74916643, 3.03852537, 3.25703721, 2.72891607, 4.30086117, 4.14324569, 3.86664723, 3.21892988, 3.2362716, 2.69662641, 2.02067441, 4.3230521, 2.76771248, 4.59250627, 3.71423041, 2.62892061, 2.54715643, 2.62903808, 2.56937398, 2.38833514, 3.026197, 3.79177256, 3.40384809, 2.71258364, 2.72956542, 3.86106623, 2.88227298, 2.83355527, 2.60463699, 2.82950795, 3.80748749, 2.27196917, 4.06532075, 2.02381417, 4.31357804, 4.20355679, 4.17353503, 2.37972788,
            3.50832608, 2.47117757, 2.10050839, 3.1282213, 3.37473814, 2.77101192, 3.27546221, 3.4879047, 2.71202278, 4.37154703, 3.08916543, 4.2152846, 3.01444646, 2.89872007, 3.09001466, 2.58111562, 3.67885139, 4.50306158, 2.38924076, 3.97848561, 3.62797096, 5.48658076, 5.1696083, 5.00088377, 4.5708264, 4.83124261, 4.9555821, 4.68348673, 5.04905451, 5.34592292, 4.65576271, 4.52539364, 5.56034832, 5.569496, 4.68987066, 4.79022115, 5.28055013, 4.96219671, 5.39120155, 5.29529544, 5.47474081, 4.8026844, 4.94805339, 4.56593656, 4.86086423, 5.34016576, 5.48101543, 4.403106, 4.86537473, 4.81100369, 4.8893996, 5.18401585, 5.04021381, 4.45947096, 5.47215469, 5.15509165, 4.88681068, 5.52744376, 5.61627163, 4.76083012, 4.52105463, 5.56323095, 4.92283162, 5.18342014, 4.92206181, 4.98301344, 5.15469649, 4.7144995, 4.95199912, 5.32952697, 4.86423371, 4.65106585, 5.13800738, 5.15023422, 5.08770112, 4.90425248, 4.72219848, 5.0694421, 4.68441128, 5.21139276, 4.53982605]

mouse = pd.DataFrame({'x_coordinate': x_coordinate,
                     'y_coordinate': y_coordinate, 'x_scaled': x_scaled, 'y_scaled': y_scaled})
# Import the kmeans and vq functions

# Generate cluster centers
cluster_centers, distortion = kmeans(mouse[['x_scaled', 'y_scaled']], 3)

# Assign cluster labels
mouse['cluster_labels'], distortion_list = vq(
    mouse[['x_scaled', 'y_scaled']], cluster_centers)

# Plot clusters
sns.scatterplot(x='x_scaled', y='y_scaled', hue='cluster_labels', data=mouse)
plt.show()


fifa = pd.read_csv('Cluster Analysis in Python/fifa_18_sample_data.csv')

# Scale wage and value
fifa['scaled_def'] = whiten(fifa['def'])
fifa['scaled_phy'] = whiten(fifa['phy'])

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
- Uses: Identifying features in satellite images

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

# Find Standard Deviations
r_std, g_std, b_std = pixels[['red', 'green', 'blue']].std()

# Scale actual RGB values in range of 0-1
for cluster_center in cluster_centers:
    scaled_r, scaled_g, scaled_b = cluster_center
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
batman_image = img.imread('Cluster Analysis in Python/batman.jpg')
print(batman_image.shape)

r = []
g = []
b = []

# Store RGB values of all pixels in lists r, g and b
for pixel in batman_image:
    for temp_r, temp_g, temp_b in pixel:
        r.append(temp_r)
        g.append(temp_g)
        b.append(temp_b)


distortions = []
num_clusters = range(1, 7)

batman_df = pd.DataFrame({'red': r, 'blue': b, 'green': g})

# Scale red, blue and green
batman_df['scaled_red'] = whiten(batman_df['red'])
batman_df['scaled_blue'] = whiten(batman_df['blue'])
batman_df['scaled_green'] = whiten(batman_df['green'])

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

colors = []

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

TD-IDF (Term Frequency - Inverse Document Frequency)
- A weighted measure: evaluate how important a word is to a document in a collection

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=50, min_df=0.2, tokenizer=remove_noise)
tfidf_matrix = tfidf_vectorizer.fit_transform(data)

Clustering with sparse matrix
- kmeans() in SciPy doesn't not support sparse matrices
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


stop_words = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'youre', 'youve', 'youll', 'youd', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'shes', 'her', 'hers', 'herself', 'it', 'its', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'thatll', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from',
              'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'dont', 'should', 'shouldve', 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', 'arent', 'couldn', 'couldnt', 'didn', 'didnt', 'doesn', 'doesnt', 'hadn', 'hadnt', 'hasn', 'hasnt', 'haven', 'havent', 'isn', 'isnt', 'ma', 'mightn', 'mightnt', 'mustn', 'mustnt', 'needn', 'neednt', 'shan', 'shant', 'shouldn', 'shouldnt', 'wasn', 'wasnt', 'weren', 'werent', 'won', 'wont', 'wouldn', 'wouldnt']


def remove_noise(text, stop_words=[]):
    tokens = word_tokenize(text)
    cleaned_tokens = []

    for token in tokens:
        token = re.sub('[^A-Za-z0-9]+', '', token)
        if len(token) > 1 and token.lower() not in stop_words:
            # Get lowercase
            cleaned_tokens.append(token.lower())

    return cleaned_tokens


movies_plots = pd.read_csv('Cluster Analysis in Python\movies_plot.csv')

plots = movies_plots['Plot']
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

fifa = pd.read_csv('Cluster Analysis in Python/fifa_18_sample_data.csv')

fifa = fifa.iloc[:250, :]

fifa['cluster_labels'] = [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 2, 0, 1, 1, 0, 1, 1, 2, 1, 1, 1, 1, 0, 1, 2, 2, 2, 2, 0, 1, 2, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 1, 1, 1, 0, 2, 1, 1, 2, 2, 0, 0, 2, 1, 2, 2, 0, 0, 1, 2, 1, 1, 2, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 2, 1, 0, 2, 1, 0, 0, 2, 0, 1, 0, 2, 2, 2, 1, 1, 2, 0, 1, 2, 1, 0, 2, 1, 2, 2, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 2,
                          0, 2, 2, 1, 1, 0, 1, 1, 0, 2, 1, 1, 0, 2, 1, 1, 0, 1, 1, 0, 2, 2, 2, 1, 2, 0, 0, 1, 1, 2, 1, 2, 0, 1, 1, 2, 0, 0, 0, 0, 0, 1, 0, 2, 2, 2, 1, 0, 1, 1, 0, 0, 2, 0, 1, 1, 0, 0, 0, 0, 2, 1, 1, 0, 0, 2, 0, 0, 0, 2, 0, 1, 2, 1, 1, 0, 1, 1, 1, 2, 0, 1, 0, 1, 2, 1, 1, 0, 1, 1, 1, 0, 2, 0, 0, 1, 1, 0, 2, 1, 2, 2, 0, 1, 0, 1, 1, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 0, 1, 1, 2, 2, 1, 1, 1, 0, 2, 0, 1]

# Print the size of the clusters
print(fifa.groupby('cluster_labels')['ID'].count())

# Print the mean value of wages in each cluster
print(fifa.groupby('cluster_labels')['eur_wage'].mean())


# Creates Scale_features
fifa['scaled_pac'] = whiten(fifa['pac'])
fifa['scaled_sho'] = whiten(fifa['sho'])
fifa['scaled_pas'] = whiten(fifa['pas'])
fifa['scaled_dri'] = whiten(fifa['dri'])
fifa['scaled_def'] = whiten(fifa['def'])
fifa['scaled_phy'] = whiten(fifa['phy'])

scaled_features = ['scaled_pac', 'scaled_sho',
                   'scaled_pas', 'scaled_dri', 'scaled_def', 'scaled_phy']

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
