''' Introduction to Sampling '''

'''
Sampling and point estimates
Estimating the population of France
- A census asks every household how many people live there.

Sampling households
- Cheaper to ask a small number of households and use statistics to estimate the population
- Working with a subset of the whole population is called sampling

Population vs. sample
The population is the complete dataset
* Doesn't have to refer to people
* Typically, don't know what the whole population is

-The sample is the subset of data you calculate on

Coffee rating dataset
total_cup_points variety country_of origin aroma flavor aftertaste  body balance
90.58                 NA          Ethiopia  8.67   8.83       8.67  8.50    8.42
89.92              Other          Ethiopia  8.75   8.67       8.50  8.42    8.42
...                  ...               ...   ...    ...        ...   ...     ...
73.75                 NA           Vietnam  6.75   6.67       6.50  6.92    6.83

- Each row represents 1 coffee
- 1338 rows

Points vs flavor: population
pts_vs_flavor_pop = coffee_ratings[['total_cup_points', 'flavor']] -> in

    total_cup_points flavor
0              90.58   8.83
1              89.92   8.67
2              89.75   8.50
3              89.00   8.58
4              88.83   8.50
...              ...    ...
1333           78.75   7.58
1334           78.08   7.67
1335           77.17   7.33
1336           75.08   6.83
1337           73.75   6.67
[1338 rows x  2 columns] -> out

Points vs flavor: 10 row sample
pts_vs_flavor_samp = pts_vs_flavour_pop.sample(n = 10) -> in

        total_cup_points  flavor
1088               80.33    7.17
1157               79.67    7.42
1267               76.17    7.33
506                83.00    7.67
659                82.50    7.42
817                81.92    7.50
1050               80.67    7.42
685                82.42    7.50
1027               80.92    7.25
62                 85.58    8.17 
[10 rows x  2 columns] -> out

Python sampling for Series
* Use .sample() for pandas DataFrames and Series

cup_points_samp = coffee_ratings['total_cup_points'].sample(n = 10) -> in

        total_cup_points
1088               80.33
1157               79.67
1267               76.17
...                  ...
685                82.42
1027               80.92
62                 85.58
Name: total_cup_points, dtype: float64 -> out

Population parameters & point estimates
- A population parameter is a calculation made on the population dataset

import numpy as np
np.mean(pts_vs_flavor_pop['total_cup_points']) -> in

82.15120328849028 -> out

- A point estimate or sample statistic is a calculation made on the sample dataset

np.mean(cup_points_samp) -> in

81.31800000000001 -> out

Points estimates with pandas
pts_vs_flavor_pop['flavor'].mean() -> in

7.526046337817639 -> out

pts_vs_flavor_samp['flavor'].mean() -> in

7.485000000000001 -> out
'''

# Sample 1000 rows from spotify_population
spotify_sample = spotify_population.sample(n=1000)

# Print the sample
print(spotify_sample)

# Calculate the mean duration in mins from spotify_population
mean_dur_pop = spotify_population['duration_minutes'].mean()

# Calculate the mean duration in mins from spotify_sample
mean_dur_samp = spotify_sample['duration_minutes'].mean()

# Print the means
print(mean_dur_pop)
print(mean_dur_samp)


# Create a pandas Series from the loudness column of spotify_population
loudness_pop = spotify_population['loudness']

# Sample 100 values of loudness_pop
loudness_samp = loudness_pop.sample(n=100)

# Calculate the mean of loudness_pop
mean_loudness_pop = np.mean(loudness_pop)

# Calculate the mean of loudness_samp
mean_loudness_samp = np.mean(loudness_samp)

# Print the means
print(mean_loudness_pop)
print(mean_loudness_samp)


'''
Convenience sampling
The Literary Digest election prediction
* Prediction: Landon get 57%; Roosevelt gets 43%
* Actual results: Landon got 38%; Roosevelt got 62%
* Sample not  representative of population, causing sample bias
* Collecting data by the easiest method is called convenience sampling ( prone to sample bias) 

Finding the mean age of French people
* SUrvey 10 people at Disneyland Paris
* Mean age of 24.6 years
* Will this be a ood estimate for all of France?

How accurate was the survey?
Year Average French Age
1975               31.6
1985               33.6
1995               36.2
2005               38.9
2015               41.2

* 24.6 years is a poor estimate
* People who visit DIsneyland aren't representative of the whole population

Convenience sampling coffee ratings
coffee_ratings['total_cup_points'].mean() -> in

82.15120328849028 -> out

coffee_ratings_first10 = coffee_ratings.head(10)

coffee_ratings_first10['total_cup_points'].mean() -> in

89.1 -> out

Visualizing selection bias
import matplotlib.pyplot as plt
import numpy as np

coffee_ratings['total_cup_points'].hist(bins = np.arange(59, 93, 2))

coffee_ratings_first10['total_cup_points'].hist(bins = np.arange(59, 93, 2))
plt.show()

Visualizing selection bias for a random sample
coffe_sample = coffee_ratings.sample(n = 10)

coffee_sample['total_cup_points'].hist(bins = np.arange(59, 93, 2))
plt.show()
'''

# Visualize the distribution of acousticness with a histogram
spotify_population['acousticness'].hist(bins=np.arange(0, 1.01, 0.01))
plt.show()

# Update the histogram to use spotify_mysterious_sample
spotify_mysterious_sample['acousticness'].hist(bins=np.arange(0, 1.01, 0.01))
plt.show()


# Visualize the distribution of duration_minutes as a histogram
spotify_population['duration_minutes'].hist(bins=np.arange(0, 15.5, 0.5))
plt.show()

# Update the histogram to use spotify_mysterious_sample2
spotify_mysterious_sample2['duration_minutes'].hist(
    bins=np.arange(0, 15.5, 0.5))
plt.show()


'''
Pseudo-random number generation
What does random mean?
{adjective} made, done, happening, or chosen without method or conscious decision.

True random numbers
* Generated from physical processes, like flipping coins
* Hotbits uses radioactive decay
* RANDOM.ORG uses atmospheric noise
* True randomness is expensive

Pseudo-random number generation
* Pseudo-random number generation is cheap and fast
* Next 'random' number is calculated from previous 'random' number
* The first 'random' number is calculated from a seed
* The same seed value yields the same random numbers

Pseudo-random number generation example
seed = 1
calc_next_random(seed) -> in

3 -> out

calc_next_random(3) -> in

2 -> out

calc_next_random(2) -> in

6 -> out

Random number generating functions
* Prepend with numpy.random, such as numpy.random.beta()

Function            Distribution
.beta               Beta
.binomial           Binomial
.chisquare          Chi-squared
.exponential        Exponential
.f                  F
.gamma              Gamma
.geometric          Geometric
.hypergeometric     Hypergeometric
.lognormal          Lognormal
.negative_binomial  Negative binomial
.normal             Normal
.poisson            Poisson
.standard_t         t
.uniform            Unifrom

Visualizing random numbers
randoms = np.random.beta(a=2, b=2, size=5000)
randoms -> in

array([0.6208281, 0.73216171, 0.44298403, ...,
        0.13411873, 0.52198411, 0.72355098]) -> out

plt.hist(randoms, bins=np.arange(0, 1, 0.05))
plt.show()

Random numbers seeds
np.random.seed(20000229)

np.random.normal(loc=2, scale=1.5, size=2) -> in

array([-0.59030264, 1.87821258]) -> out

np.random.normal(loc=2, scale=1.5, size=2) -> in

array([2.52619561, 4.9684949 ]) -> out

Using a different seed
np.random.seed(2041004)

np.random.normal(loc=2, scale=1.5, size=2) -> in

array([1.09364337, 4.55285159]) -> out

np.random.normal(loc=2, scale=1.5, size=2) -> in

array([2.67038916, 2.36677492]) -> out
'''

# Generate random numbers from a Uniform(-3, 3)
uniforms = np.random.uniform(low=-3, high=3, size=5000)

# Plot a histogram of uniform values, binwidth 0.25
plt.hist(uniforms, bins=np.arange(-3, 3.025, 0.25))
plt.show()


# Generate random numbers from a Normal(5, 2)
normals = np.random.normal(loc=5, scale=2, size=5000)

# Plot a histogram of normal values, binwidth 0.5
plt.hist(normals, bins=np.arange(-2, 13.5, 0.5))
plt.show()


''' Sampling Methods '''

'''
Simple random and systematic sampling
Simple random sampling with pandas
coffee_ratings.sample(n=5, random_state=19000113) -> in

    total_cup_points           variety country_of_origin aroma flavor aftertaste    body balance
437            83.25              None          Colombia  7.92   7.75       7.25    7.83    7.58
285            83.83    Yellow Bourbon            Brazil  7.92   7.50       7.33    8.17    7.50
784            82.08              None          Colombia  7.50   7.42       7.42    7.67    7.42
648            82.58           Caturra          Colombia  7.58   7.50       7.42    7.67    7.42
155            84.58           Caturra          Colombia  7.42   7.67       7.75    8.08    7.83 -> out

Systematic sampling - defining the interval
sample_size = 5
pop_size = len(coffee_ratings)
print(pop_size) -> in

1338 -> out

interval = pop_size // sample_size

NB: ( // ) when used will return an integer division
print(interval) -> in

267 -> out

Systematic sampling - selecting the rows
coffee_ratings.iloc[::interval] -> in

    total_cup_points    variety country_of_origin  aroma  flavor aftertaste   body balance
0              90.58       None          Ethiopia   8.67    8.83       8.67   8.50    8.42
267            83.92       None          Colombia   7.83    7.75       7.58   7.75    7.75
534            82.92    Bourbon       El Salvador   7.50    7.50       7.75   7.92    7.83
801            82.00     Typica            Taiwan   7.33    7.50       7.17   7.50    7.33
1068           80.50      Other            Taiwan   7.17    7.17       7.17   7.17    7.25 -> out

The trouble with systematic sampling
coffee_ratings_with_id = coffee_ratings.reset_index()
coffee_ratings_with_id.plot(x='index', y='aftertaste', kind='scatter')
plt.show()

- Systematic sampling is only safe if we don't see a pattern in the data

Making systematic sampling safe
shuffled = coffee_ratings.sample(frac=1)
shuffled = shuffled.reset_index(drop = True).reset_index()
shuffled.plot(x='index', y='aftertaste', kind='scatter')
plt.show()

- Shuffling rows + systematic sampling is the same as simple random sampling 
'''

# Sample 70 rows using simple random sampling and set the seed
attrition_samp = attrition_pop.sample(n=70, random_state=18900217)

# Print the sample
print(attrition_samp)


# Set the sample size to 70
sample_size = 70

# Calculate the population size from attrition_pop
pop_size = len(attrition_pop)

# Calculate the interval
interval = pop_size // sample_size

# Systematically sample 70 rows
attrition_sys_samp = attrition_pop.iloc[::interval]

# Print the sample
print(attrition_sys_samp)


# Shuffle the rows of attrition_pop
attrition_shuffled = attrition_pop.sample(frac=1)

# Reset the row indexes and create an index column
attrition_shuffled = attrition_shuffled.reset_index(drop=True).reset_index()

# Plot YearsAtCompany vs. index for attrition_shuffled
attrition_shuffled.plot(x='index', y='YearsAtCompany', kind='scatter')
plt.show()


'''
Stratified and weighted random sampling
- It is a technique that allows us to sample a population that contains subgroups.

Coffees by country
top_counts = coffee_ratings['country_of_origin'].value_counts()
top_counts.head(6) -> in

country_of_origin
Mexico                  236
Colombia                183
Guatemala               181
Brazil                  132
Taiwan                   75
United States (Hawaii)   73
dtype: int64 -> out

Filtering for 6 countries
top_counted_countries = ['Mexico', 'Colombia', 'Guatemala', 'Brazil', 'Taiwan', 'United States (Hawaii)']

top_counted_subset = coffee_ratings['country_of_origin'].isin(top_counted_countries)

coffee_ratings_top = coffee_ratings[top_counted_subset]

Counts of a simple random sample
coffee_ratings_samp = coffee_ratings_top.sample(frac=0.1, random_state=2021)

coffee_ratings_samp['country_of_origin'].value_counts(normalize=True) -> in

country_of_origin
Mexico                  0.250000
Guatemala               0.204545
Colombia                0.181818
Brazil                  0.181818
United States (Hawaii)  0.102273
Taiwan                  0.079545
dtype: float64 -> out

Proportional stratified sampling
coffee_ratings_strat = coffee_ratings_top.groupby('country_of_origin')\
.sample(frac=0.1, random_state=2021)

coffee_ratings_strat['country_of_origin'].value_counts(normalize=True) -> in

Mexico                  0.272727
Guatemala               0.204545
Colombia                0.104545
Brazil                  0.147727
Taiwan                  0.090909
United States (Hawaii)  0.079545
dtype: float64 -> out

Equal counts stratified sampling
coffee_ratings_eq = coffee_ratings_top.groupby('country_of_origin')\
.sample(n=15, random_state=2021)

coffee_ratings_eq['country_of_origin'].value_counts(normalize=True) -> in

Taiwan                  0.166667
Brazil                  0.166667
United States (Hawaii)  0.166667
Guatemala               0.166667
Mexico                  0.166667
Colombia                0.166667
Name: country_of_origin, dtype: float64 -> out

Weighted random sampling
* Specify weights to adjust the relative probability of a row being sampled
import numpy as np
coffee_ratings_weight = coffee_ratings_top
condition = coffee_ratings_weight['country_of_origin'] == Taiwan'

coffee_ratings_weight['weight'] = np.where(condition, 2, 1)

Coffee_ratings_weight = coffee_ratings_weight.sample(frac=0.1, weights='weight')

coffee_ratings_weight['country_of_origin'].value_counts(normalize=True) -> in

Brazil                  0.261364
Mexico                  0.204545
Guatemala               0.204545
Taiwan                  0.170455
Colombia                0.090909
United States (Hawaii)  0.068182
Name: country_of_origin, dtype: float64 -> out
'''

# Proportion of employees by Education level
education_counts_pop = attrition_pop['Education'].value_counts(normalize=True)

# Print education_counts_pop
print(education_counts_pop)

# Proportional stratified sampling for 40% of each Education group
attrition_strat = attrition_pop.groupby(
    'Education').sample(frac=0.4, random_state=2022)

# Calculate the Education level proportions from attrition_strat
education_counts_strat = attrition_strat['Education'].value_counts(
    normalize=True)

# Print education_counts_strat
print(education_counts_strat)


# Get 30 employees from each Education group
attrition_eq = attrition_pop.groupby(
    'Education').sample(n=30, random_state=2022)

# Get the proportions from attrition_eq
education_counts_eq = attrition_eq['Education'].value_counts(normalize=True)

# Print the results
print(education_counts_eq)


# Plot YearsAtCompany from attrition_pop as a histogram
attrition_pop['YearsAtCompany'].hist(bins=np.arange(0, 41, 1))
plt.show()

# Sample 400 employees weighted by YearsAtCompany
attrition_weight = attrition_pop.sample(n=400, weights="YearsAtCompany")

# Plot YearsAtCompany from attrition_weight as a histogram
attrition_weight.hist('YearsAtCompany', bins=np.arange(0, 41, 1))
plt.show()


'''
Cluster sampling
Stratified sampling vs. cluster sampling
Stratified sampling
* Split the population into subgroups
* Use simple random sampling on every subgroup
Cluster sampling
* Use simple random sampling to pick some subgroups
* Use simple random sampling on only those subgroups

Varieties of coffee
varieties_pop = list(coffee_ratings['variety'].unique()) -> in

[None, 'Other', 'Bourbon', 'Catimor', 'Ethiopian Yirgacheffe', 'Caturra', 'SL14', 'Sumatra', 'SL34', 'Hawaiian Kona', 'Yellow Bourbon', 'SL28', 'Gesha', 'Catuai', 'Pacamara', 'Typica', 'Sumatra Lintong', 'Mundo Novo', 'Java', 'Peaberry', 'Pacas', 'Mandheling', 'Ruiru 11', 'Arusha', 'Etiopian Heirlooms', 'Moka Peaberry', 'Sulawesi', 'Blue Mountain', 'Marigojipe', 'Pache Comun'] -> out

Stage 1: sampling for subgroups
import random
varieties_samp = random.sample(varieties_pop, k=3) -> in

['Hawaiian Kona', 'Bourbon', 'SL28'] -> out

Stage 2: sampling each group
variety_condition = coffee_ratings['variety'].isin(varieties_samp)
coffee_ratings_cluster = coffee_ratings[variety_condition]

coffee_ratings_cluster['variety'] = coffee_ratings_cluster['variety'].cat.remove_unused_categories()

coffee_ratings_cluster.groupby('variety').sample(n=5, random_state=2021) -> in

Stage 2 output
                    total_cup_points         variety             country_of_origin ...
variety
Bourbon         575            82.83            Bourbon                  Guatemala
                560            82.83            Bourbon                  Guatemala
                524            83.00            Bourbon                  Guatemala
                1140           79.83            Bourbon                  Guatemala
                318            83.67            Bourbon                     Brazil
Hawaiian Kona   1291           73.67      Hawaiian Kona     United States (Hawaii)
                1266           76.25      Hawaiian Kona     United States (Hawaii)
                488            83.08      Hawaiian Kona     United States (Hawaii)
                461            83.17      Hawaiian Kona     United States (Hawaii)
                117            84.83      Hawaiian Kona     United States (Hawaii)
SL28            137            84.67               SL28                      Kenya
                452            83.17               SL28                      Kenya
                224            84.17               SL28                      Kenya
                66             85.50               SL28                      Kenya
                559            82.83               SL28                      Kenya -> out

Multistage sampling
- Cluster sampling is a type of multistage sampling
- Can have > 2 stages
- E.g., countrywide surveys may sample states, countries, cities and neighborhoods
'''

# Create a list of unique JobRole values
job_roles_pop = list(attrition_pop['JobRole'].unique())

# Randomly sample four JobRole values
job_roles_samp = random.sample(job_roles_pop, k=4)

# Filter for rows where JobRole is in job_roles_samp
jobrole_condition = attrition_pop['JobRole'].isin(job_roles_samp)
attrition_filtered = attrition_pop[jobrole_condition]

# Remove categories with no rows
attrition_filtered['JobRole'] = attrition_filtered['JobRole'].cat.remove_unused_categories()

# Randomly sample 10 employees from each sampled job role
attrition_clust = attrition_filtered.groupby(
    'JobRole').sample(n=10, random_state=2022)

# Print the sample
print(attrition_clust)


'''
Comparing sampling methods
Review of sampling techniques - setup
top_counted_countries = ['Mexico', 'Colombia', 'Guatemala', 'Brazil', 'Taiwan', 'United States (Hawaii)']

subset_condition = coffee_ratings['country_of_origin'].isin(top_counted_countries)
coffee_ratings_top = coffee_ratings[subset_condition]

coffee_ratings_top.shape -> in

(880, 8) -> out

Review of simple random sampling
coffee_ratings_srs = coffee_ratings_top.sample(frac=1/3, random_state=2021)

coffee_ratings_srs.shape -> in

(293, 8) -> out

Review of stratified sampling
coffee_ratings_strat = coffee_ratings_top.groupby('country_of_origin').sample(frac=1/3, random_state=2021) 

coff_ratings_strat.shape -> in

(293, 8) -> out

Review of cluster sampling
import random
top_countries_samp = random.sample(top_counted_countries, k=2)
top_condition = coffee_ratings_top['country_of_origin'].isin(top_countries_samp)
coffee_ratings_cluster = coffee_ratings_top[top_condition]
coffee_ratings_cluster['country_of_origin'] = coffee_ratings_cluster['country_of_origin'].cat.remove_unused_categories()

coffee_ratings_clust = coffee_ratings_cluster.groupby('country_of_origin').sample(n=len(coffee_ratings_top) // 6)

coffee_ratings_clust.shape -> in

(292, 8) -> out

Calculating mean cup points
Population
coffee_ratings_top['total_cup_points'].mean()

81.94700000000002 -> out

Simple random sample
coffee_ratings_srs['total_cup_points'].mean()

81.95982935153583 -> out

Stratified sample
coffee_ratings_strat['total_cup_points'].mean()

81.92566552901025 -> out

Cluster sample
coffee_ratings_clust['total_cup_points'].mean()

82.03246575342466 -> out

Mean cup points by country
Population:
coffee_ratings_top.groupby('country_of_origin')['total_cup_points'].mean()

country_of_origin
Brazil                  82.405909
Colombia                83.106557
Guatemala               81.846575
Mexico                  80.890085
Taiwan                  82.001333
United States (Hawaii)  81.820411
Name: total_cup_points, dtype: float64 -> out

Simple random sample
coffee_ratings_srs.groupby('country_of_origin')['total_cup_points'].mean()

country_of_origin
Brazil                  82.414878
Colombia                82.925536
Guatemala               82.045385
Mexico                  81.100714
Taiwan                  81.744333
United States (Hawaii)  82.008000
Name: total_cup_points, dtype: float64 -> out

Stratified sample
coffee_ratings_strat.groupby('country_of_origin')['total_cup_points'].mean()

country_of_origin
Brazil                  82.499773
Colombia                83.288197
Guatemala               81.727667
Mexico                  80.994684
Taiwan                  81.846800
United States (Hawaii)  81.051667
Name: total_cup_points, dtype: float64 -> out

Cluster sample
coffee_ratings_clust.groupby('country_of_origin')['total_cup_points'].mean()

country_of_origin
Colombia    83.128904
Mexico      80.936027
Name: total_cup_points, dtype: float64 -> out
'''

# Perform simple random sampling to get 0.25 of the population
attrition_srs = attrition_pop.sample(frac=1/4, random_state=2022)


# Perform stratified sampling to get 0.25 of each relationship group
attrition_strat = attrition_pop.groupby(
    'RelationshipSatisfaction').sample(frac=1/4, random_state=2022)


# Create a list of unique RelationshipSatisfaction values
satisfaction_unique = list(attrition_pop['RelationshipSatisfaction'].unique())

# Randomly sample 2 unique satisfaction values
satisfaction_samp = random.sample(satisfaction_unique, k=2)

# Filter for satisfaction_samp and clear unused categories from RelationshipSatisfaction
satis_condition = attrition_pop['RelationshipSatisfaction'].isin(
    satisfaction_samp)
attrition_clust_prep = attrition_pop[satis_condition]
attrition_clust_prep['RelationshipSatisfaction'] = attrition_clust_prep['RelationshipSatisfaction'].cat.remove_unused_categories()

# Perform cluster sampling on the selected group, getting 0.25 of attrition_pop
attrition_clust = attrition_clust_prep.groupby(
    'RelationshipSatisfaction').sample(n=len(attrition_pop) // 4, random_state=2022)


# Mean Attrition by RelationshipSatisfaction group
mean_attrition_pop = attrition_pop.groupby(
    'RelationshipSatisfaction')['Attrition'].mean()

# Print the result
print(mean_attrition_pop)

# Calculate the same thing for the simple random sample
mean_attrition_srs = attrition_srs.groupby(
    'RelationshipSatisfaction')['Attrition'].mean()

# Print the result
print(mean_attrition_srs)

# Calculate the same thing for the stratified sample
mean_attrition_strat = attrition_strat.groupby(
    'RelationshipSatisfaction')['Attrition'].mean()

# Print the result
print(mean_attrition_strat)

# Calculate the same thing for the cluster sample
mean_attrition_clust = attrition_clust.groupby(
    'RelationshipSatisfaction')['Attrition'].mean()

# Print the result
print(mean_attrition_clust)


''' Sampling Distributions '''

'''
Relative error of point estimates
Sample size is number of rows
len(coffee_ratings.sample(n=300)) -> in

300 -> out

len(coffee_ratings.sample(frac=0.25)) -> in

334 -> out

Various sample sizes
coffee_ratings['total_cup_points'].mean() -> in

82.15120328849028 -> out

coffee_ratings.sample(n=10)['total_cup_points'].mean() -> in

83.027 -> out

coffee_ratings.sample(n=100)['total_cup_points'].mean() -> in

82.4897 -> out

coffee_ratings.sample(n=1000)['total_cup_points'].mean() -> in

82.1186 -> out

Relative errors
Population parameter:
population_mean = coffee_ratings['total_cup_points'].mean()

Point estimate:
sample_mean = coffee_ratings.sample(n=sample_size)['total_cup_points'].mean()

Relative error as a percentage:
rel_error_pct = 100 * abs(populaion_mean - sample_mean) / population_mean

Relative error vs sample size
import matplotlib.pyplot as plt
errors.plot(x='sample_size', y='relative_error', kind='line')
plt.show()

Properties:
* Really noisy, particulrly for small samples
* Amplitude is initially steep, then flattens
* Relative error decreases to zero (when the sample size = population)
'''

# Generate a simple random sample of 50 rows, with seed 2022
attrition_srs50 = attrition_pop.sample(n=50, random_state=2022)

# Calculate the mean employee attrition in the sample
mean_attrition_srs50 = attrition_srs50['Attrition'].mean()

# Calculate the relative error percentage
rel_error_pct50 = 100 * abs(mean_attrition_pop -
                            mean_attrition_srs50) / mean_attrition_pop

# Print rel_error_pct50
print(rel_error_pct50)


# Generate a simple random sample of 100 rows, with seed 2022
attrition_srs100 = attrition_pop.sample(n=100, random_state=2022)

# Calculate the mean employee attrition in the sample
mean_attrition_srs100 = attrition_srs100['Attrition'].mean()

# Calculate the relative error percentage
rel_error_pct100 = 100 * abs(mean_attrition_pop -
                             mean_attrition_srs100) / mean_attrition_pop

# Print rel_error_pct100
print(rel_error_pct100)


'''
Creating a sampling distribution
Same code, different answer
coffee_ratings.sample(n=30)['total_cup_points'].mean() -> in

82.53066666666668 -> out

coffee_ratings.sample(n=30)['total_cup_points'].mean() -> in

81.97566666666667 -> out

coffee_ratings.sample(n=30)['total_cup_points'].mean() -> in

82.68 -> out

coffee_ratings.sample(n=30)['total_cup_points'].mean() -> in

81.675 -> out

Same code, 1000 times
mean_cup_points_1000 = []
for i in range(1000):
    mean_cup_points_1000.append(coffee_ratings.sample(n=30)['total_cup_points'].mean())

print(mean_cup_points_1000) -> in

[82.119333333333333, 82.55300000000001, 82.0726666666668, 81.7696666666666666667, ..., 82.7416666666666, 82.45033333333335, 81.7719999999999, 82.81633333333333] -> out

Distribution of sample means for size 30
import matplotlib.pyplot as plt
plt.hist(mean_cup_points_1000, bins=30)
plt.show()

- A sampling distribution is a distribution of replicates of point estimates.
'''

# Create an empty list
mean_attritions = []
# Loop 500 times to create 500 sample means
for i in range(500):
    mean_attritions.append(attrition_pop.sample(n=60)['Attrition'].mean())

# Create a histogram of the 500 sample means
plt.hist(mean_attritions, bins=16)
plt.show()


'''
Approximate sampling distributions
4 dice
dice = expand_grid({'die1': [1, 2, 3, 4, 5, 6], 'die2': [1, 2, 3, 4, 5, 6], 'die3': [1, 2, 3, 4, 5, 6], 'die4': [1, 2, 3, 4, 5, 6]}) -> in

    die1 die2 die3 die4
0      1    1    1    1
1      1    1    1    2
2      1    1    1    3
3      1    1    1    4
4      1    1    1    5
...  ...  ...  ...  ...
1291   6    6    6    2
1292   6    6    6    3
1293   6    6    6    4
1294   6    6    6    5
1295   6    6    6    6
[1296 rows x 4 columns] -> out

Mean roll
dic['mean_roll'] = (dice['die1'] + dice['die2'] + dice['die3'] + dice['die4']) / 4

print(dice) -> in

    die1 die2 die3 die4 mean_roll
0      1    1    1    1      1.00
1      1    1    1    2      1.25
2      1    1    1    3      1.50
3      1    1    1    4      1.75
4      1    1    1    5      2.00
...  ...  ...  ...  ...       ...
1291   6    6    6    2      5.00
1292   6    6    6    3      5.25
1293   6    6    6    4      5.50
1294   6    6    6    5      5.75
1295   6    6    6    6      6.00
[1296 rows x 5 columns] -> out

Exact sampling distribution
dice['mean_roll'] = dice['mean_roll'].astype('category')
dice['mean_roll'].value_counts(sort = False).plot(kind='bar')
plt.show()

The number of outcomes increases fast
n_dice = list(range(1, 101))
n_outcomes = []
for n in n_dice:
    n_outcomes.append(6**n)

outcomes = pd.DateFrame({'n_dice': n_dice, 'n_outcomes': n_outcomes})

outcomes.plot(x='n_dice', y='n_outcomes', kind='scatter')
plt.show()

Simulating the mean of four dice rolls
import numpy as np
sample_means_1000 = []
for i in range(1000):
    sample_means_1000.append(np.random.choice(list(range(1, 7)), size=4, replace=True).mean())
print(sample_means_1000) -> in

[3.25, 3.25, 1.75, 2.0, 2.0, 1.0, 1.0, 2.75, 2.5, 3.0, 2.0, 2.75, ... 1.25, 2.0, 2.5, 2.5, 3.75, 1.5, 1.75, 2.25, 2.0, 1.5, 3.25, 3.0, 3.5] -> out

Approximate sampling distribution
plt.hist(sample_means_1000, bins=20)
plt.show()
'''

# Expand a grid representing 5 8-sided dice
dice = expand_grid(
    {'die1': [1, 2, 3, 4, 5, 6, 7, 8], 'die2': [1, 2, 3, 4, 5, 6, 7, 8], 'die3': [1, 2, 3, 4, 5, 6, 7, 8], 'die4': [1, 2, 3, 4, 5, 6, 7, 8], 'die5': [1, 2, 3, 4, 5, 6, 7, 8]})

# Add a column of mean rolls and convert to a categorical
dice['mean_roll'] = (dice['die1'] + dice['die2'] +
                     dice['die3'] + dice['die4'] + dice['die5']) / 5
dice['mean_roll'] = dice['mean_roll'].astype('category')

# Draw a bar plot of mean_roll
dice['mean_roll'].value_counts(sort=False).plot(kind='bar')
plt.show()


# Sample one to eight, five times, with replacement
five_rolls = np.random.choice(range(1, 9), size=5, replace=True)

# Print the mean of five_rolls
print(five_rolls.mean())

# Replicate the sampling code 1000 times
sample_means_1000 = []
for i in range(1000):
    sample_means_1000.append(np.random.choice(
        list(range(1, 9)), size=5, replace=True).mean())

# Draw a histogram of sample_means_1000 with 20 bins
plt.hist(sample_means_1000, bins=20)
plt.show()


'''
Standard errors and the Central Limit Theorem
Consequences of the central limit theorem
- Averages of independent samples have approximately normal distributions.

- As the sample size increases,
* The distribution of the averages gets closer to being normally distributed
* the width of the sampling distribution gets narrower

Population & sampling distribution means
coffee_ratings['total_cup_points'].mean() -> in

82.15120328849028 -> out

Use np.mean() on each approximate sampling distribution:

Sample size  Mean sample mean
5           82.18420719999999
20                 82.1558634
80          82.14510154999999
320              82.154017925

Population & sampling distribution standard deviations
coffee_ratings['total_cup_points'].std(ddof = 0) -> in

2.685858187306438 -> out

Use np.std(ddof=1) on each approximate sampling distribution:

Sample size  Std dev sample mean
5             1.1886358227738543
20            0.5940321141669805
80            0.2934024263916487
320          0.13095083089190876

* Specify ddof = 0 when calling .std() on populations
* .std() calculates a sample standard deviation by default
* Specify ddof = 1 when calling np.std() on samples or sampling distributions

Population mean over square root sample size
Sample size  Std dev sample mean                    Calculation    Result
5             1.1886358227738543    2.685858187306438 / sqrt(5)     1.201
20            0.5940321141669805   2.685858187306438 / sqrt(20)     0.601
80            0.2934024263916487   2.685858187306438 / sqrt(80)     0.300
320          0.13095083089190876  2.685858187306438 / sqrt(320)     0.150

Result = Estimate of the Std dev of the sampling distribution for the sample size

Standard error
- Standard deviation of the sampling distribution

- It is useful in a variety of contexts
* Estimating population standrd deviation
* Setting expectations on what level of variabilty to expect from the sampling process.

-Important tool in understanding sampling variability
'''

# Calculate the mean of the mean attritions for each sampling distribution
mean_of_means_5 = np.mean(sampling_distribution_5)
mean_of_means_50 = np.mean(sampling_distribution_50)
mean_of_means_500 = np.mean(sampling_distribution_500)

# Print the results
print(mean_of_means_5)
print(mean_of_means_50)
print(mean_of_means_500)


# Calculate the std. dev. of the mean attritions for each sampling distribution
sd_of_means_5 = np.std(sampling_distribution_5, ddof=1)
sd_of_means_50 = np.std(sampling_distribution_50, ddof=1)
sd_of_means_500 = np.std(sampling_distribution_500, ddof=1)

# Print the results
print(sd_of_means_5)
print(sd_of_means_50)
print(sd_of_means_500)


''' Bootstrap Distributions '''

'''
Introduction to bootstrapping
With or without
Sampling without replacement e.g deck of Cards distribution

Sampling with replacement ('resampling') e.g Rolling dice

Why sample with replacement?
- coffee_ratings: The dataset is a sample of a larger population of all coffees
- Each coffee in our sample represents many different hypothetical population coffeess
- Sampling with replaceement is a proxy

Coffee data preparation
coffee_focus = coffee_ratings[['variety', 'country_of_origin', 'flavor']]
coffee_focus = coffee_focus.reset_index() -> in

    index   variety     country_of_origin   flavor
0       0      None              Ethiopia     8.83
1       1     Other              Ethiopia     8.67
2       2   Bourbon             Guatemala     8.50
3       3      None              Ethiopia     8.58
4       4     Other              Ethiopia     8.50
...   ...       ...                   ...      ...
1333 1333      None               Ecuador     7.58 
1334 1334      None               Ecuador     7.67
1335 1335      None         united States     7.33
1336 1336      None                 India     6.83
1337 1337      None               Vietnam     6.67
[1338 rows x 4 columns] -> out

Resampling with .sample()
coffee_resamp = coffee_focus.sample(frac=1, replace=True) -> in

        index   variety     country_of_origin   flavor
1140     1140   Bourbon             Guatemala     7.25
57         57   Bourbon             Guatemala     8.00
1152     1152   Bourbon                Mexico     7.08
621       621   Caturra              Thailand     7.50
44         44      SL28                 Kenya     8.08
...       ...       ...                   ...      ...
996       996    Typica                Mexico     7.33
1090     1090   Bourbon             Guatemala     7.33
918       918     Other             Guatemala     7.42
249       249   Caturra              Colombia     7.67
467       467   Caturra              Colombia     7.50
[1338 rows x 4 columns] -> out

repeated coffees
coffee_resamp['index'].value_counts() -> in

658       5
167       4
363       4
357       4
1047      4
        ...
717       1
770       1
766       1
764       1
0         1
Name: index, Length: 868, dtype: int64 -> out

Missing coffees
num_unique_coffees = len(coffee_resamp.drop_duplicates(subset='index')) -> in

868 -> out

len(coff_ratings) - num_unique_coffees -> in

470 -> out

Bootstrapping
Its the opposite of sampling from a population
- Sampling: going from a population to a smaller sample
- Bootstrapping:  building up a theoretical population from the sample

Bootstrapping use case:
- Develop understanding of sampling variability using a single sample

Bootstrapping process
1. Make a resample (randomly sample with replacement) of the same size as the original sample
2. Calculate the statistic of interest (e.g mean etc) for this bootstrap sample
3. Repeat steps 1 and 2 many times
The resulting statistics are bootstrap statistics, and they form a bootstrap distribution

Bootstrapping coffee mean flavor
import numpy as np
mean_flavors_1000 = []
for i in range(1000):
    mean_flavors_1000.append(np.mean(coffee_sample.sample(frac=1, replace=true)['flavor']))

Bootstrap distribution histogram
import matplotlib.pyplot as plt
plt.hist(mean_flavours_1000)
plt.show()
'''

# Generate 1 bootstrap resample
spotify_1_resample = spotify_sample.sample(frac=1, replace=True)

# Print the resample
print(spotify_1_resample)

# Calculate of the danceability column of spotify_1_resample
mean_danceability_1 = np.mean(spotify_1_resample['danceability'])

# Print the result
print(mean_danceability_1)

# Replicate this 1000 times
mean_danceability_1000 = []
for i in range(1000):
    mean_danceability_1000.append(
        np.mean(spotify_sample.sample(frac=1, replace=True)['danceability']))

# Draw a histogram of the resample means
plt.hist(mean_danceability_1000)
plt.show()


'''
Comparing sampling and bootstrap distributions
Coffee focused subset
coffee_sample = coffee_ratings[['variety', 'country_of_origin', 'flavor']].reset_index().sample(n = 500) -> in

    index        variety         country_of_origin  flavor
132   132          Other                Costa Rica    7.58
51     51           None    United States (Hawaii)    8.17
42     42 Yellow Bourbon                    Brazil    7.92
569   569        Bourbon                 Guatemala    7.67
...   ...            ...                       ...     ...
643   643         Catuai                Costa Rica    7.42
356   356        Caturra                  Colombia    7.58
494   494           None                 Indonesia    7.58
169   169           None                    Brazil    7.81
[500 rows x 4 columns] -> out

The bootstrap of mean coffee flavors
import numpy as np
mean_flavors_5000 = []
for i in range(5000):
    mean_flavors_5000.append(np.mean(coffee_sample.sample(frac=1, replace=True)['flavor']))

bootstrap_distn = mean_flavors_5000

Mean flavor bootstrap distribution
import matplotlib.pyplot as plt
plt.hist(bootstrap_distn, bins = 15)
plt.show()

Sample, bootstrap distribution, population means
Sample mean:
coffee_sample['flavor'].mean() -> in

7.5132200000000005 -> out

True population mean:
coffee_ratings['flavor'].mean() -> in

7.526046337817639 -> out

Estimated population mean:
np.mean(bootstrap_distn) -> in

7.513357731999999 -> out

Interpreting the means
Bootstrap distribution mean:
- They are usually close to the sample mean
- They may not be a good estimate of the popultion mean
* Bootstrapping cannot correct biases from sampling

Sample sd vs bootstrap distribution sd
Sample standard deviation:
coffee_sample['flavor'].std() -> in

0.3540883911928703 -> out

Estimated population standard deviation:
standard_error = np.std(bootstrap_distn, ddof=1) -> in

0.015768474367958217 -> out

-Standard error is the standard deviation of the statistic of interest

std = standard_error * np.sqrt(500) -> in

0.3525938058821761 -> out

- Standard error times square root of sample size estimates the population standard deviation

True standard deviation:
coffee_ratings['flavor'].std(ddof=0) -> in

0.34125481224622645 -> out

Interpreting the standard errors
* Estimated standard error -> standard deviation of the bootstrap distribution for a sample statistic
* Population std. dev = Std. Error x sqrt(Sample size)
'''

mean_popularity_2000_samp = []

# Generate a sampling distribution of 2000 replicates
for i in range(2000):
    mean_popularity_2000_samp.append(
        # Sample 500 rows and calculate the mean popularity
        np.mean(spotify_population.sample(n=500)['popularity'])
    )

# Print the sampling distribution results
print(mean_popularity_2000_samp)


mean_popularity_2000_boot = []

# Generate a bootstrap distribution of 2000 replicates
for i in range(2000):
    mean_popularity_2000_boot.append(
        # Resample 500 rows and calculate the mean popularity
        np.mean(spotify_sample.sample(n=500, replace=True)['popularity'])
    )

# Print the bootstrap distribution results
print(mean_popularity_2000_boot)


# Calculate the population mean popularity
pop_mean = spotify_population['popularity'].mean()

# Calculate the original sample mean popularity
samp_mean = spotify_sample['popularity'].mean()

# Calculate the sampling dist'n estimate of mean popularity
samp_distn_mean = np.mean(sampling_distribution)

# Calculate the bootstrap dist'n estimate of mean popularity
boot_distn_mean = np.mean(bootstrap_distribution)

# Print the means
print([pop_mean, samp_mean, samp_distn_mean, boot_distn_mean])


# Calculate the population std dev popularity
pop_sd = spotify_population['popularity'].std(ddof=0)

# Calculate the original sample std dev popularity
samp_sd = spotify_sample['popularity'].std()

# Calculate the sampling dist'n estimate of std dev popularity
samp_distn_sd = np.std(sampling_distribution, ddof=1) * np.sqrt(5000)

# Calculate the bootstrap dist'n estimate of std dev popularity
boot_distn_sd = np.std(bootstrap_distribution, ddof=1) * np.sqrt(5000)

# Print the standard deviations
print([pop_sd, samp_sd, samp_distn_sd, boot_distn_sd])


'''
Confidence intervals
- 'Values within one standard deviation of the mean' includes a large number of values from each of these distributions
- We'll define a related concept called a confidence interval

Predicting the weather
- Rapid City, South Dakota in the United States has the least predictable weather
- Predict the high temperature there tomorrow

Our weather prediction
- Point estimate = 47F(8.3C)
- Range of plausible high temperature values = 40 to 54F (4.4 to 12.8C)

We just reported a confidence interval!
- 40 to 54F is a confidence interval
- Sometimes written as 47F (40F, 54F) or 47F [40F, 54F]
- ... or, 47 +- 7F
- 7F is the margin of error

Bootstrap distribution of mean flavor
import matplotlib.pyplot as plt
plt.hist(coffee_boot_distn, bins=15)
plt.show()

Mean of the resamples
import numpy as np
np.mean(coffee_boot_distn) -> in

7.513452892 -> out

Mean plus or minus one standard deviation
np.mean(coffee_boot_distn) -> in

7.513452892 -> out

np.mean(coffee_boot_distn) - np.std(coffee_boot_distn, ddof=1) -> in

7.497385709174466 -> out

np.mean(coffee_boot_distn) + np.std(coffee_boot_distn, ddof=1) -> in

7.529520074825534 -> out

Quantile method for confidence intervals
np.quantile(coffee_boot_distn, 0.025) -> in

7.4817195 -> out

np.quantile(coffee_boot_distn, 0.975) -> in

7.5448805 -> out

Inverse cumulative distribution function
- PDF: the bell curve
- CDF: using calculus to integrate to get area under bell curve
- Inverse CDF: flip x and y axes

Implemented in Python with
from scipy.stats import norm
norm.ppf(quantile, loc=0, scale=1)

Standard error method for confidence interval
point_estimate = np.mean(coffee_boot_distn) -> in

7.513452892 -> out

std_error = np.std(coffee_boot_distn, ddof=1) -> in

0.016067182825533724 -> out

from scipy.stats import norm
lower = norm.ppf(0.025, loc=point_estimate, 
scale=std_error)
upper = norm.ppf(0.975, loc=point_estimate, scale=std_error)
print((lower, upper)) -> in

(7.481961792328933, 7.544943991671067) -> out
'''

# Generate a 95% confidence interval using the quantile method
lower_quant = np.quantile(bootstrap_distribution, 0.025)
upper_quant = np.quantile(bootstrap_distribution, 0.975)

# Print quantile method confidence interval
print((lower_quant, upper_quant))


# Find the mean and std dev of the bootstrap distribution
point_estimate = np.mean(bootstrap_distribution)
standard_error = np.std(bootstrap_distribution, ddof=1)

# Find the lower limit of the confidence interval
lower_se = norm.ppf(0.025, loc=point_estimate, scale=standard_error)

# Find the upper limit of the confidence interval
upper_se = norm.ppf(0.975, loc=point_estimate, scale=standard_error)

# Print standard error method confidence interval
print((lower_se, upper_se))

'''
The most important things
- The standard deviation of a bootstrap statistic is a good approximation of the standard error
- Can assume bootstrap distributions are normally distributed for confidence intervals
'''
