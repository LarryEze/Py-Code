# import necessary packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import uniform, binom, norm, poisson, expon


''' Summary Statistics '''

'''
What is statistics?
- The field of statistics - The practice and study of collecting and analyzing data
- A summary statistic - A fact about or summary of some data

Types of statistics
- Descriptive statistics 
    Describe and summarize data

- Inferential statistics
    Use a sample of data to make inferences about a larger population

Types of data
- Numeric (Quantitative)
*   Continuous (Measured) e.g Airplane speed, Time spent waiting in line
*   Discrete (Counted) e.g Number of pets, Number of packages shipped

- Categorical (Qualitative)
*   Nominal (Unordered) e.g Married / unmarried, country of residence
*   Ordinal (Ordered) e.g Strongly disagree, Somewhat disagree, Neither agree nor disagree, Somewhat agree, Strongly agree

Categorical data can be represented as numbers
*   Nominal (Unordered) e.g Married / unmarried ( 1 / 0 ), country of residence (1, 2, ...)
*   Ordinal (Ordered) e.g Strongly disagree ( 1 ), Somewhat disagree ( 2 ), Neither agree nor disagree ( 3 ), Somewhat agree ( 4 ), Strongly agree ( 5 )

Summary statistics   Plots
*   Numeric data
Mean - Scatter plots
e.g
import numpy as np
np.mean(df['column_name'])

*   Categorical data
Counts - Bar plots
e.g
df['column_name'].value_counts()
'''


'''
Measures of center
3 different definitions, or measures, of center:
- Mean
- Median
- Mode

Measures of center: Mean / Average
e.g
import numpy as np
np.mean(df['column_name'])

Measures of center: Median / Middle value
e.g
import numpy as np
np.median(df['column_name'])

Measures of center: Mode / Most frequent value
e.g
df['column_name'].value_counts()
import statistics
statistics.mode(df['column_name'])

Adding an outlier
df[df['column_name'] == 'str' ]['column_name'].agg([np.mean, np.median])

# Note: 
- Since Mean are more sensitive to extreme values, it works better for symmetrical data.
- For skewed ( not symmetrical ) data, the Median is usually better to use.
- If the data in a plot is piled up on the right, with a tail on the left, its called Left-skewed data.
- If the data in a plot is piled up on the left, with a tail on the right, its called Right-skewed data.
'''

food_consumption = pd.read_csv(
    'Introduction to Statistics in Python/food_consumption.csv')

# Import numpy with alias np

# Filter for Belgium
be_consumption = food_consumption[food_consumption['country'] == 'Belgium']

# Filter for USA
usa_consumption = food_consumption[food_consumption['country'] == 'USA']

# Calculate mean and median consumption in Belgium
print(be_consumption.agg(np.mean))
print(be_consumption.agg(np.median))

# Calculate mean and median consumption in USA
print(usa_consumption.agg(np.mean))
print(usa_consumption.agg(np.median))


# Import numpy as np

# Subset for Belgium and USA only
be_and_usa = food_consumption[(food_consumption['country'] == 'Belgium') | (
    food_consumption['country'] == 'USA')]

# Group by country, select consumption column, and compute mean and median
print(be_and_usa.groupby('country')['consumption'].agg([np.mean, np.median]))


# Import matplotlib.pyplot with alias plt

# Subset for food_category equals rice
rice_consumption = food_consumption[food_consumption['food_category'] == 'rice']

# Histogram of co2_emission for rice and show plot
plt.hist(rice_consumption['co2_emission'])
plt.show()

# Subset for food_category equals rice
rice_consumption = food_consumption[food_consumption['food_category'] == 'rice']

# Calculate mean and median of co2_emission with .agg()
print(rice_consumption['co2_emission'].agg([np.mean, np.median]))


'''
Measures of spread
It describes how spread apart or close together the data points are.

Measures of spread: Variance
Its the average distance from each data point to the data's mean
e.g
import numpy as np
np.var(df['column_name'], ddof=1)
#   Note: without ddof = 1, Population variance is calculated instead of Sample variance.

Measures of spread: Standard deviation
Its the square root of the variance
e.g
np.sqrt(np.var(df['column_name'], ddof=1)
or
np.std(df['column_name'], ddof=1)

Measures of spread: Mean absolute deviation
Its the absolute value of the distances to the mean, and then takes the mean of those differences.
e.g
dists = df['column_name'] - mean(df$sleep_total)
np.mean(np.abs(dists))

Standard deviation (More common) vs Mean absolute deviation
- Standard deviation squares distances, penalizing longer distances more than shorter ones 
- Mean absolute deviation penalizes each distance equally

Quantiles / Percentiles
It split up the data into some number of equal parts.
e.g
np.quantile(df['column_name'], 0.5)
0.5 quantile = median

Quartiles:
It is gotten when a list of numbers is passed to get multiple quantiles at once
e.g
np.quantile(df['column_name'], [0, 0.25, 0.5, 0.75, 1])

Boxplots use Quartiles
import matplotlib.pyplot as plt
plt.boxplot(df['column_name'])
plt.show()
#   Note:
- First quartile (Q1) - The bottom of the box
- Second quartile (Q2, median) - The middle line
- Third quartile (Q3) - The top of the box

Quantiles using np.linspace()
np.linspace(start, stop, num)
e.g
np.quantile(df['column_name'], np.linspace[0, 1, 5])

Measures of spread: Interquartile range (IQR)
It is the distance between the 25th and 75th percentile, which is also the height of the box in a boxplot.
e.g
IQR = np.quantile(df['column_name'], 0.75) - np.quantile(df['column_name'], 0.25)
# OR
from scipy.stats import iqr
iqr(df['column_name'])

Outliers
- This are data points that are substantially different from the others
- A data point is an outlier if : 
data < Q1 - 1.5 x IQR or
data > Q3 + 1.5 x IQR

Finding outliers
e.g
from scipy.stats import iqr
iqr = iqr(df['column_name'])
lower_threshold = np.quantile(df['column_name'], 0.25) - 1.5 * iqr
upper_threshold = np.quantile(df['column_name'], 0.75) + 1.5 * iqr

df[ (df['column_name'] < lower_threshold) | (df['column_name'] > upper_threshold) ]

Summary Statistics Shortcut
df['column_name'].describe()
#   Note: It will return values for count, mean, std, min, 25%, 50%, 75%, max
'''

# Calculate the quartiles of co2_emission
print(np.quantile(food_consumption['co2_emission'], np.linspace(0, 1, 5)))

# Calculate the quintiles of co2_emission
print(np.quantile(food_consumption['co2_emission'], np.linspace(0, 1, 6)))

# Calculate the deciles of co2_emission
print(np.quantile(food_consumption['co2_emission'], np.linspace(0, 1, 11)))


# Print variance and sd of co2_emission for each food_category
print(food_consumption.groupby('food_category')
      ['co2_emission'].agg([np.var, np.std]))

# Import matplotlib.pyplot with alias plt

# Create histogram of co2_emission for food_category 'beef'
plt.hist(
    food_consumption[food_consumption['food_category'] == 'beef']['co2_emission'])
# Show plot
plt.show()

# Create histogram of co2_emission for food_category 'eggs'
plt.hist(
    food_consumption[food_consumption['food_category'] == 'eggs']['co2_emission'])
# Show plot
plt.show()


# Calculate total co2_emission per country: emissions_by_country
emissions_by_country = food_consumption.groupby('country')[
    'co2_emission'].sum()
print(emissions_by_country)

# Compute the first and third quantiles and IQR of emissions_by_country
q1 = np.quantile(emissions_by_country, 0.25)
q3 = np.quantile(emissions_by_country, 0.75)
iqr = q3 - q1

# Calculate the lower and upper cutoffs for outliers
lower = q1 - 1.5 * iqr
upper = q3 + 1.5 * iqr

# Subset emissions_by_country to find outliers
outliers = emissions_by_country[(emissions_by_country < lower) | (
    emissions_by_country > upper)]
print(outliers)


''' Random Numbers and Probability '''

'''
What are the chances?
Measuring chance
e.g
whats's the probability of an event?
P(event) = No of ways event can happen / total no of possible outcomes

Sampling from a DataFrame
print(df)   <- in
    name n_sales
0   Amir     178
1  Brain     128
2 Claire      75
3 Damian      69    <- out

df.sample()     <- in
    name n_sales
1  Brain     128    <- out

df.sample()     <- in
    name n_sales
2 Claire      75    <- out

Setting a random seed - To ensure repetition of results
e.g
np.random.seed(int)
df.sample()     <- in
    name n_sales
1  Brain     128    <- out

np.random.seed(int)
df.sample()     <- in
    name n_sales
1  Brain     128     <- out

Note: Ensure the int(integer / number) is same whenever sampling to get the same result

Sampling twice in Python
e.g
df.sample(2)    <- in
    name n_sales
1  Brain     128
2 Claire      75     <- out

Sampling with/without replacement in Python
e.g
df.sample(5, replace = True)    <- in
    name  n_sales
1  Brain      128
2 Claire       75
1  Brain      128
3 Damian       69
0   Amir      178     <- out

Independent events
-   Two events are independent if the probability of the second event isn't affected by the outcome of the first event.
-   Sampling with replacement = each pick is independent

Dependent events
-   Two events are dependent if the probability of the second event is affected by the outcome of the first event.
-   Sampling without replacement = each pick is dependent
'''

amir_deals = pd.read_csv(
    'Introduction to Statistics in Python/amir_deals.csv')

# Count the deals for each product
counts = amir_deals['product'].value_counts()
print(counts)

# Calculate probability of picking a deal with each product
probs = counts / amir_deals['product'].count()
print(probs)


# Set random seed
np.random.seed(24)

# Sample 5 deals without replacement
sample_without_replacement = amir_deals.sample(5, replace=False)
print(sample_without_replacement)

# Set random seed
np.random.seed(24)

# Sample 5 deals with replacement
sample_with_replacement = amir_deals.sample(5, replace=True)
print(sample_with_replacement)


'''
Discrete distributions
Probability distribution
-   This describes the probability of each possible outcome in a scenario.
e.g
Probability distribution of a fair die roll = 1/6, 1/6, 1/6, 1/6, 1/6, 1/6

Expected value: This is the mean of a probability distribution
e.g
Expected value of a fair die roll = (1 x 1/6) + (2 x 1/6) + (3 x 1/6) + (4 x 1/6) + (5 x 1/6) + (6 x 1/6) = 3.5

Expected value of uneven die roll  = (1 x 1/6) + (2 x 0) + (3 x 1/3) + (4 x 1/6) + (5 x 1/6) + (6 x 1/6) = 3.67
#   The 2 was turned to 3 in the die

Discrete probability distributions
-   This describe probabilities for discrete outcomes (i.e fair die & uneven die)
-   Discrete uniform distributions is when all outcomes have the same probability = Fair die

Sampling from discrete distributions
e.g
print(die)  <- in
    number      prob
0       1   0.166667
1       2   0.166667
2       3   0.166667
3       4   0.166667
4       5   0.166667
5       6   0.166667    <- out

np.mean(die['number'])  <- in
3.5     <- out

rolls_10 = die.sample(10, replace=True)
rolls_10    <- in
    number        prob
0       1     0.166667
0       1     0.166667
4       5     0.166667
1       2     0.166667
0       1     0.166667
0       1     0.166667
5       6     0.166667
5       6     0.166667
...     <- out

np.mean(rolls_10['number'])     <- in 
3.0     <- out

visualizing a sample
rolls_10['number'].hist(bins=np.linspace(1, 7, 7))
plt.show()

Law of Large numbers
This states that as the size of your sample increases, the sample mean will approach the expected value
e.g
Sample_size Mean
10          3.00
100         3.40
1000        3.48
'''

restaurant_groups = pd.read_csv(
    'Introduction to Statistics in Python/restaurant_groups.csv')

# Create a histogram of restaurant_groups and show plot
restaurant_groups['group_size'].hist(bins=[2, 3, 4, 5, 6])
plt.show()

# Create probability distribution
'''
size_dist = restaurant_groups['group_size'].value_counts() / restaurant_groups['group_size'].count()
'''
# or
size_dist = restaurant_groups['group_size'].value_counts(
) / restaurant_groups.shape[0]

# Reset index and rename columns
size_dist = size_dist.reset_index()
size_dist.columns = ['group_size', 'prob']
print(size_dist)

# Calculate expected value
expected_value = np.sum(size_dist['group_size'] * size_dist['prob'])
print(expected_value)

# Subset groups of size 4 or more
groups_4_or_more = size_dist[size_dist['group_size'] >= 4]

# Sum the probabilities of groups_4_or_more
prob_4_or_more = np.sum(groups_4_or_more['prob'])
print(prob_4_or_more)


'''
Continuous distributions
e.g
P(wait time <= 7)
from scipy.stats import uniform
uniform.cdf(7, 0, 12)

P(wait time >= 7) = 1 - P(wait time >= 7)
from scipy.stats import uniform
1 - uniform.cdf(7, 0, 12)

P(4 <= wait time >= 7)
from scipy.stats import uniform
uniform.cdf(7, 0, 12) - uniform.cdf(4, 0, 12)

# Uniform.cdf(value, lower_limit, upper_limit)

Generating random numbers according to uniform distribution
e.g
from scipy.stats import uniform
uniform.rvs(0, 5, 10)

# Uniform.rvs(lower_limit, upper_limit, size)

Other special types of distributions
- Normal distribution
- Exponential distribution
'''

# Min and max wait times for back-up that happens every 30 min
min_time = 0
max_time = 30

# Import uniform from scipy.stats
#   from scipy.stats import uniform

# Calculate probability of waiting less than 5 mins
prob_less_than_5 = uniform.cdf(5, 0, 30)
print(prob_less_than_5)

# Calculate probability of waiting more than 5 mins
prob_greater_than_5 = 1 - uniform.cdf(5, 0, 30)
print(prob_greater_than_5)

# Calculate probability of waiting 10-20 mins
prob_between_10_and_20 = uniform.cdf(20, 0, 30) - uniform.cdf(10, 0, 30)
print(prob_between_10_and_20)


# Set random seed to 334
np.random.seed(334)

# Import uniform

# Generate 1000 wait times between 0 and 30 mins
wait_times = uniform.rvs(0, 30, size=1000)

# Create a histogram of simulated times and show plot
plt.hist(wait_times)
plt.show()


'''
The binomial distribution
# Coin flip = Binary outcomes (probability of each (Head / Tail) occurring is 50%)

A single flip
binom.rvs(no of coins, probability of heads/success, size=no of trials)
e.g
from scipy.stats import binom
binom.rvs(1, 0.5, size=1) <- in
array([1]) <- out

binom.rvs(1, 0.5, size=8) <- in
array([0, 1, 1, 0, 1, 0, 1, 1]) <- out

Many flips one time (e.g 8 coin flips 1 time)
e.g
binom.rvs(8, 0.5, size=1) <- in
array([5]) <- out

Many flips many times (e.g 3 coin flips 10 times)
e.g
binom.rvs(3, 0.5, size=10) <- in
array([0, 3, 2, 1, 3, 0, 2, 2, 0, 0]) <- out

Other probabilities (e.g prob Head = 25%, Tail = 75%)
e.g
binom.rvs(3, 0.25, size=10) <- in
array([1, 1, 1, 1, 0, 0, 2, 0, 1, 0]) <- out

Binomial distribution
- This describes the probability distribution of the number of successes in a sequence of independent trials 
e.g Number of heads in a sequence of coin flips
Described by n and p
- n: total number of trials being preformed
- p: probability of success
i.e
binom.rvs(no of coins, p = probability of heads/success, n = size=no of trials)
binom.rvs(no of coins, p, n)

What's the probability of 7 heads?
e.g
p(heads =  7)
# binom.pmf(num heads / successes, num trials, prob of heads)
binom.pmf(7, 10, 0.5) <- in
0.1171875 <- out i.e 12% chance of success

What's the probability of 7 or fewer heads?
e.g
p(heads <=  7)
# binom.cdf(num heads / successes, num trials, prob of heads)
binom.cdf(7, 10, 0.5)
0.9453125 <- out i.e 95% chance of success

What's the probability of more than 7 heads?
e.g
p(heads >  7)
# 1 - binom.cdf(num heads / successes, num trials, prob of heads)
1 - binom.cdf(7, 10, 0.5)
0.0546875 <- out i.e 5% chance of success

Expected value
Expected value = n x p
e.g
Expected number of heads out of 10 flips = 10 x 0.5 = 5

Note: If trials are not independent, the binomial distribution does not apply!
'''

# Import binom from scipy.stats

# Set random seed to 10
np.random.seed(10)

# Simulate a single deal
print(binom.rvs(1, 0.3, size=1))

# Simulate 1 week of 3 deals
print(binom.rvs(3, 0.3, size=1))

# Simulate 52 weeks of 3 deals
deals = binom.rvs(3, 0.3, size=52)

# Print mean deals won per week
print(np.mean(deals))


# Probability of closing 3 out of 3 deals
prob_3 = binom.pmf(3, 3, 0.3)

print(prob_3)

# Probability of closing <= 1 deal out of 3 deals
prob_less_than_or_equal_1 = binom.cdf(1, 3, 0.3)

print(prob_less_than_or_equal_1)

# Probability of closing > 1 deal out of 3 deals
prob_greater_than_1 = 1 - binom.cdf(1, 3, 0.3)

print(prob_greater_than_1)


# Expected number won with 30% win rate
won_30pct = 3 * 0.3
print(won_30pct)

# Expected number won with 25% win rate
won_25pct = 3 * 0.25
print(won_25pct)

# Expected number won with 35% win rate
won_35pct = 3 * 0.35
print(won_35pct)


''' More Distributions and the Central Limit Theorem '''

'''
The normal distribution
The normal distribution shape is commonly referred to as a 'bell curve'
Important properties of a normal distribution:
- It is symmetrical i.e left side is a mirror image of the right.
- Area = 1 just like any continuous distribution
- The probability is never = 0
- It is described by Mean and Standard deviation

Note: 
# When a normal distribution has mean = 0 and a standard deviation = 1, It's a special distribution called the Standard Normal Distribution.
# For the Normal distribution, 68-95-99.7 rule.  
- 68% of the area is within 1 Standard deviation of the mean
- 95% of the area is within 2 Standard deviation of the mean
- 99.7% of the area is within 3 Standard deviation of the mean

what percent of women are shorter than 154cm ?
e.g
mean = 161cm
standard deviation = 7
from scipy.stats import norm
norm.cdf(154, 161, 7) <- in
0.158655 <- out i.e about 16% of women

what percent of women are taller than 154cm ?
e.g
mean = 161cm
standard deviation = 7
from scipy.stats import norm
1 - norm.cdf(154, 161, 7) <- in
0.841345 <- out i.e about 84% of women

what percent of women are 154-157cm ?
e.g
mean = 161cm
standard deviation = 7
from scipy.stats import norm
norm.cdf(157, 161, 7) - norm.cdf(154, 161, 7) <- in
0.1252 <- out i.e about 12% of women

What height are 90% of women shorter than?
e.g
mean = 161cm
standard deviation = 7
from scipy.stats import norm
norm.ppf(0.9, 161, 7) <- in
169.97086 <- out i.e about 170cm

What height are 90% of women taller than?
e.g
mean = 161cm
standard deviation = 7
from scipy.stats import norm
norm.ppf((1 - 0.9), 161, 7) <- in
152.029 <- out i.e about 152cm

Generating random numbers
e.g
mean = 161cm
standard deviation = 7
# Generate 10 random heights
norm.rvs(161, 7, size=10) <- in
array([155.5758223, 155.13133235, 160.06377097, 168.33345778, 165.92273375, 163.32677057, 165.13280753, 146.36133538, 149.07845021, 160.5790856 ]) <- out
'''

# Histogram of amount with 10 bins and show plot
amir_deals['amount'].hist(bins=10)
plt.show()


# Probability of deal < 7500
prob_less_7500 = norm.cdf(7500, 5000, 2000)

print(prob_less_7500)

# Probability of deal > 1000
prob_over_1000 = 1 - norm.cdf(1000, 5000, 2000)

print(prob_over_1000)

# Probability of deal between 3000 and 7000
prob_3000_to_7000 = norm.cdf(7000, 5000, 2000) - norm.cdf(3000, 5000, 2000)

print(prob_3000_to_7000)

# Calculate amount that 25% of deals will be less than
pct_25 = norm.ppf(0.25, 5000, 2000)

print(pct_25)


# Calculate new average amount
new_mean = 5000 + 5000 * 0.20

# Calculate new standard deviation
new_sd = 2000 + 2000 * 0.30

# Simulate 36 new sales
new_sales = norm.rvs(new_mean, new_sd, size=36)

# Create histogram and show
plt.hist(new_sales)
plt.show()


'''
The central limit theorem
- This states that the sampling distribution of a statistic becomes closer to the normal distribution as the number of trials increases.
- This only applies when the samples are taken randomly and are independent.

Standard deviation and the CLT
e.g
die = pd.series([1, 2, 3, 4, 5, 6])
sample_sds = []
for i in range(1000):
    sample_sds.append(np.std(die.sample(5, replace=True)))

Proportions and the CLT
e.g
sales_team = pd.series(['Amir', 'Brian', 'Claire', 'Damian'])
sales_team.sample(10, replace=True)

Mean of sampling distribution
# Estimate expected value of die
np.mean(sample_means) <- in
3.48 <- out

# Estimate proportion of 'Claire's
np.mean(sample_props) <- in
0.26 <- out
'''

# Create a histogram of num_users and show
amir_deals['num_users'].hist()
plt.show()

# Set seed to 104
np.random.seed(104)

# Sample 20 num_users with replacement from amir_deals
samp_20 = amir_deals['num_users'].sample(20, replace=True)

# Take mean of samp_20
print(np.mean(samp_20))

# Sample 20 num_users with replacement from amir_deals and take mean
samp_20 = amir_deals['num_users'].sample(20, replace=True)
np.mean(samp_20)

sample_means = []
# Loop 100 times
for i in range(100):
    # Take sample of 20 num_users
    samp_20 = amir_deals['num_users'].sample(20, replace=True)
    # Calculate mean of samp_20
    samp_20_mean = np.mean(samp_20)
    # Append samp_20_mean to sample_means
    sample_means.append(samp_20_mean)

print(sample_means)

# Convert to Series and plot histogram
sample_means_series = pd.Series(sample_means)
sample_means_series.hist()

# Show plot
plt.show()


# Set seed to 321
np.random.seed(321)

sample_means = []
# Loop 30 times to take 30 means
for i in range(30):
    # Take sample of size 20 from num_users col of all_deals with replacement
    cur_sample = all_deals['num_users'].sample(20, replace=True)
    # Take mean of cur_sample
    cur_mean = np.mean(cur_sample)
    # Append cur_mean to sample_means
    sample_means.append(cur_mean)

# Print mean of sample_means
print(np.mean(sample_means))

# Print mean of num_users in amir_deals
print(np.mean(amir_deals['num_users']))


'''
The Poisson distribution
- This is a process where events appear to happen at a certain rate, but completely at random
- e.g
*   Number of animals adopted from an animal shelter per week
*   Number of people arriving at a restaurant per hour
*   Number of earthquakes in California per year
- Time unit is irrelevant, as long as you use the same unit when talking about the same situation
- It describes the probability of some number of events happening over a fixed period of time
- It is described by a value called Lambda, which represents the average number of events per time period. (it is also the Expected value of the distribution.)
i.e
Lambda = average number of events per time interval

Probability of a single value
e.g
If the average number of adoptions per week is 8, what is P(no of adoptions in a week = 5)?
poisson.pmf(value, mean)
i.e
from scipy.stats import poisson
poisson.pmf(5, 8) <- in
0.09160366 <- out about 9% chance

Probability of less than or equal to
e.g
If the average number of adoptions per week is 8, what is P(no of adoptions in a week <= 5)?
poisson.cdf(value, mean)
i.e
from scipy.stats import poisson
poisson.cdf(5, 8) <- in
0.1912361 <- out about 20% chance

Probability of greater than
e.g
If the average number of adoptions per week is 8, what is P(no of adoptions in a week > 5)?
1 - poisson.cdf(value, mean)
i.e
from scipy.stats import poisson
1 - poisson.cdf(5, 8) <- in
0.8087639 <- out about 80% chance

Sampling from a Poisson distribution
poisson.rvs(mean, size=sample_size)
e.g
from scipy.stats import poisson
poisson.rvs(8, size=10) <- in
array([9, 9, 8, 7, 11, 3, 10, 6, 8, 14]) <- out

Note: Central Limit Theorem applies in Poisson distributions
'''

# Import poisson from scipy.stats

# Probability of 5 responses
prob_5 = poisson.pmf(5, 4)

print(prob_5)

# Probability of 5 responses
prob_coworker = poisson.pmf(5, 5.5)

print(prob_coworker)

# Probability of 2 or fewer responses
prob_2_or_less = poisson.cdf(2, 4)

print(prob_2_or_less)

# Probability of > 10 responses
prob_over_10 = 1 - poisson.cdf(10, 4)

print(prob_over_10)


'''
More probability distributions
Exponential distribution
- This represents the probability of a certain time passing between Poisson events.
- e.g
*   probability of >1 day between adoptions
*   probability of <10 minutes between restaurant arrivals
*   probability of 6-8 months between earthquakes
- It also uses Lambda which represents the rate, that the Poisson distribution does.
- It is Continuous, unlike the Poisson distribution, since it represents time.

Customer service requests
On average, one customer service ticket is created every 2 minutes
i.e
Lambda = 0.5 customer service tickets created each minute

Expected value of Exponential distribution
- In terms of rate (Poisson):
Lambda = 0.5 requests per minute

- In terms of time (exponential):
1 /  Lambda = 1 request per 2 minutes

How long until a new request is created?
e.g
from scipy.stats import expon

P(wait < 1 min) = expon.cdf(1, scale=0.5) <- in
0.8646647167633873 <- out about 86% chance

P(wait > 3 min) = 1 - expon.cdf(3, scale=0.5) <- in
0.0024787521766663767 <- out about 0.2% chance

P(1 min < wait < 3 min) = expon.cdf(3, scale=0.5) - expon.cdf(1, scale=0.5) <- in
0.13285653105994633 <- out about 13% chance


(Student's) T - Distribution
- It has similar shape as the normal distribution with some degree of freedom which makes their tails thicker. i.e Their observations are more likely to fall further from the mean.

Degrees of freedom (df)
- they affect the thickness of the tails
i.e
*   Lower df = Thicker tails, Higher standard deviation
*   Higher df = Closer to normal distribution


Log-Normal distribution
- Variables that follow a log-normal distribution have a logarithm that is normally distributed.
- They result in distributions that are skewed, unlike the normal distribution.
e.g
*   Length of chess games
*   Adult blood pressure
*   Number of hospitalizations in the 2003 SARS outbreak
'''

# Import expon from scipy.stats

# Print probability response takes < 1 hour
print(expon.cdf(1, scale=2.5))

# Print probability response takes > 4 hours
print(1 - expon.cdf(4, scale=2.5))

# Print probability response takes 3-4 hours
print(expon.cdf(4, scale=2.5) - expon.cdf(3, scale=2.5))


''' Correlation and Experimental Design '''\

'''
Correlation
Relationships between 2 variables
*   x-axis =  Explanatory / Independent variable
*   y-axis = Response / Dependent variable

Correlation coefficient
- Quantifies the linear relationship between 2 variables
- Number between -1 and 1
- Magnitude corresponds to strength of relationship
- Sign ( + or - ) corresponds to direction of relationship

Magnitude = Strength of relationship
*   0.99 = Very strong relationship
*   0.75 = Strong relationship
*   0.56 = Moderate relationship
*   0.21 = Weak relationship
*   0.04 = No relationship (i.e Knowing the value of x doesn't tell us anything about y)

Sign = Direction
*   + / positive = As x increases, y increases
*   - / negative = As x increases, y decreases

Visualizing relationships
Scatterplot
e.g
import seaborn as sns
sns.scatterplot(x='column_name', y='column_name1', data=df)
plt.show()

Adding a trendline
e.g
import seaborn as sns
sns.lmplot(x='column_name', y='column_name1', data=df, ci=None)
plt.show()

Computing correlation
df['column_name'].corr(df['column_name1'])

Many ways to calculate correlation
*   Used in this course: Pearson product-moment correlation (r) which is the most common
- Variations on this formula:
*   Kendall's tau
*   Spearman's rho
'''

world_happiness = pd.read_csv(
    'Introduction to Statistics in Python\world_happiness.csv')

# Create a scatterplot of happiness_score vs. life_exp and show
sns.scatterplot(x='life_exp', y='happiness_score',  data=world_happiness)

# Show plot
plt.show()

# Create scatterplot of happiness_score vs life_exp with trendline
sns.lmplot(x='life_exp', y='happiness_score', data=world_happiness, ci=None)

# Show plot
plt.show()

# Correlation between life_exp and happiness_score
cor = world_happiness['life_exp'].corr(world_happiness['happiness_score'])

print(cor)


'''
Correlation caveats
Non-Linear relationships (e.g Quadratic relationship)
e.g
r = 0.18

Correlation only accounts for linear relationships
Note: The correlation coefficient measures the strength of linear relationships, and linear relationships only.
- Correlation shouldn't be used blindly
- Always visualize your data

Log transformation
This is applied when a data is highly skewed i.e have a weak linear relationship
e.g
df['column_name2'] = np.log(df['column_name1'])

sns.lmplot(x = 'column_name2', y = 'column_name', data = df, ci = None)
plt.show()

df['column_name2'].corr(df['column_name2'])

Other transformations
- Log transformation ( log(x) )
- Square root transformation ( sqrt(x) )
- Reciprocal transformation ( 1 / x )
- Combinations of these, e.g:
*   log(x) and log(y)
*   sqrt(x) and 1 / y

Why use a transformation?
- Certain statistical methods rely on variables having a linear relationship
e.g
*   Correlation coefficient
*   Linear regression

Correlation does not imply causation
x is correlated with y does not mean x causes y

Confounding
- This phenomenon can lead to spurious correlations.
- In statistics, a spurious correlation (or spuriousness) refers to a connection between two variables that appears to be causal but is not. With spurious correlation, any observed dependencies between variables are merely due to chance or are both related to some unseen confounder.
'''

# Scatterplot of gdp_per_cap and life_exp
sns.scatterplot(x='gdp_per_cap', y='life_exp', data=world_happiness)

# Show plot
plt.show()

# Correlation between gdp_per_cap and life_exp
cor = world_happiness['gdp_per_cap'].corr(world_happiness['life_exp'])

print(cor)


# Scatterplot of happiness_score vs. gdp_per_cap
sns.scatterplot(x='gdp_per_cap', y='happiness_score', data=world_happiness)
plt.show()

# Calculate correlation
cor = world_happiness['happiness_score'].corr(world_happiness['gdp_per_cap'])
print(cor)

# Create log_gdp_per_cap column
world_happiness['log_gdp_per_cap'] = np.log(world_happiness['gdp_per_cap'])

# Scatterplot of log_gdp_per_cap and happiness_score
sns.scatterplot(x='log_gdp_per_cap', y='happiness_score', data=world_happiness)
plt.show()

# Calculate correlation
cor = world_happiness['log_gdp_per_cap'].corr(
    world_happiness['happiness_score'])
print(cor)


# Scatterplot of grams_sugar_per_day and happiness_score
sns.scatterplot(x='grams_sugar_per_day',
                y='happiness_score', data=world_happiness)
plt.show()

# Correlation between grams_sugar_per_day and happiness_score
cor = world_happiness['grams_sugar_per_day'].corr(
    world_happiness['happiness_score'])
print(cor)


'''
Design of experiments
Vocabulary
Experiment aims to answer: What is the effect of the treatment on the response?
*   Treatment: explanatory / independent variable
*   Response: response / dependent variable
e.g
What is the effect of an advertisement on the number of products purchased?
*   Treatment: Advertisement
*   Response: Number of products purchased

Controlled experiments
Participants are assigned by researchers to either treatment group or control group
*   Treatment group sees advertisement
*   Control group does not
Groups should be comparable so that causation can be inferred
If groups are not comparable, this could lead to confounding (bias)
e.g
*   Treatment group average age: 25
*   Control group average age: 50
*   Age is a potential confounder

The gold standard of experiments will use...
- Randomized controlled trial
*   Participants are assigned to treatment / control randomly, not based on any other characteristics
*   Choosing randomly helps ensure that groups are comparable
- Placebo
*   Resembles treatment, but has no effect
*   Participants will not know which group they're in
*   In clinical trials, a sugar pill ensures that the effect of the drug is actually due to the drug itself and not the idea of receiving the drug
- Double-blind trial
*   Person administering the treatment / running the study doesn't know whether the treatment is real or a placebo
*   Prevents bias in the response and / or analysis of results

Fewer opportunities for bias = more reliable conclusion about causation


Observational studies
- Participants are not assigned randomly to groups
*   Participants assign themselves, usually based on pre-existing characteristics
- Many research questions are not conducive to a controlled experiment
*   You can't force someone to smoke or have a disease
*   You can't make someone have certain past behaviour
- Establish association, not causation
*   Effects can be confounded by factors that got certain people into the control or treatment group
*   There are ways to control for confounders to get more reliable conclusions about association


Longitudinal vs Cross-Sectional studies
Longitudinal Study
- Participants are followed over a period of time to examine effect of treatment on response
*   Effect of age on height is not confounded by generation
*   More expensive, results take longer

Cross-sectional study
- Data on participants is collected from a single snapshot in time
*   Effect of age on height is confounded by generation
*   Cheaper, faster, more convenient
'''
