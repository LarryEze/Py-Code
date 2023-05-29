''' Introduction to Hypothesis Testing '''

'''
A/B testing
- This uses Treatment and Control groups for testing
- It provides a way to check outcomes of competing scenarios and decide which way to proceed.
- A/B testing lets you compare scenarios to see which best achieves some goal.

Hypothesis tests and Z-scores
e.g
Stack Overflow Developer Survey 2020
import pandas as pd
print(stack_overflow)

Hypothesizing about the mean
A hypothesis:
The mean anual compensation of the population of data scientists is $110,000

The point estimate (sample statistic):
mean_comp_samp = stack_overflow['converted_comp'].mean() <- in

119574.71738168952 <- out

Generating a bootstrap distribution
import numpy as np
# Step 3. Repeat steps 1 & 2 many times, appending to a list
so_boot_distn = []
for i in range(5000):
    so_boot_distn.append(
        # Step 2. Calculate point estimate
        np.mean(
            # Step 1. Resample
            stack_overflow.sampe(frac=1, replace=True)['converted_comp']
        )
    )

Visualizing the bootstrap distribution
import matplotlib.pyplot as plt
plt.hist(so_boot_distn, bins=50)
plt.show()

Standard error
std_error = np.std(so_boot_distn, ddof=1) <- in

5607.997577378606 <- out

Z-scores
standardized value = (value - mean) / standard deviation
#   OR
Z = (Sample Statistic - Hypothesized parameter value) / Standard Error

mean_comp_samp = stack_overflow['converted_comp'].mean() <- in

119574.71738168952 <- out

mean_comp_hyp = 110000

std_error = np.std(so_boot_distn, ddof=1) <- in

5607.997577378606 <- out

Z_score = (mean_comp_samp - mean_comp_hyp) / std_error <- in 

1.7073326529796957 <- out

- The z-score is a standardized measure of the difference between the sample statistic and the hypothesized statistic.

Testing the hypothesis
Hypothsis testing use case:
Determine whether sample statistics are too close or far away from expected (or "hypothesized" values)

Standard normal ( Z ) distribution
Standard normal distribution: normal distribution with mean  0 + standard deviation = 1
'''

# Print the late_shipments dataset
print(late_shipments)

# Calculate the proportion of late shipments
late_prop_samp = (late_shipments['late'] == 'Yes').mean()

# Print the results
print(late_prop_samp)


# Hypothesize that the proportion is 6%
late_prop_hyp = 0.06

# Calculate the standard error
std_error = np.std(late_shipments_boot_distn)

# Find z-score of late_prop_samp
z_score = (late_prop_samp - late_prop_hyp) / std_error

# Print z_score
print(z_score)


'''
p-values

Age of first programming experience
- age_first_code_cut classifies when Stack Overflow user first started programming
* 'adult' means they started at 14 or older
* 'child' means they started before 14
- Previous research: 35% od software developers started programming as children

- Evidence that a greater proportion of data scientists starting programming as children?

Definitions
- A hypothesis is a statement about an unknown population parameter
- A hypothesis test is a test of two competing hypotheses
* The null hypothesis (Ho) is the existing idea
* The alternative hypothesis (Ha) is the new 'challenger' idea of the researcher
For our problem:
* Ho: The proportion of data scientists starting programming as children is 35%
* Ha: The proportion of data scientists starting programming as children is greater than 35%

# "Naught" is British English for "zero". For historical reasons, "H-naught" is the international convention for pronouncing the null hypothesis.

Hypothesis Testing
- Either Ha or H0 is true (not both)
_ Initially, Ho is assumed to be true
_ The test ends in either 'reject Ho' or 'fail to reject Ho'
- If the evidence from the sample is 'significant' that Ha is true, reject Ho, else choose Ho

# Significance level is 'beyond a reasonable doubt' for hypothesis testing

One-tailed and two-tailed tests
- The tails of a distribution are the left and right edges of its PDF
- Hypothesis tests check if the sample statistics lie in the tails of the null distribution, which is the distribution of the statistic if the null hypothesis was true.
- There are 3 types of tests, and the phrasing of the alternative hypothesis determines which type we should use.
Test                                    Tails
* alternative different from null   =   2-tailed
* alternative greater than null     =   right-tailed
* alternative less than null        =   left tailed

e.g
Ha: The proportion of data scientists starting programming as children is greater than 35%
- This is a right-tailed test 

P-values
- They measure the strength of support for the null hypothesis
- p-values: probability of obtaining a result, assuming the null hypothesis is true
* Large p-value, large support for Ho
- Statistic likely not in the tail of the null distribution
* Small p-value, strong evidence against Ho
- Statistic likely in the tail of the null distribution

# "p" in p-value -> probability
"small" means "close to zero"

Calculating the z-score
prop_child_samp = (stack_overflow['age_first_code_cut'] == 'child').mean() <- in
0.39141972578505085 <- out

prop_child_hyp = 0.35

std_error = np.std(first_code_boot_distn, ddof=1) <- in

0.010351057228878566 <- out

z_score = (prop_child_samp - prop_child_hyp) / std_error <- in

4.001497129152506  -> out

Calculating the p-value
- norm.cdf() is normal CDF from scipy.stats
- Left-tailed test -> use norm.cdf()
- Right-tailed test -> use 1 - norm.cdf()
i.e
from scipy.stats import norm
1 - norm.cdf(z_score, loc=0, scale=1) <- in

3.1471479512323874e-05 <- out
'''

# Calculate the z-score of late_prop_samp
z_score = (late_prop_samp - late_prop_hyp) / std_error

# Calculate the p-value
p_value = 1 - norm.cdf(z_score, loc=0, scale=1)

# Print the p-value
print(p_value)


'''
Statistical significance
p-value recap
- p-values quantify evidence for the null hypothesis
- Large p-value -> fail to reject null hypothesis
- Small p-value -> reject null hypothesis

where is the cutoff point ?

Significance level
The significance level of a hypothesis test (alpha) is the threshold point for 'beyond a reasonable doubt'
- common values of alpha are 0.2, 0.1, 0.05 and 0.01
- If p <= alpha, reject Ho, else fail to reject Ho
- Alpha should be set prior to conducting the hypothesis test

Calculating the p-value
alpha = 0.05

prop_child_samp = (stack_overflow['age_first_code_cut'] == 'child').mean()

prop_child_hyp = 0.35

std_error = np.std(first_code_boot_distn, ddof=1)

z_score = (prop_child_samp - prop_child_hyp) / std error

p_value = 1 - norm.cdf(z_score, loc=0, scale=1) -> in

3.1471479512323874e-05 -> out

Making a decison
alpha = 0.05
print(p_value) -> in

3.1471479512323874e-05 -> out

p_value <= alpha
True
# Reject Ho in favour of Ha

Confidence intervals
For a significance level of alpha, it's common to choose a confidence interval of (1 - alpha)
- alpha = 0.05 -> 95% confidence interval
i.e
import numpy as np
lower = np.quantile(first_code_boot_distn, 0.025)
upper = np.quantile(first_code_boot_distn, 0.975)
print((lower, upper)) -> -in

(0.37063246351172047, 0.41132242370632466) -> out


Types of errors
            actual Ho           actual Ha
chosen Ho   correct             false negative
chosen Ha   false positive      correct

# False positives are Type I errors; False negatives are Type II errors.

Possible errors in our example
If p <= alpha, we reject Ho:
* A false positive (Type I) error: data scientists didn't start coding as children at a higher rate
If p > alpha, we fail to reject Ho:
* A false positive (Type II) error: data scientists started coding as children at a higher rate
'''

# Calculate 95% confidence interval using quantile method
lower = np.quantile(late_shipments_boot_distn, 0.025)
upper = np.quantile(late_shipments_boot_distn, 0.975)

# Print the confidence interval
print((lower, upper))


''' Two-Sample and ANOVA Tests '''

'''
Performing t-tests
Two-sample problems
- Compare sample statistics across groups of a variable
* 'converted_comp' is a numeriacal variable
* 'age_first_code_cut' is a categorical variable with levels ('child' and 'adult')

Are users who first programmed as a child compensated higher than those that started as adults?

Hypotheses
Ho: The mean compensation (in USD) is the same for those that coded first as a child and those that coded first as an adult
Ho: Mu(child) = Mu(adult)
# OR
Ho: Mu(child) - Mu(adult) = 0

Ha: The mean compensation (in USD) is greater for those that coded first as a child compared to those that coded first as an adult
Ho: Mu(child) > Mu(adult)
# OR
Ho: Mu(child) - Mu(adult) > 0

Calculating groupwise summary statistics
stack_overflow.groupby('age_first_code_cut')['converted_comp'].mean() -> in
age_first_code_cut
adult    111313.311047
child     132419.570621
Name: converted_comp, dtype: float64 -> out

Test statistics
- Sample mean estimates the population mean
- x-bar is used to denote a sample mean
* x-bar(child) - sample mean compensation for coding first as a child
* x-bar(adult) - sample mean compensation for coding first as an adult
* x-bar(child) - x-bar(adult) - a test statistic
- z-score - a (standardized) test statistic

Standardizing the test statistic
z = (sample statistic - population parameter) / standard error

t = (difference in sample statistic - difference in population parameter) / standard error

t = ( ( x-bar(child) - x-bar(adult) )  - ( Mu(child) - Mu(adult) )  ) / SE( x-bar(child) - x-bar(adult) )

Standard Error
SE( x-bar(child) - x-bar(adult) ) is equal to approximate value of the square root of the (child)Standard deviation square divided by the (child)sample size + (adult)Standard deviation square divided by the (adult)sample size.

* s is the standard deviation of the variables
* n is the sample size (number of observations/rows in sample)

Assuming the null hypothesis is true
t = ( ( x-bar(child) - x-bar(adult) )  - ( Mu(child) - Mu(adult) )  ) / SE( x-bar(child) - x-bar(adult) )

Ho: Mu(child) - Mu(adult) = 0 -> t = ( x-bar(child) - x-bar(adult) ) / SE( x-bar(child) - x-bar(adult) )

t = ( x-bar(child) - x-bar(adult) ) / np.sqrt( s^2(child) / n(child) + s^2(adult) / n(adult) )

Calculations assuming the null hypothesis is true
e.g
xbar = stack_overflow.groupby('age_first_code_cut')['converted_comp'].mean() -> in
adult    111313.311047
child     132419.570621
Name: converted_comp, dtype: float64 age_first_code_cut -> out

s = stack_overflow.groupby('age_first_code_cut')['converted_comp'].std() -> in
adult    271546.521729
child     255585.240115
Name: converted_comp, dtype: float64 age_first_code_cut -> out

n = s = stack_overflow.groupby('age_first_code_cut')['converted_comp'].count() -> in
adult    1376
child      885
Name: converted_comp, dtype: float64 age_first_code_cut -> out

Calculating the test statistic
import numpy as np
numerator = xbar_child - xbar_adult
denominator = np.sqrt(s_child ** 2 / n_child + s_adult ** 2 / n_adult)
t_stat = numerator / denominator -> in
1.8699313316221844 -> out
'''

# Calculate the numerator of the test statistic
numerator = xbar_no - xbar_yes

# Calculate the denominator of the test statistic
denominator = np.sqrt(s_no ** 2 / n_no + s_yes ** 2 / n_yes)

# Calculate the test statistic
t_stat = numerator / denominator

# Print the test statistic
print(t_stat)


'''
Calculating p-values from t-statistics
t-distribution
- t statistic follow a t-distribution
- Have a parameter named degrees of freedom, or df
- Look like normal distributions, with fatter tails

Degrees of freedom
- Larger degrees of freedom -> t-distribution gets closer to the normal distribution
- Normal distribution -> t-distribution with infinite df
- Degrees of freedom: maximum number of logically independent values in the data sample.

Calculating degrees of freedom
e.g
Dataset has 5 independent observations
- four of the values are 2, 6, 8, and 5
- The sample mean is 5
- The last value must be 4
- i.e Here, there are 4 degrees of freedom

- df = n(child) + n(adult) - 2 

Significance level
alpha = 0.1
if p <= alpha, then reject Ho.

Calculating p-values: one proportion vs. a value
from scipy.stats import norm
1 - norm.cdf(z_score)

- z-statistic: needed when using one sample statistic to estimate a population parameter
- t-statistic: needed when using multiple sample statistics to estimate a population parameter

Calculating p-values: two means from different groups
numerator = xbar_child - xbar_adult

denominator = np.sqrt(s_child ** 2 / n_child + s_adult ** 2 / n_adult)

t_stat = numerator / denominator <- in

1.8699313316221844 <- out

degrees_of_freedom = n_child + n_adult - 2 <- in

2259 <- out

- Test statistic standard error used an approximation (not bootstrapping).
- Use t-distribution CDF not normal CDF
from scipy.stats import t
1 - t.cdf(t_stat, df=degrees_of_freedom) <- in

0.030811302165157595 <- out

- Evidence that Stack Overflow data scientists who started coding as a child earn more. 

NB: Using a sample standard deviation to estimate the standard error is computationally easier than using bootstrapping. However, to correct for the approximation, you need to use a t-distribution when transforming the test statistic to get the p-value.
'''

# Calculate the degrees of freedom
degrees_of_freedom = n_no + n_yes - 2

# Calculate the p-value from the test stat
p_value = t.cdf(t_stat, df=degrees_of_freedom)

# Print the p_value
print(p_value)


'''
Paired t-tests
Hypotheses Question
Was the percentage of Republican candidate votes lower in 2008 than 2012?
Formula:
Ho: Mu(2008) - Mu(2012) = 0
Ha: Mu(2008) - Mu(2012) < 0

Set alpha = 0.05 significance level
Note:
* Data is paired -> each voter percentage refers to the same county
e.g want to capture voting patterns in model

From two samples to one
sample_data = repub_votes_potus_08_12
sample_data['diff'] = sample_data['repub_percent_08'] - sample_data['repub_percent_12']

impor matplotlib.pyplot as plt
sample_data['diff'].hist(bins = 20)

Calculate sample statistics of the difference
xbar_diff = sample_data['diff'].mean() <- in

-2.877109041242944 <- out

Revised hypotheses
Old hypotheses:
Ho: Mu(2008) - Mu(2012) = 0
Ha: Mu(2008) - Mu(2012) < 0

New hypotheses:
Ho: Mu(diff) = 0
Ha: Mu(diff) < 0

t = ( x-bar(diff) - Mu(diff) ) / square root (  S^2 (diff) / n(diff) )

df = n(diff) - 1

Calculating the p-value
n_diff = len(sample_data) <- in

100 <- out

Mu(diff) =  0

s_diff = sample_data['diff'].std()

t_stat = ( xbar_diff - 0 ) / np.sqrt( s_diff**2 / n_diff) <- in

-5.601043121928489 <- out

degrees_of_freedom = n_diff - 1 <- in

99 <- out

from scipy.stats import t
p_value = t.cdf( t_stat, df= n_diff - 1) <- in

9.572537285272411e-08 <- out

Therefore: Reject the Ho (null hypotheses) in favour of the Ha (alternative hypotheses) that the Repulican candidates got a smaller percentage of the vote in 2008 compared to 2012.

Testing differences between two means using ttest()
import pingouin
pingouin.ttest(x=sample_data['diff'], y=0, alternative='less') <- in

        T dof alternative        p-val         CI95%  cohen-d      BF10 power
T-test     99        less 9.572537e-08 [-inf, -2.02] 0.560104 1.323e+05   1.0 <- out
# OR
ttest() with paired=True
pingouin.ttest(x=sample_data['repub_percent_08'], y=sample_data['repub_percent_12'], paired=True, alternative='less') <- in

        T dof alternative        p-val         CI95%  cohen-d      BF10 power
T-test     99        less 9.572537e-08 [-inf, -2.02] 0.560104 1.323e+05   1.0 <- out

Note: Unpaired t-tests on paired data increases the chances of false negative errors
'''

# Calculate the differences from 2012 to 2016
sample_dem_data['diff'] = sample_dem_data['dem_percent_12'] - \
    sample_dem_data['dem_percent_16']

# Find the mean of the diff column
xbar_diff = sample_dem_data['diff'].mean()

# Find the standard deviation of the diff column
s_diff = sample_dem_data['diff'].std()

# Plot a histogram of diff with 20 bins
sample_dem_data['diff'].hist(bins=20)
plt.show()


# Conduct a t-test on diff
test_results = pingouin.ttest(
    x=sample_dem_data['diff'],  y=0,  alternative="two-sided")

# Conduct a paired t-test on dem_percent_12 and dem_percent_16
paired_test_results = pingouin.ttest(
    x=sample_dem_data['dem_percent_12'], y=sample_dem_data['dem_percent_16'], paired=True,  alternative="two-sided")

# Print the paired test results
print(paired_test_results)


'''
ANOVA tests

Job satisfaction: 5 categories
stack_overflow['job_sat'].value_counts() <- in
Very satisfied          879
Slightly satisfied      680
Slightly dissatisfied   342
Neither                 201   
Very dissatisfied       159
Name: job_sat,  dtype: int64 <- out

Visualizing multiple distributions
Is mean annual compensation different for different levels of job satisfaction ?

import seaborn as sns
import matplotlib.pyplot as plt
sns.boxplot(x='converted_comp', y='job_sat', data=stack_overflow)
plt.show()

Analysis of variance (ANOVA)
- A test for differences between groups
* dv = Depedent variable
* between = Column of groups to calculate 
* p-unc = p-val
e.g
alpha = 0.2

pingouin.anova(data=stack_overflow,  dv='converted_comp', between='job_sat') <- in
    Source ddof1 ddof2        F    p-unc      np2
0  job_sat     4  2256 4.480485 0.001315 0.007882 <- out

- 0.001315 < alpha
- Therefore, at least two categories have significantly different compensation

Pairwise tests()
alpha = 0.2 

pingouin.pairwise_tests(data=stack_overflow,  dv='converted_comp', between='job_sat', padjust='none') <- in

    Contrast                  A                     B Paired Parametric ...         dof alternative    p-unc    BF10    hedges
0    job_sat Slightly satisfied        Very satisfied  False       True ... 1478.622799   two-sided 0.000064 158.564 -0.192931
1    job_sat Slightly satisfied               Neither  False       True ...  258.204546   two-sided 0.484088   0.114 -0.068513
2    job_sat Slightly satisfied     Very dissatisfied  False       True ...  187.153329   two-sided 0.215179   0.208 -0.145624
3    job_sat Slightly satisfied Slightly dissatisfied  False       True ...  569.926329   two-sided 0.969491   0.074 -0.002719
4    job_sat     Very satisfied               Neither  False       True ...  328.326639   two-sided 0.097286   0.337  0.120115
5    job_sat     Very satisfied     Very dissatisfied  False       True ...  221.666205   two-sided 0.455627   0.126  0.063479
6    job_sat     Very satisfied Slightly dissatisfied  False       True ...  821.303063   two-sided 0.002166    7.43  0.173247
7    job_sat            Neither     Very dissatisfied  False       True ...  321.165726   two-sided 0.585481   0.135 -0.058537
8    job_sat   Neither Slightly          dissatisfied  False       True ...  367.730081   two-sided 0.547406   0.118  0.055707
9    job_sat  Very dissatisfied Slightly dissatisfied  False       True ...  247.570187   two-sided 0.259590   0.197  0.119131
[10 rows x 11 columns] <- out
i.e
3 values are less than alpha (Significance level) of 0.2

In this case, there are 5 groups, resulting in 10 pairs
Note: As the number of groups increases, the number of pairs - and hence the number of hypothesis tests we must perform - increases quadratically.
- The more tests we run, the higher the chance that at least one of them will give a false positive significant result.
- With a significance level of 0.2, if we run one test, the chance of a false positive result is 0.2
- With 5 groups and 10 tests, the probability of at least one false positive is around 0.7
- With 20 groups, it's almost guaranteed that we'll get at least one false positive.
- The solution to this is to apply an adjustment to increase the p-values, reducing the chance of getting a false positive. (one common adjustment is the Bonferroni correction).

Bonferroni correction 
pingouin.pairwise_tests(data=stack_overflow,  dv='converted_comp', between='job_sat', padjust='none') <- in

  Contrast                  A                     B ...    p-unc   p-corr p-adjust    BF10    hedges
0  job_sat Slightly satisfied        Very satisfied ... 0.000064 0.000638     bonf 158.564 -0.192931
1  job_sat Slightly satisfied               Neither ... 0.484088 1.000000     bonf   0.114 -0.068513
2  job_sat Slightly satisfied     Very dissatisfied ... 0.215179 1.000000     bonf   0.208 -0.145624
3  job_sat Slightly satisfied Slightly dissatisfied ... 0.969491 1.000000     bonf   0.074 -0.002719
4  job_sat     Very satisfied               Neither ... 0.097286 0.972864     bonf   0.337  0.120115
5  job_sat     Very satisfied     Very dissatisfied ... 0.455627 1.000000     bonf   0.126  0.063479
6  job_sat     Very satisfied Slightly dissatisfied ... 0.002166 0.021659     bonf    7.43  0.173247
7  job_sat            Neither     Very dissatisfied ... 0.585481 1.000000     bonf   0.135 -0.058537
8  job_sat            Neither Slightly dissatisfied ... 0.547406 1.000000     bonf   0.118  0.055707
9  job_sat  Very dissatisfied Slightly dissatisfied ... 0.259590 1.000000     bonf   0.197  0.119131
[10 rows x 11 columns] <- out
i.e
2 values are less than alpha (Significance level) of 0.2

* p-unc = Uncorrected p-val
* p-corr = Corrected p-val

More methods
padjust : string
Method used for testing and adjustment of pvalues.
* 'none' : no correction [default]
* 'bonf' : one-step Bonferroni correction
* 'sidak' : one-step Sidak correction
* 'holm' : step-down method using Bonferroni adjustments
* 'fdr_bh' : Benjamini / Hochberg FDR correction
* 'fdr_by' : Benjamini / Yekutieli FDR correction
'''

# Calculate the mean pack_price for each shipment_mode
xbar_pack_by_mode = late_shipments.groupby(
    "shipment_mode")['pack_price'].mean()

# Calculate the standard deviation of the pack_price for each shipment_mode
s_pack_by_mode = late_shipments.groupby("shipment_mode")['pack_price'].std()

# Boxplot of shipment_mode vs. pack_price
sns.boxplot(x='pack_price', y='shipment_mode', data=late_shipments)
plt.show()


# Run an ANOVA for pack_price across shipment_mode
anova_results = pingouin.anova(
    data=late_shipments, dv='pack_price', between='shipment_mode')

# Print anova_results
print(anova_results)


# Modify the pairwise t-tests to use Bonferroni p-value adjustment
pairwise_results = pingouin.pairwise_tests(
    data=late_shipments, dv="pack_price", between="shipment_mode", padjust="bonf")

# Print pairwise_results
print(pairwise_results)


''' Proportion Tests '''

'''
One-sample proportion tests
Chapter 1 recap
- Is a claim about an unknown population proportion feasible?
i.e
* Standard error of sample statistic from bootstrap distribution
* Compute a standardized test statistic - z score
* Calculate a p-value
* Decide which hypothesis made most sense

Standardized test statistic for proportions
P: population proportion (unknown population parameter)
P-hat: sample proportion (sample statistic)
Po: hypothesized population proportion

z =  ( P-hat -  mean( P-hat ) ) / SE( P-hat ) = ( P-hat -  P ) /  SE( P-hat )

Assuming Ho is true, P = Po, so
z = ( P-hat -  Po ) /  SE( P-hat )

Simplifying the standard error calculations
SE( P-hat ) = square root ( ( Po x ( 1 - Po ) ) / n ) -> Under Ho, SE( P-hat ) depends on hypothesized ( Po ) and sample size ( n )

Assuming Ho is true,

z = ( P-hat - Po ) / square root ( ( Po x ( 1 - Po ) ) / n )
* Only uses sample information ( P-hat and n ) and the hypothesized parameter ( Po )

Why Z instead of t?
e.g
t = ( X-bar(child) - X-bar(adult) ) / square root ( ( S^2(child) / n(child) ) + ( S^2(adult) / n(adult) ) )

- S is calculated from X-bar
* X-bar estimates the population mean
* S estimates the population standard deviation
* Increased uncertainty in out eestimate of the parameter
- t-distribution - fatter tails than a normal distribution
- P-hat only appears in the numerator, so z-scores are fine

Stack Overflow age categories
Ho: Proportion of Stack Overflow users under thirty = 0.5
Ha: Proportion of Stack Overflow users under thirty /= (not equal) 0.5
e.g
alpha = 0.01
stack_overflow['age_cat'].value_counts(normalize=True) <- in
Under 30     0.535604
At least 30  0.464396
Name: age_cat, dtype: float64 <- out

Variables for z
p_hat = (stack_overflow['age_cat'] == 'under 30').mean() <- in
0.5356037151702786 <- out

p_0 = 0.5

n = len(stack_overflow) <- in
2261 <- out

Calculatig the z-score
import numpy as np
numerator =  p_hat - p_0
denominator = np.sqrt(p_0 * ( 1 - p_0) / n)
z_score = numerator / denominator <- in
3.385911440783662 <- out

Calculating the p-value
Left-tailed ('less than'):
from scipy.stats import norm
p_value = norm.cdf(z_score)

Right-tailed ('greater than'):
p_value = 1 - norm.cdf(z_score)

Two-tailed ('not equal'):
p_value = norm.cdf(-z_score) + 1 - norm.cdf(z_score)

p_value = 2 * ( 1 - norm.cdf(z_score) ) <- in
0.0007094227368100725 <- out

p_value <= alpha <- in
True <- out
i.e
Here, the p-value is less than the significance level of 0.01, so we reject the null hypothesis, concluding that the proportion of users under 30 is not equal to 0.5
'''

# Hypothesize that the proportion of late shipments is 6%
p_0 = 0.06

# Calculate the sample proportion of late shipments
p_hat = (late_shipments['late'] == "Yes").mean()

# Calculate the sample size
n = len(late_shipments)

# Calculate the numerator and denominator of the test statistic
numerator = p_hat - p_0
denominator = np.sqrt(p_0 * (1 - p_0) / n)

# Calculate the test statistic
z_score = numerator / denominator

# Calculate the p-value from the z-score
p_value = 1 - norm.cdf(z_score)

# Print the p-value
print(p_value)


'''
Two-sample proportion tests
Comparing 2 proportions
Ho: Proportion of hobbyist users is the same for those under thirty as those at least thirty
Ho: P ( >= 30 ) - P ( < 30 ) = 0

Ha: Proportion of hobbyist users is different for those under thirty to those at least thirty
Ha: P ( >= 30 ) - P ( < 30 ) != (not equal) 0

alpha = 0.05

Calculating the z-score
* Z-score equation for a proportion test:
z = ( ( P-hat (>= 30) - P-hat (< 30) ) - 0 (p_0) ) / SE( P-hat (>= 30) - P-hat (< 30) )

* Standard error equation:
SE( P-hat (>= 30) - P-hat (< 30) ) = square root ( ( ( P-hat x (1 - P-hat) ) / n (>= 30) ) + ( ( P-hat x (1 - P-hat) ) / n (< 30) ) )

* P-hat -> weighted mean of P-hat (>= 30) and P-hat (< 30)
P-hat = ( n (>= 30) x P-hat (>= 30) + n (< 30) x P-hat (< 30) ) / ( n (>= 30) + n (< 30) )

* Only require P-hat (>= 30), P-hat (< 30), n (>= 30), N (< 30) from the sample to calculate the z-score

Getting the numbers for the z-score
p_hats = stack_overflow.groupby('age_cat')['hobbyist'].value_counts(normalize = True) <- in
age_cat      hobbyist
At least 30  Yes          0.773333
             No           0.226667
Under 30     Yes          0.843105
             No           0.156895
Name: hobbyist, dtype: float64 <- out

p_hat_at_least_30 = p_hats[ ('At least 30', 'Yes') ]
p_hat_under_30 = p_hats[ ('Under 30', 'Yes') ]
print( p_hat_at_least_30, p_hat_under_30 ) <- in
0.773333  0.843105 <- out

n = stack_overflow.groupby('age_cat')['hobbyist'].count() <- in
age_cat
At least 30  1050
Under 30     1211
Name: hobbyist, dtype: int64 <- out

n_at_least_30 = n['At least 30']
n_under_30 = n['Under 30']
print( n_at_least_30, n_under_30 ) <- in
1050  1211 <- out

Getting the numbers for the z-score
p_hat = ( n_at least_30 * p_hat_at_least_30 + n_under_30 * p_hat_under_30 ) / ( n_at_least_30 + n_under_30 )

std_error = np.sqrt( p_hat * (1 - p_hat) / n_at_least_30 + p_hat * (1 - p_hat) / n_under_30 )

z_score = ( p_hat_at_least_30 - p_hat_under_30 ) / std_error
print(z_score) <- in
-4.223718652693034 <- out

Proportion tests using proportions_ztest()
stack_overflow.groupby('age_cat')['hobbyist'].value_counts() <- in
age_cat      hobbyist
At least 30  Yes          812
             No           238
Under 30     Yes          1021
             No           190
Name: hobbyist, dtype: int64 <- out

n_hobbyists = np.array( [ 812, 1021] )
n_rows = np.array( [812 + 238, 1021 + 190] )
from statsmodels.stats.proportion import proportions_ztest
z_score, p_value = proportions_ztest( count= n_hobbyists, nobs= n_rows, alternative= 'two-sided' ) <- in
( -4.223691463320559, 2.403330142685068e-05 ) <- out 
i.e
The p-value is smaller than the 0.5 significance level we specified, so we can conclude that there is a difference in the proportion of hobbyists between the two age groups.
'''

# Calculate the pooled estimate of the population proportion
p_hat = (p_hats["reasonable"] * ns["reasonable"] + p_hats["expensive"]
         * ns["expensive"]) / (ns["reasonable"] + ns["expensive"])

# Calculate p_hat one minus p_hat
p_hat_times_not_p_hat = p_hat * (1 - p_hat)

# Divide this by each of the sample sizes and then sum
p_hat_times_not_p_hat_over_ns = p_hat_times_not_p_hat / \
    ns["expensive"] + p_hat_times_not_p_hat / ns["reasonable"]

# Calculate the standard error
std_error = np.sqrt(p_hat_times_not_p_hat_over_ns)

# Calculate the z-score
z_score = (p_hats["expensive"] - p_hats["reasonable"]) / std_error

# Calculate the p-value from the z-score (right test)
p_value = 1 - norm.cdf(z_score)

# Print p_value
print(p_value)


# Count the late column values for each freight_cost_group
late_by_freight_cost_group = late_shipments.groupby("freight_cost_group")[
    'late'].value_counts()

# Create an array of the "Yes" counts for each freight_cost_group
success_counts = np.array([45, 16])

# Create an array of the total number of rows in each freight_cost_group
n = np.array([500 + 45, 439 + 16])

# Run a z-test on the two proportions
stat, p_value = proportions_ztest(
    count=success_counts, nobs=n, alternative='larger')

# Print the results
print(stat, p_value)


'''
Chi-square test of independence
This extends proportion tests to more than 2 groups.

Independence of variables
Previous hypothesis test result: evidence that 'hobbyist' and 'age_cat' are associated

Statistical independence - If the proportion of successes in the response variable is the same across all categories of the explanatory variable, the two variables are statistically independent.

Test for independence of variables 
e.g
import pingouin
expected, observed, stats = pingouin.chi2_independence(data=stack_overflow, x='hobbyist', y='age_cat', correction=False)
print(stats) <- in

                test    lambda      chi2 dof     pval   cramer    power
0            pearson  1.000000 17.839570 1.0 0.000024 0.088826 0.988205
1       cressie-read  0.666667 17.818114 1.0 0.000024 0.088773 0.988126
2     log-likelihood  0.000000 17.802653 1.0 0.000025 0.088734 0.988069
3      freeman-tukey -0.500000 17.815060 1.0 0.000024 0.088765 0.988115
4 mod-log-likelihood -1.000000 17.848099 1.0 0.000024 0.088848 0.988236
5             neyman -2.000000 17.976656 1.0 0.000022 0.089167 0.988694 <- out

NB: The correction argument specifies whether or not to apply Yates's continuity correction, which is a fudge factor for when the sample size is very small and the degrees of freedom is one.

X^2 statistic = 17.839570 = (-4.223691463320559)**2 = ( Z-score )^2

Job satisfaction and age category
stack_overflow['age_cat'].value_counts() <- in
Under 30        1211
At least 30     1050
Name: age_cat, dtype: int64 <- out

stack_overflow['job_sat'].value_counts() <- in
Very satisfied          879
Slightly satisfied      680
Slightly dissatisfied   342
Neither                 201
Very dissatisfied       159
Name: job_sat, dtype: int64 <- out

Declaring the hypotheses
Ho: Age categories are independent of job satisfaction levels
Ha: Age categories are not independent of job satisfaction levels

alpha = 0.1

* Test statistic donoted X^2
* Assuming independence, how far away are the observed results from the expected values?

Exploratory visualization: proportional stacked bar plot
props = stack_overflow.groupby('job_sat')['age_cat'].value_counts(normalize=True)
wide_props = props.unstack()
wide_props.plot(kind='bar', stacked=True)

Chi-square independence test
import pingouin
expected, observed, stats = pingouin.chi2_independence(data=stack_overflow, x='job_sat', y='age_cat')
print(stats) <- in

                test    lambda     chi2 dof     pval   cramer    power
0            pearson  1.000000 5.552373 4.0 0.235164 0.049555 0.437417
1       cressie-read  0.666667 5.554106 4.0 0.235014 0.049563 0.437545
2     log-likelihood  0.000000 5.558529 4.0 0.234632 0.049583 0.437871
3      freeman-tukey -0.500000 5.562688 4.0 0.234274 0.049601 0.438178
4 mod-log-likelihood -1.000000 5.567570 4.0 0.233854 0.049623 0.438538
5             neyman -2.000000 5.579519 4.0 0.232828 0.049676 0.439419 <- out

Degress of freedom:
( No. of response categories - 1 ) x ( No. of explanatory categories - 1 ) 
( 2 - 1 ) * ( 5 - 1 ) = 4

NB: P-value is 0.23, which is above the significance level we set, so we conclude that age categories are independent of job satisfaction

Swapping the variables?
props = stack_overflow.groupby('age_cat')['job_sat'].value_counts(normalize=True)
wide_props = props.unstack()
wide_props.plot(kind='bar', stacked=True)

Chi-square both ways
expected, observed, stats = pingouin.chi2_independence(data=stack_overflow, x='age_cat', y='job_sat')
print(stats[stats['test'] == 'pearson']) <- in

     test   lambda     chi2 dof     pval   cramer    power
0 pearson 1.000000 5.552373 4.0 0.235164 0.049555 0.437417 <- out

Ask: Are the variables X and Y independent?
Not: Is variable X independent from variable Y?
Since the order doesn't matter

What about direction and tails?
* Observed and expected counts squared must be non-negative
* chi-square (X^2)  tests are almost always right-tailed

NB: Left-tailed chi-square tests are used in statistical forensics to detect if a fit is suspiciously good because the data was fabricated. Chi-square tests of variance can be two-tailed. These are niche uses, though.
'''

# Proportion of freight_cost_group grouped by vendor_inco_term
props = late_shipments.groupby('vendor_inco_term')[
    'freight_cost_group'].value_counts(normalize=True)

# Convert props to wide format
wide_props = props.unstack()

# Proportional stacked bar plot of freight_cost_group vs. vendor_inco_term
wide_props.plot(kind="bar", stacked=True)
plt.show()

# Determine if freight_cost_group and vendor_inco_term are independent
expected, observed, stats = pingouin.chi2_independence(
    data=late_shipments, x='freight_cost_group', y='vendor_inco_term')

# Print results
print(stats[stats['test'] == 'pearson'])


'''
Chi-square goodness of fit tests
This is another variant of the chi-square test used to compare a single categorical variable to a hypothesized distribution.

Purple links
How do you feel when you discover that you've already visited the top resource?

purple_link_counts = stack_overflow['purple_link'].value_counts()
purple_link_counts = purple_link_counts.rename_axis( 'purple_link' )\.reset_index( name='n' )\.sort_values( 'purple_link' ) <- in

          purple_link           n
2   Amused                    368
3   Annoyed                   263
0   Hello, old friend        1225
1   Indifferent               405 <- out

Declaring the hypotheses
hypothesized = pd.DataFrame( { 'purple_link': ['Amused', 'Annoyed', 'Hello, old friend', 'Indifferent'], 'prop': [1/6, 1/6, 1/2, 1/6] } ) <- in

    purple_link                 prop
0   Amused                  0.166667
1   Annoyed                 0.166667
2   Hello, old friend       0.500000
3   Indifferent             0.166667 <- out

Ho: The sample matches the hypothesized distribution
Ha: The sample does not match the hypothesized distribution

X^2 measures how far observed results are from expectations in each group

alpha = 0.01

Hypothesized counts by category
n_total = len(stack_overflow)
hypothesized['n'] = hypothesized['prop'] * n_total <- in 
    purple_link              prop            n
0   Amused               0.166667   376.833333
1   Annoyed              0.166667   376.833333
2   Hello, old friend    0.500000  1130.500000  
3   Indifferent          0.166667   376.833333 <- out

Visualizing counts
import matplotlib.pyplot as plt

plt.bar( purple_link_counts['purple_link'], purple_link_counts['n'], color='red', label='Observed' )
plt.bar( hypothesized['purple_link'], hypothesized['n'], alpha=0.5, color='blue', label='hypothesized' )

plt.legend()
plt.show()

chi-square goodness of fit test
NB: The one-sample chi-square test is called a Goodness of Fit Test, as we're testing how well our hypothesized data fits the observed data.

from scipy.stats import chisquare
chisquare( f_obs=purple_link_counts['n'], f_exp=hypothesized['n'] ) <- in
Power_divergenceResult( statistic= 44.59840778416629, pvalue=1.1261810719413759e-09 ) <- out

Therefore, since the p-value returned by the function is much lower than the significance level of 0.01, so we conclude that the sample distribution of proportions is different from the hypothesized distribution.
'''

# Find the number of rows in late_shipments
n_total = len(late_shipments)

# Create n column that is prop column * n_total
hypothesized["n"] = hypothesized["prop"] * n_total

# Plot a red bar graph of n vs. vendor_inco_term for incoterm_counts
plt.bar(incoterm_counts['vendor_inco_term'],
        incoterm_counts['n'], color="red", label="Observed")

# Add a blue bar plot for the hypothesized counts
plt.bar(hypothesized['vendor_inco_term'], hypothesized['n'],
        alpha=0.5, color='blue',  label="Hypothesized")
plt.legend()
plt.show()

# Perform a goodness of fit test on the incoterm counts n
gof_test = chisquare(f_obs=incoterm_counts['n'], f_exp=hypothesized['n'])

# Print gof_test results
print(gof_test)


''' Non-Parametric Tests '''

'''
Assumptions in hypothesis testing
Randomness
Assumption
- The samples are random subsets of larger populations
Consequence
- Sample is not representative of population
How to check this
- Understand how your data was collected
- Speak to the data collector / domain expert


Independence of observations
Assumption
- Each observation (row) in the dataset is independent
Consequence
- Increased chance of false negative / positive error
How to check this
- Understand how our data was collected


Large sample size
Assumption
- The sample is big enough to mitigate uncertainty, so that the Central Limit Theorem applies
Consequence
- Wider confidence intervals
- Increased chance of false negative / positive errors
How to check this
- It depends on the test


Large sample size: t-test
One sample
- At least 30 observations in the sample
n >= 30
n: sample size

Two samples
- At least 30 observations in each sample
n(1) >= 30, n2 >= 30
n(i): sample size for group i

ANOVA
- At least 30 observations in each sample
n(i) >= 30 for all values of i

Paired samples
- At least 30 pairs observations across the samples
Number of rows in our data >= 30


Large sample size: proportion tests
One sample
- Number of successes in sample is greater than or equal to 10
n x P-hat >= 10

- Number of failures in sample is greater than or equal to 10
n x ( 1 - P-hat ) >= 10
n: sample size
P-hat: proportion of successes in sample

Two samples
- Number of successes in each sample is greater than or equal to 10
n(1) x P-hat(1) >= 10
n(2) x P-hat(2) >= 10

- Number of failures in each sample is greater than or equal to 10
n(1) x ( 1 - P-hat(1) ) >= 10
n(2) x ( 1 - P-hat(2) ) >= 10


Large sample size: chi-square tests
- The number of successes in each group is greater than or equal to 5
n(i) x P-hat(i) >= 5 for all values of i

- The number of failures in each group is greater than or equal to 5
n(i) x ( 1 - P-hat(i) ) >= 5 for all values of i
n(i): sample size for group i
P-hat(i): proportion of successes in sample group i


Sanity check
If the bootstrap distribution doesn't look normal, assumptions likely aren't valid
* Revisit data collection to check for randomness, independence, and sample size
'''

# Count the freight_cost_group values
counts = late_shipments['freight_cost_group'].value_counts()

# Print the result
print(counts)

# Inspect whether the counts are big enough ( for a two sample t-test )
print((counts >= 30).all())


# Count the late values
counts = late_shipments['late'].value_counts()

# Print the result
print(counts)

# Inspect whether the counts are big enough ( for a one sample proportion test )
print((counts >= 10).all())


# Count the values of freight_cost_group grouped by vendor_inco_term
counts = late_shipments.groupby('vendor_inco_term')[
    'freight_cost_group'].value_counts()

# Print the result
print(counts)

# Inspect whether the counts are big enough ( for a chi-square independence test )
print((counts >= 5).all())


# Count the shipment_mode values
counts = late_shipments['shipment_mode'].value_counts()

# Print the result
print(counts)

# Inspect whether the counts are big enough ( for an ANOVA test )
print((counts >= 30).all())


'''
Non-parametric tests
Parametric tests
- z-test, t-test and ANOVA are all parametric tests
- Assume a normal distribution
- Require sufficiently large sample sizes
- The Central Limit Theorem applies

Smaller Republican votes data
e.g
print(repub_votes_small) <- in
            state     county repub_prercent_08 repub_prercent_12
80          Texas  Red River         68.507522         69.944817
84          Texas     Walker         60.707197         64.971903
33       Kentucky     Powell         57.059533         61.727293
81          Texas Schleicher         74.386503         77.384464
93  West Virginia     Morgan         60.857614         64.068711 <- out

Results with pingouin.ttest()
- 5 pairs is not enough to meet the sample size condition for the paired t-test:
* At least 30 pairs of observations across the samples.
e.g
alpha = 0.01
import pingouin
pingouin.ttest(x=repub_votes_small['repub_prercent_08'], y=repub_votes_small['repub_prercent_12'], paired=True, alternative='less') <- in
                T dof alternative    p-val         CI95%  cohen-d   BF10    power
T-test  -5.875753   4        less 0.002096 [-inf, -2.11] 0.500068 26.468 0.239034 <- out

Non-parametric tests
- Non-parametric tests avoid the parametric assumptions and conditions

- Many no-parametric tests use ranks of the data
e.g
x = [ 1, 15, 3, 10, 6 ]
from scipy.stats import rankdata
rankdata(x) <- in
array([1., 5., 2., 4., 3.]) <- out

- Non-parametric tests are more reliable than parametric tests for small sample sizes and when data isn't normally distributed

Wilcoxon-signed rank test
- Developed by Frank Wilcoxon in 1945
- One of the first non-parametric procedures

Wilcoxon-signed rank test (step 1)
- Works on the ranked absolute differences between the pairs of data
e.g
repub_votes_small['diff'] = repub_votes_small['repub_prercent_08'] - repub_votes_small['repub_prercent_12']
print(repub_votes_small) <- in
            state     county repub_prercent_08 repub_prercent_12      diff 
80          Texas  Red River         68.507522         69.944817 -1.437295 
84          Texas     Walker         60.707197         64.971903 -4.264705 
33       Kentucky     Powell         57.059533         61.727293 -4.667760 
81          Texas Schleicher         74.386503         77.384464 -2.997961 
93  West Virginia     Morgan         60.857614         64.068711 -3.211097 <- out

Wilcoxon-signed rank test (step 2)
- Works on the ranked absolute differences between the pairs of data
e.g
repub_votes_small['abs_diff'] = repub_votes_small['diff'].abs()
print(repub_votes_small) <- in
            state     county repub_prercent_08 repub_prercent_12      diff  abs_diff 
80          Texas  Red River         68.507522         69.944817 -1.437295  1.437295 
84          Texas     Walker         60.707197         64.971903 -4.264705  4.264705 
33       Kentucky     Powell         57.059533         61.727293 -4.667760  4.667760 
81          Texas Schleicher         74.386503         77.384464 -2.997961  2.997961 
93  West Virginia     Morgan         60.857614         64.068711 -3.211097  3.211097 <- out

Wilcoxon-signed rank test (step 3)
- Works on the ranked absolute differences between the pairs of data
e.g
from scipy.stats import rankdata
repub_votes_small['rank_abs_diff'] = rankdata( repub_votes_small['abs_diff'] )
print(repub_votes_small) <- in
            state     county repub_prercent_08 repub_prercent_12      diff  abs_diff   rank_abs_diff
80          Texas  Red River         68.507522         69.944817 -1.437295  1.437295             1.0
84          Texas     Walker         60.707197         64.971903 -4.264705  4.264705             4.0
33       Kentucky     Powell         57.059533         61.727293 -4.667760  4.667760             5.0
81          Texas Schleicher         74.386503         77.384464 -2.997961  2.997961             2.0
93  West Virginia     Morgan         60.857614         64.068711 -3.211097  3.211097             3.0 <- out

Wilcoxon-signed rank test (step 4)
- Involves calculating a test statistic called W
- Incorporate the sum of the ranks for negative and positive differences ( i.e ['diff'] column )
e.g
T_minus = 1 + 4 + 5 + 2 + 3
T_plus = 0
W = np.min( [T_minus, T_plus] ) <- in
0 <- out

Implementation with pingouin.wilcoxon()
e.g
alpha = 0.01
pingouin.wilcoxon(x=repub_votes_small['repub_prercent_08'], y=repub_votes_small['repub_prercent_12'], alternative='less') <- in
            W-val alternative   p-val  RBC CLES
Wilcoxon      0.0        less 0.03125 -1.0 0.72 <- out

i.e The p-value of around 0.3 percent, which is over ten times larger than the p-value from the t-test, so we should feel more confident with this result given the small sample size.
- Fail to reject Ho, since 0.03125 > 0.01
'''

# Conduct a paired t-test on dem_percent_12 and dem_percent_16
paired_test_results = pingouin.ttest(
    x=sample_dem_data['dem_percent_12'], y=sample_dem_data['dem_percent_16'], paired=True, alternative='two-sided')

# Print paired t-test results
print(paired_test_results)


# Conduct a Wilcoxon test on dem_percent_12 and dem_percent_16
wilcoxon_test_results = pingouin.wilcoxon(
    x=sample_dem_data['dem_percent_12'], y=sample_dem_data['dem_percent_16'], alternative='two-sided')

# Print Wilcoxon test results
print(wilcoxon_test_results)


'''
Non-parametric ANOVA and unpaired t-tests
Wilcoxon-Mann-Whitney test
- ALso known as the Mann Whitney U test
- A t-test on the ranks of the numeric input
- Works on unpaired data

Wilcoxon-Mann-Whitney test setup
e.g
age_vs_comp = stack_overflow[ ['converted_comp', 'age_first_code_cut'] ]
age_vs_comp_wide = age_vs_comp.pivot(columns='age_first_code_cut', values='converted_comp')

Wilcoxon-Mann-Whitney test
alpha = 0.01
import pingouin
pingouin.mwu( x=age_vs_comp_wide['child'], y=age_vs_comp_wide['adult'], alternative='greater' ) <- in
        U-val alternative        p-val       RBC     CLES
MWU  744365.5     greater 1.902723e-19 -0.222516 0.611258 <- out

Kruskal-Wallis test
Kruskal-Wallis test is to Wilcoxon-Mann-Whitney test as ANOVA is to t-test
i.e its used to extend test for more than 2 pair groups
e.g
alpha = 0.01
pingouin.kruskal( data=stack_overflow, dv='converted_comp', between='job_sat' ) <- in
            source ddof1         H        p-unc
Kruskal    job_sat     4 72.814939 5.772915e-15 <- out

i.e
Since the P-value is smaller than the significance level, this provides evidence that at least one of the mean compensation totals is different than the others across these 5 job satisfaction groups.
'''

# Select the weight_kilograms and late columns
weight_vs_late = late_shipments[['weight_kilograms', 'late']]

# Convert weight_vs_late into wide format
weight_vs_late_wide = weight_vs_late.pivot(
    columns='late', values='weight_kilograms')

# Run a two-sided Wilcoxon-Mann-Whitney test on weight_kilograms vs. late
wmw_test = pingouin.mwu(
    x=weight_vs_late_wide['No'], y=weight_vs_late_wide['Yes'], alternative='two-sided')

# Print the test results
print(wmw_test)


# Run a Kruskal-Wallis test on weight_kilograms vs. shipment_mode
kw_test = pingouin.kruskal(
    data=late_shipments, dv='weight_kilograms', between='shipment_mode')

# Print the results
print(kw_test)
