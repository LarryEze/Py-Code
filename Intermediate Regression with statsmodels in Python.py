# import necessary packages
from statsmodels.formula.api import ols, logit
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from itertools import product
from scipy.optimize import minimize
from scipy.stats import logistic


''' Parallel Slopes '''

'''
Parallel slopes linear regression
From simple regression to multiple regression
- Multiple regression is a regression model with more than one explanatory variable.
- More explanatory variables can give more insight and better predictions.

The fish dataset
fish -> in

mass_g length_cm  species
242.0       23.2    Bream
5.9          7.5    Perch
200.0       30.0     Pike
40.0        12.9    Roach -> out

* Each row represents a fish
* mass_g is the response variable
* 1 numeric, 1 categorical explanatory variable.

One explanatory variable at a time
from statsmodels.formula.api import ols

mdl_mass_vs_length = ols('mass_g ~ length_cm', data = fish).fit()
print(mdl_mass_vs_length.params) -> in

Intercept       -536.223947
length_cm         34.899245
dtype: float64 -> out

* 1 intercept coefficient
* 1 slope coefficient

mdl_mass_vs_species = ols('mass_g ~ species + 0', data = fish).fit()
print(mdl_mass_vs_species.params) -> in

species[Bream]      617.828571
species[Perch]      382.239286
species[Pike]       718.705882
species[Roach]      152.050000
dtype: float64 -> out

* 1 intercept coefficient for each category

Both variables at the same time
mdl_mass_vs_both = ols('mass_g ~ length_cm + species + 0', data=fish).fit()
print(mdl_mass_vs_both.params) -> in

species[Bream]      -672.241866
species[Perch]      -713.292859
species[Pike]      -1089.456053
species[Roach]      -726.777799
length_cm 42.568554
dtype: float64 -> out

* 1 slope coefficient
* 1 intercept coefficient for each category

Visualization: 1 numeric explanatory variable
import matplotlib.pyplot as plt
import seaborn as sns

sns.regplot(x='length_cm', y='mass_g', data=fish, ci=None)
plt.show()

Visualization: 1 categorical explanatory variable
sns.boxplot(x='species', y='mass_g', data=fish, showmeans=True)
plt.show()

Visualization: both explanatory variables
coeffs = mdl_mass_vs_both.params
print(coeffs) -> in

species[Bream]      -672.241866
species[Perch]      -713.292859
species[Pike]      -1089.456053
species[Roach]      -726.777799
length_cm             42.568554 -> out

ic_bream, ic_perch, ic_pike, ic_roach, sl = coeffs

sns.scatterplot(x='length_cm', y='mass_g', hue='species', data=fish)

plt.axline(xy1=(0, ic_bream), slope=sl, color='blue')
plt.axline(xy1=(0, ic_perch), slope=sl, color='green')
plt.axline(xy1=(0, ic_pike), slope=sl, color='red')
plt.axline(xy1=(0, ic_roach), slope=sl, color='orange')

plt.show()
'''

taiwan_real_estate = pd.read_csv(
    'Intermediate Regression with statsmodels in Python/taiwan_real_estate2.csv')

# Import ols from statsmodels.formula.api

# Fit a linear regression of price_twd_msq vs. n_convenience
mdl_price_vs_conv = ols("price_twd_msq ~ n_convenience",
                        data=taiwan_real_estate).fit()

# Fit a linear regression of price_twd_msq vs. house_age_years, no intercept
mdl_price_vs_age = ols("price_twd_msq ~ house_age_years + 0",
                       data=taiwan_real_estate).fit()

# Fit a linear regression of price_twd_msq vs. n_convenience plus house_age_years, no intercept
mdl_price_vs_both = ols(
    "price_twd_msq ~ n_convenience + house_age_years + 0", data=taiwan_real_estate).fit()

# Print the coefficients
print(mdl_price_vs_both.params)


# Import matplotlib.pyplot and seaborn

# Create a scatter plot with linear trend line of price_twd_msq vs. n_convenience
sns.regplot(x='n_convenience', y='price_twd_msq',
            data=taiwan_real_estate, ci=None)

# Show the plot
plt.show()


# Import matplotlib.pyplot and seaborn

# Create a boxplot of price_twd_msq vs. house_age_years
sns.boxplot(x='house_age_years', y='price_twd_msq', data=taiwan_real_estate)

# Show the plot
plt.show()


# Extract the model coefficients, coeffs
coeffs = mdl_price_vs_both.params

# Assign each of the coeffs
ic_0_15, ic_15_30, ic_30_45, slope = coeffs

# Draw a scatter plot of price_twd_msq vs. n_convenience, colored by house_age_years
sns.scatterplot(x="n_convenience", y="price_twd_msq",
                hue="house_age_years", data=taiwan_real_estate)

# Add three parallel lines for each category of house_age_years
# Color the line for ic_0_15 blue
plt.axline(xy1=(0, ic_0_15), slope=slope, color="blue")
# Color the line for ic_15_30 orange
plt.axline(xy1=(0, ic_15_30), slope=slope, color="orange")
# Color the line for ic_30_45 green
plt.axline(xy1=(0, ic_30_45), slope=slope, color="green")

# Show the plot
plt.show()


'''
Predicting parallel slopes
The prediction workflow
import pandas as pd
import numpy as np

expl_data_length = pd.DataFrame({'length_cm': np.arange(5, 61, 5)})
print(expl_data_length) -> in

    length_cm
0           5
1          10
2          15
3          20
4          25
5          30
6          35
7          40
8          45
9          50
10         55
11         60 -> out

[A. B, C] x [1, 2] ==> [A1, B1, C1, A2, B2, C2]

from itertools import product
product(['A', 'B'. 'C'], [1, 2])

length_cm = np.arange(5, 61, 5)
species = fish['species'].unique()

p = product(length_cm, species)

expl_data_both = pd.DataFrame(p, columns=['length_cm', 'species']) -> in

    length_cm   species
0           5     Bream
1           5     Roach
2           5     Perch
3           5      Pike
4          10     Bream
5          10     Roach
6          10     Perch
...
41         55     Roach
42         55     Perch
43         55      Pike
44         60     Bream
45         60     Roach
46         60     Perch
47         60      Pike -> out

Predict mass_g from length_cm only
prediction_data_length = expl_data_length.assign(mass_g = mdl_mass_vs_length.predict(expl_data)) -> in

    length_cm      mass_g
0           5   -361.7277
1          10   -187.2315
2          15    -12.7353
3          20    161.7610 
4          25    336.2572
5          30    510.7534
... # number of rows: 12 -> out

Predict mass_g from both explanatory variables
prediction_data_both = expl_data_both.assign(mass_g = mdl_mass_vs_both.predict(expl_data)) -> in

    length_cm   species     mass_g
0           5     Bream  -459.3991
1           5     Roach  -513.9350
2           5     Perch  -500.4501
3           5      Pike  -876.6133
4          10     Bream  -246.5563
5          10     Roach  -301.0923
... # number of rows: 48 -> out

Visualizing the predictions
plt.axline(xy1=(0, ic_bream), slope=sl, color='blue')
plt.axline(xy1=(0, ic_perch), slope=sl, color='green')
plt.axline(xy1=(0, ic_pike), slope=sl, color='red')
plt.axline(xy1=(0, ic_roach), slope=sl, color='orange')

sns.scatterplot(x='length_cm', y='mass_g', hue='species', data=fish)

sns.scatterplot(x='length_cm', y='mass_g', colour='black', data=prediction_data)

plt.show()

Manually calculating predictions for linear regression
coeffs = mdl_mass_vs_length.params
print(coeffs) -> in

Intercept     -536.223947
length_cm       34.899245 -> out

intercept, slope = coeffs

explanatory_data = pd.DataFrame({'length_cm': np.arange(5, 61, 5)})

prediction_data =  explanatory_data.assign(mass_g = intercept + slope * explanatory_data ) 

print(prediction_data) -> in

    length_cm        mass_g
0           5   -361.727721
1          10   -187.231494
2          15    -12.735268
3          20    161.760959
4          25    336.257185
5          30    510.753412
...
9          50   1208.738318
10         55   1383.234545
11         60   1557.730771 -> out

Manually calculating predictions for multiple regression
coeffs = mdl_mass_vs_both.params
print(coeffs) -> in

species[Bream]      -672.241866
species[Perch]      -713.292859
species[Pike]      -1089.456053
species[Roach]      -726.777799
length_cm             42.568554 -> out

ic_bream, ic_perch, ic_pike, ic_roach, slope = coeffs

np.select()
conditions = [condition_1, condition_2, # ..., condition_n]

choices = [list_of_choices] # same length as conditions

np.select(conditions, choices) 

Choosing an intercept with np.select()
conditions = [explanatory_data['species'] == 'Bream', explanatory_data['species'] == 'Perch', explanatory_data['species'] == 'Pike', explanatory_data['species'] == 'Roach']

choices = [ic_bream, ic_perch, ic_pike, ic_roach]

intercept = np.select(conditions, choices)

print(intercept) -> in

[ -672.24, -726.78, -713.29, -1089.46
  -672.24, -726.78, -713.29, -1089.46 
  -672.24, -726.78, -713.29, -1089.46
  -672.24, -726.78, -713.29, -1089.46
  -672.24, -726.78, -713.29, -1089.46
  -672.24, -726.78, -713.29, -1089.46
  -672.24, -726.78, -713.29, -1089.46
  -672.24, -726.78, -713.29, -1089.46
  -672.24, -726.78, -713.29, -1089.46
  -672.24, -726.78, -713.29, -1089.46
  -672.24, -726.78, -713.29, -1089.46
  -672.24, -726.78, -713.29, -1089.46 ] -> out

The final prediction step
prediction_data = explanatory_data.assign(intercept = np.select(conditions, choices), mass_g = intercept + slope * explanatory_data['length_cm'])

print(prediction_data) -> out

    length_cm   species     intercept      mass_g
0           5     Bream     -672.2419   -459.3991
1           5     Roach     -726.7778   -513.9350
2           5     Perch     -713.2929   -500.4501
3           5      Pike    -1089.4561   -876.6133
4          10     Bream     -672.2419   -246.5563
5          10     Roach     -726.7778   -301.0923
6          10     Perch     -713.2929   -287.6073
7          10      Pike    -1089.4561   -663.7705
8          15     Bream     -672.2419    -33.7136
...
40         55     Bream     -672.2419   1669.0286
41         55     Roach     -726.7778   1614.4927
42         55     Perch     -713.2929   1627.9776
43         55      Pike    -1089.4561   1251.8144
44         60     Bream     -672.2419   1881.8714
45         60     Roach     -726.7778   1827.3354
46         60     Perch     -713.2929   1840.8204 
47         60      Pike    -1089.4561   1464.6572 -> out

Compare to .predict()
mdl_mass_vs_both.predict(explanatory_data) -> in

0   -459.3991
1   -513.9350
2   -500.4501
3   -876.6133
4   -246.5563
5   -301.0923
...
43  1251.8144
44  1881.8714
45  1827.3354
46  1840.8204 
47  1464.6572 -> out
'''

# Create n_convenience as a range of numbers from 0 to 10
n_convenience = np.arange(0, 11)

# Extract the unique values of house_age_years
house_age_years = taiwan_real_estate["house_age_years"].unique()

# Create p as all combinations of values of n_convenience and house_age_years
p = product(n_convenience, house_age_years)

# Transform p to a DataFrame and name the columns
explanatory_data = pd.DataFrame(
    p, columns=['n_convenience', 'house_age_years'])

# Add predictions to the DataFrame
prediction_data = explanatory_data.assign(
    price_twd_msq=mdl_price_vs_both.predict(explanatory_data))

print(prediction_data)


# Extract the model coefficients, coeffs
coeffs = mdl_price_vs_both.params

# Assign each of the coeffs
ic_0_15, ic_15_30, ic_30_45, slope = coeffs

# Create the parallel slopes plot
plt.axline(xy1=(0, ic_0_15), slope=slope, color="green")
plt.axline(xy1=(0, ic_15_30), slope=slope, color="orange")
plt.axline(xy1=(0, ic_30_45), slope=slope, color="blue")
sns.scatterplot(x="n_convenience", y="price_twd_msq",
                hue="house_age_years", data=taiwan_real_estate)

# Add the predictions in black
sns.scatterplot(x="n_convenience", y="price_twd_msq",
                color="black", data=prediction_data)

plt.show()


# Define conditions
conditions = [explanatory_data['house_age_years'] == "0 to 15", explanatory_data['house_age_years']
              == "15 to 30", explanatory_data['house_age_years'] == "30 to 45"]

# Define choices
choices = [ic_0_15, ic_15_30, ic_30_45]

# Create array of intercepts for each house_age_year category
intercept = np.select(conditions, choices)

# Create prediction_data with columns intercept and price_twd_msq
prediction_data = explanatory_data.assign(intercept=np.select(
    conditions, choices), price_twd_msq=intercept + slope * explanatory_data['n_convenience'])

print(prediction_data)


'''
Assessing model performance
Model performance metrics
- Coefficient of determination (R-squared): how well the linear regression line fits the observed values.
* Larger is better
- Residual standard error (RSE): the typical size of the residuals.
* Smaller is better

Getting the coefficient of determination
print(mdl_mass_vs_length.rsquared) -> in

0.8225689502644215 -> out

print(mdl_mass_vs_species.rsquared) -> in

0.25814887709499157 -> out

print(mdl_mass_vs_both.rsquared) -> in

0.9200433561156649 -> out

Adjusted coefficient of determination
- More explanatory variables increases Coefficient of determination (R^2).
- Too many explanatory variables causes overfitting.
- Adjusted coefficient of determination penalizes more explanatory variables.
- Adjusted coefficient of determination = 1 - ( 1 - R^2) * ( n(obs) - 1 / n(obs) - n(var) - 1 )

* R^2 : Coefficient of determination
* n(obs) : Number of observations
* n(var) : Number of explanatory variables

- Penalty is noticeable when R^2 is small, or n(var) is large fraction of n(obs).
- In statsmodels, it's contained in the rsquared_adj attribute.

Getting the adjusted coefficient of determination
print('rsq_length: ', mdl_mass_vs_length.rsquared)
print('rsq_adj_length: ', mdl_mass_vs_length.rsquared_adj) -> in

rsq_length:         0.8225689502644215
rsq_adj_length:     0.8211607673300121 -> out

print('rsq_species: ', mdl_mass_vs_species.rsquared)
print('rsq_adj_species: ', mdl_mass_vs_species.rsquared_adj) -> in

rsq_species:        0.25814887709499157
rsq_adj_species:    0.24020086605696722 -> out

print('rsq_both: ', mdl_mass_vs_both.rsquared)
print('rsq_adj_both: ', mdl_mass_vs_both.rsquared_adj) -> in

rsq_both:           0.9200433561156649
rsq_adj_both:       0.9174431400543857 -> out

Getting the residual standard error
rse_length = np.sqrt(mdl_mass_vs_length.mse_resid)
print('rse_length: ', rse_length) -> in

rse_length: 152.12092835414788 -> out

* mse : mean squared error
* rse : np.sqrt(mse_resid)

rse_species = np.sqrt(mdl_mass_vs_species.mse_resid)
print('rse_species: ', rse_species) -> in

rse_species: 313.5501156682592 -> out

rse_both = np.sqrt(mdl_mass_vs_both.mse_resid)
print('rse_both: ', rse_both) -> in

rse_both: 103.35563303966488 -> out
'''

# Print the coeffs of determination for mdl_price_vs_conv
print("rsquared_conv: ", mdl_price_vs_conv.rsquared)
print("rsquared_adj_conv: ", mdl_price_vs_conv.rsquared_adj)

# Print the coeffs of determination for mdl_price_vs_age
print("rsquared_age: ", mdl_price_vs_age.rsquared)
print("rsquared_adj_age: ", mdl_price_vs_age.rsquared_adj)

# Print the coeffs of determination for mdl_price_vs_both
print("rsquared_both: ", mdl_price_vs_both.rsquared)
print("rsquared_adj_both: ", mdl_price_vs_both.rsquared_adj)


# Print the RSE for mdl_price_vs_conv
print("rse_conv: ", np.sqrt(mdl_price_vs_conv.mse_resid))

# Print the RSE for mdl_price_vs_age
print("rse_age: ", np.sqrt(mdl_price_vs_age.mse_resid))

# Print RSE for mdl_price_vs_both
print("rse_both: ", np.sqrt(mdl_price_vs_both.mse_resid))


'''
Interactions

Models for each category
Four categories
print(fish['species'].unique()) -> in

array(['Bream', 'Roach', 'Perch', 'Pike'], dtype=object) -> out

Splitting the dataset
bream = fish[fish['species'] == 'Bream']
perch = fish[fish['species'] == 'Perch']
pike = fish[fish['species'] == 'Pike']
roach = fish[fish['species'] == 'Roach']

Four models
mdl_bream = ols('mass_g ~ length_cm', data = bream).fit()
print(mdl_bream.params) -> in

Intercept       -1035.3476
length_cm          54.5500 -> out

mdl_perch = ols('mass_g ~ length_cm', data = perch).fit()
print(mdl_perch.params) -> in

Intercept       -619.1751
length_cm         38.9115 -> out

mdl_pike = ols('mass_g ~ length_cm', data = pike).fit()
print(mdl_pike.params) -> in

Intercept       -1540.8243
length_cm          53.1949 -> out

mdl_roach = ols('mass_g ~ length_cm', data = roach).fit()
print(mdl_roach.params) -> in

Intercept       -329.3762
length_cm         23.3193 -> out

Explanatory data
explanatory_data = pd.DataFrame({'length_cm': np.arange(5, 61, 5)})
print(explanatory_data) -> in

    length_cm
0           5
1          10
2          15
3          20
4          25
5          30
6          35
7          40
8          45
9          50
10         55
11         60 -> out

Making predictions
prediction_data_bream = explanatory_data.assign(mass_g = mdl_bream.predict(explanatory_data), species = 'Bream')

prediction_data_perch = explanatory_data.assign(mass_g = mdl_perch.predict(explanatory_data), species = 'Perch')

prediction_data_pike = explanatory_data.assign(mass_g = mdl_pike.predict(explanatory_data), species = 'Pike')

prediction_data_roach = explanatory_data.assign(mass_g = mdl_roach.predict(explanatory_data), species = 'Roach')

Concatenating predictions
prediction_data = pd.concat([prediction_data_bream, prediction_data_roach, prediction_data_perch, prediction_data_pike]) -> in

    length_cm        mass_g   species
0           5   -762.597660     Bream
1          10   -489.847756     Bream
2          15   -217.097851     Bream
3          20     55.652054     Bream
4          25    328.401958     Bream
5          30    601.151863     Bream
...
3          20   -476.926955      Pike
4          25   -210.952626      Pike
5          30     55.021703      Pike
6          35    320.996032      Pike
7          40    586.970362      Pike
8          45    852.944691      Pike
9          50   1118.919020      Pike
10         55   1384.893349      Pike
11         60   1650.867679      Pike -> out

Visualizing predictions
sns.lmplot(x='length_cm', y='mass_g', data=fish, hue='species', ci=None)

sns.scatterplot(x='length_cm', y='mass_g', data=prediction_data, hue='species', ci=None, legend=False)

plt.show()

Coefficient of determination
mdl_fish = ols('mass_g ~ length_cm + species', data=fish).fit()
print(mdl_fish.rsquared_adj) -> in

0.917 -> out

print(mdl_bream.rsquared_adj) -> in

0.874 -> out

print(mdl_perch.rsquared_adj) -> in

0.917 -> out

print(mdl_pike.rsquared_adj) -> in

0.941 -> out

print(mdl_roach.rsquared_adj) -> in

0.815 -> out

Residual standard error
print(np.sqrt(mdl_fish.mse_resid)) -> in

103 -> out

print(np.sqrt(mdl_bream.mse_resid)) -> in

74.2 -> out

print(np.sqrt(mdl_perch.mse_resid)) -> in

100 -> out

print(np.sqrt(mdl_pike.mse_resid)) -> in

120 -> out

print(np.sqrt(mdl_roach.mse_resid)) -> in

38.2 -> out
'''

# Filter for rows where house age is 0 to 15 years
taiwan_0_to_15 = taiwan_real_estate[taiwan_real_estate["house_age_years"] == "0 to 15"]

# Filter for rows where house age is 15 to 30 years
taiwan_15_to_30 = taiwan_real_estate[taiwan_real_estate["house_age_years"] == "15 to 30"]

# Filter for rows where house age is 30 to 45 years
taiwan_30_to_45 = taiwan_real_estate[taiwan_real_estate["house_age_years"] == "30 to 45"]

# Model price vs. no. convenience stores using 0 to 15 data
mdl_0_to_15 = ols('price_twd_msq ~ n_convenience', data=taiwan_0_to_15).fit()

# Model price vs. no. convenience stores using 15 to 30 data
mdl_15_to_30 = ols('price_twd_msq ~ n_convenience', data=taiwan_15_to_30).fit()

# Model price vs. no. convenience stores using 30 to 45 data
mdl_30_to_45 = ols('price_twd_msq ~ n_convenience', data=taiwan_30_to_45).fit()

# Print the coefficients
print(mdl_0_to_15.params)
print(mdl_15_to_30.params)
print(mdl_30_to_45.params)


# Create explanatory_data, setting no. of conv stores from  0 to 10
explanatory_data = pd.DataFrame({'n_convenience': np.arange(0, 11)})

# Add column of predictions using "0 to 15" model and explanatory data
prediction_data_0_to_15 = explanatory_data.assign(
    price_twd_msq=mdl_0_to_15.predict(explanatory_data))

# Same again, with "15 to 30"
prediction_data_15_to_30 = explanatory_data.assign(
    price_twd_msq=mdl_15_to_30.predict(explanatory_data))

# Same again, with "30 to 45"
prediction_data_30_to_45 = explanatory_data.assign(
    price_twd_msq=mdl_30_to_45.predict(explanatory_data))

print(prediction_data_0_to_15)
print(prediction_data_15_to_30)
print(prediction_data_30_to_45)


# Plot the trend lines of price_twd_msq vs. n_convenience for each house age category
sns.lmplot(x="n_convenience", y="price_twd_msq", data=taiwan_real_estate,
           hue="house_age_years", ci=None, legend_out=False)

# Add a scatter plot for prediction_data
sns.scatterplot(x="n_convenience", y="price_twd_msq",
                data=prediction_data, hue="house_age_years", legend=False)

plt.show()


mdl_all_ages = ols('price_twd_msq ~ n_convenience',
                   data=taiwan_real_estate).fit()

# Print the coeff. of determination for mdl_all_ages
print("R-squared for mdl_all_ages: ", mdl_all_ages.rsquared)

# Print the coeff. of determination for mdl_0_to_15
print("R-squared for mdl_0_to_15: ", mdl_0_to_15.rsquared)

# Print the coeff. of determination for mdl_15_to_30
print("R-squared for mdl_15_to_30: ", mdl_15_to_30.rsquared)

# Print the coeff. of determination for mdl_30_to_45
print("R-squared for mdl_30_to_45: ", mdl_30_to_45.rsquared)


# Print the RSE for mdl_all_ages
print("RSE for mdl_all_ages: ", np.sqrt(mdl_all_ages.mse_resid))

# Print the RSE for mdl_0_to_15
print("RSE for mdl_0_to_15: ", np.sqrt(mdl_0_to_15.mse_resid))

# Print the RSE for mdl_15_to_30
print("RSE for mdl_15_to_30: ", np.sqrt(mdl_15_to_30.mse_resid))

# Print the RSE for mdl_30_to_45
print("RSE for mdl_30_to_45: ", np.sqrt(mdl_30_to_45.mse_resid))


'''
One model with an interaction
What is an interaction?
In the fish dataset
* Different fish species have different mass to length ratios
* The effect of length on the expected mass is different for different species.

More generally
The effect of one explanatory variable on the expected response changes depending on the value of another explanatory variable.

Specifying interactions
No interactions
response ~ explntry1 + explntry2

With interactions (implicit)
response_var ~ explntry1 * explntry2

With interactions (explicit) 
response ~ explntry1 + explntry2 + explntry1:explntry2

No interactions
mass_g ~ length_cm + species

With interactions (implicit)
mass_g ~ length_cm * species

With interactions (explicit) 
mass_g ~ length_cm + species + length_cm:species

Running the model
mdl_mass_vs_both = ols('mass_g ~ length_cm * species', data=fish).fit()

print(mdl_mass_vs_both.params) -> in

Intercept                       -1035.3476
species[T.Perch]                  416.1725
species[T.Pike]                  -505.4767
species[T.Roach]                  705.9714
length_cm                          54.5500
length_cm:species[T.Perch]        -15.6385
length_cm:species[T.Pike]          -1.3551
length_cm:species[T.Roach]        -31.2307 -> out

Easier to understand coefficients
mdl_mass_vs_both_inter = ols('mass_g ~ species + species:length_cm + 0', data=fish).fit()

print(mdl_mass_vs_both_inter.params) -> in

species[Bream]              -1035.3476
species[Perch]               -619.1751
species[Pike]               -1540.8243
species[Roach]               -329.3762
species[Bream]:length_cm       54.5500
species[Perch]:length_cm       38.9115
species[Pike]:length_cm        53.1949
species[Roach]:length_cm       23.3193 -> out
'''

# Model price vs both with an interaction using "times" syntax
mdl_price_vs_both_inter = ols(
    'price_twd_msq ~ n_convenience * house_age_years', data=taiwan_real_estate).fit()

# Print the coefficients
print(mdl_price_vs_both_inter.params)


# Model price vs. both with an interaction using "colon" syntax
mdl_price_vs_both_inter = ols(
    'price_twd_msq ~ n_convenience + house_age_years + n_convenience:house_age_years', data=taiwan_real_estate).fit()

# Print the coefficients
print(mdl_price_vs_both_inter.params)


# Model price vs. house age plus an interaction, no intercept
mdl_readable_inter = ols(
    ' price_twd_msq ~ house_age_years + house_age_years:n_convenience + 0', data=taiwan_real_estate).fit()

# Print the coefficients for mdl_0_to_15
print("mdl_0_to_15 coefficients:", "\n", mdl_0_to_15.params)

# Print the coefficients for mdl_15_to_30
print("mdl_15_to_30 coefficients:", "\n", mdl_15_to_30.params)

# Print the coefficients for mdl_30_to_45
print("mdl_30_to_45 coefficients:", "\n", mdl_30_to_45.params)

# Print the coefficients for mdl_readable_inter
print("\n", "mdl_readable_inter coefficients:", "\n", mdl_readable_inter.params)


'''
Making predictions with interactions
The model with the interaction
mdl_mass_vs_both_inter = ols('mass_g ~ species + species:length_cm + 0', data=fish).fit()

print(mdl_mass_vs_both_inter.params) -> in

species[Bream]              -1035.3476
species[Perch]               -619.1751
species[Pike]               -1540.8243
species[Roach]               -329.3762
species[Bream]:length_cm       54.5500
species[Perch]:length_cm       38.9115
species[Pike]:length_cm        53.1949
species[Roach]:length_cm       23.3193 -> out

The prediction flow
from itertools import product

length_cm = np.arange(5, 61, 5)
species = fish['species'].unique()
p = product(length_cm, species) 

explanatory_data = pd.DataFrame(p, columns=['length_cm', 'species'])

prediction_data = explanatory_data.assign(mass_g = mdl_mass_vs_both_inter.predict(explanatory_data))

print(prediction_data) -> in

    length_cm   species     mass_g
0           5     Bream  -762.5977
1           5     Roach  -212.7799
2           5     Perch  -424.6178
3           5      Pike -1274.8499
4          10     Bream  -489.8478
5          10     Roach   -96.1836
6          10     Perch  -230.0604
7          10      Pike -1008.8756
8          15     Bream  -217.0979
...
40         55     Bream  1964.9014
41         55     Roach   953.1833
42         55     Perch  1520.9556
43         55      Pike  1384.8933
44         60     Bream  2237.6513
45         60     Roach  1069.7796
46         60     Perch  1715.5129
47         60      Pike  1650.8677 -> out

Visualizing the predictions
sns.lmplot(x='length_cm', y='mass_g', data=fish, hue='species', ci=None)

sns.scatterplot(x='length_cm', y='mass_g', data=prediction_data, hue='species')

plt.show()

Manually calculating the predictions
coeffs = mdl_mass_vs_both_inter.params -> in

species[Bream]              -1035.3476
species[Perch]               -619.1751
species[Pike]               -1540.8243
species[Roach]               -329.3762
species[Bream]:length_cm       54.5500
species[Perch]:length_cm       38.9115
species[Pike]:length_cm        53.1949
species[Roach]:length_cm       23.3193 -> out

ic_bream, ic_perch, ic_pike, ic_roach, slope_bream, slope_perch, slope_pike, slope_roach = coeffs

conditions = [explanatory_data['species'] == 'Bream', explanatory_data['species'] == 'Perch', explanatory_data['species'] == 'Pike', explanatory_data['species'] == 'Roach']

ic_choices = [ic_bream, ic_perch, ic_pike, ic_roach]
intercept = np.select(conditions, ic_choices)

slope_choices = [slope_bream, slope_perch, slope_pike, slope_roach]
slope = np.select(conditions, slope_choices)

Manually calculating the predictions
prediction_data = explanatory_data.assign(mass_g = intercept + slope * explanatory_data['length_cm'])

print(prediction_data) -> in

    length_cm   species     mass_g
0           5     Bream  -762.5977
1           5     Roach  -212.7799
2           5     Perch  -424.6178
3           5      Pike -1274.8499
4          10     Bream  -489.8478
5          10     Roach   -96.1836
...
43         55      Pike  1384.8933
44         60     Bream  2237.6513
45         60     Roach  1069.7796
46         60     Perch  1715.5129
47         60      Pike  1650.8677 -> out
'''

# Create n_convenience as an array of numbers from 0 to 10
n_convenience = np.arange(0, 11)

# Extract the unique values of house_age_years
house_age_years = taiwan_real_estate["house_age_years"].unique()

# Create p as all combinations of values of n_convenience and house_age_years
p = product(n_convenience, house_age_years)

# Transform p to a DataFrame and name the columns
explanatory_data = pd.DataFrame(
    p, columns=["n_convenience", "house_age_years"])

# Add predictions to the DataFrame
prediction_data = explanatory_data.assign(
    price_twd_msq=mdl_price_vs_both_inter.predict(explanatory_data))

# Plot the trend lines of price_twd_msq vs. n_convenience colored by house_age_years
sns.lmplot(x='n_convenience', y='price_twd_msq',
           hue='house_age_years', data=taiwan_real_estate, ci=None)

# Add a scatter plot for prediction_data
sns.scatterplot(x='n_convenience', y='price_twd_msq',
                hue='house_age_years', data=prediction_data, legend=False)

# Show the plot
plt.show()


# Get the coefficients from mdl_price_vs_both_inter
coeffs = mdl_price_vs_both_inter.params

# Assign each of the elements of coeffs
ic_0_15, ic_15_30, ic_30_45, slope_0_15, slope_15_30, slope_30_45 = coeffs

# Create conditions
conditions = [explanatory_data["house_age_years"] == "0 to 15", explanatory_data["house_age_years"]
              == "15 to 30", explanatory_data["house_age_years"] == "30 to 45"]

# Create intercept_choices
intercept_choices = [ic_0_15, ic_15_30, ic_30_45]

# Create slope_choices
slope_choices = [slope_0_15, slope_15_30, slope_30_45]

# Create intercept and slope
intercept = np.select(conditions, intercept_choices)
slope = np.select(conditions, slope_choices)

# Create prediction_data with columns intercept and price_twd_msq
prediction_data = explanatory_data.assign(
    price_twd_msq=intercept + slope * explanatory_data['n_convenience'])

# Print it
print(prediction_data)


'''
Simpson's Paradox
A most ingenious paradox!
Simpson's Paradox occurs when the trend of a model on the whole dataset is very different from the trends shown by models on subsets of the dataset.

trend = slope coefficient

Synthetic Simpson data

simpsons_paradox -> in

        x          y    group
62.24344    70.60840    D
52.33499    14.70577    B
56.36795    46.39554    C
66.80395    66.17487    D
66.53605    89.24658    E
62.38129    91.45260    E -> out

* 5 groups of data, labeled 'A' to 'E'

Linear regressions
Whole dataset
mdl_whole = ols('y ~ x', data=simpsons_paradox).fit()

print(mdl_whole.params) -> in

Intercept   -38.554
x             1.751 -> out

By group
mdl_by_group = ols('y ~ group + group:x + 0', data =  simpsons_paradox).fit()

print(mdl_by_group.params) -> in

    groupA  groupB   groupC      groupD    groupE
32.5051    67.3886  99.6333    132.3932  123.8242
    groupA:x    groupB:x    groupC:x    groupD:x    groupE:x
-0.6266          -1.0105     -0.9940     -0.9908     -0.5364 -> out

Plotting the whole dataset
sns.regplot(x='x', y='y', data=simpsons_paradox, ci=None)

Plotting by group
sns.lmplot(x='x', y='y', data=simpsons_paradox, hue='group', ci=None)

Reconciling the difference
Good advice
* If possible, try to plot the dataset

Common advice
* You can't choose the best model in general - it depends on the dataset and the question you are trying to answer

More good advice
* Articulate a question before you start modeling

Reconciling the difference
- Usually (but not always) the grouped model contains more insight
- Are you missing explanatory variables?
- Context is important ( i.e what your dataset means and what question you are trying to answer)

Simpson's paradox in real datasets
- The paradox is usually less obvious
- You may see a zero slope rather than a complete change in direction
- It may not appear in every group
'''

auctions = pd.read_csv(
    'Intermediate Regression with statsmodels in Python/auctions.csv')

# Take a glimpse at the dataset
print(auctions.info())

# Model price vs. opening bid using auctions
mdl_price_vs_openbid = ols("price ~ openbid", data=auctions).fit()

# See the result
print(mdl_price_vs_openbid.params)

# Plot the scatter plot of price vs. openbid with a linear trend line
sns.regplot(x='openbid', y='price', data=auctions, ci=None)

# Show the plot
plt.show()


# Fit linear regression of price vs. opening bid and auction type, with an interaction, without intercept
mdl_price_vs_both = ols(
    "price ~ auction_type + openbid:auction_type + 0", data=auctions).fit()

# Using auctions, plot price vs. opening bid colored by auction type as a scatter plot with linear regr'n trend lines
sns.lmplot(x='openbid', y='price', hue='auction_type', data=auctions, ci=None)

# Show the plot
plt.show()


''' Multiple Linear Regression '''

'''
Two numeric explanatory variables
Visualizing three numeric variables
- 3D scatter plot
- 2D scatter plot with response as color

Another column for the fish dataset
fish -> in

species mass_g length_cm height_cm
Bream     1000      33.5     18.96
Bream      925      36.2     18.75
Roach      290      24.0      8.88
Roach      390      29.5      9.48 
Perch     1100      39.0     12.80
Perch     1000      40.2     12.60
Pike      1250      52.0     10.69
Pike      1650      59.0     10.81 -> out

2D scatter plot, color for response
sns.scatterplot(x='length_cm', y='height_cm', data=fish, hue='mass_g')

Modeling with two numeric explanatory variables
mdl_mass_vs_both = ols('mass_g ~ length_cm + height_cm', data=fish).fit()

print(mdl_mass_vs_both.params) -> in

Intercept   -622.150234
length_cm     28.968405
height_cm     26.334804 -> out

The prediction flow
from itertools import product

length_cm = np.arange(5, 61, 5)
height_cm = np.arange(2, 21, 2)

p = product(length_cm, height_cm)

explanatory_data = pd.DataFrame(p, columns=['length_cm', 'height_cm'])

prediction_data = explanatory_data.assign(mass_g = mdl_mass_vs_both.predict(explanatory_data))

print(prediction_data) -> in

    length_cm  height_cm        mass_g
0           5          2   -424.638603
1           5          4   -371.968995
2           5          6   -319.299387
3           5          8   -266.629780
4           5         10   -213.960172
...       ...        ...           ...
115        60         12   1431.971694
116        60         14   1484.641302
117        60         16   1537.310909
118        60         18   1589.980517
119        60         20   1642.650125
[120 rows x 3 columns] -> out

Plotting the predictions
sns.scatterplot(x = 'length_cm', y = 'height_cm', data = fish, hue = 'mass_g')

sns.scatterplot(x = 'length_cm', y = 'height_cm', data = prediction_data, hue = 'mass_g', legend = False, marker = 's')

plt.show()

Including an interaction
mdl_mass_vs_both_inter = ols('mass_g ~ length_cm * height_cm', data=fish).fit()

print(mdl_mass_vs_both_inter.params) -> in

Intercept               159.107480
length_cm                 0.301426
height_cm               -78.125178
length_cm:height_cm       3.545435 -> out

The prediction flow with an interaction
length_cm = np.arange(5, 61, 5)
height_cm = np.arange(2, 21, 2)

p = product(length_cm, height_cm)

explanatory_data = pd.DataFrame(p, columns=['length_cm', 'height_cm'])

prediction_data = explanatory_data.assign(mass_g = mdl_mass_vs_both_inter.predict(explanatory_data))

print(prediction_data) 

Plotting the predictions
sns.scatterplot(x = 'length_cm', y = 'height_cm', data = fish, hue = 'mass_g')

sns.scatterplot(x = 'length_cm', y = 'height_cm', data = prediction_data, hue = 'mass_g', legend = False, marker = 's')

plt.show()
'''

# Transform dist_to_mrt_m to sqrt_dist_to_mrt_m
taiwan_real_estate["sqrt_dist_to_mrt_m"] = np.sqrt(
    taiwan_real_estate.dist_to_mrt_m)

# Draw a scatter plot of sqrt_dist_to_mrt_m vs. n_convenience colored by price_twd_msq
sns.scatterplot(x='n_convenience', y='sqrt_dist_to_mrt_m',
                data=taiwan_real_estate, hue='price_twd_msq')

# Show the plot
plt.show()


# Fit linear regression of price vs. no. of conv. stores and sqrt dist. to nearest MRT, no interaction
mdl_price_vs_conv_dist = ols(
    "price_twd_msq ~ n_convenience + sqrt_dist_to_mrt_m", data=taiwan_real_estate).fit()

# Create n_convenience as an array of numbers from 0 to 10
n_convenience = np.arange(0, 11)

# Create sqrt_dist_to_mrt_m as an array of numbers from 0 to 80 in steps of 10
sqrt_dist_to_mrt_m = np.arange(0, 81, 10)

# Create p as all combinations of values of n_convenience and sqrt_dist_to_mrt_m
p = product(n_convenience, sqrt_dist_to_mrt_m)

# Transform p to a DataFrame and name the columns
explanatory_data = pd.DataFrame(
    p, columns=['n_convenience', 'sqrt_dist_to_mrt_m'])

# Add column of predictions
prediction_data = explanatory_data.assign(
    price_twd_msq=mdl_price_vs_conv_dist.predict(explanatory_data))

# See the result
print(prediction_data)


# Create scatter plot of taiwan_real_estate
sns.scatterplot(x='n_convenience', y='sqrt_dist_to_mrt_m',
                data=taiwan_real_estate, hue='price_twd_msq')

# Create scatter plot of prediction_data without legend
sns.scatterplot(x='n_convenience', y='sqrt_dist_to_mrt_m',
                data=prediction_data, hue='price_twd_msq', legend=False, marker='s')

# Show the plot
plt.show()


# Convert to mdl_price_vs_conv_dist_inter
mdl_price_vs_conv_dist_inter = ols(
    "price_twd_msq ~ n_convenience * sqrt_dist_to_mrt_m", data=taiwan_real_estate).fit()

# Use mdl_price_vs_conv_dist_inter to make predictions
n_convenience = np.arange(0, 11)
sqrt_dist_to_mrt_m = np.arange(0, 81, 10)

p = product(n_convenience, sqrt_dist_to_mrt_m)

explanatory_data = pd.DataFrame(
    p, columns=["n_convenience", "sqrt_dist_to_mrt_m"])

prediction_data = explanatory_data.assign(
    price_twd_msq=mdl_price_vs_conv_dist_inter.predict(explanatory_data))

sns.scatterplot(x="n_convenience", y="sqrt_dist_to_mrt_m",
                data=taiwan_real_estate, hue="price_twd_msq", legend=False)

sns.scatterplot(x="n_convenience", y="sqrt_dist_to_mrt_m",
                data=prediction_data, hue="price_twd_msq", marker="s")

plt.show()


'''
More than two explanatory variables
From last time
sns.scatterplot(x='length_cm', y='height_cm', data=fish, hue='mass_g')

Faceting by species
grid = sns.FacetGrid(data=fish, col='species', hue='mass_g', col_wrap=2, palette='plasma')

grid.map(sns.scatterplot, 'length_cm', 'height_cm')

plt.show()

Faceting by species
* It's possible to use more than one categorical variable for faceting
* Beware of faceting overuse
* Plotting becomes harder with increasing number of variables

Different levels of interaction
No interactions

ols('mass_g ~ length_cm + height_cm + species + 0', data=fish).fit()

Two-way interactions between pairs of variables

ols('mass_g ~ length_cm + height_cm + species + length_cm:height_cm + length_cm:species + height_cm:species + 0', data=fish).fit()

# Same as

Only two-way interactions
ols('mass_g ~ (length_cm + height_cm + species) ** 2 + 0', data=fish).fit()

Three-way interactions between all three variables

ols('mass_g ~ length_cm + height_cm + species + length_cm:height_cm + length_cm:species + height_cm:species + length_cm:height_cm:species + 0', data=fish).fit()

# Same as

All the interactions
ols('mass_g ~ length_cm * height_cm * species + 0', data=fish).fit()

The prediction flow
mdl_mass_vs_all = ols('mass_g ~ length_cm * height_cm * species + 0', data=fish).fit()

length_cm = np.arange(5, 61, 5)
height_cm = np.arange(2, 21, 2)
species = fish['species'].unique()

p =product(length_cm, height_cm, species)

explanatory_data = pd.DataFrame(p, columns=['length_cm', 'height_cm', 'species'])

prediction_data = explanatory_data.assign(mass_g = mdl_mass_vs_all.predict(explanatory_data))

print(prediction_data) -> in

    length_cm   height_cm   species          mass_g
0           5           2     Bream     -570.656437 
1           5           2     Roach       31.449145
2           5           2     Perch       43.789984
3           5           2      Pike      271.270093
4           5           4     Bream     -451.127405
...
475        60          18      Pike     2690.346384
476        60          20     Bream     1531.618475
477        60          20     Roach     2621.797668
478        60          20     Perch     3041.931709
479        60          20      Pike     2926.352397
[480 rows x 4 columns] -> out
'''

# Prepare the grid using taiwan_real_estate, for each house age category, colored by price_twd_msq
grid = sns.FacetGrid(data=taiwan_real_estate,
                     col='house_age_years', hue='price_twd_msq', palette="plasma")

# Plot the scatterplot with sqrt_dist_to_mrt_m on the x-axis and n_convenience on the y-axis
grid.map(sns.scatterplot, 'sqrt_dist_to_mrt_m', 'n_convenience')

# Show the plot (brighter colors mean higher prices)
plt.show()


# Model price vs. no. of conv. stores, sqrt dist. to MRT station & house age, no global intercept, no interactions
mdl_price_vs_all_no_inter = ols(
    'price_twd_msq ~ n_convenience + sqrt_dist_to_mrt_m + house_age_years + 0', data=taiwan_real_estate).fit()

# See the result
print(mdl_price_vs_all_no_inter.params)

# Model price vs. sqrt dist. to MRT station, no. of conv. stores & house age, no global intercept, 3-way interactions
mdl_price_vs_all_3_way_inter = ols(
    'price_twd_msq ~ n_convenience * sqrt_dist_to_mrt_m * house_age_years + 0', data=taiwan_real_estate).fit()

# See the result
print(mdl_price_vs_all_3_way_inter.params)

# Model price vs. sqrt dist. to MRT station, no. of conv. stores & house age, no global intercept, 2-way interactions
mdl_price_vs_all_2_way_inter = ols(
    'price_twd_msq ~ (n_convenience + sqrt_dist_to_mrt_m + house_age_years) ** 2 + 0', data=taiwan_real_estate).fit()

# See the result
print(mdl_price_vs_all_2_way_inter.params)


# Create n_convenience as an array of numbers from 0 to 10
n_convenience = np.arange(0, 11)

# Create sqrt_dist_to_mrt_m as an array of numbers from 0 to 80 in steps of 10
sqrt_dist_to_mrt_m = np.arange(0, 81, 10)

# Create house_age_years with unique values
house_age_years = taiwan_real_estate['house_age_years'].unique()

# Create p as all combinations of n_convenience, sqrt_dist_to_mrt_m, and house_age_years, in that order
p = product(n_convenience, sqrt_dist_to_mrt_m, house_age_years)

# Transform p to a DataFrame and name the columns
explanatory_data = pd.DataFrame(
    p, columns=['n_convenience', 'sqrt_dist_to_mrt_m', 'house_age_years'])

# See the result
print(explanatory_data)

# Add column of predictions
prediction_data = explanatory_data.assign(
    price_twd_msq=mdl_price_vs_all_3_way_inter.predict(explanatory_data))

# See the result
print(prediction_data)


'''
How linear regression works
A metric for the best fit
- The simplest idea (which doesn't work)
* Take the sum of all the residuals
* Some residuals are negative
- The next simplest idea (which does work)
* Take the square of each residual, and add up those squares
* This is called the sum of squares

A detour into numerical optimization
A line plot of a quadratic equation

x = np.arange(-4, 5, 0.1)
y = x ** 2 -  x + 10

xy_data = pd.DataFrame({'x': x, 'y': y})

sns.lineplot(x = 'x', y = 'y', data = xy_data)
plt.show()

using calculus to solve the equation
y = x^2 - x + 10
dy / dx = 2x - 1
0 = 2x - 1
x = 0.5
y = 0.5^2 - 0.5 + 10 = 9.75

* Not all equations can be solved like this
* You can let Python figure it out

NB: Don't worry if this doesn't make sense, you won't need it for the exercises

minimize()
from scipy.optimize import minimize

def calc_quadratic(x):
    y = x ** 2 - x + 10
    return y

minimize(fun=calc_quadratic, x0=3) -> in

fun: 9.75
hess_inv: array([[0.5]])
jac: array([0.])
message: 'Optimization terminated successfully.'
nfev: 6
nit: 2
njev: 3
status: 0
success: True
x: array([0.49999998]) -> out

A linear regression algorithm
Define a function to calculate the sum of squares metric

def function_name(args):
    # some calculations with the args
    return outcome

def calc_sum_of_squares(coeffs):
    # Unpack coeffs
    intercept, slope = coeffs
    # More calculation!
    # Calculate predicted y-values
    y_pred = intercept + slope * x_actual
    # Calculate differences between y_actual and y_pred
    y_diff = y_pred - y_actual
    # Calculate sum of squares
    sum_sq = np.sum(y_diff ** 2)
    # Return sum of squares
    return sum_sq

call minimize() to find coefficients that minimize this function

minimize(fun=calc_sum_of_squares, x0=0)
'''

x_actual = taiwan_real_estate['n_convenience']
y_actual = taiwan_real_estate['price_twd_msq']

# Complete the function


def calc_sum_of_squares(coeffs):
    # Unpack coeffs
    intercept, slope = coeffs
    # Calculate predicted y-values
    y_pred = intercept + slope * x_actual
    # Calculate differences between y_actual and y_pred
    y_diff = y_pred - y_actual
    # Calculate sum of squares
    sum_sq = np.sum(y_diff ** 2)
    # Return sum of squares
    return sum_sq


# Test the function with intercept 10 and slope 1
print(calc_sum_of_squares([10, 1]))

# Call minimize on calc_sum_of_squares
print(minimize(fun=calc_sum_of_squares, x0=[0, 0]))

# Compare the output with the ols() call.
print(ols("price_twd_msq ~ n_convenience", data=taiwan_real_estate).fit().params)


''' Multiple Logistic Regression '''

'''
Multiple logistic regression
Bank churn dataset

has_churned     time_since_first_purchase   time_since_last_purchase
0                               0.3993247                 -0.5158691
1                              -0.4297957                  0.6780654
0                               3.7383122                  0.4082544
0                               0.6032289                 -0.6990435
...                                   ...                        ...
response length of relationship recency of activity

* https://www.rdocumentation.org/packages/bayesQR/topics/Churn

logit()
from statsmodels.formula.api import logit

logit('response ~ explanatory', data=dataset).fit()

logit('response ~ explanatory + explanatory2', data=dataset).fit() # without interactions

logit('response ~ explanatory * explanatory2', data=dataset).fit() # with interactions

The four outcomes
                predicted false     predicted true
actual false            correct     false positive
actual true      false negative            correct

conf_matrix = mdl_logit.pred_table()

print(conf_matrix) -> in

[[102.  98.]
 [ 53. 147.]] -> out

Recall the following definitions:
- Accuracy is the proportion of predictions that are correct.
accuracy = (TN + TP) / (TN + FN + FP + TP)

- Sensitivity is the proportion of true observations that are correctly predicted by the model as being true.
sensitivity = TP / (TP + FN)

- Specificity is the proportion of false observations that are correctly predicted by the model as being false.
specificity = TN / (TN + FP)

where: true positives (TP), true negatives (TN), false positives (FP), and false negatives (FN)

Prediction flow
from itertools import product

explanatory1 = some_values
explanatory2 = some_values

p = product(explanatory1, explanatory2)

explanatory_data = pd.DataFrame(p, columns = ['explanatory1', 'explanatory2'])

prediction_data = explanatory_data.assign( mass_g = mdl_logit.predict(explanatory_data))

Visualization
prediction_data['most_likely_outcome'] = np.round(prediction_data['has_churned'])

sns.scatterplot(..., data=churn, hue='has_churned', ...)

sns.scatterplot(..., data=prediction_data, hue='most_likely_outcome', ...)
'''

churn = pd.read_csv(
    'Intermediate Regression with statsmodels in Python\churn.csv')

# Import logit

# Fit a logistic regression of churn status vs. length of relationship, recency, and an interaction
mdl_churn_vs_both_inter = logit(
    'has_churned ~ time_since_first_purchase * time_since_last_purchase', data=churn).fit()

# Print the coefficients
print(mdl_churn_vs_both_inter.params)


# Create time_since_first_purchase
time_since_first_purchase = np.arange(-2, 4.1, 0.1)

# Create time_since_last_purchase
time_since_last_purchase = np.arange(-1, 6.1, 0.1)

# Create p as all combinations of values of time_since_first_purchase and time_since_last_purchase
p = product(time_since_first_purchase, time_since_last_purchase)

# Transform p to a DataFrame and name the columns
explanatory_data = pd.DataFrame(
    p, columns=["time_since_first_purchase", "time_since_last_purchase"])

# Create prediction_data
prediction_data = explanatory_data.assign(
    has_churned=mdl_churn_vs_both_inter.predict(explanatory_data))

# Create most_likely_outcome
prediction_data["most_likely_outcome"] = np.round(
    prediction_data['has_churned'])

# See the result
print(prediction_data)


# Using churn, plot recency vs. length of relationship, colored by churn status
sns.scatterplot(x="time_since_first_purchase",
                y="time_since_last_purchase", data=churn, hue="has_churned")

# Using prediction_data, plot recency vs. length of relationship, colored by most_likely_outcome
sns.scatterplot(x="time_since_first_purchase", y="time_since_last_purchase",
                data=prediction_data, hue="most_likely_outcome", alpha=0.2, legend=False)

# Show the plot
plt.show()


# Create conf_matrix
conf_matrix = mdl_churn_vs_both_inter.pred_table()

# Extract TN, TP, FN and FP from conf_matrix
TN = 102.
TP = 147.
FN = 53.
FP = 98.

# Calculate and print the accuracy
accuracy = (TN + TP) / (TN + FN + FP + TP)
print("accuracy", accuracy)

# Calculate and print the sensitivity
sensitivity = TP / (TP + FN)
print("sensitivity", sensitivity)

# Calculate and print the specificity
specificity = TN / (TN + FP)
print("specificity", specificity)


'''
The logistic distribution
Gaussian / Normal distribution probability density function (PDF)
from scipy.stats import norm

x = np.arange(-4, 4.05, 0.05)

gauss_dist = pd.DataFrame({'x': x, 'gauss_pdf': norm.pdf(x)})

sns.lineplot(x= 'x', y= 'gauss_pdf', data=gauss_dist)
plt.show()

Gaussian / Normal distribution cumulative  distribution function (CDF)
x = np.arange(-4, 4.05, 0.05)

gauss_dist = pd.DataFrame({'x': x, 'gauss_pdf': norm.pdf(x), 'gauss_cdf': norm.cdf(x)})

sns.lineplot(x= 'x', y= 'gauss_cdf', data=gauss_dist)
plt.show()

Gaussian inverse CDF
p = np.arange(0.001. 1, 0.001)

gauss_dist_inv = pd.DataFrame({'p': p, 'gauss_inv_cdf': norm.ppf(p)})

* The inverse CDF is also known as percent point function (PPF) or quantile function.

sns.lineplot(x= 'p', y= 'gauss_inv_cdf', data= gauss_dist_inv)

logistics PDF
from scipy.stats import logistic

x = np.arange(-4, 4.05, 0.05)

logistic_dist = pd.DataFrame({'x': x, 'log_pdf': logistic.pdf(x)})

sns.lineplot(x= 'x', y= 'log_pdf', data= logistic_dist)

Logistic distribution
* Logistic distribution CDF is also called the logistic function.
* cdf(x) = 1 / (1 + np.exp(-x) )
* Logistic distribution inverse CDF is also called the logit function
* inverse_cdf(p) = log(p / (1-p) )

NB:
- The gaussian distribution's probability density function has a "bell" shape
- The logistic distribution's cumulative distribution function has an "S" shape, known as a sigmoid curve
'''

# Import logistic

# Create x ranging from minus ten to ten in steps of 0.1
x = np.arange(-10, 10.1, 0.1)

# Create logistic_dist
logistic_dist = pd.DataFrame(
    {"x": x, "log_cdf": logistic.cdf(x), "log_cdf_man": 1 / (1 + np.exp(-x))})

# Using logistic_dist, plot log_cdf vs. x
sns.lineplot(x='x', y='log_cdf', data=logistic_dist)

# Show the plot
plt.show()


# Create p ranging from 0.001 to 0.999 in steps of 0.001
p = np.arange(0.001, 1, 0.001)

# Create logistic_dist_inv
logistic_dist_inv = pd.DataFrame(
    {"p": p, "logit": logistic.ppf(p), "logit_man": np.log(p / (1 - p))})

# Using logistic_dist_inv, plot logit vs. p
sns.lineplot(x='p', y='logit', data=logistic_dist_inv)

# Show the plot
plt.show()


'''
How logistic regression works
Sum of squares doesn't work
np.sum((y_pred - y_actual) **2)

y_pred : Predicted response ( is between 0 and 1)
y_actual : Actual response ( is always 0 or 1)
* There is a better metric than sum of squares.

Likelihood metric
np.sum(y_pred * y_actual + (1 - y_pred) * (1- y_actual))

when y_actual = 1
y_pred * 1 + (1 - y_pred) * (1-1) = y_pred

when y_actual = 0
y_pred * 0 + (1 - y_pred) * (1-0) = 1 - y_pred

Log-likelihood
* Computing likelihood involves adding many very small numbers, leading to numerical error
* Log-likelihood is easier to compute

log_likelihood = np.log(y_pred) * y_actual + np.log(1 - y_pred) * (1 - y_actual)

Both equations give the same coefficients answer.

Negative log-likelihood
Maximizing log-likelihood is the same as minimizing negative log-likelihood

-np.sum(log_likelihood)

Logistic regression algorithm
def calc_neg_log_likelihood(coeffs)
    intercept, slope = coeffs
    # More calculation!

from scipy.optimize import minimize

minimize(fun=calc_neg_log_likelihood, x0=[0, 0])
'''

# Complete the function


def calc_neg_log_likelihood(coeffs):
    # Unpack coeffs
    intercept, slope = coeffs
    # Calculate predicted y-values
    y_pred = logistic.cdf(intercept + slope * x_actual)
    # Calculate log-likelihood
    log_likelihood = np.log(y_pred) * y_actual + \
        np.log(1 - y_pred) * (1 - y_actual)
    # Calculate negative sum of log_likelihood
    neg_sum_ll = -np.sum(log_likelihood)
    # Return negative sum of log_likelihood
    return neg_sum_ll


# Call minimize on calc_sum_of_squares
print(minimize(fun=calc_neg_log_likelihood, x0=[0, 0]))

# Compare the output with the logit() call.
print(logit("has_churned ~ time_since_last_purchase", data=churn).fit().params)
