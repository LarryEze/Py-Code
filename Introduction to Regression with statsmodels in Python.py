''' Simple Linear Regression Modeling '''

'''
A tale of two variables
Swedish motor insurance data
- Each row represents one geographic region in sweden
- There are 63 rows

n_claims    total_payment_sek
108                     392.5
19                       46.2
13                       15.7
124                     422.2
40                      119.4
...                       ...

Descriptive statistics
import pandas as pd
print(swedish_motor_insurance.mean()) -> in

n_claims  22.904762
total_payment_sek  98.187302
dtype: float64 -> out

print(swedish_motoe_insurance['n_claims'].corr(swedish_motoe_insurance['total_payment_sek'])) -> in

0.9128782350234068 -> out

What is regression?
- Statistical models to explore the relationship between a response variable and some explanatory variables.
- Given values of explanatory variables, you can predict the values of the response variable.

n_claims    total_payment_sek
108                     392.5
19                       46.2
13                       15.7
124                     422.2
40                      119.4
200                       ???

Jargon
Response variable (a.k.a. dependent variable)
The variable that you want to predict

Explanatory variables (a.k.a. independent variables)
The variables that explain how the response variable will change

Linear regression and logistic regression
Linear regression
- The response variable is numeric

Logistic regression
- The response variable is logical

Simple linear / logistic regression
- There is only one explanatory variable

Visualizing pairs of variables
import matplotlib.pyplot as plt
import seaborn as sns

sns.scatterplot(c='n_claims', y='total_payment_sek', data=swedish_motor_insurance)
plt.show()

Adding a linear trend line
sns.regplot(c='n_claims', y='total_payment_sek', data=swedish_motor_insurance, ci=None)
plt.show()

Python packages for regression
- statsmodels
* Optimized for insight (focus in this course)
- scikit-learn
* Optimized for prediction (focus in other DataCamp courses)
'''

# Import seaborn with alias sns
# import seaborn as sns

# Import matplotlib.pyplot with alias plt
# import matplotlib.pyplot as plt

# Draw the scatter plot
import numpy as np
sns.scatterplot(x='n_convenience', y='price_twd_msq', data=taiwan_real_estate)

# Show the plot
plt.show()

# Import seaborn with alias sns

# Import matplotlib.pyplot with alias plt

# Draw the scatter plot
sns.scatterplot(x="n_convenience",
                y="price_twd_msq",
                data=taiwan_real_estate)

# Draw a trend line on the scatter plot of price_twd_msq vs. n_convenience
sns.regplot(x="n_convenience",
            y="price_twd_msq",
            data=taiwan_real_estate,
            ci=None,
            scatter_kws={'alpha': 0.5})

# Show the plot
plt.show()


'''
Fitting a linear regression
Straight lines are defined by two things
Intercept
The y value at the point when x is zero

Slope
The amount the y value increases if you increase x by one

Equation
y = intercept + slope * x

Running a model
ols stands for ordinary least squares
ols(formula = 'response ~ explanatory', data = data)

from statsmodels.formula.api import ols

mdl_payment_vs_claims = ols('total_payment_sek ~ n_claims', data = swedish_motor_insurance)
mdl_payment_vs_claims = mdl_payment_vs_claims.fit()
print(mdl_payment_vs_claims.params)    -> in

Intercept       19.994486
n_Claims         3.413824
dtype: float64   -> out

Equation
total_payment_sek = 19.99 + 3.41 * n_claims
'''

# Import the ols function

# Create the model object
mdl_price_vs_conv = ols('price_twd_msq ~ n_convenience',
                        data=taiwan_real_estate)

# Fit the model
mdl_price_vs_conv = mdl_price_vs_conv.fit()

# Print the parameters of the fitted model
print(mdl_price_vs_conv.params)


'''
Categorical explanatory variables
Fish dataset
- Each row represents one fish.
- There are 128 rows in the dataset.
- There are 4 species of fish.
* Common Bream
* European Perch
* Northern Pike
* Common Roach

species     mass_g
Bream        242.0
Perch          5.9
Pike         200.0
Roach         40.0
...            ...

Visualizing 1 numeric and 1 categorical variable
import matplotlib.pyplot as plt
import seaborn as sns

sns.hisplot(data=fish, x='mass_g', col='species', col_wrap=2, bins=9)
plt.show()

Summary statistics: mean mass by species
summary_stats = fish.groupby('species')['mass_g').mean() 
print(summary_stats)  -> in

species
Bream           617.828571
Perch           382.239286
Pike            718.705882
Roach           152.050000
Name: mass_g, dtype: float64   -> out

Linear regression
from statsmodels.formula.api import ols

mdl_mass_vs_species = ols('mass_g ~ species', data=fish).fit() -> in

Intercept                617.828571   
species[T.Perch]        -235.589286    
species[T.Pike]          100.877311   
species[T.Roach]        -465.778571    -> out

No intercept
mdl_mass_vs_species = ols('mass_g ~ species + 0', data=fish).fit() -> in

Intercept               617.828571   
species[T.Perch]        382.239286    
species[T.Pike]         718.705882   
species[T.Roach]        152.050000    -> out
'''

# Histograms of price_twd_msq with 10 bins, split by the age of each house
sns.displot(data=taiwan_real_estate,
            x='price_twd_msq',
            col='house_age_years',
            bins=10)

# Show the plot
plt.show()


# Calculate the mean of price_twd_msq, grouped by house age
mean_price_by_age = taiwan_real_estate.groupby(
    'house_age_years')['price_twd_msq'].mean()

# Print the result
print(mean_price_by_age)


# Create the model, fit it
mdl_price_vs_age = ols(
    'price_twd_msq ~ house_age_years', data=taiwan_real_estate).fit()

# Print the parameters of the fitted model
print(mdl_price_vs_age.params)

# Update the model formula to remove the intercept
mdl_price_vs_age0 = ols(
    "price_twd_msq ~ house_age_years + 0", data=taiwan_real_estate).fit()

# Print the parameters of the fitted model
print(mdl_price_vs_age0.params)


''' Predictions and model objects '''

'''
Making predictions
Te fish dataset: bream
bream = fish[fish['species'] == 'Bream']
print(bream.head()) -> in

    species   mass_g    length_cm
0   Bream      242.0         23.2
1   Bream      290.0         24.0
2   Bream      340.0         23.9
3   Bream      363.0         26.3
4   Bream      430.0         26.5  -> out

Plotting mass vs length
sns.regplot(c='length_cm', y='mass_g', data=bream, ci=None)
plt.show()

Running the model
mdl_mass_vs_length <- ols('mass_g ~ length_cm', data = bream).fit()
print(mdl_mass_vs_length.params) -> in

Intercept       -1035.347565
length_cm          54.549981
dtype: float64  -> out

Data on explanatory values to predict
If i set the explanatory variables to these values, what value would the response variable have?

explanatory_data = pd.DataFrame({'length_cm': np.arange(20, 41)})  -> in

    length_cm
0          20
1          21
2          22
3          23
4          24
5          25
    ...

Call predict()
predict(mdl_mass_vs_length.predict(explanatory_data)) -> in

0         55.652054
1        110.202035         
2        164.752015         
3        219.301996         
4        273.851977
    ...
16       928.451749        
17       983.001730        
18      1037.551710
19      1092.101691        
20      1146.651672  
Length: 21, dtype: float64  ->out

Predicting inside a data frame
explanatory_data = pd.DataFrame({'length_cm': np.arange(20, 41)})  

prediction_data = explanatory_data.assign(mass_g=mdl_mass_vs_length.predict(explanatory_data))
print(prediction_data) -> in

    length_cm        mass_g
0          20     55.652054
1          21    110.202035
2          22    164.752015
3          23    219.301996
4          24    273.851977
...     ...         ...
16         36    928.451749
17         37    983.001730
18         38   1037.551710
19         39   1092.101691
20         40   1146.651672  -> out

Showing predictions
import matplotlib.pyplot as plt
import seaborn as sns

fig = plt.figure()
sns.regplot(x='length_cm', y='mass_g', ci=None, data=bream)
sns.scatterplot(x='length_cm', y='mass_g', data=prediction_data, color='red', marker='s')
plt.show()

Extrapolating
Extrapolating means, making predictions outside the range of observed data.

little_bream = pd.DataFrame({'length_cm': [10]})

pred_little_bream =
little_bream.assign(mass_g=mdl_mass_vs_length.predict(little_bream)) -> in

    length_cm        mass_g
0          10   -489.847756   -> out

- Extrapolation is sometimes appropriate, but can lead to misleading or ridiculous results.

explanatory_data = pd.DataFrame({"explanatory_var": list_of_values})
predictions = model.predict(explanatory_data)
prediction_data = explanatory_data.assign(response_var=predictions)
'''

# Import numpy with alias np

# Create explanatory_data
explanatory_data = pd.DataFrame({'n_convenience': np.arange(0, 11)})

# Use mdl_price_vs_conv to predict with explanatory_data, call it price_twd_msq
price_twd_msq = mdl_price_vs_conv.predict(explanatory_data)

# Create prediction_data
prediction_data = explanatory_data.assign(
    price_twd_msq=price_twd_msq)

# Print the result
print(prediction_data)


# Create a new figure, fig
fig = plt.figure()

sns.regplot(x="n_convenience",
            y="price_twd_msq",
            data=taiwan_real_estate,
            ci=None)
# Add a scatter plot layer to the regplot
sns.scatterplot(x="n_convenience",
                y="price_twd_msq",
                data=prediction_data,
                color='red')

# Show the layered plot
plt.show()


# Define a DataFrame impossible
impossible = pd.DataFrame({'n_convenience': [-1, 2.5]})


'''
Working with model objects
.params attribute
from statsmodels.formula.api import ols
mdl_mass_vs_length = ols('mass_g ~ length_cm', data=bream).fit()
print(mdl_mass_vs_length.params)

Intercept     -1035.347565
length_cm        54.549981
dtype: float64   -> out

.fittedvalues attribute
Fitted values: predictions on the original dataset

print(mdl_mass_vs_length.fittedvalues) 

or equivalently
explanatory_data = bream['length_cm']
print(mdl_mass_vs_length.predict(explanatory_data))  -> in

0        230.211993
1        273.851977  
2        268.396979   
3        399.316934   
4        410.226930
...
30      873.901768
31      873.901768  
32      939.361745    
33     1004.821722     
34     1037.551710
Length: 35, dtype: float64  -> out

.resid attribute
Residuals: actual response values minus predicted response values

print(mdl_mass_vs_length.resid) -> in

or equivalently
print(bream['mass_g'] - mdl_mass_vs_length.fittedvalues)

0       11.788007
1       16.148023
2       71.603021
3      -36.316934
4       19.773070
...     -> out

.summary()
mdl_mass_vs_length.summary()  -> in

                                OLS Regression Results                            
Dep. Variable:                 mass_g       R-squared:                       0.878
Model:                            OLS       Adj. R-squared:                  0.874
Method:                 Least Squares       F-statistic:                     237.6
Date:                Thu, 29 Oct 2020       Prob (F-statistic):           1.22e-16
Time:                        13:23:21       Log-Likelihood:                -199.35
No. Observations:                  35       AIC:                             402.7
Df Residuals:                      35       BIC:                             405.8
Df Model:                           1                                         
Covariance Type:            nonrobust                                         

                        coef    std err          t      P>|t|      [0.025      0.975]
Intercept         -1035.3476    107.973     -9.589      0.000   -1255.020    -815.676
n_convenience        54.5500      3.539     15.415      0.000      47.350      61.750

Omnibus:                        7.314       Durbin-Watson:                   1.478
Prob(Omnibus):                  0.026       Jarque-Bera (JB):               10.857
Skew:                          -0.252       Prob(JB):                      0.00439
Kurtosis:                       5.682       Cond. No.                         263.  -> out
'''

# Print the model parameters of mdl_price_vs_conv
print(mdl_price_vs_conv.params)

# Print the fitted values of mdl_price_vs_conv
print(mdl_price_vs_conv.fittedvalues)

# Print the residuals of mdl_price_vs_conv
print(mdl_price_vs_conv.resid)

# Print a summary of mdl_price_vs_conv
print(mdl_price_vs_conv.summary())


# Get the coefficients of mdl_price_vs_conv
coeffs = mdl_price_vs_conv.params

# Get the intercept
intercept = coeffs['Intercept']

# Get the slope
slope = coeffs['n_convenience']

# Manually calculate the predictions
price_twd_msq = intercept + slope * explanatory_data
print(price_twd_msq)

# Compare to the results from .predict()
print(price_twd_msq.assign(
    predictions_auto=mdl_price_vs_conv.predict(explanatory_data)))


'''
Regression to the mean
The concept
- Response value = fitted value + residual
- "The stuff you explained" + "The stuff you couldn"t explain"
- Residuals exist due to problems in the model and fundamental randomness
- Extreme cases are often due to randomness
- Regression to the mean means extreme cases don"t persist over time

Pearson"s father son dataset
- 1078 father / son pairs
- Do tall fathers have tall sons?

father_height_cm    son_height_cm
165.2                       151.8
160.7                       160.6
165.0                       160.9
167.0                       159.5
155.3                       163.3
...                         ...

Scatter plot
fig = plt.figure()

sns.scatterplot(x='father_height_cm', y='son_height_cm', data=father_son)

plt.axline(xy1=(150, 150), slope=1, linewidth=2, color='green')

plt.axis('equal')
plt.show()

Adding a regression line
fig = plt.figure()

sns.regplot(x='father_height_cm', y='son_height_cm', data=father_son, ci=None, line_kws={'color': 'black'})

plt.axline(xy1=(150, 150), slope=1, linewidth=2, color='green')

plt.axis('equal')
plt.show()

Running a regression
mdl_son_vs_father = ols('son_height_cm ~ father_height_cm', data = father_son)
print(mdl_son_vs_father.params) -> in

Intercept              86.071975
father_height_cm        0.514093
dtype: float64     -> out

Making predictions
really_tall_father = pd.DataFrame({'father_height_cm': [190]})

mdl_son_vs_father.predict(really_tall_father) -> in

183.7 -> out

really_short_father = pd.DataFrame({'father_height_cm': [150]})

mdl_son_vs_father.predict(really_short_father) -> in

163.2 -> out
'''

# Create a new figure, fig
fig = plt.figure()

# Plot the first layer: y = x
plt.axline(xy1=(0, 0), slope=1, linewidth=2, color="green")

# Add scatter plot with linear regression trend line
sns.regplot(x='return_2018', y='return_2019',
            data=sp500_yearly_returns, ci=None)

# Set the axes so that the distances along the x and y axes look the same
plt.axis('equal')

# Show the plot
plt.show()


mdl_returns = ols("return_2019 ~ return_2018", data=sp500_yearly_returns).fit()

# Create a DataFrame with return_2018 at -1, 0, and 1
explanatory_data = pd.DataFrame({'return_2018': [-1, 0, 1]})

# Use mdl_returns to predict with explanatory_data
print(mdl_returns.predict(explanatory_data))


'''
Transforming variables
Perch dataset
perch = fish[fish['species'] == 'Perch')
print(perch.head()) -> in

    species     mass_g      length_cm
55  Perch          5.9            7.5
56  Perch         32.0           12.5 
57  Perch         40.0           13.8
58  Perch         51.5           15.0
59  Perch         70.0           15.7
...   ...          ...            ...  -> out

It"s not a linear relationship
sns.regplot(x='length_cm', y='mass_g', data=perch, ci=None)
plt.show()

Plotting mass vs length cubed
perch['length_cm_cubed'] = perch['length_cm'] ** 3

sns.regplot(x='length_cm_cubed', y='mass_g', data=perch, ci=None)
plt.show()

Modeling mass vs length cubed
perch['length_cm_cubed'] = perch['length_cm'] ** 3

mdl_perch = ols('mass_g ~ length_cm_cubed', data=perch).fit()
mdl_perch.params -> in

Intercept          -0.117478
length_cm_cubed     0.016796
dtype: float64  -> out

Predicting mass vs length cubed
explanatory_data = pd.DataFrame({'length_cm_cubed': np.arange(10, 41, 5) ** 3, 'length_cm': np.arange(10, 40, 5)})

prediction_data = explanatory_data.assign(mass_g=mdl_perch.predict(explanatory_data))
print(prediction_data) -> in

    length_cm_cubed     length_cm        mass_g
0              1000            10     16.678235
1              3375            15     56.567717
2              8000            20    134.247429
3             15625            25    262.313982
4             27000            30    453.364084
5             42875            35    719.994447
6             64000            40   1074.801781  -> out

Plotting mass vs length cubed
fig = plt.figure()

sns.regplot(x='length_cm_cubed', y='mass_g', data=perch, ci=None)

sns.scatterplot(x='length_cm_cubed', y='mass_g', data=prediction_data, color='red', marker='s')

Facebook advertising dataset
How advertising works
1. Pay Facebook to shows sads.
2. People see the adds ('impressions')
3. Some people who see it, click it.

-   936 rows
-   Each row represents 1 advert

spent_usd   n_impressions   n_clicks
1.43                 7350          1
1.82                17861          2
1.25                 4259          1
1.29                 4133          1
4.77                15615          3
...                 ...         ...

Plot is cramped
sns.regplot(x='spent_usd', y='n_impressions', data=ad_conversion, ci=None)

Square root vs square root
ad_conversion['sqrt_spent_usd'] = np.sqrt(ad_conversion['spent_usd'])

ad_conversion['sqrt_n_impressions'] = np.sqrt(ad_conversion['n_impressions'])

sns.regplot(x='sqrt_spent_usd', y='sqrt_n_impressions', data=ad_conversion, ci=None)

Modeling and predicting
mdl_ad = ols('sqrt_n_impressions ~ sqrt_spent_usd', data=ad_conversion).fit()

explanatory_data = pd.DataFrame({'sqrt_spent_usd': np.sqrt(np.arange(0, 601, 100)), 'spent_usd': np.arange(0, 601, 100)})

prediction_data = explanatory_data.assign(sqrt_n_impressions=mdl_ad.predict(explanatory_data), n_impressions=mdl_ad.predict(explanatory_data) ** 2)
print(prediction_data)  -> in

    sqrt_spent_usd  spent_usd   sqrt_n_impressions      n_impressions
0         0.000000          0            15.319713       2.346936e+02
1        10.000000        100           597.736582       3.572890e+05
2        14.142136        200           838.981547       7.038900e+05
3        17.320508        300          1024.095320       1.048771e+06
4        20.000000        400          1180.153450       1.392762e+06
5        22.360680        500          1317.643422       1.736184e+06
6        24.494897        600          1441.943858       2.079202e+06  -> out
'''

# Create sqrt_dist_to_mrt_m
taiwan_real_estate["sqrt_dist_to_mrt_m"] = np.sqrt(
    taiwan_real_estate["dist_to_mrt_m"])

# Run a linear regression of price_twd_msq vs. square root of dist_to_mrt_m using taiwan_real_estate
mdl_price_vs_dist = ols(
    'price_twd_msq ~ sqrt_dist_to_mrt_m', data=taiwan_real_estate).fit()

# Print the parameters
print(mdl_price_vs_dist.params)

explanatory_data = pd.DataFrame({"sqrt_dist_to_mrt_m": np.sqrt(np.arange(0, 81, 10) ** 2),
                                "dist_to_mrt_m": np.arange(0, 81, 10) ** 2})

# Create prediction_data by adding a column of predictions to explantory_data
prediction_data = explanatory_data.assign(
    price_twd_msq=mdl_price_vs_dist.predict(explanatory_data)
)

# Print the result
print(prediction_data)

fig = plt.figure()
sns.regplot(x="sqrt_dist_to_mrt_m", y="price_twd_msq",
            data=taiwan_real_estate, ci=None)

# Add a layer of your prediction points
sns.scatterplot(x='sqrt_dist_to_mrt_m', y='price_twd_msq',
                data=prediction_data, color='red')
plt.show()


# Create qdrt_n_impressions and qdrt_n_clicks
ad_conversion["qdrt_n_impressions"] = ad_conversion["n_impressions"] ** 0.25
ad_conversion["qdrt_n_clicks"] = ad_conversion["n_clicks"] ** 0.25

plt.figure()

# Plot using the transformed variables
sns.regplot(x='qdrt_n_impressions', y='qdrt_n_clicks',
            data=ad_conversion, ci=None)
plt.show()

# Run a linear regression of your transformed variables
mdl_click_vs_impression = ols(
    'qdrt_n_clicks ~ qdrt_n_impressions', data=ad_conversion).fit()

explanatory_data = pd.DataFrame({"qdrt_n_impressions": np.arange(0, 3e6+1, 5e5) ** .25,
                                "n_impressions": np.arange(0, 3e6+1, 5e5)})

# Complete prediction_data
prediction_data = explanatory_data.assign(
    qdrt_n_clicks=mdl_click_vs_impression.predict(explanatory_data)
)

# Print the result
print(prediction_data)


# Back transform qdrt_n_clicks
prediction_data["n_clicks"] = prediction_data["qdrt_n_clicks"] ** 4

# Plot the transformed variables
fig = plt.figure()
sns.regplot(x="qdrt_n_impressions", y="qdrt_n_clicks",
            data=ad_conversion, ci=None)

# Add a layer of your prediction points
sns.scatterplot(x="qdrt_n_impressions", y="qdrt_n_clicks",
                data=prediction_data, color='red')
plt.show()


''' Assessing model fit '''

'''
Quantifying model fit
Coefficient of determination
Sometimes called 'r-squared' (simple linear regression) or 'R-squared' (more than one explanatory variable).
- It is the proportion of the variance in the response variable that is predictable from the explanatory variable
* 1 means a perfect fit
* 0 means the worst possible fit

.summary()
Look at the value titled 'R-Squared'

mdl_bream = ols('mass_g ~ length_cm', data = bream).fit()

print(mdl_bream.summary()) -> in

# Some lines of output omitted

                                OLS Regression Results                            
Dep. Variable:                 mass_g       R-squared:                       0.878
Model:                            OLS       Adj. R-squared:                  0.874
Method:                 Least Squares       F-statistic:                     237.6

.rsquared attribute
print(mdl_bream.rsquared)  -> in

0.8780627095147174  -> out

It's just correlation squared
coeff_determination = bream['length_cm'].corr(bream['mass_g']) ** 2
print(coeff_determination)  -> in

0.8780627095147174  -> out

Residual standard error (RSE)
a 'typical' difference between a prediction and an observed response
- It has the same unit as the response variable.
- MSE = RSE^2

.mse_resid attribute
mse = mdl_bream.mse_resid
print('mse: ', mse)  -> in

mse:  5498.555084973521  -> out

rse = np.sqrt(mse)
print('rse: ', rse)  -> in

rse:  74.15224261594197  -> out

Calculating RSE: residuals squared
residuals_sq = mdl_bream.resid ** 2

print('residuals sq: \n', residuals_sq)  -> in

residuals sq:
0        138.957118
1        260.758635
2       5126.992578
3       1318.919660
4        390.974309
    ...
30      2125.047026
31      6576.923291
32       206.259713
33       889.335096
34      7665.302003
Length: 35, dtype: float64 -> out

resid_sum_of_sq = sum(residuals_sq)  

print('resid sum of sq :', resid_sum_of_sq)  -> in

resid sum of sq : 181452.31780412616  -> out

deg_freedom = len(bream.index) - 2

print('deg freedom: ', deg_freedom)  -> in

deg freedom: 33  -> out
* Degrees of freedom equals the number of observations minus the number of model coefficients.

rse = np.sqrt(resid_sum_of_sq / deg_freedom)

print('res :', rse)  -> in

rse : 74.15224261594197  -> out

Interpreting RSE
mdl_bream has an RSE of 74.
- The difference between predicted bream masses and observed bream masses is typically about 74g.

Root-mean-square error (RMSE)
residuals_sq = mdl_bream.resid ** 2

resid_sum_of_sq = sum(residuals_sq)  

n_obs = len(bream.index)

rse = np.sqrt(resid_sum_of_sq / n_obs)

print('rmes :', rmse)  -> in

rmse : 72.00244396727619  -> out
'''

# Print a summary of mdl_click_vs_impression_orig
print(mdl_click_vs_impression_orig.summary())

# Print a summary of mdl_click_vs_impression_trans
print(mdl_click_vs_impression_trans.summary())

# Print the coeff of determination for mdl_click_vs_impression_orig
print(mdl_click_vs_impression_orig.rsquared)

# Print the coeff of determination for mdl_click_vs_impression_trans
print(mdl_click_vs_impression_trans.rsquared)


# Calculate mse_orig for mdl_click_vs_impression_orig
mse_orig = mdl_click_vs_impression_orig.mse_resid

# Calculate rse_orig for mdl_click_vs_impression_orig and print it
rse_orig = np.sqrt(mse_orig)
print("RSE of original model: ", rse_orig)

# Calculate mse_trans for mdl_click_vs_impression_trans
mse_trans = mdl_click_vs_impression_trans.mse_resid

# Calculate rse_trans for mdl_click_vs_impression_trans and print it
rse_trans = np.sqrt(mse_trans)
print("RSE of transformed model: ", rse_trans)


'''
Visualizing model fit
Hoped for properties of residuals
- Residuals are normally distributed
- The mean of the residuals is zero

Bream and perch again
Bream: the 'good' model
mdl_bream = ols('mass_g ~ length_cm', data=bream).fit()

Perch: the 'bad' model
mdl_perch = ols('mass_g ~ length_cm', data=perch).fit()

* Residuals vs fitted values (x-axis=Fitted values, y-axis=Residuals)
* Q-Q plot (x-axis=Theoretical Quantiles, y-axis=Sample Quantiles)
* Scale-location plot (x-axis=Fitted values, y-axis=sqrt of Standardized residuals)

residplot()
sns.residplot(x='length_cm', y='mass_g', data=bream, lowess=True)
plt.xlabe('Fitted values')
plt.ylabel('Residuals')
plt.show()

qqplot()
from statsmodels.api import qqplot
qqplot(data=mdl_bream.resid, fit=True, line='45')
plt.show()

Scale-location plot
model_norm_residuals_bream = mdl_bream.get_influence().resid_studentized_internal
model_norm_residuals_abs_sqrt_bream = np.sqrt(np.abs(model_norm_residuals_bream))
sns.regplot(x=mdl_bream.fittedvalues, y=model_norm_residuals_abs_sqrt_bream, ci=None, lowess=True)
plt.xlabe('Fitted values')
plt.ylabel('Sqrt of abs val of stdized residuals')
plt.show()
'''

# Plot the residuals vs. fitted values
sns.residplot(x='n_convenience', y='price_twd_msq',
                data=taiwan_real_estate, lowess=True)
plt.xlabel("Fitted values")
plt.ylabel("Residuals")

# Show the plot
plt.show()

# Import qqplot
# from statsmodels.api import qqplot

# Create the Q-Q plot of the residuals
qqplot(data=mdl_price_vs_conv.resid, fit=True, line="45")

# Show the plot
plt.show()

# Preprocessing steps
model_norm_residuals = mdl_price_vs_conv.get_influence().resid_studentized_internal
model_norm_residuals_abs_sqrt = np.sqrt(np.abs(model_norm_residuals))

# Create the scale-location plot
sns.regplot(x=mdl_price_vs_conv.fittedvalues,
            y=model_norm_residuals_abs_sqrt, ci=None, lowess=True)
plt.xlabel("Fitted values")
plt.ylabel("Sqrt of abs val of stdized residuals")

# Show the plot
plt.show()


'''
Outliers, leverage, and influence
Roach dataset
roach = fish[fish['species'] == 'Roach']
print(roach.head())

    species   mass_g       length_cm
35  Roach       40.0            12.9
36  Roach       69.0            16.5
37  Roach       78.0            17.5
38  Roach       87.0            18.2
39  Roach      120.0            18.6
...              ...             ...

Which points are outliers?
* The technical term for an unusual data point is an outlier

sns.regplot(x='length_cm', y='mass_g', data=roach, ci=None)
plt.show()

Extreme explanatory values
roach['extreme_l']  = (roach['length_cm'] < 15) | (roach['length_cm'] > 26))

fig = plt.figure()
sns.regplot(x='length_cm', y='mass_g', data=roach, ci=None)

sns.scatterplot(x='length_cm', y='mass_g', data=roach, hue='extreme_l')

Response values away from the regression line
roach['extreme_m']  = roach['mass_g'] < 1

fig = plt.figure()
sns.regplot(x='length_cm', y='mass_g', data=roach, ci=None)

sns.scatterplot(x='length_cm', y='mass_g', data=roach, hue='extreme_l', style='extreme_m')

Leverage and influence
- Leverage is a measure of how extreme the explanatory variable values are.
- Influence measures how much the model would change if you left the observation out of the dataset when modeling.

.get_influence() and .summary_frame()
mdl_roach = ols('mass_g ~ length_cm', data=roach).fit()
summary_roach = mdl_roach.get_influence().summary_frame()
roach['leverage'] = summary_roach['hat_diag']

print(roach.head())  -> in

    species   mass_g     length_cm      leverage
35  Roach       40,0          12.9      0.313729
36  Roach       69.0          16.5      0.125538
37  Roach       78.0          17.5      0.093487
38  Roach       87.0          18.2      0.076283
39  Roach      120.0          18.6      0.068387  -> out

Cook"s distance
Cook"s distance is the most common measure of influence

roach['cooks_dist'] = summary_roach[;cooks_d']
print(roach.head())  -> in

    species   mass_g     length_cm      leverage  cooks_dist
35  Roach       40,0          12.9      0.313729    1.074015
36  Roach       69.0          16.5      0.125538    0.010429
37  Roach       78.0          17.5      0.093487    0.000020
38  Roach       87.0          18.2      0.076283    0.001980
39  Roach      120.0          18.6      0.068387    0.006610  -> out

Most influential roaches
print(roach.sort_values('cooks_dist', ascending=False) -> in

    species   mass_g    length_cm       leverage    cooks_dist
35  Roach       40.0         12.9       0.313729      1.074015     # really short roach
35  Roach      390.0         29.5       0.394740      0.365782     # really long roach
35  Roach        0.0         19.0       0.061897      0.311852     # zero mass roach
35  Roach      290.0         24.0       0.099488      0.150064
35  Roach      180.0         23.6       0.088391      0.061209
...   ...        ...          ...            ...           ...
43  Roach      150.0         20.4       0.050264      0.000257
44  Roach      145.0         20.5       0.050092      0.000256
42  Roach      120.0         19.4       0.056815      0.000199
47  Roach      160.0         21.1       0.050910      0.000137  -> out

Removing the most influential roach
roach_not_short = roach[roach['length_cm'] != 12.9]

sns.regplot(x='length_cm', y='mass_g', data=roach, ci=None, line_kws={'color': 'green'})

sns.scatterplot(x='length_cm', y='mass_g', data=roach_not_short, ci=None, line_kws={'color': 'red'})
'''

# Create summary_info
summary_info = mdl_price_vs_dist.get_influence().summary_frame()

# Add the hat_diag column to taiwan_real_estate, name it leverage
taiwan_real_estate["leverage"] = summary_info['hat_diag']

# Sort taiwan_real_estate by leverage in descending order and print the head
print(taiwan_real_estate.sort_values("leverage", ascending=False).head())

# Add the cooks_d column to taiwan_real_estate, name it cooks_dist
taiwan_real_estate["cooks_dist"] = summary_info["cooks_d"]

# Sort taiwan_real_estate by cooks_dist in descending order and print the head.
print(taiwan_real_estate.sort_values("cooks_dist", ascending=False).head())


''' Simple logistic regression '''

'''


Why you need logistic regression
Bank churn dataset
has_churned     time_since_first_purchase   time_since_last_purchase
0                               0.3993247                 -0.5158691
1                              -0.4297957                  0.6780654
0                               3.7383122                  0.4082544
0                               0.6032289                 -0.6990435
...                                   ...                        ...

response          length of relationship         recency of activity

Churn vs recency: a linear model
mdl_churn_vs_recency_lm = ols('has_churned ~ time_since_last_purchase', data=churn).fit()

print(mdl_churn_vs_recency_lm.params)  -> in

Intercept                       0.490780
time_since_last_purchase        0.063783
dtype: float64  ->  out

intercept, slope = mdl_churn_vs_recency_lm.params

Visualizing the linear model
sns.scatterplot(x='time_since_last_purchase', y='has_churned', data=churn)

plt.axline(xy1=(0, intercept), slope=slope)

plt.show()

- Predictions are probabilities of churn, not amounts of churn.

Zooming out
sns.scatterplot(x='time_since_last_purchase', y='has_churned', data=churn)

plt.axline(xy1=(0, intercept), slope=slope)

plt.xlim(-10, 10)
plt.ylim(-0.2,  1.2)
plt.show()

What is logistic regression?
- Another type of generalized linear model.
- Used when the response variable is logical
- The reponses follow logistic (S-shaped) curve.

Logistic regression using logit()
from statsmodels.formula.api import logit

mdl_churn_vs_recency_logit = logit('has_churned ~ time_since_last_purchase', data=churn).fit()
print(mdl_churn_vs_recency_logit.params)  -> in

Intercept                      -0.035019
time_since_last_purchase        0.269215
dtype: float64  -> in

Visualizing the logistic model
sns.regplot(x='time_since_last_purchase', y='has_churned', data=churn, ci=None, logistic=True)

plt.axline(xy1=(0, intercept), slope=slope, color='black')

plt.show()
'''

# Create the histograms of time_since_last_purchase split by has_churned
sns.displot(x='time_since_last_purchase', col='has_churned', data=churn)

plt.show()

# Redraw the plot with time_since_first_purchase
sns.displot(x='time_since_first_purchase', col='has_churned', data=churn)

plt.show()


# Draw a linear regression trend line and a scatter plot of time_since_first_purchase vs. has_churned
sns.regplot(x="time_since_first_purchase",
            y="has_churned",
            data=churn,
            ci=None,
            line_kws={"color": "red"})

# Draw a logistic regression trend line and a scatter plot of time_since_first_purchase vs. has_churned
sns.regplot(x="time_since_first_purchase",
            y="has_churned",
            data=churn,
            ci=None,
            line_kws={"color": "blue"},
            logistic=True)

plt.show()


# Import logit
# from statsmodels.formula.api import logit

# Fit a logistic regression of churn vs. length of relationship using the churn dataset
mdl_churn_vs_relationship = logit(
    'has_churned ~ time_since_first_purchase', data=churn).fit()

# Print the parameters of the fitted model
print(mdl_churn_vs_relationship.params)


'''
Predictions and odds ratios
Making predictions
mdl_recency = logit('has_churned ~ time_since_last_purchase', data=churn).fit()

explanatory_data = pd.DataFrame({'time_since_last_purchase': np.arange(-1, 6.25, 0.25))

prediction_data = explanatory_data.assign(has_churned = mdl_recency.predict(explanatory_data))

Adding point predictions
sns.regplot(x="time_since_last_purchase", y="has_churned", data=churn, ci=None, logistic=True)

sns.scatterplot(x="time_since_last_purchase", y="has_churned", data=prediction_data, color='red')

plt.show()

Getting the most likely outcome
prediction_data = explanatory_data.assign(has_churned = mdl_recency.predict(explanatory_data))

prediction_data['most_likely_outcome'] = np.round(prediction_data['has_churned'])

Visualizing most likely outcome
sns.regplot(x="time_since_last_purchase", y="has_churned", data=churn, ci=None, logistic=True)

sns.scatterplot(x="time_since_last_purchase", y="most_likely_outcome", data=prediction_data, color='red')

plt.show()

Odds ratios
Odds ratio is the probability of something happening divided by the probability thst it doesn"t

odds_ratio = probability / (1 - probability)

odds_ratio = 0.25 / (1 - 0.25) = 1/3

Calculating odds ratio
prediction_data['odds_ratio'] = prediction_data['has_churned'] / (1 - prediction_data['has_churned']) 

Visualizing odds ratio
sns.lineplot(x="time_since_last_purchase", y="odds_ratio", data=prediction_data)

plt.axhline(y=1, linestyle='dotted')

plt.show()

Visualizing log odds ratio
sns.lineplot(x="time_since_last_purchase", y="odds_ratio", data=prediction_data)

plt.axhline(y=1, linestyle='dotted')

plt.yscale('log')

plt.show()

Calculating log odds ratio
prediction_data['log_odds_ratio'] = np.log(prediction_data['odds_ratio'])

All predictions together
time_since_last_prchs     has_churned     most_likely_rspns     odds_ratio      log_odds_ratio      
0                         0.491                   0             0.966                   -0.035
2                         0.623                   1             1.654                    0.503
4                         0.739                   1             2.834                    1.042
6                         0.829                   1             4.856                    1.580
...                         ...                 ...               ...                      ...

Comparing scales
Scale                   Are values easy to interpret?   Are changes easy to interpret?  Is precise?
Probability                                         +                                x            +
Most likely outcome                                ++                                +            x
Odds ratio                                          +                                x            +
Log odds ratio                                      x                                +            +
'''


# Create prediction_data
prediction_data = explanatory_data.assign(
    has_churned=mdl_churn_vs_relationship.predict(explanatory_data)
)

fig = plt.figure()

# Create a scatter plot with logistic trend line
sns.scatterplot(x='time_since_first_purchase',
                y='has_churned', data=prediction_data)

# Overlay with prediction_data, colored red
sns.regplot(x='time_since_first_purchase', y='has_churned',
            data=churn, ci=None, color='red', logistic=True)

plt.show()


# Update prediction data by adding most_likely_outcome
prediction_data["most_likely_outcome"] = np.round(
    prediction_data["has_churned"])

fig = plt.figure()

# Create a scatter plot with logistic trend line (from previous exercise)
sns.regplot(x="time_since_first_purchase",
            y="has_churned",
            data=churn,
            ci=None,
            logistic=True)

# Overlay with prediction_data, colored red
sns.scatterplot(x="time_since_first_purchase",
                y="most_likely_outcome",
                data=prediction_data,
                color='red')

plt.show()


# Update prediction data with odds_ratio
prediction_data["odds_ratio"] = prediction_data["has_churned"] / \
    (1 - prediction_data["has_churned"])

fig = plt.figure()

# Create a line plot of odds_ratio vs time_since_first_purchase
sns.lineplot(x='time_since_first_purchase',
             y='odds_ratio', data=prediction_data)

# Add a dotted horizontal line at odds_ratio = 1
plt.axhline(y=1, linestyle="dotted")

plt.show()


# Update prediction data with log_odds_ratio
prediction_data["log_odds_ratio"] = np.log(prediction_data["odds_ratio"])

fig = plt.figure()

# Update the line plot: log_odds_ratio vs. time_since_first_purchase
sns.lineplot(x="time_since_first_purchase",
             y="log_odds_ratio",
             data=prediction_data)

# Add a dotted horizontal line at log_odds_ratio = 0
plt.axhline(y=0, linestyle="dotted")

plt.show()


'''
Quantifying logistic regression fit
The four outcomes    
                predicted false     predicted true
actual false            correct     false negative
actual true      false positive            correct  

Confusion matrix: counts of outcomes
actual_response =  churn['has_churned']

predicted_response = np.round(mdl_recency.predict())

outcomes = pd.DataFrame({'actual_response': actual_response, 'predicted_response': predicted_response})

print(outcomes.value_counts(sort=False))  -> in

actual_response  predicted_response    
0           0.0                 141
            1.0                  59
1           0.0                 111
            1.0                  89  -> out

Visualizing the confusion matrix: mosaic plot
conf_matrix = mdl_recency.pred_table()

print(conf_matrix)  -> in

[[141.         59.]     
 [111.         89.]]  -> out

true negative  false positive
false negative  true positive

from statsmodels.graphics.mosaicplot import mosaic

mosaic(conf_matrix)

Accuracy
- Accuracy is the proportion of correct predictions.
- Higher accuracy is better

accuracy = (TN + TP) / (TN + FN + FP + TP)

[[141.         59.]     
 [111.         89.]  

TN = conf_matrix[0,0]
TP = conf_matrix[1,1]
FN = conf_matrix[1,0]
FP = conf_matrix[0,1]

acc = (TN + TP) / (TN + FN + FP + TP)
print(acc)  -> in

0.575   -> out

Sensitivity
- Sensitivity is the proportion of true positives.
- Higher sensitivity is better

sensitivity = TP / (FN + TP)

[[141.         59.]     
 [111.         89.]  

TN = conf_matrix[0,0]
TP = conf_matrix[1,1]
FN = conf_matrix[1,0]
FP = conf_matrix[0,1]

sens = TP / (FN + TP)
print(sens)  -> in

0.445   -> out

Specificity
- Sensitivity is the proportion of true negatives.
- Higher specificity is better, though there is often a a trade-off where improving specificity will decrease sensitivity and vice versa.

specificity = TN / (TN + FP)

[[141.         59.]     
 [111.         89.]  

TN = conf_matrix[0,0]
TP = conf_matrix[1,1]
FN = conf_matrix[1,0]
FP = conf_matrix[0,1]

sens = TN / (TN + FP)
print(spec)  -> in

0.705   -> out
'''

# Get the actual responses
actual_response = churn['has_churned']

# Get the predicted responses
predicted_response = np.round(mdl_churn_vs_relationship.predict())

# Create outcomes as a DataFrame of both Series
outcomes = pd.DataFrame(
    {'actual_response': actual_response, 'predicted_response': predicted_response})

# Print the outcomes
print(outcomes.value_counts(sort=False))


# Import mosaic from statsmodels.graphics.mosaicplot
# from statsmodels.graphics.mosaicplot import mosaic

# Calculate the confusion matrix conf_matrix
conf_matrix = mdl_churn_vs_relationship.pred_table()

# Print it
print(conf_matrix)

# Draw a mosaic plot of conf_matrix
mosaic(conf_matrix)
plt.show()


# Extract TN, TP, FN and FP from conf_matrix
TN = conf_matrix[0, 0]
TP = conf_matrix[1, 1]
FN = conf_matrix[1, 0]
FP = conf_matrix[0, 1]

# Calculate and print the accuracy
accuracy = (TN + TP) / (TN + FN + FP + TP)
print("accuracy: ", accuracy)

# Calculate and print the sensitivity
sensitivity = TP / (FN + TP)
print("sensitivity: ", sensitivity)

# Calculate and print the specificity
specificity = TN / (TN + FP)
print("specificity: ", specificity)
