''' Introduction to Seaborn '''

'''
Seaborn Introduction
Pandas
- Pandas is a foundational library for analyzing data
- It also supports basic plotting capability
e.g
import pandas as pd
df = pd.read_csv('wines.csv')
df['column_name'].plot.hist()
plt.show()

Matplotlib
- Matplotlib provides the raw building blocks for Seaborn's visualizations
- It can also be used on its own to plot data
e.g
import matplotlib.pyplot as plt
import pandas as pd
df = pd.read_csv('wines.csv')
fig, ax = plt.subplots()
ax.hist(df['column_name'])
plt.show()

Introduction to Seaborn
- Seaborn supports complex visualizations  of data
- It is built on matplotlib and works best with pandas' dataframes
e.g
import seaborn as sns
sns.histplot(df['column_name'])
plt.show()

Seaborn displot
- The displot leverages the histplot and other functions for distribution plots
- By default, it generates a histogram but can also generate other plot types
e.g
import seaborn as sns
sns.displot(df['column_name'], kind='kde')
plt.show()

# KDE   -   Kernel Density Element
'''

# import all modules
#   import pandas as pd
#   import seaborn as sns
#   import matplotlib.pyplot as plt

# Read in the DataFrame
df = pd.read_csv(grant_file)


# Display pandas histogram
df['Award_Amount'].plot.hist()
plt.show()

# Clear out the pandas histogram
plt.clf()

# Display a Seaborn displot
sns.displot(df['Award_Amount'])
plt.show()

# Clear the displot
plt.clf()


'''
Using the distribution plot
Creating a histogram
- The displot function has multiple optional arguments
- You can overlay a KDE plot on the histogram and specify the number of bins to use
e.g
sns.displot(df['column_name'], kde=True, bins=10)

Alternative data distributions
- A rug plot is an alternative way to view the distribution of data by including small tickmarks along the x axis
- A kde curve and rug plot can be combined
e.g
sns.displot(df['column_name'], kind='kde', rug=True, fill=True)

Further plot types
- The displot function uses several functions including kdeplot, rugplot and ecdfplot
- The ecdfplot shows the cummulative distribution of the data
e.g
sns.displot(df['column_name'], kind='ecdf')

# ECDF  -   Estimated Cumulative Density Function

# displot, kdeplot, rugplot, ecdfplot are often referred to as univariate analysis because we only look at one variable
'''

# Create a displot
sns.displot(df['Award_Amount'], bins=20)

# Display the plot
plt.show()


# Create a displot of the Award Amount
sns.displot(df['Award_Amount'], kind='kde', rug=True, fill=True)

# Plot the results
plt.show()


'''
Regression Plots in Seaborn
#   Regression analysis is bivariate because we are looking for relationships between two variables

Introduction to regplot
- The regplot function generates a scatter plot with a regression line
- Usage is similar to the displot
- The data, and  x and y variables must be defined
e.g
sns.regplot(data=df, x='column_name', y='column_name')

lmplot() builds on top of the base regplot()
#   regplot - low level
e.g
sns.regplot(data=df, x='column_name', y='column_name')

#   lmplot - high level
e.g
sns.lmplot(data=df, x='column_name', y='column_name')

lmplot faceting
- Organize data by colors ( hue )
e.g
sns.lmplot(data=df, x='column_name', y='column_name', hue='column_name')

-Organize data by columns ( col )
e.g
sns.lmplot(data=df, x='column_name', y='column_name', col='column_name')

#   The use of plotting multiple graphs while changing a single variable is often called FACETING.
'''

# Create a regression plot of premiums vs. insurance_losses
sns.regplot(x="insurance_losses", y="premiums", data=df)

# Display the plot
plt.show()

# Create an lmplot of premiums vs. insurance_losses
sns.lmplot(x="insurance_losses", y="premiums", data=df)

# Display the second plot
plt.show()


# Create a regression plot using hue
sns.lmplot(data=df, x="insurance_losses", y="premiums", hue="Region")

# Show the results
plt.show()


# Create a regression plot with multiple rows
sns.lmplot(data=df, x="insurance_losses", y="premiums", row="Region")

# Show the plot
plt.show()


''' Customizing Seaborn Plots '''

'''
Using Seaborn Styles
Setting Styles
- Seaborn has default configurations that can be applied with sns.set()
- These styles can override matplotlib and pandas plots as well
e.g
sns.set()
df['column_name'].plot.hist()

Theme examples with sns.set_style()
e.g
for style in ['whit', 'dark', 'whitegrid', 'darkgrid', 'ticks'] :
    sns.set_style(style)
    sns.displot(df['column_name'])
    plt.show()

Removing axes with despine()
- Sometimes plots are improved by removing elements
- Seaborn contains a shortcut for removing the spines of a plot
e.g
sns.set_style('white')
sns.displot(df['column_name'])
sns.despine(left=True)
'''

# Plot the pandas histogram
df['fmr_2'].plot.hist()
plt.show()
plt.clf()

# Set the default seaborn style
sns.set()

# Plot the pandas histogram again
df['fmr_2'].plot.hist()
plt.show()
plt.clf()


sns.set_style('dark')
sns.displot(df['fmr_2'])
plt.show()
plt.clf()

sns.set_style('whitegrid')
sns.displot(df['fmr_2'])
plt.show()
plt.clf()


# Set the style to white
sns.set_style('white')

# Create a regression plot
sns.lmplot(data=df, x='pop2010', y='fmr_2')

# Remove the spines
sns.despine()

# Show the plot and clear the figure
plt.show()
plt.clf()


'''
Colors in Seaborn
Defining a color for a plot
- Seaborn supports assigning colors to plots using matplotlib color codes
e.g
sns.set(color_codes = True)
sns.displot(df['column_name'], color='g')

Palettes
- Seaborn uses the set_palette() function to define a palette
e.g
palettes = ['deep', 'muted', 'pastel', 'bright', 'dark', 'colourblind']
for p in palettes :
    sns.set_palette(p)
    sns.displot(df['column_name'])

Displaying Palettes
- sns.palplot() function displays a palette
- sns.color_palette() returns the current palette
e.g
palettes = ['deep', 'muted', 'pastel', 'bright', 'dark', 'colourblind']
for p in palettes :
    sns.set_palette(p)
    sns.palplot(sns.color_palette())
    plt.show()

Defining Custom Palettes
- Circular colors = when the data is not ordered
e.g
sns.palplot(sns.color_palette('Paired', 12))

- Sequential colors = when the data has a consistent range from high to low
e.g
sns.palplot(sns.color_palette('Blues', 12))

- Diverging colors = when both the low and high values are interesting
e.g
sns.palplot(sns.color_palette('BrBG', 12))
'''

# Set style, enable color code, and create a magenta displot
sns.set(color_codes=True)
sns.displot(df['fmr_3'], color='m')

# Show the plot
plt.show()


# Loop through differences between bright and colorblind palettes
for p in ['bright', 'colorblind']:
    sns.set_palette(p)
    sns.displot(df['fmr_3'])
    plt.show()

    # Clear the plots
    plt.clf()


sns.palplot(sns.color_palette('Purples', 8))
plt.show()

sns.palplot(sns.color_palette('husl', 10))
plt.show()

sns.palplot(sns.color_palette('coolwarm', 6))
plt.show()


'''
Customizing with matplotlib
Matplotlib Axes
- Most customization available through matplotlib Axes objects
- Axes can be passed to seaborn functions
- The axes object supports many common customizations
e.g
fig, ax = plt.subplots()
sns.histplot(df['column_name'], ax = ax)
ax.set(xlabel='Write x label', ylabel='Write y label', xlim=(0, 50000), title='Write plot title')

Combining Plots
- It is possible to combine and configure multiple plots
fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(7, 4))

sns.histplot(df['column_name'], stat='column_name', ax=ax0)
sns.histplot(df.query(' column_name == 'column_value' ')['column_name'], stat='column_name', ax=ax1)

ax1.set(xlabel = 'Write x label', xlim=(0, 70000))
ax1.axvline(x= 20000, label='Write label', linestyle='--')
ax1.legend()
'''

# Create a figure and axes
fig, ax = plt.subplots()

# Plot the distribution of data
sns.histplot(df['fmr_3'], ax=ax)

# Create a more descriptive x axis label
ax.set(xlabel="3 Bedroom Fair Market Rent")

# Show the plot
plt.show()


# Create a figure and axes
fig, ax = plt.subplots()

# Plot the distribution of 1 bedroom rents
sns.histplot(df['fmr_1'], ax=ax)

# Modify the properties of the plot
ax.set(xlabel="1 Bedroom Fair Market Rent", xlim=(100, 1500), title="US Rent")

# Display the plot
plt.show()


# Create a figure and axes. Then plot the data
fig, ax = plt.subplots()
sns.histplot(df['fmr_1'], ax=ax)

# Customize the labels and limits
ax.set(xlabel="1 Bedroom Fair Market Rent", xlim=(100, 1500), title="US Rent")

# Add vertical lines for the median and mean
ax.axvline(x=df['fmr_1'].median(), color='m',
           label='Median', linestyle='--', linewidth=2)
ax.axvline(x=df['fmr_1'].mean(), color='b',
           label='Mean', linestyle='-', linewidth=2)

# Show the legend and plot the data
ax.legend()
plt.show()


# Create a plot with 1 row and 2 columns that share the y axis label
fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, sharey=True)

# Plot the distribution of 1 bedroom apartments on ax0
sns.histplot(df['fmr_1'], ax=ax0)
ax0.set(xlabel="1 Bedroom Fair Market Rent", xlim=(100, 1500))

# Plot the distribution of 2 bedroom apartments on ax1
sns.histplot(df['fmr_2'], ax=ax1)
ax1.set(xlabel="2 Bedroom Fair Market Rent", xlim=(100, 1500))

# Display the plot
plt.show()


'''
Categorical Data
- Data which takes on a limited and fixed number of values
- Normally combined with numeric data
- Examples include:
*   Geography (country, state, region)
*   Gender
*   Ethnicity
*   Blood type
*   Eye color

Categorical Plot Types
Plot types - show each observation
#   The first group includes the Stripplot() and the Swarmplot(), which show all the individual observations on the plot.

Plot types - abstract representations
#   These includes the Boxplot(), Boxenplot() and the Violinplot(), which show an abstract representation of the categorical data.

Plot types - statistical estimates
#   These includes the Barplot(), Pointplot() and Countplot(), which show statistical estimates of the categorical variables.
#   The Barplot() and Pointplot() contain useful summaries of data.
#   Countplot() shows the number of instances of each observation.

Plots of each observation - stripplot() 
This shows every observation in the datset.
e.g
sns.stripplot(data=df, y='column_name', x='column_name', jitter=True)

Plots of each observation - swarmplot() 
This uses a complex algorithm to place the observations in a manner where they do not overlap.
e.g
sns.swarmplot(data=df, y='column_name', x='column_name')

Plots of each observation - boxplot()
This shows several measures related to the distribution of the data, including the median, upper and lower quartiles, as well as outliers.
e.g
sns.boxplot(data=df, y='column_name', x='column_name')

Plots of each observation - violinplot()
This is a combination of kdeplot & boxplot and can be suitable for providing an alternative view of the distribution of data.
e.g
sns.violinplot(data=df, y='column_name', x='column_name')         

Plots of each observation - boxenplot()
This is an enhanced box plot and can scale more effectively to large datasets. It is a hybrid between the violin & boxplot
e.g
sns.boxenplot(data=df, y='column_name', x='column_name')        

Plots of each observation - barplot()
This shows an estimate of the value as well as a confidence interval.
e.g
sns.barplot(data=df, y='column_name', x='column_name', hue='column_name')

Plots of each observation - pointplot()
This is similar to the barplot() in that it shows a summary measure and confidence interval
e.g
sns.pointplot(data=df, y='column_name', x='column_name', hue='column_name')

Plots of each observation - countplot()
They display the number of instances of each variable
e.g
sns.countplot(data=df, y='column_name', x='column_name', hue='column_name')
'''

# Create the stripplot
sns.stripplot(data=df, x='Award_Amount', y='Model Selected', jitter=True)

plt.show()

# Create and display a swarmplot with hue set to the Region
sns.swarmplot(data=df, x='Award_Amount', y='Model Selected', hue='Region')

plt.show()


# Create a boxplot
sns.boxplot(data=df, x='Award_Amount', y='Model Selected')

plt.show()
plt.clf()

# Create a violinplot with the husl palette
sns.violinplot(data=df, x='Award_Amount', y='Model Selected', palette='husl')

plt.show()
plt.clf()

# Create a boxenplot with the Paired palette and the Region column as the hue
sns.boxenplot(data=df, x='Award_Amount', y='Model Selected',
              palette='Paired', hue='Region')

plt.show()
plt.clf()


# Show a countplot with the number of models used with each region a different color
sns.countplot(data=df, y="Model Selected", hue="Region")

plt.show()
plt.clf()

# Create a pointplot and include the capsize in order to show caps on the error bars
sns.pointplot(data=df, y='Award_Amount', x='Model Selected', capsize=.1)

plt.show()
plt.clf()

# Create a barplot with each Region shown as a different color
sns.barplot(data=df, y='Award_Amount', x='Model Selected', hue='Region')

plt.show()
plt.clf()


'''
Regression Plots
Plotting with regplot()
e.g
sns.regplot(data=df, x='column_name', y='column_name', marker='+')

Evaluating regression with residplot()
- A residual plot is useful for evaluating the fit of a model
- It is a very useful plot for understanding the appropriateness of a regression model.
- Seaborn supports through residplot function
e.g
sns.residplot(data=df, x='column_name', y='column_name')

Polynomial regression
- Seaborn supports polynomial regression using the order parameter
If a value greater than 1 is passed to the order parameter of regplot(), then Seaborn will attempt a polynomial fit using underlying NumPy functions.
e.g
sns.regplot(data=df, x='column_name', y='column_name', order=2)

Residplot with polynomial regression
e.g
sns.residplot(data=df, x='column_name', y='column_name', order=2)

Categorical values
e.g
sns.regplot(data=df, x='column_name', y='column_name', x_jitter=.1, order=2)

Estimators
- In some cases, an x_estimator can be useful for highlighting trends
e.g
sns.regplot(data=df, x='column_name', y='column_name', x_estimator=np.mean, order=2)

Binning the data
- x_bins can be used to divide the data into discrete bins
- The regression line is still fit against all the data 
e.g
sns.regplot(data=df, x='column_name', y='column_name', x_bins=4)
'''

# Display a regression plot for Tuition
sns.regplot(data=df, y='Tuition', x='SAT_AVG_ALL', marker='^', color='g')

plt.show()
plt.clf()

# Display the residual plot
sns.residplot(data=df, y='Tuition', x='SAT_AVG_ALL', color='g')

plt.show()
plt.clf()


# Plot a regression plot of Tuition and the Percentage of Pell Grants
sns.regplot(data=df, y='Tuition', x='PCTPELL')

plt.show()
plt.clf()

# Create another plot that estimates the tuition by PCTPELL
sns.regplot(data=df, y='Tuition', x='PCTPELL', x_bins=5)

plt.show()
plt.clf()

# The final plot should include a line using a 2nd order polynomial
sns.regplot(data=df, y='Tuition', x='PCTPELL', x_bins=5, order=2)

plt.show()
plt.clf()


# Create a scatter plot by disabling the regression line
sns.regplot(data=df, y='Tuition', x='UG', fit_reg=False)

plt.show()
plt.clf()

# Create a scatter plot and bin the data into 5 bins
sns.regplot(data=df, y='Tuition', x='UG', x_bins=5)

plt.show()
plt.clf()

# Create a regplot and bin the data into 8 bins
sns.regplot(data=df, y='Tuition', x='UG', x_bins=8)

plt.show()
plt.clf()


'''
Matrix plots
The heatmap is the most common type of matrix plot and can be easily created by Seaborn and can be useful for quickly seeing trends in a dataset.

Getting data in the right format
- Seaborn heatmap() function requires data to be in a grid format
- Pandas crosstab() is frequently used to manipulate the data
e.g
pd.crosstab(df['column_name'], df['column_name'],  values=df['column_name'], aggfunc='mean').round(0)

Build a heatmap
e.g
sns.heatmap(pd.crosstab(df['column_name'], df['column_name'], values=df['column_name'], aggfunc='mean'))

Customize a heatmap
e.g
sns.heatmap(df_crosstab, annot=True, fmt='d', cmap='YlGnBu', cbar=False, linewidths=.5)

*   annot - To turn on / off annotations in the individual cells.
*   fmt - Ensures that the results are displayed as integers
*   cmap - To change the shading (YlGnBu = Yellow Green Blue)
*   cbar - To turn on / off the color bar
*   linewidths - To put small spacing between the cells so that the values are simpler to view

Centering a heatmap
- Seaborn support centering the heatmap colors on a specific value
e.g
sns.heatmap(df_crosstab,  annot=True, fmt='d', cmap='YlGnBu', cbar=True, center=df_crosstab.loc[9, 6])

PLotting a correlation matrix
- Pandas corr function calculates correlations between columns in a dataframe
- The output can be converted to a heatmap with seaborn
e.g
cols = ['A', 'B', 'C', 'D']
sns.heatmap(df[cols].corr(), cmap='YlGnBu')
'''

# Create a crosstab table of the data
pd_crosstab = pd.crosstab(df["Group"], df["YEAR"])
print(pd_crosstab)

# Plot a heatmap of the table
sns.heatmap(pd_crosstab)

# Rotate tick marks for visibility
plt.yticks(rotation=0)
plt.xticks(rotation=90)

plt.show()


# Create the crosstab DataFrame
pd_crosstab = pd.crosstab(df["Group"], df["YEAR"])

# Plot a heatmap of the table with no color bar and using the BuGn palette
sns.heatmap(pd_crosstab, cbar=False, cmap="BuGn", linewidths=0.3)

# Rotate tick marks for visibility
plt.yticks(rotation=0)
plt.xticks(rotation=90)

# Show the plot
plt.show()
plt.clf()


''' Creating Plots on Data Aware Grids '''

'''
Using FacetGrid, catplot and lmplot
Tidy data
- Seaborn's grid plots require data in 'tidy format'
- One observation per row of data

FacetGrid
- THe FacetGrid is foundational for many data aware grids
- It allows the user to control how data is distributed across columns, rows and hue
- One a FacetGrid is created, the plot type must be mapped to the grid
e.g
g = sns.FacetGrid(df, col='column_name')
g.map(sns.boxplot, 'column_name', order = ['A', 'B', 'C', 'D'])

Catplot()
- The catplot is a simpler way to use a FacetGrid for categorical data
- Combines the facetting and mapping process into 1 function
e.g
sns.catplot(x='column_name', data= df, col='column_name', kind='box')

FacetGrid for regression
- FacetGrid() can also be used for scatter or regrssion plots
e.g
g = sns.FacetGrid(df, col='column_name')
g.map(plt.scatter, 'column_name', 'column_name')

lmplot
- lmplot plots scatter and regression plots on a FacetGrid
e.g
sns.lmplot(data= df, x='column_name', y='column_name', col='column_name', fit_reg=False)

lmplot with regression
e.g
sns.lmplot(data= df, x='column_name', y='column_name', col='column_name', row='column_name')
'''

# Create FacetGrid with Degree_Type and specify the order of the rows using row_order
g2 = sns.FacetGrid(df, row="Degree_Type", row_order=[
                   'Graduate', 'Bachelors', 'Associates', 'Certificate'])

# Map a pointplot of SAT_AVG_ALL onto the grid
g2.map(sns.pointplot, 'SAT_AVG_ALL')

# Show the plot
plt.show()
plt.clf()


# Create a factor plot that contains boxplots of Tuition values
sns.catplot(data=df, x='Tuition', kind='box', row='Degree_Type')

plt.show()
plt.clf()

# Create a facetted pointplot of Average SAT_AVG_ALL scores facetted by Degree Type
sns.catplot(data=df, x='SAT_AVG_ALL', kind='point', row='Degree_Type',
            row_order=['Graduate', 'Bachelors', 'Associates', 'Certificate'])

plt.show()
plt.clf()


# Create a FacetGrid varying by column and columns ordered with the degree_order variable
g = sns.FacetGrid(df, col="Degree_Type", col_order=degree_ord)

# Map a scatter plot of Undergrad Population compared to PCTPELL
g.map(plt.scatter, 'UG', 'PCTPELL')

plt.show()
plt.clf()

# Re-create the previous plot as an lmplot
sns.lmplot(data=df, x='UG', y='PCTPELL',
           col="Degree_Type", col_order=degree_ord)

plt.show()
plt.clf()

# Create an lmplot that has a column for Ownership, a row for Degree_Type and hue based on the WOMENONLY column
sns.lmplot(data=df, x='SAT_AVG_ALL', y='Tuition', col="Ownership", row='Degree_Type',
           row_order=['Graduate', 'Bachelors'], hue='WOMENONLY', col_order=inst_ord)

plt.show()
plt.clf()


'''
Using PairGrid and pairplot

Pairwise relationships
- PairGrid shows pairwise relationships between data elements

Creating a PairGrid
e.g
g = sns.PairGrid(df, vars=['column_name', 'column_name'])
g = g.map(sns.scatterplot)

Customizing the PairGrid diagonals
g = sns.PairGrid(df, vars=['column_name', 'column_name'])
g = g.map_diag(sns.histplot)
g = g.map_offdiag(sns.scatterplot)

Pairplot
- Pairplot is a shortcut for the PairGrid
e.g
sns.pairplot(df, vars=['column_name', 'column_name'], kind='reg', diag_kind='hist')

Customizing a pairplot
e.g
sns.pairplot(df.query('column_name < int'), vars=['column_name', 'column_name', 'column_name'], hue = 'column_name', palette='husl', plot_kws={'alpha': 0.5})
'''

# Create a PairGrid with a scatter plot for fatal_collisions and premiums
g = sns.PairGrid(df, vars=["fatal_collisions", "premiums"])
g2 = g.map(sns.scatterplot)

plt.show()
plt.clf()

# Create the same PairGrid but map a histogram on the diag
g = sns.PairGrid(df, vars=["fatal_collisions", "premiums"])
g2 = g.map_diag(sns.histplot)
g3 = g2.map_offdiag(sns.scatterplot)

plt.show()
plt.clf()


# Create a pairwise plot of the variables using a scatter plot
sns.pairplot(data=df, vars=["fatal_collisions", "premiums"], kind='scatter')

plt.show()
plt.clf()

# Plot the same data but use a different color palette and color code by Region
sns.pairplot(data=df, vars=["fatal_collisions", "premiums"],
             kind='scatter', hue='Region', palette='RdBu', diag_kws={'alpha': .5})

plt.show()
plt.clf()


# Build a pairplot with different x and y variables
sns.pairplot(data=df, x_vars=["fatal_collisions_speeding", "fatal_collisions_alc"], y_vars=[
             'premiums', 'insurance_losses'], kind='scatter', hue='Region', palette='husl')

plt.show()
plt.clf()

# plot relationships between insurance_losses and premiums
sns.pairplot(data=df, vars=["insurance_losses", "premiums"],
             kind='reg', palette='BrBG', diag_kind='kde', hue='Region')

plt.show()
plt.clf()


'''
Using JointGrid and jointplot
- A JointGrid() allows us to compare the distribution of data between two variables.
- It makes use of Scatterplot, Regression lines, as well as Histograms, Distribution plots and Kernel Density estimates to give us insights into our data.

Basic JointGrid
e.g
g = sns.JointGrid(data=df, x='column_name', y='column_name')
g.plot(sns.regplot, sns.histplot)

Advanced JointGrid
e.g
g = sns.JointGrid(data=df, x='column_name', y='column_name')
g = g.plot_joint(sns.kdeplot)
g = g.plot_marginals(sns.kdeplot, shade=True)

Jointplot()
They are easier to use but provides fewer customization capabilities.
e.g
sns.jointplot(data=df, x='column_name', y='column_name', kind='hex')

Customizing a jointplot
supports:
- scatter plot
- hex plot
- residual plot
- regression line
- kde plot
- overlay plot (* to enhance the final output)
e.g
g = (sns.jointplot(x='column_name', y='column_name', kind='scatter', xlim=(0, 25000), data=df.query(' column_name < int & column_name == 'str' '))
.plot_joint(sns.kdeplot))
'''

# Build a JointGrid comparing humidity and total_rentals
sns.set_style("whitegrid")
g = sns.JointGrid(x="hum", y="total_rentals", data=df, xlim=(0.1, 1.0))

g.plot(sns.regplot, sns.histplot)

plt.show()
plt.clf()

# Create a jointplot similar to the JointGrid
sns.jointplot(x="hum", y="total_rentals", kind='reg', data=df)

plt.show()
plt.clf()


# Plot temp vs. total_rentals as a regression plot
sns.jointplot(x="temp", y="total_rentals", kind='reg',
              data=df, order=2, xlim=(0, 1))

plt.show()
plt.clf()

# Plot a jointplot showing the residuals
sns.jointplot(x="temp", y="total_rentals", kind='resid', data=df, order=2)

plt.show()
plt.clf()


# Create a jointplot of temp vs. casual riders
# Include a kdeplot over the scatter plot
g = sns.jointplot(x="temp", y="casual", kind='scatter',
                  data=df, marginal_kws=dict(bins=10))
g.plot_joint(sns.kdeplot)

plt.show()
plt.clf()

# Replicate the above plot but only for registered riders
g = sns.jointplot(x="temp", y="registered", kind='scatter',
                  data=df, marginal_kws=dict(bins=10))
g.plot_joint(sns.kdeplot)

plt.show()
plt.clf()


'''
Selecting Seaborn Plots
How they are built :
pairplot    -   PairGrid

jointplot   -   JointGrid

displot -   FacetGrid
lmplot  -   FacetGrid
catplot -   FacetGrid

kdeplot -   Matplotlib
rugplot -   Matplotlib
residplot   -   Matplotlib
histplot    -   Matplotlib
ecdfplot    -   Matplotlib
regplot -   Matplotlib
pointplot   -   Matplotlib
barplot -   Matplotlib
countplot   -   Matplotlib
boxplot -   Matplotlib
violinplot  -   Matplotlib
boxenplot   -   Matplotlib
swarmplot   -   Matplotlib
stripplot   -   Matplotlib
heatmap -   Matplotlib
palplot -   Matplotlib 

Univariate Distribution Analysis
- This is used in analyzing numerical data distributions.
- displot() is best place to start for this analysis
- rugplot(), kdeplot() and ecdfplot() can be useful alternatives

Regression Analysis
- This shows the relationship between 2 variables
- lmplot() performs regression analysis and its also the best function to determine the linear relationships between data and supports facetting .
- scatterplot(), regplot(), residplot() and FacetGrid() can be useful alternatives

Categorical Plots
- boxplot() or violinplot() are best to examine the distribution of the variables.
- Then, follow up with statistical estimation plots such as the pointplot(), barplot() or countplot()
- Explore data and facet across rows or columns with catplot()

pairplot() and jointplot()
They are most useful after preliminary analysis of regressions or distributions of the data.
- Perform regression analysis with lmplot
- Analyze distributions with displot
'''
