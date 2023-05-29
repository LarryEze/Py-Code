# Importing the course packages
import pandas as pd
import numpy as np
from numpy import median
import matplotlib.pyplot as plt
import seaborn as sns

# Importing the course datasets
country_data = pd.read_csv(
    'Introduction to Data Visualization with Seaborn\countries-of-the-world.csv', decimal=",")
mpg = pd.read_csv('Introduction to Data Visualization with Seaborn\mpg.csv')
student_data = pd.read_csv(
    'Introduction to Data Visualization with Seaborn\student-alcohol-consumption.csv', index_col=0)
survey_data = pd.read_csv(
    'Introduction to Data Visualization with Seaborn\young-people-survey-responses.csv', index_col=0)


''' INTRODUCTION TO SEABORN '''

'''
Introduction to Seaborn
Getting started
import seaborn as sns
import matplotlib.pyplot as plt

e.g (Scatter plot)
height = [62, 64, 69, 75, 66]
weight = [120, 136, 148, 175, 137]
sns.scatterplot(x = height, y = weight)
plt.show()

e.g (Create a count plot)
gender = ['Female', 'Female', 'Male', 'Male', 'Male']
sns.countplot(x = gender)
plt.show()
'''

gdp = country_data['GDP ($ per capita)']
phones = country_data['Phones (per 1000)']
percent_literate = country_data['Literacy (%)']

# Create scatter plot with GDP on the x-axis and number of phones on the y-axis
sns.scatterplot(x=gdp, y=phones)

# Show plot
plt.show()

# Change this scatter plot to have percent literate on the y-axis
sns.scatterplot(x=gdp, y=percent_literate)

# Show plot
plt.show()


region = country_data['Region']

# Import Matplotlib and Seaborn

# Create count plot with region on the y-axis
sns.countplot(y=region)

# Show plot
plt.show()


'''
Using pandas with Seaborn
Working with DataFrames
import pandas as pd
df = pd.read_csv('masculinity.csv')

Using DataFrames with countplot()
import matplotlib.pyplot as plt
import seaborn as sns
df
e.g (Using DataFrames with countplot())
sns.countplot(x='column_name', data = df_name)
plt.show()
'''

csv_filepath = 'Introduction to Data Visualization with Seaborn\young-people-survey-responses.csv'

# Import Matplotlib, pandas, and Seaborn

# Create a DataFrame from csv file
df = pd.read_csv(csv_filepath)

# Create a count plot with "Spiders" on the x-axis
sns.countplot(x='Spiders', data=df)

# Display the plot
plt.show()


'''
Adding a third variable with hue
Tips dataset
import pandas as pd
matplotlib.pyplot as plt
import seaborn as sns
tips = sns.load_dataset('tips')

e.g (A basic scatter plot)
sns.scatterplot(x='column_name', y='column_name', data=df_name, hue ='column_name', hue_order=['Yes', 'No'])
plt.show()

Specifying hue colors
hue_colors = {'Yes': 'black', 'No': 'red'}
sns.scatterplot(x='column_name', y='column_name', data=df_name, hue='column_name', palette=hue_colors)
plt.show()
'''

# Import Matplotlib and Seaborn

# Change the legend order in the scatter plot
sns.scatterplot(x="absences", y="G3", data=student_data,
                hue="location", hue_order=['Rural', 'Urban'])

# Show plot
plt.show()


# Import Matplotlib and Seaborn

# Create a dictionary mapping subgroup values to colors
palette_colors = {'Rural': "green", 'Urban': "blue"}

# Create a count plot of school with location subgroups
sns.countplot(x='school', data=student_data,
              hue='location', palette=palette_colors)

# Display plot
plt.show()


''' VISUALIZING TWO QUANTITATIVE VARIABLES '''

'''
Introduction to relational plots and subplots
Introducing relplot() - relational plot
They are useful for Scatter and Line plots

scatterplot() vs relplot()
Using scatterplot()
import seaborn as sns
import matplotlib.pyplot as plt
sns.scatterplot(x='column_name', y='column_name', data=df_name)
plt.show()
###
Using relplot()
import seaborn as sns
import matplotlib.pyplot as plt
sns.relplot(x='column_name', y='column_name', data=df_name, kind='scatter')
plt.show()

Subplots in columns
sns.relplot(x='column_name', y='column_name', data=df_name, kind='scatter', col='column_name')
plt.show()

Subplots in rows
sns.relplot(x='column_name', y='column_name', data=df_name, kind='scatter', row='column_name')
plt.show()

Subplots in rows and columns
sns.relplot(x='column_name', y='column_name', data=df_name, kind='scatter', col='column_name', row='column_name')
plt.show()

Wrapping columns
sns.relplot(x='column_name', y='column_name', data=df_name, kind='scatter', col='column_name', col_wrap=2)
plt.show()

Ordering columns
sns.relplot(x='column_name', y='column_name', data=df_name, kind='scatter', col='column_name', col_wrap=2, col_order=['A', 'B', 'C'])
plt.show()
'''

# Change this scatter plot to arrange the plots in rows
sns.relplot(x="absences", y="G3", data=student_data,
            kind="scatter", row="study_time")

# Show plot
plt.show()


# Adjust further to add subplots
sns.relplot(x="G1", y="G3", data=student_data, kind="scatter", col="schoolsup",
            col_order=["yes", "no"], row='famsup', row_order=["yes", "no"])

# Show plot
plt.show()


'''
Customizing scatter plots
Subgroups with point size
import seaborn as sns
import matplotlib.pyplot as plt
sns.relplot(x='column_name', y='column_name', data=df_name, kind='scatter', size='column_name')
plt.show()

point size and hue
sns.relplot(x='column_name', y='column_name', data=df_name, kind='scatter', size='column_name', hue='column_name')
plt.show()

subgroups with point style
sns.relplot(x='column_name', y='column_name', data=df_name, kind='scatter', hue='column_name', style='column_name')
plt.show()

Changing point transparency
sns.relplot(x='column_name', y='column_name', data=df_name, kind='scatter', alpha=0.4)
plt.show()
'''

# Import Matplotlib and Seaborn

# Create scatter plot of horsepower vs. mpg
sns.relplot(x="horsepower", y="mpg", data=mpg,
            kind="scatter", size="cylinders", hue='cylinders')

# Show plot
plt.show()


# Import Matplotlib and Seaborn

# Create a scatter plot of acceleration vs. mpg
sns.relplot(x='acceleration', y='mpg', data=mpg,
            kind='scatter', style='origin', hue='origin')

# Show plot
plt.show()


'''
Introduction to line plots
In seaborn, there are 2 types of relational plots: scatter plots and line plots
Scatter plots - Each plot point is an independent observation
Line plots - Each plot point represents the same 'thing' typically tracked over time

Line plot
import matplotlib.pyplot as plt
import seaborn as sns
sns.relplot(x='column_name', y='column_name', data=df_name, kind='line')
plt.show()

subgroups by location
sns.relplot(x='column_name', y='column_name', data=df_name, kind='line', style='column_name', hue='column_name')
plt.show()

Adding markers
sns.relplot(x='column_name', y='column_name', data=df_name, kind='line', style='column_name', hue='column_name', markers=True)
plt.show()

Turning off line style
sns.relplot(x='column_name', y='column_name', data=df_name, kind='line', style='column_name', hue='column_name', markers=True, dashes=False)
plt.show()

#   If a line plot is given multiple observations per x-value, it will aggregate them into a single summary measure (mean)
#   Seaborn will automatically calculate a confidence interval for the mean in a line plot displayed by a shaded region

Replacing confidence interval with standard deviation
sns.relplot(x='column_name', y='column_name', data=df_name, kind='line', ci='sd')
plt.show()

Turning off confidence interval
sns.relplot(x='column_name', y='column_name', data=df_name, kind='line', ci=None)
plt.show()
'''

# Import Matplotlib and Seaborn

# Create line plot
sns.relplot(x='model_year', y='mpg', data=mpg, kind='line')

# Show plot
plt.show()


# Make the shaded area show the standard deviation
sns.relplot(x="model_year", y="mpg", data=mpg, kind="line", ci='sd')

# Show plot
plt.show()


# Import Matplotlib and Seaborn

# Add markers and make each line have the same style
sns.relplot(x="model_year", y="horsepower",
            data=mpg, kind="line",
            ci=None, style="origin",
            hue="origin", markers=True, dashes=False)

# Show plot
plt.show()


''' Visualizing a Categorical and a Quantitative Variable '''

'''
Count plots and bar plots
#   Count plots and Bar plots are 2 types of visualizations that seaborn calls 'categorical plots'.

catplot - Used to create categorical plots

countplot() vs catplot()
Using countplot()
import matplotlib.pyplot as plt
import seaborn as sns
sns.countplot(x='column_name', data=df_name)
plt.show()
###
Using catplot()
sns.catplot(x='column_name', data=df_name, kind='count')
plt.show()

Changing the order
category_order = ['A', 'B', 'C']
sns.catplot(x='column_name', data=df_name, kind='count', order = category_order)
plt.show()

Bar plots
Displays mean of quantitative variable per category
sns.catplot(x='column_name', y='column_name', data=df_name, kind='bar')
plt.show()

Turning off confidence intervals
sns.catplot(x='column_name', y='column_name', data=df_name, kind='count', ci=None)
plt.show()
'''

# Assuming survey_data is your DataFrame with an 'Age' column
categories = ['Less than 21', '21+']
bins = [0, 21, float('inf')]

# Create the 'Age Category' column based on the 'Age' column
survey_data['Age Category'] = pd.cut(
    survey_data['Age'], bins=bins, labels=categories, right=False)

# Create count plot of internet usage
sns.catplot(x='Internet usage', data=survey_data, kind='count')

# Show plot
plt.show()

# Change the orientation of the plot
sns.catplot(y="Internet usage", data=survey_data, kind="count")

# Show plot
plt.show()

# Separate into column subplots based on age category
sns.catplot(y="Internet usage", data=survey_data,
            kind="count", col='Age Category')

# Show plot
plt.show()


survey_data['Interested in Math'] = survey_data['Mathematics'] > 3

# Create a bar plot of interest in math, separated by gender
sns.catplot(x='Gender', y='Interested in Math', data=survey_data, kind='bar')

# Show plot
plt.show()


# List of categories from lowest to highest
category_order = ["<2 hours",
                  "2 to 5 hours",
                  "5 to 10 hours",
                  ">10 hours"]

# Turn off the confidence intervals
sns.catplot(x="study_time", y="G3",
            data=student_data,
            kind="bar",
            order=category_order, ci=None)

# Show plot
plt.show()


'''
Box plots
#   This shows the distribution of quantitative data
#   The coloured box shows the 25th to 75th percentile
#   It also shows the median, spread, skewness and outliers

How to create a box plot
import matplotlib.pyplot as plt
import seaborn as sns
g = sns.catplot(x='column_name', y='column_name', data=df_name, kind='box')
plt.show()

Change the order of categories
g = sns.catplot(x='column_name', y='column_name', data=df_name, kind='box', order=['A', 'B'])
plt.show()

Omitting the outliers using 'sym'
g = sns.catplot(x='column_name', y='column_name', data=df_name, kind='box', sym="")
plt.show()

Changing the whiskers using 'whis'
- By default, the whiskers extend to 1.5 times the interquartile range
- Make them extend to 2.0 * IQR: whis = 2.0
- Show the 5th and 95th percentiles: whis = [5, 95]
- Show min and max values: whis = [0, 100]
e.g
g = sns.catplot(x='column_name', y='column_name', data=df_name, kind='box', whis = [0, 100])
plt.show()
'''

# Specify the category ordering
study_time_order = ["<2 hours", "2 to 5 hours",
                    "5 to 10 hours", ">10 hours"]

# Create a box plot and set the order of the categories
sns.catplot(x='study_time', y='G3', data=student_data,
            kind='box', order=study_time_order)

# Show plot
plt.show()


# Create a box plot with subgroups and omit the outliers
sns.catplot(x='internet', y='G3', data=student_data,
            kind='box', hue='location', sym='')

# Show plot
plt.show()


# Set the whiskers to 0.5 * IQR
sns.catplot(x="romantic", y="G3",
            data=student_data,
            kind="box", whis=0.5)

# Show plot
plt.show()

# Extend the whiskers to the 5th and 95th percentile
sns.catplot(x="romantic", y="G3",
            data=student_data,
            kind="box",
            whis=[5, 95])

# Show plot
plt.show()

# Set the whiskers at the min and max values
sns.catplot(x="romantic", y="G3",
            data=student_data,
            kind="box",
            whis=[0, 100])

# Show plot
plt.show()


'''
Point plots
#   They show the mean of quantitative variable for the observations in each category, plotted as a single point.
#   Vertical lines shows 95% confidence intervals
Difference between a Line plot and Point plot 
-   Line plot has quantitative variable (usually time) on x-axis
-   Point plot has categorical variable on x-axis

Creating a point plot
import matplotlib.pyplot as plt
import seaborn as sns
sns.catplot(x='column_name', y='column_name', data=df_name, hue='column_name', kind='point')
plt.show()

Disconnecting the points
sns.catplot(x='column_name', y='column_name', data=df_name, hue='column_name', kind='point', join=False)
plt.show()

Displaying the median
import matplotlib.pyplot as plt
import seaborn as sns
from numpy import median
sns.catplot(x='column_name', y='column_name', data=df_name, kind='point', estimator=median)
plt.show()

Customizing the confidence intervals
sns.catplot(x='column_name', y='column_name', data=df_name, kind='point', capsize=0.2)
plt.show()

Turning off confidence intervals
sns.catplot(x='column_name', y='column_name', data=df_name, kind='point', ci=None)
plt.show()
'''
# Remove the lines joining the points
sns.catplot(x="famrel", y="absences",
            data=student_data,
            kind="point",
            capsize=0.2, join=False)

# Show plot
plt.show()


# Import median function from numpy

# Plot the median number of absences instead of the mean
sns.catplot(x="romantic", y="absences",
            data=student_data,
            kind="point",
            hue="school",
            ci=None, estimator=median)

# Show plot
plt.show()


''' Customizing Seaborn Plots '''

'''
Changing plot style and color

Changing the figure style
-   Figure 'style' includes background and axes
-   5 Preset options: 'white', 'dark', 'whitegrid', 'darkgrid', 'ticks'
-   sns.set_style()
-   Default figure style ('white')

Figure style: 'whitegrid'
e.g
sns.set_style('whitegrid')
sns.catplot(x='column_name', y='column_name', data=df_name, hue='column_name', kind='point')
plt.show()

Changing the palette
-   Figure 'palette' changes the color of the main elements of the plot
-   sns.set_palette()
-   Use preset palettes or create a custom palette
Diverging palettes
RdBu - Red to Blue
PRGn - Purple to Green
RdBu_r - Red to Blue in reverse
PRGn_r - Purple to Green in reverse
e.g
sns.set_palette('RdBu')
category_order = ['A', 'B', 'C']
sns.catplot(x='column_name', data=df_name, kind='count', , order=category_order)
plt.show()

Sequential palettes
Colors move from light to dark values and are great for emphasizing a variable on a continuous scale.
'Greys'
'Blues'
'PuRd'
'GnBu'

Custom palettes
custom_palette = ['red', 'green', 'blue', ,yellow']
sns.set_palette(custom_palette)

Changing the scale
-   Figure 'context' changes the scale of the plot elements and labels
-   sns.set_context()
-   Smallest to largest: 'paper', 'notebook', 'talk', 'poster'
-   default context is 'paper'
e.g
Larger context: 'talk'
sns.set_context('talk')
sns.catplot(x='column_name', y='column_name', data=df_name, hue='column_name', kind='point')
plt.show()
'''

# Change the color palette to "RdBu"
sns.set_style("whitegrid")
sns.set_palette("RdBu")

# Create a count plot of survey responses
category_order = ["Never", "Rarely", "Sometimes",
                  "Often", "Always"]

sns.catplot(x="Parents Advice",
            data=survey_data,
            kind="count",
            order=category_order)

# Show plot
plt.show()


# Change the context to "poster"
sns.set_context("poster")

# Create bar plot
sns.catplot(x="Number of Siblings", y="Feels Lonely",
            data=survey_data, kind="bar")

# Show plot
plt.show()


# Set the style to "darkgrid"
sns.set_style('darkgrid')

# Set a custom color palette
sns.set_palette(['#39A7D0', '#36ADA4'])

# Create the box plot of age distribution by gender
sns.catplot(x="Gender", y="Age",
            data=survey_data, kind="box")

# Show plot
plt.show()


'''
Adding titles and labels: Part 1

FacetGrid vs AxesSubplot objects
Seaborn plots create two different types of objects: FacetGrid and AxesSubplot

To figure out the type of object
g = sns.scatterplot(x='column_name', y='column_name', data=df_name)
type(g)

(1) Object Type - FacetGrid
Plot Types - relplot(), catplot()
Characteristics - Can create subplots

(2) Object Type - AxesSubplot
Plot Types - scatterplot(), countplot() etc
Characteristics - Only creates a single plot

Adding a title to FacetGrid
g = sns.catplot(x='column_name', y='column_name', data=df_name, kind='box')
g.fig.suptitle('New Title', y=1.03)
plt.show()
'''

# Create scatter plot
g = sns.relplot(x="weight",
                y="horsepower",
                data=mpg,
                kind="scatter")

# Identify plot type
type_of_g = type(g)

# Print type
print(type_of_g)


# Create scatter plot
g = sns.relplot(x="weight",
                y="horsepower",
                data=mpg,
                kind="scatter")

# Add a title "Car Weight vs. Horsepower"
g.fig.suptitle("Car Weight vs. Horsepower")

# Show plot
plt.show()


'''
Adding Titles and labels: Part 2

Adding a title to AxesSubplot
e.g
g = sns.boxplot(x='column_name', y='column_name', data=df_name)
g.set_title('New Title', y=1.03)
plt.show()

Adding a title to subplots
e.g
g = sns.catplot(x='column_name', y='column_name', data=df_name, kind='box', col='column_name')
g.fig.suptitle('New Title', y=1.03)
g.set_titles('This is {col_name}')

Adding axis labels
e.g
g = sns.catplot(x='column_name', y='column_name', data=df_name, kind='box')
g.set(xlabel='New X Label', ylabel='New Y Label')
plt.show()

Rotating x-axis tick labels
e.g
g = sns.catplot(x='column_name', y='column_name', data=df_name, kind='box')
plt.xticks(rotation=90)
plt.show()
'''

mpg_mean = mpg[['model_year', 'origin', 'mpg']]

# Create line plot
g = sns.lineplot(x="model_year", y="mpg_mean",
                 data=mpg_mean,
                 hue="origin")

# Add a title "Average MPG Over Time"
g.set_title("Average MPG Over Time")

# Add x-axis and y-axis labels
g.set(xlabel="Car Model Year", ylabel="Average MPG")

# Show plot
plt.show()


# Create point plot
sns.catplot(x="origin",
            y="acceleration",
            data=mpg,
            kind="point",
            join=False,
            capsize=0.1)

# Rotate x-tick labels
plt.xticks(rotation=90)

# Show plot
plt.show()


# Set palette to "Blues"
sns.set_palette('Blues')

# Adjust to add subgroups based on "Interested in Pets"
g = sns.catplot(x="Gender",
                y="Age", data=survey_data,
                kind="box", hue='Interested in Pets')

# Set title to "Age of Those Interested in Pets vs. Not"
g.fig.suptitle('Age of Those Interested in Pets vs. Not')

# Show plot
plt.show()


# Set the figure style to "dark"
sns.set_style('dark')

# Adjust to add subplots per gender
g = sns.catplot(x="Village - town", y="Likes Techno",
                data=survey_data, kind="bar",
                col='Gender')

# Add title and axis labels
g.fig.suptitle("Percentage of Young People Who Like Techno", y=1.02)
g.set(xlabel="Location of Residence",
      ylabel="% Who Like Techno")

# Show plot
plt.show()
