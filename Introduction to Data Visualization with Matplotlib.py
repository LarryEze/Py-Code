# Importing the course packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Importing the course datasets
climate_change = pd.read_csv(
    'Introduction to Data Visualization with Matplotlib\climate_change.csv', parse_dates=["date"], index_col="date")
medals = pd.read_csv(
    'Introduction to Data Visualization with Matplotlib\medals_by_country_2016.csv', index_col=0)
summer_2016 = pd.read_csv(
    'Introduction to Data Visualization with Matplotlib\summer2016.csv')
austin_weather = pd.read_csv(
    'Introduction to Data Visualization with Matplotlib/austin_weather.csv', index_col="DATE")
weather = pd.read_csv(
    'Introduction to Data Visualization with Matplotlib\seattle_weather.csv', index_col="DATE")

# Some pre-processing on the weather datasets, including adding a month column
seattle_weather = weather[weather["STATION"] == "USW00094290"]
month = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
         "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
seattle_weather["MONTH"] = month
austin_weather["MONTH"] = month


""" INTRODUCTION TO Matplotlib """

"""
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
plt.show()
figure - container that holds everything seen on the page
axis - container that holds the data

e.g
ax.plot(table_name["column_name"], table_name["column_name"])
plt.show()
"""

# Create a Figure and an Axes with plt.subplots
fig, ax = plt.subplots()

# Call the show function to show the result
plt.show()


# Create a Figure and an Axes with plt.subplots
fig, ax = plt.subplots()

# Plot MLY-PRCP-NORMAL from seattle_weather against the MONTH
ax.plot(seattle_weather["MONTH"], seattle_weather["MLY-PRCP-NORMAL"])

# Plot MLY-PRCP-NORMAL from austin_weather against MONTH
ax.plot(austin_weather["MONTH"], austin_weather["MLY-PRCP-NORMAL"])

# Call the show function
plt.show()


""" CUSTOMIZING DATA APPEARANCE """

"""
ax.plot(table_name["column_name"], table_name["column_name"], marker = "o", linestyle = "--", color = "r")
ax.set_xlabel("Time (months)")
ax.set_ylabel("Average temperature (Fahrenheit degrees)")
ax.set_title("Weather in Seattle")
plt.show()

markers
"o" = circle markers
"v" = downward facing triangle marker etc

linestyles
"None" = To remove connecting lines
"--" = dashed connecting lines etc

colours
"r" = red etc
"""

# Plot Seattle data, setting data appearance
ax.plot(
    seattle_weather["MONTH"],
    seattle_weather["MLY-PRCP-NORMAL"],
    color="b",
    marker="o",
    linestyle="--",
)

# Plot Austin data, setting data appearance
ax.plot(
    austin_weather["MONTH"],
    austin_weather["MLY-PRCP-NORMAL"],
    color="r",
    marker="v",
    linestyle="--",
)

# Call show to display the resulting plot
plt.show()


# Plot Seattle and Austin data
ax.plot(seattle_weather["MONTH"], seattle_weather["MLY-PRCP-NORMAL"])
ax.plot(austin_weather["MONTH"], austin_weather["MLY-PRCP-NORMAL"])

# Customize the x-axis label
ax.set_xlabel("Time (months)")

# Customize the y-axis label
ax.set_ylabel("Precipitation (inches)")

# Add the title
ax.set_title("Weather patterns in Austin and Seattle")

# Display the figure
plt.show()


""" SMALL MULTIPLES """

"""
import matplotlib.pyplot as plt

fig, ax = plt.subplots(no of rows, no of columns, sharey = True)

sharey
To ensure all subplots share the same range of y-axis value

ax.shape
This will tell the shape (no of rows and columns) of the subplots

e.g
fig, ax = plt.subplots(3, 2)
This will produce a plot with 3 rows and 2 columns

ax[0,0].plot(table_name["column_name"], table_name["column_name"], color = "b")
plt.show()  # To plot for the first row and column 

fig, ax = plt.subplots(2, 1)
This will produce a plot with 2 rows and 1 column
ax[0].plot(table_name["column_name"], table_name["column_name"], color = "r")
ax[0].plot(table_name["column_name"], table_name["column_name"], linestyle = "--", color = "r")
ax[0].plot(table_name["column_name"], table_name["column_name"], linestyle = "--", color = "r")
ax[1].plot(table_name["column_name"], table_name["column_name"], color = "r")
ax[1].plot(table_name["column_name"], table_name["column_name"], linestyle = "--", color = "r")
ax[1].plot(table_name["column_name"], table_name["column_name"], linestyle = "--", color = "r")
ax[0].set_xlabel("Time (months)")
ax[1].set_ylabel("Average temperature (Fahrenheit degrees)")
ax[1].set_ylabel("Average temperature (Fahrenheit degrees)")
ax.set_title("Weather in Seattle")
plt.show()
"""

# Create a Figure and an array of subplots with 2 rows and 2 columns
fig, ax = plt.subplots(2, 2)

# Addressing the top left Axes as index 0, 0, plot month and Seattle precipitation
ax[0, 0].plot(seattle_weather["MONTH"], seattle_weather["MLY-PRCP-NORMAL"])

# In the top right (index 0,1), plot month and Seattle temperatures
ax[0, 1].plot(seattle_weather["MONTH"], seattle_weather["MLY-TAVG-NORMAL"])

# In the bottom left (1, 0) plot month and Austin precipitations
ax[1, 0].plot(austin_weather["MONTH"], austin_weather["MLY-PRCP-NORMAL"])

# In the bottom right (1, 1) plot month and Austin temperatures
ax[1, 1].plot(austin_weather["MONTH"], austin_weather["MLY-TAVG-NORMAL"])
plt.show()


# Create a figure and an array of axes: 2 rows, 1 column with shared y axis
fig, ax = plt.subplots(2, 1, sharey=True)

# Plot Seattle precipitation data in the top axes
ax[0].plot(seattle_weather["MONTH"],
           seattle_weather["MLY-PRCP-NORMAL"], color="b")

ax[0].plot(
    seattle_weather["MONTH"],
    seattle_weather["MLY-PRCP-25PCTL"],
    color="b",
    linestyle="--",
)

ax[0].plot(
    seattle_weather["MONTH"],
    seattle_weather["MLY-PRCP-75PCTL"],
    color="b",
    linestyle="--",
)

# Plot Austin precipitation data in the bottom axes
ax[1].plot(austin_weather["MONTH"],
           austin_weather["MLY-PRCP-NORMAL"], color="r")

ax[1].plot(
    austin_weather["MONTH"],
    austin_weather["MLY-PRCP-25PCTL"],
    color="r",
    linestyle="--",
)
ax[1].plot(
    austin_weather["MONTH"],
    austin_weather["MLY-PRCP-75PCTL"],
    color="r",
    linestyle="--",
)

plt.show()


""" PLOTTING TIME-SERIES """

"""
import pandas as pd
table_name = pd.read_csv('climate_change.csv', parse_dates=["date"], index_col="date")

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
e.g
ax.plot(table_name.index, table_name['column_name'])
ax.set_xlabel('xlabel_title')
ax.set_ylabel('ylabel_title')
plt.show()

zooming in on a decade
sixties = table_name["1960-01-01":"1969-12-31"]
fig, ax = plt.subplots()
ax.plot(sixties.index, sixties['co2'])
ax.set_xlabel('Time')
ax.set_ylabel('ylabel_title')
plt.show()

zooming in on one year
sixty_nine = table_name["1969-01-01":"1969-12-31"]
fig, ax = plt.subplots()
ax.plot(sixty_nine.index, sixty_nine['co2'])
ax.set_xlabel('Time')
ax.set_ylabel('ylabel_title')
plt.show()
"""

# Read the data from file using read_csv
climate_change = pd.read_csv(
    'Introduction to Data Visualization with Matplotlib\climate_change.csv', parse_dates=["date"], index_col="date"
)

# import matplotlib.pyplot as plt
fig, ax = plt.subplots()

# Add the time-series for "relative_temp" to the plot
ax.plot(climate_change.index, climate_change["relative_temp"])

# Set the x-axis label
ax.set_xlabel("Time")

# Set the y-axis label
ax.set_ylabel("Relative temperature (Celsius)")

# Show the figure
plt.show()


# Use plt.subplots to create fig and ax
fig, ax = plt.subplots()

# Create variable seventies with data from "1970-01-01" to "1979-12-31"
seventies = climate_change["1970-01-01":"1979-12-31"]

# Add the time-series for "co2" data from seventies to the plot
ax.plot(seventies.index, seventies["co2"])

# Show the figure
plt.show()


""" PLOTTING TIME-SERIES WITH DIFFERENT VARIABLES """

""" 
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.plot(table_name.index, table_name['column_name'], color = 'color_name')
ax.set_xlabel('xlabel_title')
ax.set_ylabel('ylabel_title', color = 'color_name')
ax.tick_params('y', color = 'color_name')

ax2 = ax.twinx()
ax.plot(table_name.index, table_name['column_name'], color = 'color_name')
ax.set_ylabel('ylabel_title', color = 'color_name')
ax.tick_params('y', color = 'color_name')
plt.show()

TO DEFINE A FUNCTION
def plot_timeseries(axes, x, y, color, xlabel, ylabel):
axes.plot(x, y, color = color)
axes.set_xlabel(xlabel)
axes.set_ylabel(ylabel, color = color)
axes.tick_params('y', colors = color)

e.g
fig, ax =  plt.subplots()
plot_timeseries(ax, table_name.index, table_name['column_name'], 'color_name', 'xlabel_title', 'ylabel_title')
ax2 = ax.twinx()
plot_timeseries(ax, table_name.index, table_name['column_name'], 'color_name', 'xlabel_title', 'ylabel_title')
plt.show()
"""

# Initialize a Figure and Axes
fig, ax = plt.subplots()

# Plot the CO2 variable in blue
ax.plot(climate_change.index, climate_change["co2"], color="blue")

# Create a twin Axes that shares the x-axis
ax2 = ax.twinx()

# Plot the relative temperature in red
ax2.plot(climate_change.index, climate_change["relative_temp"], color="red")

plt.show()


# Define a function called plot_timeseries
def plot_timeseries(axes, x, y, color, xlabel, ylabel):
    # Plot the inputs x,y in the provided color
    axes.plot(x, y, color=color)

    # Set the x-axis label
    axes.set_xlabel(xlabel)

    # Set the y-axis label
    axes.set_ylabel(ylabel, color=color)

    # Set the colors tick params for y-axis
    axes.tick_params("y", colors=color)


# Initialize a Figure and Axes
fig, ax = plt.subplots()

# Plot the CO2 levels time-series in blue
plot_timeseries(
    ax,
    climate_change.index,
    climate_change["co2"],
    "blue",
    "Time (years)",
    "CO2 levels",
)

# Create a twin Axes object that shares the x-axis
ax2 = ax.twinx()

# Plot the relative temperature data in red
plot_timeseries(
    ax,
    climate_change.index,
    climate_change["relative_temp"],
    "red",
    "Time (years)",
    "Relative temperature (Celsius)",
)

plt.show()


""" ANNOTATING TIME-SERIES DATA """

"""
fig, ax =  plt.subplots()
plot_timeseries(ax, table_name.index, table_name['column_name'], 'color_name', 'xlabel_title', 'ylabel_title')

ax2 = ax.twinx()
plot_timeseries(ax, table_name.index, table_name['column_name'], 'color_name', 'xlabel_title', 'ylabel_title')

ax2.annotate("text_label", xy=(x_co-ordinate, y_co-ordinate))
plt.show()

e.g
ax2.annotate(">1 degree", xy=(pd.Timestamp("2015-10-06"), 1))

Positioning the text
ax2.annotate(">1 degree", xy=(pd.Timestamp("2015-10-06"), 1), xytext=("2008-10-06"), -0.2))

adding arrows to annotation
ax2.annotate(">1 degree", xy=(pd.Timestamp("2015-10-06"), 1), xytext=("2008-10-06"), -0.2), arrowprops={})

customizing arrow properties
ax2.annotate(">1 degree", xy=(pd.Timestamp("2015-10-06"), 1), xytext=("2008-10-06"), -0.2), arrowprops={"arrowstyle":"->", "color":"gray"})
"""

# Initialize a Figure and Axes
fig, ax = plt.subplots()

# Plot the relative temperature data
ax.plot(climate_change.index, climate_change["relative_temp"])

# Annotate the date at which temperatures exceeded 1 degree
ax.annotate(">1 degree", xy=(pd.Timestamp("2015-10-06"), 1))

plt.show()


# Initialize a Figure and Axes
fig, ax = plt.subplots()

# Plot the CO2 levels time-series in blue
plot_timeseries(
    ax,
    climate_change.index,
    climate_change["co2"],
    "blue",
    "Time (years)",
    "CO2 levels",
)

# Create an Axes object that shares the x-axis
ax2 = ax.twinx()

# Plot the relative temperature data in red
plot_timeseries(
    ax2,
    climate_change.index,
    climate_change["relative_temp"],
    "red",
    "Time (years)",
    "Relative temp (Celsius)",
)

# Annotate point with relative temperature >1 degree
ax2.annotate(
    ">1 degree",
    xy=(pd.Timestamp("2015-10-06"), 1),
    xytext=(pd.Timestamp("2008-10-06"), -0.2),
    arrowprops={"arrowstyle": "->", "color": "gray"},
)

plt.show()


""" QUANTITATIVE COMPARISONS AND STATISTICAL VISUALIZATIONS """

"""
QUANTITATIVE COMPARISONS: BAR-CHARTS
ax.bar - This will create a bar chart

To rotate the tick labels
fig, ax = plt.subplots()
ax.bar(table_name.index, table_name['column_name'])
ax.set_xticklabels(table_name.index, rotation = 90) # To rotate the x-axis names by 90 degree
ax.set_ylabel("y_title")
plt.show()

To create a stacked bar chart
fig, ax = plt.subplots()
ax.bar(table_name.index, table_name['column_name1'], label='column_name1')
ax.bar(table_name.index, table_name['column_name2'], bottom=table_name['column_name1'], label='column_name2') # stack 2 bars

ax.bar(table_name.index, table_name['column_name1'], label='column_name1')
ax.bar(table_name.index, table_name['column_name2'], bottom=table_name['column_name1'], label='column_name2')
ax.bar(table_name.index, table_name['column_name3'], bottom=table_name['column_name1'] + table_name['column_name2'], label='column_name3') # stack 3 bars

ax.set_xticklabels(table_name.index, rotation = 90) # To rotate the x-axis names by 90 degree
ax.set_ylabel("y_title")
ax.legend()
"""

# Initialize a Figure and Axes
fig, ax = plt.subplots()

# Plot a bar-chart of gold medals as a function of country
ax.bar(medals.index, medals["Gold"])

# Set the x-axis tick labels to the country names
ax.set_xticklabels(medals.index, rotation=90)

# Set the y-axis label
ax.set_ylabel("Number of medals")

plt.show()


# Add bars for "Gold" with the label "Gold"
ax.bar(medals.index, medals["Gold"], label="Gold")

# Stack bars for "Silver" on top with label "Silver"
ax.bar(medals.index, medals["Silver"], bottom=medals["Gold"], label="Silver")

# Stack bars for "Bronze" on top of that with label "Bronze"
ax.bar(
    medals.index,
    medals["Bronze"],
    bottom=medals["Gold"] + medals["Silver"],
    label="Bronze",
)

# Display the legend
ax.legend()

plt.show()


"""
QUANTITATIVE COMPARISONS: HISTOGRAM  
ax.hist - This will create a histogram of bar type

Specify # histtype="step" # to avoid occlusion (i.e display as thin lines instead of solid bars so as to avoid plotting on each other)

fig, ax = plt.subplots()
ax.hist(table_name["column_name1"], label="text_label", bins=Integer, histtype="step")
ax.hist(table_name["column_name2"], label="text_label", bins=Integer, histtype="step")

ax.set_xlabel('xlabel_title')
ax.set_ylabel('ylabel_title')
ax.legend()
plt.show()


bins can be set as integer ( whole no ) or as a sequence of values e.g [150, 160, 170, 180, 190, 200, 210]
For integers, the bars will be shared according to the given no while for sequence of value, they will be set to be the boundaries between the bins.
"""

mens_rowing = summer_2016[(summer_2016['Sex'] == 'M')
                          & (summer_2016['Sport'] == 'Rowing')]
mens_gymnastics = summer_2016[(summer_2016['Sex'] == 'M')
                              & (summer_2016['Sport'] == 'Gymnastics')]

# Initialize a Figure and Axes
fig, ax = plt.subplots()
# Plot a histogram of "Weight" for mens_rowing
ax.hist(mens_rowing["Weight"])

# Compare to histogram of "Weight" for mens_gymnastics
ax.hist(mens_gymnastics["Weight"])

# Set the x-axis label to "Weight (kg)"
ax.set_xlabel("Weight (kg)")

# Set the y-axis label to "# of observations"
ax.set_ylabel("# of observations")

plt.show()


# Initialize a Figure and Axes
fig, ax = plt.subplots()

# Plot a histogram of "Weight" for mens_rowing
ax.hist(mens_rowing["Weight"], label="Rowing", histtype="step", bins=5)

# Compare to histogram of "Weight" for mens_gymnastics
ax.hist(mens_gymnastics["Weight"], label="Gymnastics", histtype="step", bins=5)

ax.set_xlabel("Weight (kg)")
ax.set_ylabel("# of observations")

# Add the legend and show the Figure
ax.legend()
plt.show()


"""
STATISTICAL PLOTTING
Statistical plotting is a set of methods for using visualization to make comparisons.

2 of the techniques to do this include
- ADDING ERROR BARS TO  PLOTS / BAR CHARTS
Error plots / bars summarize the distribution of the data in one number, such as the standard deviation of the values.
e.g 
fig, ax = plt.subplots()

ax.bar("axis_label", table_name["column_name"].mean(), yerr = table_name["column_name"].std())
ax.bar("axis_label", table_name["column_name"].mean(), yerr = table_name["column_name"].std())


ax.set_ylabel("axis_title")
plt.show()

OR # Adding error bar to a line plot

fig, ax = plt.subplots()

ax.errorbar(table_name["column_name"], table_name["column_name_with_meanValue"], yerr = table_name["column_name_with_stddevValue"])
ax.errorbar(table_name["column_name"], table_name["column_name_with_meanValue"], yerr = table_name["column_name_with_stddevValue"])

ax.set_ylabel("axis_title")
plt.show()

- ADDING BOXPLOTS
e.g
fig, ax = plt.subplots()

ax.boxplot([ table_name["column_name"], table_name["column_name"] ])
ax.set_xticklabels(["axis_label", "axis_label"])
ax.set_ylabel("axis_title")

plt.show()
"""

# Initialize a Figure and Axes
fig, ax = plt.subplots()

# Add a bar for the rowing "Height" column mean/std
ax.bar(
    "Rowing", mens_rowing["Height"].mean(), yerr=mens_rowing["Height"].std()
)

# Add a bar for the gymnastics "Height" column mean/std
ax.bar(
    "Gymnastics", mens_gymnastics["Height"].mean(), yerr=mens_gymnastics["Height"].std()
)

# Label the y-axis
ax.set_ylabel("Height (cm)")

plt.show()


# Initialize a Figure and Axes
fig, ax = plt.subplots()

# Add Seattle temperature data in each month with error bars
ax.errorbar(
    seattle_weather["MONTH"],
    seattle_weather["MLY-TAVG-NORMAL"],
    yerr=seattle_weather["MLY-TAVG-STDDEV"],
)

# Add Austin temperature data in each month with error bars
ax.errorbar(
    austin_weather["MONTH"],
    austin_weather["MLY-TAVG-NORMAL"],
    yerr=austin_weather["MLY-TAVG-STDDEV"],
)

# Set the y-axis label
ax.set_ylabel("Temperature (Fahrenheit)")

plt.show()


# Initialize a Figure and Axes
fig, ax = plt.subplots()

# Add a boxplot for the "Height" column in the DataFrames
ax.boxplot([mens_rowing["Height"], mens_gymnastics["Height"]])

# Add x-axis tick labels:
ax.set_xticklabels(["Rowing", "Gymnastics"])

# Add a y-axis label
ax.set_ylabel("Height (cm)")

plt.show()


"""
QUANTITATIVE COMPARISONS: SCATTER PLOTS
fig, ax= plt.subplots()
e.g
ax.scatter(table_name['column_name'], table_name['column_name'])

ax.set_xlabel('xlabel_title')
ax.set_ylabel('ylabel_title')
plt.show()

Customizing scatter plots
eighties = table_name['1980-01-01':'1989-12-31']
nineties = table_name['1990-01-01':'1999-12-31']

fig, ax= plt.subplots()
ax.scatter(eighties['column_name'], eighties['column_name'], color='red', label='eighties')
ax.scatter(nineties['column_name'], nineties['column_name'], color='red', label='nineties')

ax.set_xlabel('xlabel_title')
ax.set_ylabel('ylabel_title')
plt.show()

Encoding a third variable by color
fig, ax= plt.subplots()
ax.scatter(table_name['column_name'], table_name['column_name'], c = table_name.index)

ax.set_xlabel('xlabel_title')
ax.set_ylabel('ylabel_title')
plt.show()
"""

# Initialize a Figure and Axes
fig, ax = plt.subplots()

# Add data: "co2" on x-axis, "relative_temp" on y-axis
ax.scatter(climate_change["co2"], climate_change["relative_temp"])

# Set the x-axis label to "CO2 (ppm)"
ax.set_xlabel("CO2 (ppm)")

# Set the y-axis label to "Relative temperature (C)"
ax.set_ylabel("Relative temperature (C)")

plt.show()


# Initialize a Figure and Axes
fig, ax = plt.subplots()

# Add data: "co2", "relative_temp" as x-y, index as color
ax.scatter(
    climate_change["co2"], climate_change["relative_temp"], c=climate_change.index
)

# Set the x-axis label to "CO2 (ppm)"
ax.set_xlabel("CO2 (ppm)")

# Set the y-axis label to "Relative temperature (C)"
ax.set_ylabel("Relative temperature (C)")

plt.show()


""" SHARING VISUALIZATIONS WITH OTHERS """

"""
PREPARING YOUR FIGURES TO SHARE WITH OTHERS
# Changing plot style
plt.style.use('default') - use default style
plt.style.use('ggplot') - use ggplot style
plt.style.use('bmh') - use bmh style
plt.style.use('grayscale') - use grayscale style
plt.style.use('seaborn-colorblind') - use seaborn colourblind style
plt.style.use('Solarize_Light2') - use solarize light style

e.g
plt.style.use('ggplot')
fig, ax = plt.subplots()
ax.plot(table_name["column_name1"], table_name["column_name1"])
ax.plot(table_name["column_name1"], table_name["column_name1"])

ax.set_xlabel('xlabel_title')
ax.set_ylabel('ylabel_title')
plt.show()

GUIDELINES FOR CHOOSING PLOTTING STYLE
- Dark background are usually less visible
- Consider choosing colorblind-friendly option e.g 'seaborn-colorblind' or 'tableau-colorblind10'
- Use less ink if the figure is to be printed out
- Use the 'grayscale' style if the figure is to be printed in black-and-white
"""

# Use the "ggplot" style and create new Figure/Axes
plt.style.use("ggplot")
fig, ax = plt.subplots()
ax.plot(seattle_weather["MONTH"], seattle_weather["MLY-TAVG-NORMAL"])
plt.show()

# Use the "Solarize_Light2" style and create new Figure/Axes
plt.style.use("Solarize_Light2")
fig, ax = plt.subplots()
ax.plot(austin_weather["MONTH"], austin_weather["MLY-TAVG-NORMAL"])
plt.show()


""" 
SAVING YOUR VISUALIZATIONS
change plt.show() to fig.savefig('file_name.png', quality=50, dpi=300)

e.g
fig, ax = plt.subplots()
ax.bar(table_name.index, table_name["column_name1"])

ax.set_xticklabels(table_name.index, rotation=90)
ax.set_ylabel('ylabel_title')
fig.set_size_inches([Width, Height])
fig.savefig('file_name.png')

Different file formats
- png format provides lossless compression of the image (i.e high quality image with relatively large amounts of diskspace / bandwidth)
- jpg format provides lossy compression (i.e less diskspace and bandwidth) and are used for websites
- svg format will produce a vector graphics file where elements can be edited in detail by advanced graphics software e.g Gimp / Adobe illustrator

Quality keyword is used to set image size & loss of quality and has a range between 1 & 100

DPI (Dots Per Inch) keyword. is used to render the image and the higher the dpi, the more dense the image quality and the larger its file size

fig.set_size_inches([Width, Height]) is used to control the size of the figure thereby determining its aspect ratio
"""

# Show the figure
plt.show()

# Save as a PNG file
fig.savefig("my_figure.png")

# Save as a PNG file with 300 dpi
fig.savefig("my_figure_300dpi.png", dpi=300)

# Set figure dimensions and save as a PNG
fig.set_size_inches([3, 5])
fig.savefig("figure_3_5.png")

# Set figure dimensions and save as a PNG
fig.set_size_inches([5, 3])
fig.savefig("figure_5_3.png")


"""
AUTOMATING FIGURES FROM DATA
Getting unique values of a column
table_name['column_name']
x = table_name['column_name'].unique()
print(x)

To create a std errorbar Bar-chart of heights for all sports
summer_2016_medals['Sport']

fig, ax = plt.subplots()
for sport in sports:
    sport_df = summer_2016_medals[summer_2016_medals['Sport'] == sport]
    ax.bar(sport, sport_df['Height'].mean(), yerr=sport_df['Height'].std())
ax.set_xticklabels(sports, rotation=90)
ax.set_ylabel('Height (cm)')
plt.show()
"""

summer_2016_medals = summer_2016

# Extract the "Sport" column
sports_column = summer_2016_medals["Sport"]

# Find the unique values of the "Sport" column
sports = sports_column.unique()

# Print out the unique sports values
print(sports)


# Initialize a Figure and Axes
fig, ax = plt.subplots()

# Loop over the different sports branches
for sport in sports:
    # Extract the rows only for this sport
    sport_df = summer_2016_medals[summer_2016_medals["Sport"] == sport]
    # Add a bar for the "Weight" mean with std y error bar
    ax.bar(sport, sport_df["Weight"].mean(), yerr=sport_df["Weight"].std())

ax.set_ylabel("Weight")
ax.set_xticklabels(sports, rotation=90)

# Save the figure to file
fig.savefig("sports_weights.png")
