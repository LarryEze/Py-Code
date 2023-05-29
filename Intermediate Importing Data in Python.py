""" Importing data from the Internet """

"""
Importing flat files from the web
Can you import (download) web data?
- You can: go to the URL and click to download files
- But: It is not reproducible and not scalable because the data wasn't written in code

The urllib package
- Provides interface for fetching data across the web
- urlopen() - accepts URLs instead of file names

How to automate file download in Python
e.g
from urllib.request import urlretrieve
url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv'
urlretrieve( url, 'winequality-white.csv' )
"""

# Import package
from urllib.request import urlretrieve

# Import pandas
import pandas as pd

# Assign url of file: url
url = "https://assets.datacamp.com/production/course_1606/datasets/winequality-red.csv"

# Save file locally
urlretrieve(url, "winequality-red.csv")

# Read file into a DataFrame and print its head
df = pd.read_csv("winequality-red.csv", sep=";")
print(df.head())


# Import packages
import matplotlib.pyplot as plt
import pandas as pd

# Assign url of file: url
url = "https://assets.datacamp.com/production/course_1606/datasets/winequality-red.csv"

# Read file into a DataFrame: df
df = pd.read_csv(url, sep=";")

# Print the head of the DataFrame
print(df.head())

# Plot first column of df
df.iloc[:, 0].hist()
plt.xlabel("fixed acidity (g(tartaric acid)/dm$^3$)")
plt.ylabel("count")
plt.show()


# Import package
import pandas as pd

# Assign url of file: url
url = "https://assets.datacamp.com/course/importing_data_into_r/latitude.xls"

# Read in all sheets of Excel file: xls
xls = pd.read_excel(url, sheet_name=None)

# Print the sheetnames to the shell
print(xls.keys())

# Print the head of the first sheet (using its name, NOT its index)
print(xls["1700"].head())


"""
HTTP requests to import files from the web
URL
- It stands for Unifrom / Universal Resource Locator
- They are references to web resources
- Focus: web addresses
- web addresses URL consists of 2 parts:
* Protocol identifier - http: / https:
* Resource name - datacamp.com
- Protocol identifier + Resource name specify web addresses uniquely

HTTP
- It stands for HyperText Transfer Protocol
- HTTP is an application protocol for distributed, collaborative, hypermedia informattion systems (- wikipedia)
- HTTP is the foundation of data communication for th World Wide Web
- HTTPS - more secure form of HTTP
- Going to a website = sending HTTP request
* GET request
- urlretrieve() performs a GET request
- HTML stands for HyperText Markup Language

GET requests using urllib
from urllib.requests import urlopen, Request        # import the necessary functions
url = 'https://www.wikipedia.org/'                  # specify the URL
request = Request( url )                            # package the GET request
response = urlopen( request )                       # send the request and catch the response
html = reponse.read() 

GET requests using requests
Requests package allows you to send organic, grass-fed HTTP/1.1 requests, without the need for manual labour
- One of the most downloaded Python packages
e.g
import requests
url = 'https://www'wikipedia.org/'
r = requests.get( url )
text = r.text
response.close()
"""

# Import packages
from urllib.request import urlopen, Request

# Specify the url
url = "https://campus.datacamp.com/courses/1606/4135?ex=2"

# This packages the request: request
request = Request(url)

# Sends the request and catches the response: response
response = urlopen(request)

# Print the datatype of response
print(type(response))

# Be polite and close the response!
response.close()


# Import packages
from urllib.request import urlopen, Request

# Specify the url
url = "https://campus.datacamp.com/courses/1606/4135?ex=2"

# This packages the request
request = Request(url)

# Sends the request and catches the response: response
response = urlopen(request)

# Extract the response: html
html = response.read()

# Print the html
print(html)

# Be polite and close the response!
response.close()


# Import package
import requests

# Specify the url: url
url = "http://www.datacamp.com/teach/documentation"

# Packages the request, send the request and catch the response: r
r = requests.get(url)

# Extract the response: text
text = r.text

# Print the html
print(text)


"""
Scraping the web in Python
HTML 
- It's a micture of unstructured and structured data
- Structured data:
* Has pre-defined data model, or
* Organized in a defined manner
- Unstructured data: neither of these properties

BeautifulSoup
- It parse and extract structured data from HTML
- In web development, the term 'tag soup' reders to structurally or syntacticallyincorrect HTML code written for a web page.
- Therefore, Beautiful Soup makes tag soup beautiful again and to extract information from it with ease
e.g
from bs4 import BeautifulSoup
import requests
url = 'https://www.crummy.com/software/BeautifulSoup/'
r = requests.get( url )
html_doc = r.text
soup = BeautifulSoup(html_doc) 

Exploring BeautifulSoup
- It has many methods such as: 
print(soup.title)           # extract title

print(soup.get_text())      # extract text

- .find_all()
for link in soup.find_all( 'a' ):
    print( link.get( 'href' )
"""

# Import packages
import requests
from bs4 import BeautifulSoup

# Specify url: url
url = "https://www.python.org/~guido/"

# Package the request, send the request and catch the response: r
r = requests.get(url)

# Extracts the response as html: html_doc
html_doc = r.text

# Create a BeautifulSoup object from the HTML: soup
soup = BeautifulSoup(html_doc)

# Prettify the BeautifulSoup object: pretty_soup
pretty_soup = soup.prettify()

# Print the response
print(pretty_soup)


# Import packages
import requests
from bs4 import BeautifulSoup

# Specify url: url
url = "https://www.python.org/~guido/"

# Package the request, send the request and catch the response: r
r = requests.get(url)

# Extract the response as html: html_doc
html_doc = r.text

# Create a BeautifulSoup object from the HTML: soup
soup = BeautifulSoup(html_doc)

# Get the title of Guido's webpage: guido_title
guido_title = soup.title

# Print the title of Guido's webpage to the shell
print(guido_title)

# Get Guido's text: guido_text
guido_text = soup.get_text()

# Print Guido's text to the shell
print(guido_text)


# Import packages
import requests
from bs4 import BeautifulSoup

# Specify url
url = "https://www.python.org/~guido/"

# Package the request, send the request and catch the response: r
r = requests.get(url)

# Extracts the response as html: html_doc
html_doc = r.text

# create a BeautifulSoup object from the HTML: soup
soup = BeautifulSoup(html_doc)

# Print the title of Guido's webpage
print(soup.title)

# Find all 'a' tags (which define hyperlinks): a_tags
a_tags = soup.find_all("a")

# Print the URLs to the shell
for link in a_tags:
    print(link.get("href"))


""" Interacting with APIs to import data from the web """

"""
Introduction to APIs and JSONs
APIs
- It stands for Application Programming Interface
- It is a set of protocols and routines for building and interacting with software applications

JSONs
- It stands for JavaScript Object Notation
- A arose from the need for Real-time server-to-browser communication
- they are human readable

Loading JSONs in Python
import json
with open( 'snakes.json', 'r' ) as json_file:
    json_data = json.load( json_file )
type( json_data ) <- in
dict <- out

Exploring JSONs in Python
for key, value in json_data.items():
    print( key + ':', value )
"""

# Load JSON: json_data
with open("a_movie.json") as json_file:
    json_data = json.load(json_file)

# Print each key-value pair in json_data
for k in json_data.keys():
    print(k + ": ", json_data[k])


"""
APIs and interacting with the world wide web
Connecting to an API in Python
import requests
url = 'http://www.omdbapi.com/?t=hackers'
r = requests.get( url )
json_data = r.json()

for key, value in json_data.items():
    print(key + ':', value)

What was that URL?
- http - making an HTTP request
- www.omdbapi.com - querying the OMDB API
- ?t=hackers
* Query strings ( are parts of URLs that do not necessarily fit into conventional hierarchical path structure )
* Return data for a movie with title ( t ) 'Hackers'
http://www.omdbapi.com/?t=hackers
"""

# Import requests package
import requests

# Assign URL to variable: url
url = "http://www.omdbapi.com/?apikey=72bc447a&t=the+social+network"

# Package the request, send the request and catch the response: r
r = requests.get(url)

# Print the text of the response
print(r.text)


# Import package
import requests

# Assign URL to variable: url
url = "http://www.omdbapi.com/?apikey=72bc447a&t=social+network"

# Package the request, send the request and catch the response: r
r = requests.get(url)

# Decode the JSON data into a dictionary: json_data
json_data = r.json()

# Print each key-value pair in json_data
for k in json_data.keys():
    print(k + ": ", json_data[k])


# Import package
import requests

# Assign URL to variable: url
url = "https://en.wikipedia.org/w/api.php?action=query&prop=extracts&format=json&exintro=&titles=pizza"

# Package the request, send the request and catch the response: r
r = requests.get(url)

# Decode the JSON data into a dictionary: json_data
json_data = r.json()

# Print the Wikipedia page extract
pizza_extract = json_data["query"]["pages"]["24768"]["extract"]
print(pizza_extract)


""" Diving deep into the Twitter API """

"""
The Twitter API and Authentication
Twitter has a number of APIs
REST APIs
- It stands for Representational State Transfer
- It allows the user to 'read and write Twitter data'

The Streaming APIs
* Get statuses/sample APi - To read and process tweets ( i.e Returns a small random sample of all public statuses )
* Firehose API - To access all public statuses

Using Tweepy: Authentication
tweets.py
e.g
import tweepy, json
consumer_key = '...'
consumer_secret = '...'
access_token = '...'
access_token_secret = '...'

# Create Streaming object
stream = tweepy.Stream( consumer_key, consumer_secret, access_token, access_token_secret )

# This line filters Twitter Streams to capture data by keywords: 
stream.filter( track=[ 'apples', 'oranges' ] )

e.g
consumer_key = "nZ6EA0FxZ293SxGNg8g8aP0HM"
consumer_secret = "fJGEodwe3KiKUnsYJC3VRndj7jevVvXbK2D5EiJ2nehafRgA6i"
access_token = "1092294848-aHN7DcRP9B4VMTQIhwqOYiB14YkW92fFO8k8EPy"
access_token_secret = "X4dHmhPfaksHcQ7SCbmZa2oYBBVSD2g8uIHXsp5CTaksx"
"""

# Store credentials in relevant variables
consumer_key = "nZ6EA0FxZ293SxGNg8g8aP0HM"
consumer_secret = "fJGEodwe3KiKUnsYJC3VRndj7jevVvXbK2D5EiJ2nehafRgA6i"
access_token = "1092294848-aHN7DcRP9B4VMTQIhwqOYiB14YkW92fFO8k8EPy"
access_token_secret = "X4dHmhPfaksHcQ7SCbmZa2oYBBVSD2g8uIHXsp5CTaksx"

# Create your Stream object with credentials
stream = tweepy.Stream(consumer_key, consumer_secret, access_token, access_token_secret)

# Filter your Stream variable
stream.filter(track=["clinton", "trump", "sanders", "cruz"])


# Import package
import json

# String of path to file: tweets_data_path
tweets_data_path = "tweets.txt"

# Initialize empty list to store tweets: tweets_data
tweets_data = []

# Open connection to file
tweets_file = open(tweets_data_path, "r")

# Read in tweets and store in list: tweets_data
for line in tweets_file:
    tweet = json.loads(line)
    tweets_data.append(tweet)

# Close connection to file
tweets_file.close()

# Print the keys of the first tweet dict
print(tweets_data[0].keys())


# Import package
import pandas as pd

# Build DataFrame of tweet texts and languages
df = pd.DataFrame(tweets_data, columns=["text", "lang"])

# Print head of DataFrame
print(df.head())


# Initialize list to store tweet counts
[clinton, trump, sanders, cruz] = [0, 0, 0, 0]

import re


def word_in_text(word, text):
    word = word.lower()
    text = text.lower()
    match = re.search(word, text)

    if match:
        return True
    return False


# Iterate through df, counting the number of tweets in which each candidate is mentioned
for index, row in df.iterrows():
    clinton += word_in_text("clinton", row["text"])
    trump += word_in_text("trump", row["text"])
    sanders += word_in_text("sanders", row["text"])
    cruz += word_in_text("cruz", row["text"])


# Import packages
import matplotlib.pyplot as plt
import seaborn as sns

# Set seaborn style
sns.set(color_codes=True)

# Create a list of labels:cd
cd = ["clinton", "trump", "sanders", "cruz"]

# Plot the bar chart
ax = sns.barplot(cd, [clinton, trump, sanders, cruz])
ax.set(ylabel="count")
plt.show()
