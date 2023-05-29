''' Regular expressions & word tokenization '''

'''
Introduction to regular expressions
What is Natural Language Processing?
- It is a field of study, focused on making sense of language
* using statistics and computers.

- Basics of NLP include:
* Topic identification
* Text classification

- NLP applications include:
* Chatbots
* Translation
* Sentiment analysis
* . . . and many more!

What exactly are regular expressions?
- They are strings with a special syntax
* which allows us to match patterns in other strings
- A pattern is a series of letters or symbols which can map to an actual text or words or punctuation.
- Applications of regular expressions include:
* Find all web links in a document
* Parse email addresses
* Remove / replace unwanted characters
- Regular expressions are often referred to as Regex and can be used easily with python via the 're' library.
e.g
import re

# To match patterns in a strings
re.match('abc', 'abcdef') -> in

<_sre.SRE_Match object; span=(0, 3), match='abc'> -> out

# To match pattern in words
word_regex = '\w+'
re.match(word_regex, 'hi there!') -> in

<_sre.SRE_Match object; span=(0, 2), match='hi'> -> out

Common regex pattern
pattern         matches             example
\w+             word                'Magic'
\d              digit               9
\s              space               ' '
.*              wildcard            'username74'
+ or *          greedy match        'aaaaaa'
\S              not space           'no_spaces '
[a-z]           lowercase group     'abcdefg'

- NB*: Using these character classes as capital letters negates them

Python's re module
- re module
* split: split a string on regex
* findall: find all patterns in a string
* search: search for a pattern
* match: match an entire string or substring based on a pattern

- The syntax for the regex library is always to pass the pattern first, and the string second
* depending on the method, it may return an iterator, a new string or a match object.

e.g
re.split('\s+', 'Split on spaces.') -> in

['Split' 'on' 'spaces.'] -> out
'''

# Write a pattern to match sentence endings: sentence_endings
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer
from nltk.tokenize import regexp_tokenize
sentence_endings = r"[.?!]"

# Split my_string on sentence endings and print the result
print(re.split(sentence_endings, my_string))

# Find all capitalized words in my_string and print the result
capitalized_words = r"[A-Z]\w+"
print(re.findall(capitalized_words, my_string))

# Split my_string on spaces and print the result
spaces = r"\s+"
print(re.split(spaces, my_string))

# Find all digits in my_string and print the result
digits = r"\d+"
print(re.findall(digits, my_string))


'''
Introduction to tokenization
- Tokenization is the process of transforming a string or document into smaller chunks (tokens).
* It is usually one step in the process of preparing a text for Natural Language Processing (NLP)
- There are many different theories and rules when using tokenization
* You can create your own rules using regular expressions
- Some examples:
* Breaking out words or sentences
* Separating punctuation
* Separating all hashtags in a tweet

nltk library
- nltk: natuaral language toolkit

from nltk.tokenize import word_tokenize
word_tokenize('Hi there!') -> in

['Hi', 'there', '!'] -> out

Why tokenize?
- It can help with some simple text processing tasks like
* mapping part of speech
* mapping common words
* Removing unwanted tokens

e.g
- 'I don't like Sam's shoes.'
* 'I', 'do', 'n't', 'like', 'Sam', ''s', 'shoes', '.'

Other nltk tokenizers
- sent_tokenize: tokenize a document into sentences
- regexp_tokenize: tokenize a string or document based on a regular expression pattern
- TweetTokenizer: special class just for tweet tokenization, allowing you to separate hashtags, mentions and lots of exclamation points!!!

More regex practice
- Difference between re.search() and re.match()

import re
re.match('abc', 'abcde') -> in

<_sre.SRE_Match object; span=(0, 3), match='abc'> -> out

re.search('abc', 'abcde') -> in

<_sre.SRE_Match object; span=(0, 3), match='abc'> -> out

re.match('cd', 'abcde')

re.search('cd', 'abcde') -> in

<_sre.SRE_Match object; span=(2, 4), match='cd'> -> out

- NB*: if you need to find a pattern that might not be at the beginning of the string, use Search
* If you want to be specific about the composition of the entire string, or atleast the initial pattern, use Match
'''

# Import necessary modules

# Split scene_one into sentences: sentences
sentences = sent_tokenize(scene_one)

# Use word_tokenize to tokenize the fourth sentence: tokenized_sent
tokenized_sent = word_tokenize(sentences[3])

# Make a set of unique tokens in the entire scene: unique_tokens
unique_tokens = set(word_tokenize(scene_one))

# Print the unique tokens result
print(unique_tokens)


# Search for the first occurrence of "coconuts" in scene_one: match
match = re.search("coconuts", scene_one)

# Print the start and end indexes of match
print(match.start(), match.end())

# Write a regular expression to search for anything in square brackets: pattern1
pattern1 = r"\[.*]"

# Use re.search to find the first text in square brackets
print(re.search(pattern1, scene_one))

# Find the script notation at the beginning of the fourth sentence and print it
pattern2 = r"[\w\s]+:"
print(re.match(pattern2, sentences[3]))


'''
Advanced tokenization with NLTK and regex
Regex groups using or '|'
- OR is represented using '|'
- to use the or, you can define a group using ()
* Groups can either be a pattern or a set of characters you want to match explicitly
- You can also define explicit character ranges using []

e.g
import re
match_digits_and_words = ('(\d+|\w+)')
re.findall(match_digits_and_words, 'He has 11 cats.') -> in

['He', 'has', '11', 'cats'] -> out

Regex ranges and groups
pattern             matches                                                 example
[A-Za-z]+           upper and lowercase English alphabet                    'ABCDEFghijk'
[0-9]               numbers from 0-9                                        9
[A-Za-z\-\.]+       upper and lowercase English alphabet, - and .           'My-Website.com'
(a-z)               a, - and z                                              'a-z'
(\s+|,)             spaces or a comma                                       ', '

Character range with 're.match()'
import re
my_str = 'match lowercase spaces nums like 12, but no commas'
re.match('[a-z0-9 ]+', my_str) -> in

<_sre.SRE_Match object; span=(0, 42), match='match lowercase spaces nums like 12'> -> out

- NB*: Unlike the syntax for the regex library, with nltk_tokenize() you pass the pattern as the second argument.
'''

# To retain sentence punctuation as separate tokens, but have '#1' remain a single token.

my_string = "SOLDIER #1: Found them? In Mercea? The coconut's tropical!"
regexp_tokenize(my_string, r"(\w+|#\d|\?|!)")


# Import the necessary modules
# Define a regex pattern to find hashtags: pattern1
pattern1 = r"#\w+"
# Use the pattern on the first tweet in the tweets list
hashtags = regexp_tokenize(tweets[0], pattern1)
print(hashtags)

# Import the necessary modules
# Write a pattern that matches both mentions (@) and hashtags
pattern2 = r"([@#]\w+)"
# Use the pattern on the last tweet in the tweets list
mentions_hashtags = regexp_tokenize(tweets[-1], pattern2)
print(mentions_hashtags)

# Import the necessary modules
# Use the TweetTokenizer to tokenize all tweets into one list
tknzr = TweetTokenizer()
all_tokens = [tknzr.tokenize(t) for t in tweets]
print(all_tokens)


# Tokenize and print all words in german_text
all_words = word_tokenize(german_text)
print(all_words)

# Tokenize and print only capital words
capital_words = r"[A-ZÜ]\w+"
print(regexp_tokenize(german_text, capital_words))

# Tokenize and print only emoji
emoji = "['\U0001F300-\U0001F5FF'|'\U0001F600-\U0001F64F'|'\U0001F680-\U0001F6FF'|'\u2600-\u26FF\u2700-\u27BF']"
print(regexp_tokenize(german_text, emoji))


'''
Charting word length with NLTK
Plotting a histogram with matplotlib
from matplotlib import pyplot as plt

plt.hist([1, 5, 5, 7, 7, 7, 9]) 
plt.show()

Combining NLP data extraction with plotting
from matplotlib import pyplot as plt
from nltk.tokenize import word_tokenize

words = word_tokenize('This is a pretty cool tool!')
word_lengths = [len(w) for w in words]
plt.hist(word_lengths)
plt.show()
'''

# Split the script into lines: lines
lines = holy_grail.split('\n')

# Replace all script lines for speaker
pattern = "[A-Z]{2,}(\s)?(#\d)?([A-Z]{2,})?:"
lines = [re.sub(pattern, '', l) for l in lines]

# Tokenize each line: tokenized_lines
tokenized_lines = [regexp_tokenize(s, '\w+') for s in lines]

# Make a frequency list of lengths: line_num_words
line_num_words = [len(t_line) for t_line in tokenized_lines]

# Plot a histogram of the line lengths
plt.hist(line_num_words)

# Show the plot
plt.show()


''' Simple topic identification '''

'''
Word counts with bag-of-words
Bag-of-words
- It is a very simple and basic method for finding topics in a text
- Need to first create tokens using tokenization
* . . . and then count up all the tokens
- The theory is, the more frequent a word / token is, the more central / important it might be to the text
- It can be a great way to determine the significant words in a text based on the number of times they are used.
e.g
- Text: 'The cat is in the box. The cat likes the box. The box is over the cat.'
- Bag of words (stripped punctuation):
* 'The': 3, 'box': 3
* 'cat': 3, 'the': 3
* 'is': 2
* 'in': 1, 'likes': 1, 'over': 1

Bag-of-words in Python
from nltk.tokenize import word_tokenize
from collections import Counter
Counter(word_tokenize(''"The cat is in the box. The cat likes the box. The box is over the cat."'')) -> in

Counter({'.': 3, 'The': 3, 'box': 3, 'cat': 3, 'in': 1, . . . ,'the': 3}) -> out

Counter.most_common(2) -> in

[('The', 3), ('box', 3)] -> out

- NB*: other than ordering by token frequency, the most_common() method does not sort the tokens it returns or tell us there are more tokens with that same frequency.
'''

# Import Counter

# Tokenize the article: tokens
tokens = word_tokenize(article)

# Convert the tokens into lowercase: lower_tokens
lower_tokens = [t.lower() for t in tokens]

# Create a Counter with the lowercase tokens: bow_simple
bow_simple = Counter(lower_tokens)

# Print the 10 most common tokens
print(bow_simple.most_common(10))


'''
Simple text preprocessing
Why preprocess?
- It helps make for better input data
* when performing machine learning or other statistical methods
- Examples:
* Tokenization to create a bag of words
* Lowercasing words
- Other common techniques are things like: 
- Lemmatization / Stemming
* shorten the words to their root stems
- Removing stop words, pinctuation, or unwanted tokens
-  It good to experiment with different approaches to preprocessing and see which works best for your task and goal.

Preprocessing example
- Input text: Cats, dogs and birds are common pets. So are fish.
- Output tokens: cat, dog, bird, common, pet, fish

Text preprocessing with Python
from nltk.corpus import stopwords

text = ''"The cat is in the box. The cat likes the box. The box is over the cat."''
tokens = [w for w in word_tokenize(text.lower()) if w.isalpha()]
no_stops = [t for t in tokens if t not in stopwords.words('english')]
Counter(no_stops).most_common(2) -> in

[('cat', 3). ('box', 3)] -> out
'''

# Import WordNetLemmatizer

# Retain alphabetic words: alpha_only
alpha_only = [t for t in lower_tokens if t.isalpha()]

# Remove all stop words: no_stops
no_stops = [t for t in alpha_only if t not in english_stops]

# Instantiate the WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()

# Lemmatize all tokens into a new list: lemmatized
lemmatized = [wordnet_lemmatizer.lemmatize(t) for t in no_stops]

# Create the bag-of-words: bow
bow = Counter(lemmatized)

# Print the 10 most common tokens
print(bow.most_common(10))


'''
Introduction to gensim
- 'Gensim' ia a popular open-source NLP library
- It uses top academic models to perform complex tasks like
* building document or word vectors, corpora
* Performing topic identificatiom amd document comparisons

What is a word vector?
- A word embeding or vector is trained from a larger corpus and is a multi-dimensional representation of a word or document.
- The deep learning algorithm used to create word vectors has been able to distill this meaning based on how those words are used throughout the text.

Gensim example
- LDA stands for Latent Dirichlet Allocation. and it is a statistical model that can be applied to text using Gensim for topic analysis and modelling.
- A corpus (or if plural, sorpora) is a set of texts used to help perform NLP tasks.

from gensim.corpora.dictionary import Dictionary
from nltk.tokenize import word_tokenize

my_documents = ['The movie was about a spaceship and aliens.', 'I really liked the movie!', 'Awesome action scenes, but boring characters.', 'The movie was awful! I hate alien films.', 'Space is cool! I liked the movie.', 'More space films, please!']
tokenized_docs = [word_tokenize(doc.lower()) for doc in my_documents]
dictionary = Dictionary(tokenized_docs)
dictionary.token2id -> in

{'!': 11, ',': 17, '.': 7, 'a': 2, 'about': 4, . . . } -> out

Creating a gensim corpus
- A normal corpus is just a collection of documents
* But a gensim corpus uses a simple bag-of-words model which transforms each document into a bag of word using the token ids and the frequency of each token in the document.

corpus = [dictionary.doc2bow(doc) for doc in tokenized_docs]
corpus -> in

[[(0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1), (7, 1), (8, 1)], [(0, 1), (1, 1), (9, 1), (10, 1), (11, 1), (12, 1), . . . ] -> out

- gensim models can be easily saved, updated ad reused thanks to te extra tools we have available in Gensim.
- Our dictionary can also be updated with new texts and extract only words that meet particular thresholds. 
- This is a more advanced and feature-rich bag-of-words model which can be used for future exercises.
'''

# Import Dictionary

# Create a Dictionary from the articles: dictionary
dictionary = Dictionary(articles)

# Select the id for "computer": computer_id
computer_id = dictionary.token2id.get("computer")

# Use computer_id with the dictionary to print the word
print(dictionary.get(computer_id))

# Create a MmCorpus: corpus
corpus = [dictionary.doc2bow(article) for article in articles]

# Print the first 10 word ids with their frequency counts from the fifth document
print(corpus[4][:10])


# Save the fifth document: doc
doc = corpus[4]

# Sort the doc for frequency: bow_doc
bow_doc = sorted(doc, key=lambda w: w[1], reverse=True)

# Print the top 5 words of the document alongside the count
for word_id, word_count in bow_doc[:5]:
    print(dictionary.get(word_id), word_count)

# Create the defaultdict: total_word_count
total_word_count = defaultdict(int)
for word_id, word_count in itertools.chain.from_iterable(corpus):
    total_word_count[word_id] += word_count

# Create a sorted list from the defaultdict: sorted_word_count
sorted_word_count = sorted(total_word_count.items(),
                           key=lambda w: w[1], reverse=True)

# Print the top 5 words across all documents alongside the count
for word_id, word_count in sorted_word_count[:5]:
    print(dictionary.get(word_id), word_count)


'''
Tf-idf with gensim
What is Tf-idf?
- Tf-idf stands for Term frequency-inverse document frequency
- It allows you to determine the most important words in each document in the corpus.
- Its idea is that each corpus might have more shared words than just stopwords.
* These common words are like stopwords and should be removed or at least down-weighted in importance.
- Example from astronomy: 'Sky'
- Tf-idf take texts that share common language and ensure the most common words across the entire corpus don't show up as keywords.
- Tf-idf helps keep the document-specific frequent words weighted high and the common words across the entire corpus weighted low.

Tf-idf formula
w(i, j) = tf(i, j) * log(N / df(i))

- w(i, j) = tf-idf weight for token i in document j
- tf(i, j) = number of occurences of token i in document j
- df(i) = number of documents that contain token i
- N = total number of documents

Tf-idf with gensim
from gensim.models.tfidfmodel import TfidfModel

tfidf = TfidfModel(corpus)
tfidf[corpus[1]] -> in

[(0, 0.1746298276735174), (1, 0.1746298276735174), (9, 0.29853166221463673), (10, 0.7716931521027908), . . . ] -> out
'''

# Create a new TfidfModel using the corpus: tfidf
tfidf = TfidfModel(corpus)

# Calculate the tfidf weights of doc: tfidf_weights
tfidf_weights = tfidf[doc]

# Print the first five weights
print(tfidf_weights[:5])

# Sort the weights from highest to lowest: sorted_tfidf_weights
sorted_tfidf_weights = sorted(tfidf_weights, key=lambda w: w[1], reverse=True)

# Print the top 5 weighted words
for term_id, weight in sorted_tfidf_weights[:5]:
    print(dictionary.get(term_id), weight)


''' Named-entity recognition '''

'''
Named Entity Recognition
what is Named Entity Recognition?
- Named Entity Recognition (NER) is a NLP task used to identify important named entities in the text
* such as people, places and organizations
* they can also be Dates, states, works of art
* . . . and other categories depending on the libraries and notation used.
- NER can be used alongside topic identification
* . . . or on its own to detemine important items in a text or answer basic NLP understanding questions such as who? what? when? and where?

nltk and the Stanford CoreNLP Library
- The Stanford CoreNLP library:
* Integrated into Python via nltk by performing a few steps before use
* (Java based) i.e including installing the required Java files and setting system environment variables
* It has great support for NER as well as some related NLP tasks such as  coreference (or linking pronouns and entities together) and dependency trees to help with parsing meaning and relationships amongst words or phrases in a sentence.
- You can also use the Stanford library on its own without integrating it with nltk or operate it as an API server

Using nltk for NER
import nltk

sentence = ''"In New York, I like to ride the Metro to visit MOMA and some restaurants rated well by Ruth Reichl."''
tokenized_sent = nltk.word_tokenize(sentence)
tagged_sent = nltk.pos_tag(tokenized_sent)
tagged_sent[:3] -> in

[('In', 'IN'), ('New', 'NNP'), ('York', 'NNP')] -> out

print(nltk.ne_chunk(tagged_sent)) -> in

(   s
    In/IN
    (GPE New/NNP York/NNP)
    ,/,    
    I/PRP
    like/VBP
    to/TO
    ride/VB
    the/DT
    (ORGANIZATION Metro/NNP)
    to/TO
    visit/VB
    (ORGANIZATION MOMA/NNP)
    and/CC
    some/DT
    restaurants/NNS
    rated/VBN
    well/RB
    by/IN
    (PERSON Ruth/NNP Reichl/NNP)
    ./.) -> out
'''

# Tokenize the article into sentences: sentences
sentences = nltk.sent_tokenize(article)

# Tokenize each sentence into words: token_sentences
token_sentences = [nltk.word_tokenize(sent) for sent in sentences]

# Tag each tokenized sentence into parts of speech: pos_sentences
pos_sentences = [nltk.pos_tag(sent) for sent in token_sentences]

# Create the named entity chunks: chunked_sentences
chunked_sentences = nltk.ne_chunk_sents(pos_sentences, binary=True)

# Test for stems of the tree with 'NE' tags
for sent in chunked_sentences:
    for chunk in sent:
        if hasattr(chunk, "label") and chunk.label() == "NE":
            print(chunk)


# Create the defaultdict: ner_categories
ner_categories = defaultdict(int)

# Create the nested for loop
for sent in chunked_sentences:
    for chunk in sent:
        if hasattr(chunk, 'label'):
            ner_categories[chunk.label()] += 1

# Create a list from the dictionary keys for the chart labels: labels
labels = list(ner_categories.keys())

# Create a list of the values: values
values = [ner_categories.get(v) for v in labels]

# Create the pie chart
plt.pie(values, labels=labels, autopct='%1.1f%%', startangle=140)

# Display the chart
plt.show()


'''
Introduction to SpaCy
What is SpaCy?
- It is an NLP library similar to Gensim, but with different implementations.
* It particularly focus on creating NLP pipelines to generate models and corpora.
- It is open-source and has severa; extra libraries and tools built by the same team including:
* Displacy -a visualization tool for viewing parse trees which uses Node-js to create interactive text.
- SpaCy also has tools to build word and document vectors from text.

import spacy
nlp = spacy.load('en_core_web_sm')
nlp.entity -> in

<spacy.pipeline.EntityRecognizer at 0x7f76b75e68b8> -> out

doc = nlp(''"Berlin is the capital of Germany; and the residence of Chancellor Angela Merkel."'')
doc.ents -> in

(Berlin, Germany, ANgela Merkel) -> out

print(doc.ents[0], doc.ents[0].label_) -> in

Berlin GPE -> out

- Spacy has several other language models available, including advanced German and Chinese implementations.

Why use SpaCy for NER?
- Easy pipeline creation
- It has a different set of entity types and often labels entities differently than nltk
- SpaCy comes with informal language corpora
* allowing you to more easily find entities in documents like Tweets and chat messages,
- SpaCy is a quickly growing library!
'''

# Import spacy

# Instantiate the English model: nlp
nlp = spacy.load('en_core_web_sm', disable=['tagger', 'parser', 'matcher'])

# Create a new document: doc
doc = nlp(article)

# Print all of the found entities and their labels
for ent in doc.ents:
    print(ent.label_, ent.text)


'''
Multilingual NER with polyglot
What is polyglot?
- It is a NLP library which uses word vectors top perform simple tasks such as entity recognition.
- Why polyglot?
* It has a wide variety of languages it supports.
* It has word embeddings for more than 130 languages!
- It can be used for Transliterations i.e the ability to translate text by swapping characters from one language to another.

Spanish NER with polyglot
from polyglot.text import Text

text = ''"El presidente de la Generalitat de Cataluna, Carles Puigdemont, ha afirmado hoy a la alcadesa de Madrid, Manuela Carmena, que en su etapa de alcalde de Girona (de julio de 2011 a enero de 2016) hizo una gran promocion de Madrid."''
ptext = Text(text)
ptext.entities -> in

[   I-ORG(['Generalitat', 'de']), 
    I-LOC(['Generalitat', 'de', 'Catalina']),
    I-PER(['Carles', 'Puigdemont']), 
    I-LOC(['Madrid']),
    I-PER(['Manuela', 'Carmena']),
    I-LOC(['Girona']),
    I-LOC(['Madrid'])   ] -> out
'''

# Create a new text object using Polyglot's Text class: txt
txt = Text(article)

# Print each of the entities found
for ent in txt.entities:
    print(ent)

# Print the type of ent
print(type(ent))


# Create the list of tuples: entities
entities = [(ent.tag, ' '.join(ent)) for ent in txt.entities]

# Print entities
print(entities)


# Initialize the count variable: count
count = 0

# Iterate over all the entities
for ent in txt.entities:
    # Check whether the entity contains 'Márquez' or 'Gabo'
    if "Márquez" in ent or "Gabo" in ent:
        # Increment count
        count += 1

# Print count
print(count)

# Calculate the percentage of entities that refer to "Gabo": percentage
percentage = count / len(txt.entities)
print(percentage)


''' Building a "fake news" classifier '''

'''
Classifying fake news using supervised learning with NLP
What is supervised learning?
- It is a form of machine learning where you are given or create training data.
- The data has a label (or outcome) which you want the model or algorithm to learn

- Classification problem
* Goal: Make good hypotheses about the species based on geometric features

Sepal length    Sepal width     Petal length    Petal width     Species
5.1             3.5             1.4             0.2             I. setosa
7.0             3.2             4.77            1.4             I. versicolor
6.3             3.3             6.0             2.5             I. virginica

Supervised learning with NLP
- Need to use language instead of geometric features
- scikit-learn: Powerful open-source library
- How to create supervised learning data from text?
* Use bag-of-words models or tf-idf as features

IMDB Movie Dataset
Plot                                                    Sci-Fi  Action
In a post=apocalyptic world in human decay, a . . .     1       0
Mohei is a wandering swordsman, He arrives in . . .     0       1
#137 is a SCI/FI thriller about a girl, MArla, . . .    1       0

- Goal: predict movie genre based on plot summary
- Categorical features generated using preprocessing

Supervised learning steps
- Collect and preprocess our data
- Determine a label (Example: Movie genre)
- Split data into training and test sets
- Extract features from the text to help predict the label
* Bag-of-words vector built into scikit-learn
- Exavluate trained model using the test set
'''


'''
Building word count vectors with scikit-learn
Predicting movie genre
- Dataset consisting of movie plots and corresponding genre
- Goal: Create bag-of-word vectors for the movie plots
* Can we predict genre based on the words used in the plot summary?

Count Vectorizer with Python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

df = . . . # Load data into DataFrame
y = df['Sci -Fi']
X_train, X_test, y_train, y_test = train_test_split(df['plot'], y, test_size=0.33, random_state=53)
count_vectorizer = CountVectorizer(stop_words='english')
count_train = count_vectorizer.fit_transform(X_train.values)
count_test = count_vectorizer.transform(X_test.values)
'''

# Import the necessary modules

# Print the head of df
print(df.head())

# Create a series to store the labels: y
y = df.label

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], y, test_size=0.33, random_state=53)

# Initialize a CountVectorizer object: count_vectorizer
count_vectorizer = CountVectorizer(stop_words="english")

# Transform the training data using only the 'text' column values: count_train
count_train = count_vectorizer.fit_transform(X_train)

# Transform the test data using only the 'text' column values: count_test
count_test = count_vectorizer.transform(X_test)

# Print the first 10 features of the count_vectorizer
print(count_vectorizer.get_feature_names()[:10])


# Import TfidfVectorizer

# Initialize a TfidfVectorizer object: tfidf_vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)

# Transform the training data: tfidf_train
tfidf_train = tfidf_vectorizer.fit_transform(X_train)

# Transform the test data: tfidf_test
tfidf_test = tfidf_vectorizer.transform(X_test)

# Print the first 10 features
print(tfidf_vectorizer.get_feature_names()[:10])

# Print the first 5 vectors of the tfidf training data
print(tfidf_train.A[:5])


# Create the CountVectorizer DataFrame: count_df
count_df = pd.DataFrame(
    count_train.A, columns=count_vectorizer.get_feature_names())

# Create the TfidfVectorizer DataFrame: tfidf_df
tfidf_df = pd.DataFrame(
    tfidf_train.A, columns=tfidf_vectorizer.get_feature_names())

# Print the head of count_df
print(count_df.head())

# Print the head of tfidf_df
print(tfidf_df.head())

# Calculate the difference in columns: difference
difference = set(count_df.columns) - set(tfidf_df.columns)
print(difference)

# Check whether the DataFrames are equal
print(count_df.equals(tfidf_df))


'''
Training and testing a classification model with scikit-learn
Naive Bayes classifier
- Naive Bayes model is commonly used for testing NLP classification problems 
* because of its basis in probability.
- It attempts to answer the question, given a particular piece of data, how likely is a particular outcome?

- Examples:
* If the plot has a spaceship, how likely is it to be sci-fi?
* Given a spaceship and an alien, how likely now is it a sci-fi?
- Each word acts as a feature from our CountVectorizer helping classify our text using probability.
- Naive Bayes: Simple and effective

Naive Bayes with scikit-learn
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

nb_classifier = MultinomialNB()
nb_classifier.fit(count_train, y_train)
pred = nb_classifier.predict(count_test)
metrics.accuracy_score(y_test, pred) -> in

0.85841849389820424 -> out

Confusion matrix
metrics.confusion_matrix(y_test, pred, labels=[0, 1]) -> in

array([ [6410,  563],
        [ 864, 2242]    ]) -> out

        Action  Sci-Fi
Action  6410    563
Sci-Fi  864     2242
'''

# Import the necessary modules

# Instantiate a Multinomial Naive Bayes classifier: nb_classifier
nb_classifier = MultinomialNB()

# Fit the classifier to the training data
nb_classifier.fit(count_train, y_train)

# Create the predicted tags: pred
pred = nb_classifier.predict(count_test)

# Calculate the accuracy score: score
score = metrics.accuracy_score(y_test, pred)
print(score)

# Calculate the confusion matrix: cm
cm = metrics.confusion_matrix(y_test, pred, labels=['FAKE', 'REAL'])
print(cm)


# Create a Multinomial Naive Bayes classifier: nb_classifier
nb_classifier = MultinomialNB()

# Fit the classifier to the training data
nb_classifier.fit(tfidf_train, y_train)

# Create the predicted tags: pred
pred = nb_classifier.predict(tfidf_test)

# Calculate the accuracy score: score
score = metrics.accuracy_score(y_test, pred)
print(score)

# Calculate the confusion matrix: cm
cm = metrics.confusion_matrix(y_test, pred, labels=['FAKE', 'REAL'])
print(cm)


''' 
Simple NLP, complex problems
'''

# Create the list of alphas: alphas
alphas = np.arange(0, 1, 0.1)

# Define train_and_predict()


def train_and_predict(alpha):
    # Instantiate the classifier: nb_classifier
    nb_classifier = MultinomialNB(alpha=alpha)
    # Fit to the training data
    nb_classifier.fit(tfidf_train, y_train)
    # Predict the labels: pred
    pred = nb_classifier.predict(tfidf_test)
    # Compute accuracy: score
    score = metrics.accuracy_score(y_test, pred)
    return score


# Iterate over the alphas and print the corresponding score
for alpha in alphas:
    print('Alpha: ', alpha)
    print('Score: ', train_and_predict(alpha))
    print()


# Get the class labels: class_labels
class_labels = nb_classifier.classes_

# Extract the features: feature_names
feature_names = tfidf_vectorizer.get_feature_names()

# Zip the feature names together with the coefficient array and sort by weights: feat_with_weights
feat_with_weights = sorted(zip(nb_classifier.coef_[0], feature_names))

# Print the first class label and the top 20 feat_with_weights entries
print(class_labels[0], feat_with_weights[:20])

# Print the second class label and the bottom 20 feat_with_weights entries
print(class_labels[1], feat_with_weights[-20:])
