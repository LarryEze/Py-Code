''' Basic features and readability scores '''

'''
Introduction to NLP feature engineering
Numerical data
e.g 
Iris dataset

sepal length    sepal width     petal length    petal width     class
6.3             2.9             5.6             1.8             Iris-virginica
4.9             3.0             1.4             0.2             Iris-setosa
5.6             2.9             3.6             1.3             Iris-versicolor
6.0             2.7             5.1             1.6             Iris-versicolor
7.2             3.6             6.1             2.5             Iris-virginica

- For any ML algorithm, data fed into it must be in tabular form and all the training features must be numerical
- ML algorithms can also work with categorical data provided they are converted into numerical form through one-hot encoding. 

One-hot encoding
sex     one-hot encoding    sex_female  sex_male
female  ->                  1           0
male    ->                  0           1
female  ->                  1           0
male    ->                  0           1
female  ->                  1           0
. . .   . . .               . . .       . . . 

One-hot encoding with pandas
# Import the pandas library
import pandas as pd

# Perform one-hot encoding on the 'sex' feature of df
df = pd.get_dummies(df, columns=['sex'])

Textual data
Movie Review Dataset
review                                              class
This movie is for dog lovers. A very poignant. . .  positive
The movie is forgettable. The plot lacked. . .      negative
A truly amazing movie about dogs. A gripping. . .   positive

- Textual data cannot be utilized directly by any machine learning or ML algorithm.
* The training feature 'review' isn't numerical, neither is it categorical to perform one-hot encoding on.
** There are 2 steps to make beofre making Textual dataset suitable for ML.

Text pre-processing
Standardize the text.
e.g
- converting words to lowercase 
e.g: Reduction to reduction 
and their base form.
e.g: reduction to reduce

Vectorization
- This is the process of converting text features to numerical training features.

0       1       2       . . .   n     class
0.03    0.71    0.00    . . .   0.22  positive
0.45    0.00    0.03    . . .   0.19  negative
0.14    0.18    0.00    . . .   0.45  positive

Basic features
- Number fo words
- Number of characters
- Average length of words
- Tweets

POS tagging
Word    POS
I       Pronoun
have    Verb
a       Article
dog     Noun

- POS tagging will label each word with its corresponding part-of-speech.

Named Entity Recognition
- It is done to find out if a particular noun is referring to a person, organization or country.

Noun        NER
Brian       Person
DataCamp    Organization
'''

# Print the features of df1
print(df1.columns)

# Perform one-hot encoding
df1 = pd.get_dummies(df1, columns=['feature 5'])

# Print the new features of df1
print(df1.columns)

# Print first five rows of df1
print(df1.head())


'''
Basic feature extraction
Number of characters
'I don't know.' # 13 characters
- The number of characters is the length of the string

# Compute the number of characters
text = 'I don't know.'
num_char = len(text)

# Print the number of characters
print(num_char) -> in

13 -> out

# Create a 'num_chars' feature
df['num_chars'] = df['review'].apply(len)

Number of words
# Split the string into words
text = 'Mary had a little lamb.'
words = text.split()

# Print the list containing words
print(words) -> in

['Mary', 'had', 'a', 'little', 'lamb.'] -> out

# Print number of words
print(len(words)) -> in

5 -> out

# Function that returns number of words in string
def word_count(string):
    # Split the string into words
    words = string.split()

    # Return length of words list
    return len(words)

# Create num_words feature in df
df['num_words'] = df['review'].apply(word_count)

Average word length
def avg_word_length(x):
    # Split the string into words
    words = x.split()

    # Compute length of each word and store in a separate list
    word_lengths = [len(word) for word in words]

    # Compute average word length
    avg_word_length = sum(word_lengths) / len(words)

    # Return average word length
    return(avg_word_length)

# Create a new feature avg_word_length
df['avg_word_length'] = df['review'].apply(doc_density)

Special features
Hashtags and mentions
# Function that returns number of hashtags
def hashtag_count(string):
    # Split the string into words
    words = string.split()

    # Create a list of hashtags
    hashtags = [word for word in words if word.startswith('#')]

    # Return number of hashtags
    return len(hashtags)

hashtag_count('@janedoe This is my first tweet! #FirstTweet #Happy') -> in

2 -> out

Other features
- Number of sentences
- Number of paragraphs
- Words starting with an uppercase
- All-capital words
- Numeric quantities
'''

# Create a feature char_count
tweets['char_count'] = tweets['content'].apply(len)

# Print the average character count
print(tweets['char_count'].mean())


# Function that returns number of words in a string
def count_words(string):
    # Split the string into words
    words = string.split()

    # Return the number of words
    return len(words)


# Create a new feature word_count
ted['word_count'] = ted['transcript'].apply(count_words)

# Print the average word count of the talks
print(ted['word_count'].mean())


# Function that returns numner of hashtags in a string
def count_hashtags(string):
    # Split the string into words
    words = string.split()

    # Create a list of words that are hashtags
    hashtags = [word for word in words if word.startswith('#')]

    # Return number of hashtags
    return (len(hashtags))


# Create a feature hashtag_count and display distribution
tweets['hashtag_count'] = tweets['content'].apply(count_hashtags)
tweets['hashtag_count'].hist()
plt.title('Hashtag count distribution')
plt.show()


# Function that returns number of mentions in a string
def count_mentions(string):
    # Split the string into words
    words = string.split()

    # Create a list of words that are mentions
    mentions = [word for word in words if word.startswith('@')]

    # Return number of mentions
    return (len(mentions))


# Create a feature mention_count and display distribution
tweets['mention_count'] = tweets['content'].apply(count_mentions)
tweets['mention_count'].hist()
plt.title('Mention count distribution')
plt.show()


'''
Readability tests
Overview of readability tests
- It's used to determine the readability of an English passage
- It makes use of a scale ranging from primary school up to college graduate level
- It uses a mathematical formula utilizing word, syllable and sentence count
- It is used in fake news and opinion spam detection

Readability test examples (english language)
- Flesch reading ease
- Gunning fog index
- Simple Measure of Gobbledygook (SMOG)
- Dale-Chall score

Flesch reading ease
- It is one of the oldest and most widely used tests
- Its dependent on 2 factors:
* The Greater the average sentence length, the harder the text is to read
** 'This is a short sentence.'
** 'This is a longer sentence with more words and it is harder to follow than the first sentence.'

* The Greater the average number of syllables in a word, harder the text is to read
** 'I live in my home.'
** 'I reside in my domicile.'
- The Higher the score, the greater its readability.

Flesch reading ease score interpretation
Reading ease    score Grade Level
90 - 100        5
80 - 90         6
70 - 80         7
60 - 70         8 - 9
50 -60          10 - 12
30 - 50         College
0 - 30          College Graduate

Gunning fog index
- It was developed in 1954
- It is also dependent on the average sentence length
- The Greater the percentage of complex words, the harder the text is to read
* Here, complex words refers to all words that have three (3) or more syllables.
- The Higher the index, the lesser its readability

Gunning fog index interpretation
Fog index   Grade Level
17          College graduate
16          College senior
15          College junior
14          College sophomore
13          College freshman
12          High school senior
11          High school junior
10          High school sophomore
9           High school freshman
8           Eighth grade
7           Seventh grade
6           Sixth grade

The textatistic library
# Import the Textatistic class
from textatistic import Textatistic

# Create a Textatistic Object
readability_scores = Textatistic(text).scores

# Generate scores
print(readability_scores['flesch_score'])
print(readability_scores['gunningfog_score']) -> in

21.14 # college graduate
16.26 # college senior -> out
'''

# Import Textatistic

# Compute the readability scores
readability_scores = Textatistic(sisyphus_essay).scores

# Print the flesch reading ease score
flesch = readability_scores['flesch_score']
print("The Flesch Reading Ease is %.2f" % (flesch))


# Import Textatistic

# List of excerpts
excerpts = [forbes, harvard_law, r_digest, time_kids]

# Loop through excerpts and compute gunning fog index
gunning_fog_scores = []
for excerpt in excerpts:
    readability_scores = Textatistic(excerpt).scores
    gunning_fog = readability_scores['gunningfog_score']
    gunning_fog_scores.append(gunning_fog)

# Print the gunning fog indices
print(gunning_fog_scores)


''' Text preprocessing, POS tagging and NER '''

'''
Tokenization and Lemmatization
Text sources
- News articles
- Tweets
- Comments

Making text machine friendly
- Dogs, dog
- reduction, REDUCING, reduce
- don't, do not
- won't, will not
* all of the above text samples should be standardized since they all mean the same thing

Text preprocessing techniques
- Converting words into lowercase
- Removing leading and trailing whitespaces
- Removing punctuation
- Removing stopwords
- Expanding contractions

Tokenization
- This is the process of splitting a string into its constituent tokens.
* These tokens may be sentences, words or punctuations and is specific to a particular language.
e.g
'I have a dog. His name is Hachi.'
Tokens:
['I', 'have', 'a', 'dog', '.', 'His', 'name', 'is', 'Hachi', '.']

'Don't do this.'
Tokens:
['Do', 'n't', 'do', 'this', '.']

Tokenization using SpaCy
import spacy

# Load the en_core_web_sm model
nlp = spacy.load('en_core_web_sm')

# Initialize string
string = 'Hello! I don't know what I'm doing here.'

# Create a Doc object
doc = nlp(string)

# Generate list of tokens
tokens = [token.text for token in doc]
print(tokens) -> in

['Hello', '!', 'I', 'do', 'n't', 'know', 'what', 'I', ''m', 'doing', 'here', '.'] -> out

Lemmatization
- It is the process of converting a word into its lowercased base form or lemma.
* It is an extremely powerful process of standardization.
e.g
- reducing, reduces, reduced, reduction -> reduce
- am, are, is -> be
- n't -> not
- 've -> have 

Lemmatization using SpaCy
import spacy

# Load the en_core_web_sm model
nlp = spacy.load('en_core_web_sm')

# Initialize string
string = 'Hello! I don't know what I'm doing here.'

# Create a Doc object
doc = nlp(string)

# Generate list of tokens
lemmas = [token.lemma_ for token in doc]
print(lemmas) -> in

['Hello', '!', '-PRON-', 'do', 'not', 'know', 'what', '-PRON-', ''be', 'do', 'here', '.'] -> out

* The standard behavior is to convert evry pronoun into the string '-PRON-'
'''


# Load the en_core_web_sm model
nlp = spacy.load('en_core_web_sm')

# Create a Doc object
doc = nlp(gettysburg)

# Generate the tokens
tokens = [token.text for token in doc]
print(tokens)


# Load the en_core_web_sm model
nlp = spacy.load('en_core_web_sm')

# Create a Doc object
doc = nlp(gettysburg)

# Generate lemmas
lemmas = [token.lemma_ for token in doc]

# Convert lemmas into a string
print(' '.join(lemmas))


'''
Text cleaning
Text cleaning techniques
- Unnecessary whitespaces and escape sequences
- Punctuations
- Special characters (numbers, emojis, etc.)
- Stopwords

isalpha()
'Dog'.isalpha() -> in

True -> out

'3dogs'.isalpha() -> in

False -> out

'12347'.isalpha() -> in

False -> out

'!'.isalpha() -> in

False -> out

'?'.isalpha() -> in

False -> out

A word of caution
- Abbreviations: U.S.A, U.K etc.
- Proper Nouns: word2vec and xto10x
* using isalpha() may not be sufficient in this cases
- One can write custom function (using regex - Regular expressions) for such cases.

Removing non-alphabetic characters
string = '' 
OMG!!!! This is like     the best thing ever \t\n.
Wow, such an amazing song! I'm hooked.  Top 5 definitely. ?
''

import spacy

# Generate list of tokens
nlp = spacy.load('en_core_web_sm')
doc = nlp(string)
lemmas = [token.lemma_ for token in doc]

# Remove tokens that are not alphabetic
a_lemmas = [lemma for lemma in lemmas if lemma.isalpha() or lemma == '-PRON-']

# Print string after text cleaning
print(' '.join(a_lemmas)) -> in

'omg this be like the good thing ever wow such an amazing song -PRON- be hooked top definitely' -> out

stopwords
- This are words that occur extremely commonly
- E.g. articles such as 'a' and 'the', be verbs such as 'is' and 'am', pronouns such as 'he' and she' etc.

Removing stopwords using spaCy
# Get list of stopwords
stopwords = spacy.lang.en.stop_words.STOP_WORDS

string = '' 
OMG!!!! This is like     the best thing ever \t\n.
Wow, such an amazing song! I'm hooked.  Top 5 definitely. ?
''

import spacy

# Generate list of tokens
nlp = spacy.load('en_core_web_sm')
doc = nlp(string)
lemmas = [token.lemma_ for token in doc]

# Remove tokens that are not alphabetic
a_lemmas = [lemma for lemma in lemmas if lemma.isalpha() and lemma not in stopwords]

# Print string after text cleaning
print(' '.join(a_lemmas)) -> in

'omg like good thing wow amazing song hooked definitely' -> out

- Always exercise caution while using third party stopword lists.
* Its often advisable to create your custom stopword lists.

Other text preprocessing techniques
- Removing HTML / XML tags
- Replacing accented characters (such as `e)
- Correcting spelling errors

A word of caution
- The text preprocessing techniques you use is always dependent on the application.
- There are many applications which may find punctuations, numbers and emojis useful, so it may  be wise to not remove them.
- NB*: Always use only those text preprocessing techniques that are relevant to your application.
'''

# Load model and create Doc object
nlp = spacy.load('en_core_web_sm')
doc = nlp(blog)

# Generate lemmatized tokens
lemmas = [token.lemma_ for token in doc]

# Remove stopwords and non-alphabetic tokens
a_lemmas = [lemma for lemma in lemmas if lemma.isalpha()
            and lemma not in stopwords]

# Print string after text cleaning
print(' '.join(a_lemmas))


# Function to preprocess text
def preprocess(text):
    # Create Doc object
    doc = nlp(text, disable=['ner', 'parser'])

    # Generate lemmas
    lemmas = [token.lemma_ for token in doc]

    # Remove stopwords and non-alphabetic characters
    a_lemmas = [lemma for lemma in lemmas if lemma.isalpha()
                and lemma not in stopwords]

    return ' '.join(a_lemmas)


# Apply preprocess to ted['transcript']
ted['transcript'] = ted['transcript'].apply(preprocess)
print(ted['transcript'])


'''
Part-of-speech tagging
Applications
- Word-sense disambiguation i.e to identify the sense of a word in a sentence.
e.g
* 'The bear is a majestic animal'
* 'Please bear with me'
- Sentiment analysis
- Question answering systems
- Linguistic approaches to determining fake news and opinion spam detection.

POS tagging
- It is the process of assigning every word(or token) in a piece of text, its corresponding part-of-speech.
e.g
'Jane is an amazing guitarist.'

- POS Tagging:
* Jane      -> proper noun
* is        -> verb
* an        -> determiner
* amazing   -> adjective
* guitarist -> noun

POS tagging using spaCy
import spacy

# Load the en_core_web_sm model
nlp = spacy.load('en_core_web_sm')

# Initialize string
string = 'Jane is an amazing guitarist.'

# Create a Doc object
doc = nlp(string)

# Generate list of tokens and pos tags
pos = [(token.text, token.pos_) for token in doc]
print(pos) -> in

[('Jane', 'PROPN'),  ('is', 'VERB'), ('an', 'DET'), ('amazing', 'ADJ'), ('guitarist', 'NOUN')] -> out

- NB*:  Remember that POS tagging is not an exact science.
* spaCy infers the POS tags of these words based on the predictions given by its pre-trained models.
i.e The accuracy of the POS tagging is dependent on the data that the model has been trained on and the data that it is being used on.

POS annotation in spaCy
- PROPN -> proper noun
- DET ->
'''

# Load the en_core_web_sm model
nlp = spacy.load('en_core_web_sm')

# Create a Doc object
doc = nlp(lotf)

# Generate tokens and pos tags
pos = [(token.text, token.pos_) for token in doc]
print(pos)


nlp = spacy.load('en_core_web_sm')

# Returns number of proper nouns


def proper_nouns(text, model=nlp):
    # Create doc object
    doc = model(text)
    # Generate list of POS tags
    pos = [token.pos_ for token in doc]

    # Return number of proper nouns
    return pos.count('PROPN')


print(proper_nouns("Abdul, Bill and Cathy went to the market to buy apples.", nlp))


nlp = spacy.load('en_core_web_sm')

# Returns number of other nouns


def nouns(text, model=nlp):
    # Create doc object
    doc = model(text)
    # Generate list of POS tags
    pos = [token.pos_ for token in doc]

    # Return number of other nouns
    return pos.count('NOUN')


print(nouns("Abdul, Bill and Cathy went to the market to buy apples.", nlp))


headlines['num_propn'] = headlines['title'].apply(proper_nouns)

# Compute mean of proper nouns
real_propn = headlines[headlines['label'] == 'REAL']['num_propn'].mean()
fake_propn = headlines[headlines['label'] == 'FAKE']['num_propn'].mean()

# Print results
print("Mean no. of proper nouns in real and fake headlines are %.2f and %.2f respectively" %
      (real_propn, fake_propn))


headlines['num_noun'] = headlines['title'].apply(nouns)

# Compute mean of other nouns
real_noun = headlines[headlines['label'] == 'REAL']['num_noun'].mean()
fake_noun = headlines[headlines['label'] == 'FAKE']['num_noun'].mean()

# Print results
print("Mean no. of other nouns in real and fake headlines are %.2f and %.2f respectively" %
      (real_noun, fake_noun))


'''
Named entity recognition
Applications
- It is used in creating efficient search algorithms
- It is used in Question answering systems
- It is used in News article classification
- It is used in Customer service centers to classify and record their complaints efficiently.

Named entity recognition
- It is anything that can be denoted with a proper name or a proper noun.
* Therefore, it is the process of identifying such named entities in a piece of text and classifying them into predefined categories.
- Categories include person, organization, country etc.

'John Doe is a software engineer working at Google. He lives in France.'
- Named Entities
* John Doe  -> person
* Google    -> organization
* France    -> country (geopolitical entity)

NER using spaCy
import spacy
string = 'John Doe is a software engineer working at Google. He lives in France.'

# Load model and create Doc object
nlp = spacy.load('en_core_web_sm')
doc = nlp(string)

# Generate named entities
ne = [(ent.text, ent.label_) for ent in doc.ents]
print(ne) -> in

[('John Doe', 'PERSON'), ('Google', 'ORG'), ('France', 'GPE')] -> out

NER annotations in spaCy
- More than 15 categories of named entities
- e.g
TYPE        DESCRIPTION
PERSON      People, including fictional.
NORP        Nationalities or religious or political groups.
FAC         Buildings, airports, highways, bridges, etc.
ORG         Companies, agencies, institutions, etc.
GPE         Countries, cities, states.
. . .       . . .

A word of caution
- spaCy model is not perfect
- Its performance is dependent on training and test data
- Its better to train your models with specialized data for nnuanced cases
- spaCy model is Language specific
'''

# Load the required model
nlp = spacy.load('en_core_web_sm')

# Create a Doc instance
text = 'Sundar Pichai is the CEO of Google. Its headquarters is in Mountain View.'
doc = nlp(text)

# Print all named entities and their labels
for ent in doc.ents:
    print(ent.text, ent.label_)


def find_persons(text):
    # Create Doc object
    doc = nlp(text)

    # Identify the persons
    persons = [ent.text for ent in doc.ents if ent.label_ == 'PERSON']

    # Return persons
    return persons


print(find_persons(tc))


''' N-Gram models '''

'''
Building a bag of words model
- Vectorization is the process of converting text into vectors

Recap of data format for ML algorithms
For any ML algorithm,
- The Data must be in tabular form
- All the training features must be numerical

Bag of words model
- It is a procedure of extracting word tokens from a text document
* Compute the frequency of the word tokens
* Construct a word vector out of these frequencies and the vocabulary of the entire corpus of documents.

Bag of words model example
Corpus
'The lion is the king of the jungle'
'Lions have lifespans of a decade'
'The lion is an endangered species'

Bag of words model example
Vocabulary -> a, an, decade, endangered, have, is, jungle, king, lifespans, lion, Lions, of, species, the, The

'The lion is the king of the jungle'
[0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 2, 1]

'Lions have lifespans of a decade'
[1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0]

'The lion is an endangered species'
[0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1]

- NB*: Each value in the vector corresponds to the frequency of the corresponding word in the vocabulary.

Text preprocessing
- Lions, lion -> lion
- The, the -> the
- No punctuations
- No stopwords
- This will lead to smaller vocabularies
- Reducing the number of dimensions helps improve performance

Bag of words model using sklearn
corpus = pd.Series([
'The lion is the king of the jungle',
'Lions have lifespans of a decade',
'The lion is an endangered species'
])

# Import CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer

# Create CountVectorizer object
vectorizer = CountVectorizer()

# Generate matrix of word vectors
bow_matrix = vectorizer.fit_transform(corpus)
print(bow_matrix.toarray()) -> in

array([[0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 3],
        [0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0],
        [1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1]], dtype=int64) -> out

- CountVectorizer() automatically lowercases words and ignores single character tokens such as 'a'.
* Also, it doesn't necessarily index the vocabulary in alphabetical order.
- get_feature_names() essentially gives us a list which represents the mapping of the feature indices to the feature name in the vocabulary.
'''

# Import CountVectorizer

# Create CountVectorizer object
vectorizer = CountVectorizer()

# Generate matrix of word vectors
bow_matrix = vectorizer.fit_transform(corpus)

# Print the shape of bow_matrix
print(bow_matrix.shape)


# Import CountVectorizer

# Create CountVectorizer object
vectorizer = CountVectorizer()

# Generate matrix of word vectors
bow_lem_matrix = vectorizer.fit_transform(lem_corpus)

# Print the shape of bow_lem_matrix
print(bow_lem_matrix.shape)


# Create CountVectorizer object
vectorizer = CountVectorizer()

# Generate matrix of word vectors
bow_matrix = vectorizer.fit_transform(corpus)

# Convert bow_matrix into a DataFrame
bow_df = pd.DataFrame(bow_matrix.toarray())

# Map the column names to vocabulary
bow_df.columns = vectorizer.get_feature_names()

# Print bow_df
print(bow_df)


'''
Building a BoW Naive Bayes classifier
Spam filtering
message                                                                                                                     label
WINNER!! As a valued network customer you have been selected to receive a $900 prize reward! To claim call 09061701461      spam
Ah, work. I vaguely remember that. What does it feel like?                                                                  ham

Steps
There are 3 steps
- Text preprocessing
- Building a bag-of-words model (or representation)
- Machine learning

Text preprocessing using CountVectorizer
CountVectorizer arguments
- lowercase: False, True
- strip_accents: 'unicode', 'ascii', None
- stop_words: 'english', list, None
- token_pattern: regex
- tokenizer: function

- NB*: CountVectorizer cannot perform certain steps such as lemmatization automatically, this is where spaCy is useful.
* CountVectorizer's main job is to convert a corpus into a matrix of numerical vectors.

Building the BoW (Bag-of-Words) model
# Import CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer

# Create CountVectorizer object
vectorizer = CountVectorizer(strip_accents = 'ascii', stop_words = 'english', lowercase = False)

# Import train_test_split
from sklearn.model_selection import train_test_split

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size = 0.25)

# Generate training BoW vectors
X_train_bow = vectorizer.fit_transform(X_train)

# Generate test BoW vectors
X_test_bow = vectorizer.transform(X_test)

Training the Naive Bayes classifier
# Import MultinomialNB
from sklearn.naive_bayes import MultinomialNB

# Create MultinomialNB object
clf = MultinomialNB()

# Train clf
clf.fit(X_train_bow, y_train)

# Compute accuracy on test set
accuracy = clf.score(X_test_bow, y_test)
print(accuracy) -> in

0.760051 -> out
'''

# Import CountVectorizer

# Create a CountVectorizer object
vectorizer = CountVectorizer(lowercase=True, stop_words='english')

# Fit and transform X_train
X_train_bow = vectorizer.fit_transform(X_train)

# Transform X_test
X_test_bow = vectorizer.transform(X_test)

# Print shape of X_train_bow and X_test_bow
print(X_train_bow.shape)
print(X_test_bow.shape)


# Create a MultinomialNB object
clf = MultinomialNB()

# Fit the classifier
clf.fit(X_train_bow, y_train)

# Measure the accuracy
accuracy = clf.score(X_test_bow, y_test)
print("The accuracy of the classifier on the test set is %.3f" % accuracy)

# Predict the sentiment of a negative review
review = "The movie was terrible. The music was underwhelming and the acting mediocre."
prediction = clf.predict(vectorizer.transform([review]))[0]
print("The sentiment predicted by the classifier is %i" % (prediction))


'''
Building n-gram models
BoW shortcomings
review                                  label
'The movie was good and not boring'     positive
'The movie was not good and boring'     negative

- They have exactly the same BoW representation!
- The context of the words is lost.
- Sentiment is dependent on the position of 'not'.

n-grams
- It is a continuous sequence of n elements (or words) in a given document.
- n = 1 -> Bag-of-Words
'for you a thousand times over'

- n = 2 (bigrams) , n-grams:
[
'for you',
'you a',
'a thousand',
'thousand times'
'times over'
]

- n = 3, n-grams:
[
'for you a',
'you a thousand',
'a thousand times'
'thousand times over'
]

- n-grams are used to capture more context and also account for cases like 'not'.

Applications
- Sentence completion
- Spelling correction
- Machine translation correction
* In all of these cases, the model computes the probability of n words occuring contiguously to perform the above processes.

Building n-gram models using scikit-learn
Generates only bigrams.
bigrams = CountVectorizer(ngram_range = (2, 2))

Generates unigrams, bigrams and trigrams
ngrams = CountVectorizer(ngram_range = (1, 3))

Shortcomings
- Adding higher order n-grams increases the number of dimensions even more and while performing machine learning, leads to a problem known as the curse of dimensionality.
- n-grams for n greater than 3 becomes exceedingly rare to find in multiple documents so that feature becomes effectively useless.
* Try to restrict to n-grams where n is small
'''

# Generate n-grams upto n=1
vectorizer_ng1 = CountVectorizer(ngram_range=(1, 1))
ng1 = vectorizer_ng1.fit_transform(corpus)

# Generate n-grams upto n=2
vectorizer_ng2 = CountVectorizer(ngram_range=(1, 2))
ng2 = vectorizer_ng2.fit_transform(corpus)

# Generate n-grams upto n=3
vectorizer_ng3 = CountVectorizer(ngram_range=(1, 3))
ng3 = vectorizer_ng3.fit_transform(corpus)

# Print the number of features for each model
print("ng1, ng2 and ng3 have %i, %i and %i features respectively" %
      (ng1.shape[1], ng2.shape[1], ng3.shape[1]))


# Define an instance of MultinomialNB
clf_ng = MultinomialNB()

# Fit the classifier
clf_ng.fit(X_train_ng, y_train)

# Measure the accuracy
accuracy = clf_ng.score(X_test_ng, y_test)
print("The accuracy of the classifier on the test set is %.3f" % accuracy)

# Predict the sentiment of a negative review
review = "The movie was not good. The plot had several holes and the acting lacked panache."
prediction = clf_ng.predict(ng_vectorizer.transform([review]))[0]
print("The sentiment predicted by the classifier is %i" % (prediction))


start_time = time.time()
# Splitting the data into training and test sets
train_X, test_X, train_y, test_y = train_test_split(
    df['review'], df['sentiment'], test_size=0.5, random_state=42, stratify=df['sentiment'])

# Generating ngrams
vectorizer = CountVectorizer()
train_X = vectorizer.fit_transform(train_X)
test_X = vectorizer.transform(test_X)

# Fit classifier
clf = MultinomialNB()
clf.fit(train_X, train_y)

# Print accuracy, time and number of dimensions
print("The program took %.3f seconds to complete. The accuracy on the test set is %.2f. The ngram representation had %i features." %
      (time.time() - start_time, clf.score(test_X, test_y), train_X.shape[1]))


start_time = time.time()
# Splitting the data into training and test sets
train_X, test_X, train_y, test_y = train_test_split(
    df['review'], df['sentiment'], test_size=0.5, random_state=42, stratify=df['sentiment'])

# Generating ngrams
vectorizer = CountVectorizer(ngram_range=(1, 3))
train_X = vectorizer.fit_transform(train_X)
test_X = vectorizer.transform(test_X)

# Fit classifier
clf = MultinomialNB()
clf.fit(train_X, train_y)

# Print accuracy, time and number of dimensions
print("The program took %.3f seconds to complete. The accuracy on the test set is %.2f. The ngram representation had %i features." %
      (time.time() - start_time, clf.score(test_X, test_y), train_X.shape[1]))


''' TF-IDF and similarity scores '''

'''
Building tf-idf document vectors
n-gram modeling
- In this method, the weight of a dimension for the vector representation of a document is dependent on the number of times the word corresponding to the dimension occurs in the document.
* Document contains the word 'human' in five places.
* Dimension corresponding to 'human' has weight 5.

Motivation
- Some words occur very commonly across all documents in the corpus
- e.g Corpus of documents on the universe
* One document has 'jupiter' and 'universe' occuring 20 times each
* 'jupiter' rarely occurs in the other documents, 'universe' is common
* Give more weight to 'jupiter' on account of exclusivity.

Applications
- They can be used to automatically detect stopwordsfor the corpus instead of relying on a generic list.
- They are used in search algorithms to determine the ranking of pages containing the search query.
- They are used in recommender systems.
- In a lot of cases, this kind of weighting also generates better performance during predictive modeling.

Term frequency - inverse document frequency
- It is based on the idea that the weight of a term in a document should be proportional to its frequency
* and an inverse function of the number of documents in which it occurs.

Mathematical formula
w(i, j) = tf(i,j).log( N / df(i) )

w(i, j) -> weight of term i in document j
tf(i,j) -> term frequency of term i in document j
N -> number of documents in the corpus
df(i) -> number of documents containing term i

- The higher the tf-idf weight, the more important is the word in characterizing the document.
- A high tf-idf weight for a word in a document may imply that the word is relatively exclusive to that particular document or that the word occurs extremely commonly in the document, or both.

tf-idf using scikit-learn
# Import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Create TfidfVectorizer object
vectorizer = TfidfVectorizer()

# Generate matrix of word vectors
tfidf_matrix = vectorizer.fit_transform(corpus)
print(tfidf_matrix.toarray()) -> in

[[0.        0.         0.        0.         0.25434658 0.33443519 0.33443519 0.         0.25434658 0.         0.25434658 0.        0.76303975]
 [0.        0.46735098 0.        0.46735098 0.         0.         0.         0.46735098 0.         0.46735098 0.35543247 0.        0.        ]
. . . -> out 
'''

# Import TfidfVectorizer

# Create TfidfVectorizer object
vectorizer = TfidfVectorizer()

# Generate matrix of word vectors
tfidf_matrix = vectorizer.fit_transform(ted)

# Print the shape of tfidf_matrix
print(tfidf_matrix.shape)


'''
Cosine similarity
- The cosine similarity score of two vectors is the cosine of the angle between the vectors.
* Mathematically, it is the ratio of the dot product of the vectors and the product of the magnitude of the two vectors.

sim(A, B) = cos(θ) = A.B / ||A||||B||

The dot product
- It is computed by summing the product of values across corresponding dimensions of the vectors.
- Consider two vectors,
V = (v(1), v(2), . . . , v(n)), W = (w(1), w(2), . . . , w(n))
Then the dot product of V and W is, 
V.W = (v(1) x w(1)) + (v(2) x w(2)) + . . . + (v(n) x w(n))

Example:
A = (4, 7, 1), B = (5, 2, 3)
A.B = (4 x 5) + (7 x 2) + . . . (1 x 3)
= 20 + 14 + 3 = 37

Magnitude of a vector
- It is essentially the length of the vector.
- Mathematically, it is defined as the square root of the sum of the squares of values across all the dimensions of a vector.
- For any vector,
V = (v(1), v(2), . . . , v(n))
The magnitude is defined as,
||V|| = square root( (v(1))^2 + (v(2))^2 + . . . + (v(n))^2 )

Example:
A = (4, 7, 1) 
||A|| = square root( (4)^2 + (7)^2 + (1)^2 )
= square root( 16 + 49 + 1 )
= square root( 66 )

B = (5, 2, 3)
||A|| = square root( (5)^2 + (2)^2 + (3)^2 )
= square root( 25 + 4 + 9 )
= square root( 38 )

The cosine score
The cosine score( A, B) = 37 / ( square root( 66 ) x  square root( 38 ) )
= 0.7388

Cosine Score: points to remember
- Value between -1 and 1
- In NLP, value between 0 and 1
* 0 indicates no similarity and 1 indicates that the documents are identical.
- It is robust to document length.

Implemetation using scikit-learn 
- cosine_similarity takes in 2D arrays as arguments
* Passing in 1D arrays will throw an error

# Import the cosine_similarity
from sklearn.metrics.pairwise import cosine_similarity

# Define two 3-dimensional vectors A and B
A = (4, 7, 1)
B = (5, 2, 3)

# Compute the cosine score of A and B
score = cosine_similarity([A], [B])

# Print the cosine score
print(score) -> in

array([[ 0.73881883]]) -> out
'''

# Initialize numpy vectors
A = np.array([1, 3])
B = np.array([-2, 2])

# Compute dot product
dot_prod = np.dot(A, B)

# Print dot product
print(dot_prod)


# Initialize an instance of tf-idf Vectorizer
tfidf_vectorizer = TfidfVectorizer()

# Generate the tf-idf vectors for the corpus
tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)

# Compute and print the cosine similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
print(cosine_sim)


'''
Building a plot line based recommender
Movie recommender
Title                       Overview
Shanghai Triad              A provincial boy related to a Shanghai crime family is recruited by his uncle into cosmopolitan Shanghai in the 1930s to be a servant to a ganglord's mistress.
Cry, the Beloved Country    A South-African preacher goes to search for his wayward son who has committed a crime in the big city.

Movie recommender
get_recommendations('The Godfather') -> in

1178                The Godfather: Part II
44030   The Godfather Trilogy: 1972 - 1990
1914               The Godfather: Part III
23126                           Blood Ties
11297                     Household saints
34717                    Start Liquidation
10821                             Election
38030                           Goodfellas
17729                    Short Sharp Shock
26293                   Beck 28 - Familjen
Name: title, dtype: object -> out

Steps
- Text preprocessing
- Generate tf-idf vectors for the overviews
- Generate a cosine similarity matrix which contains the pairwise similarity scores of every movie with every other movie.

The recommender function
- It takes a movie title, cosine similarity matrix and indices series as arguments.
* The indices series is a reverse mapping of movie titles with their indices in the original dataframe.
- The function extracts the pairwise cosine similarity scores for the movie passed in with every other movie.
- It sort the scores in descending order.
- It output the titles of movies corresponding to the highest scores.
- NB*: The function ignores the highest similarity score ( of 1 ). This is because the movie most similar to a given movie is the movie itself.

Generating tf-idf vectors
# Import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Create TfidfVectorizer object
vectorizer = TfidfVectorizer()

# Generate matrix of tf-idf vectors
tfidf_matrix = vectorizer.fit_transform(movie_plots)

Generating cosine similarity matrix
# Import cosine_similarity
from sklearn.metrics.pairwise import cosine_similarity

# Generate cosine similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix) -> in

array([[1.        , 0.27435345, 0.23092036, . . . , 0.        , 0.        , 0.00758112],
    [0.27435345, 1.        , 0.1246955 , . . . , 0.        , 0.        , 0.00740494],
    . . . ,
    [0.00758112, 0.00740494, 0.        , . . . , 0.        , 0.        , 1.        ]]) -> out

The linear_kernel function
- The magnitude of a tf-idf vector is always 1
- Cosine score between two tf-idf vectors is their dot product.
- This fact can significantly improve the computation time of our cosine similarity matrix as we do not need to compute the magnitudes while working with tf-idf vectors. 
- Use linear_kernel instead of cosine_similarity since it computes the pairwise dot product of every vector with every other vector.

Generating cosine similarity matrix
# Import cosine_similarity
from sklearn.metrics.pairwise import linear_kernel

# Generate cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix) -> in

array([[1.        , 0.27435345, 0.23092036, . . . , 0.        , 0.        , 0.00758112],
    [0.27435345, 1.        , 0.1246955 , . . . , 0.        , 0.        , 0.00740494],
    . . . ,
    [0.00758112, 0.00740494, 0.        , . . . , 0.        , 0.        , 1.        ]]) -> out

The get_recommendations functions
- It can be used to generate recommendations using the cosine similarity matrix.

get_recommendations('The Lion King', cosine_sim, indices) -> in

7782                      African Cats
5877    The Lion King 2: Simba's Pride
4524                         Born Free
2719                          The Bear
4770     Once Upon a Time in China III
7070                        Crows Zero
739                   The Wizard of Oz
8926                   The Jungle Book
1749                 Shadow of a Doubt
7993                      October Baby
Name: title, dtype: object -> out
'''

# Record start time
start = time.time()

# Compute cosine similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Print cosine similarity matrix
print(cosine_sim)

# Print time taken
print("Time taken: %s seconds" % (time.time() - start))


# Record start time
start = time.time()

# Compute cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Print cosine similarity matrix
print(cosine_sim)

# Print time taken
print("Time taken: %s seconds" % (time.time() - start))


# Initialize the TfidfVectorizer
tfidf = TfidfVectorizer(stop_words='english')

# Construct the TF-IDF matrix
tfidf_matrix = tfidf.fit_transform(movie_plots)

# Generate the cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Generate recommendations
print(get_recommendations('The Dark Knight Rises', cosine_sim, indices))


# Generate mapping between titles and index
indices = pd.Series(metadata.index, index=metadata['title']).drop_duplicates()


def get_recommendations(title, cosine_sim, indices):
    # Get index of movie that matches title
    idx = indices[title]
    # Sort the movies based on the similarity scores
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    # Get the scores for 10 most similar movies
    sim_scores = sim_scores[1:11]
    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]
    # Return the top 10 most similar movies
    return metadata['title'].iloc[movie_indices]


# Initialize the TfidfVectorizer
tfidf = TfidfVectorizer(stop_words='english')

# Construct the TF-IDF matrix
tfidf_matrix = tfidf.fit_transform(transcripts)

# Generate the cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Generate recommendations
print(get_recommendations('5 ways to kill your dreams', cosine_sim, indices))


'''
Beyond n-grams: word embeddings
The problem with Bag-of- Words (BoW) and tf-idf
'I am happy'
'I am joyous'
'I am sad'

- In this scenario 'I am happy' and 'I am joyous' will have the same score as 'I am happy' and 'I am sad' regardless of how we vectorize it.
* This is because 'happy', 'joyous' and 'sad' are considered to be completely different words.
* However, we know that happy and joyous are more similar to each other than sad.

Word embeddings
- It is the process of mapping words into an n-dimensional vector space.
- This vectors are usually produced using deep learning models and huge amounts of data.
- This vectors are used to discern how similar two words are to each other
- They can also be used to detect synonyms and antonyms
- They are also capable of capturing complex relationships. 
* For instance, it can be used to detect that the following words are related to one another 
** King - Queen -> Man - Woman
** France - Paris -> Russia - Moscow

- NB*: They are not trained on user data; 
* they are dependent on the pre-trained spaCy model you're using 
* and are independent of the size of your dataset.

Word embeddings using spaCy
- NB*: It is advisable to load larger spacy models while working with word vectors.
* This is because the 'en_core_web_sm' model does not technically ship with word vectors but context specific tensors, which tend to give relatively poorer results.

import spacy

# Load model and create Doc object
nlp = spacy.load('en_core_web_lg')
doc = nlp('I am happy') 

# Generate word vectors for each token
for token is doc:
    print(token.vector) -> in

[-1.0747459e+00 4.8677087e-02 5.6630421e+00 1.6680446e+00
-1.3194644e+00 -1.5142369e+00 1.1940931e+00 -3.0168812e+00
. . . -> out

Word similarities
doc = nlp('happy joyous sad')

for token1 in doc:
    for token2 in doc:
        print(token1.text, token2.text, token1.similarity(token2)) -> in

happy happy     1.0
happy joyous    0.63244456
happy sad       0.37338886
joyous happy    0.63244456
joyous joyous   1.0
joyous sad      0.5340932
. . . -> out

Document similarities
# Generate doc objects
sent1 = nlp('I am happy')
sent2 = nlp('I am sad')
sent3 = nlp('I am joyous')

# Compute similarity between sent1 and sent2
sent1.similarity(sent2) -> in

0.9273363837282105 -> out

# Compute similarity between sent1 and sent3
sent1.similarity(sent3) -> in

0.9403554938594568 -> out

- NB*: The similarity scores are high in both cases because all sentences share 2 out of their three words, 'I' and 'am' 
'''

# Create the doc object
doc = nlp(sent)

# Compute pairwise similarity scores
for token1 in doc:
    for token2 in doc:
        print(token1.text, token2.text, token1.similarity(token2))


# Create Doc objects
mother_doc = nlp(mother)
hopes_doc = nlp(hopes)
hey_doc = nlp(hey)

# Print similarity between mother and hopes
print(mother_doc.similarity(hopes_doc))

# Print similarity between mother and hey
print(mother_doc.similarity(hey_doc))
