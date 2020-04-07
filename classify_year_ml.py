#!/usr/bin/env python

import pandas as pd
import gensim
from sklearn.metrics import classification_report
from sklearn.linear_model import SGDClassifier
import sys
from nltk.corpus import stopwords
import string
import re
from nltk.tokenize import word_tokenize
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from nltk.stem.snowball import EnglishStemmer
inputf = "./data/english_lyrics.csv"
songs = pd.read_csv(inputf)
stemmer = EnglishStemmer()
def decades(y):
    yi = int(y)
    return y - (y%10)

(row,col) = songs.shape
lyrics = songs['lyrics']
years = songs['year']

songs["year"] = songs.apply(lambda x:decades(x['year']),axis=1)
songs = songs[~(songs.year < 1970)]
songs = songs.reset_index(drop=True)
print(songs)




(row,col) = songs.shape

def pick(src):
    return src[:1729]



yearmap = {}
years = songs['year']
for i in range(0,row):
    y = years[i]
    if y not in yearmap:
        yearmap[y] = []
    yearmap[y].append(i)

need = []
for y in yearmap:
    print(y, len(yearmap[y]))
    need.extend(yearmap[y][0:800])

print(len(need))

songs = songs[songs.index.isin(need)]
songs = songs.reset_index(drop=True)
(row,col) = songs.shape

# Process text for classification modeling
def preprocessText(text, remove_stops=True):

    # Remove everything between hard brackets
    text = re.sub(pattern="\[.+?\]( )?", repl='', string=text)

    # Change "walkin'" to "walking", for example
    text = re.sub(pattern="n\\\' ", repl='ng ', string=text)

    # Remove x4 and (x4), for example
    text = re.sub(pattern="(\()?x\d+(\))?", repl=' ', string=text)

    # Fix apostrophe issues
    text = re.sub(pattern="\\x91", repl="'", string=text)
    text = re.sub(pattern="\\x92", repl="'", string=text)
    text = re.sub(pattern="<u\+0092>", repl="'", string=text)

    # Make lowercase
    text = text.lower()

    # Special cases/words
    text = re.sub(pattern="'til", repl="til", string=text)
    text = re.sub(pattern="'til", repl="til", string=text)
    text = re.sub(pattern="gon'", repl="gon", string=text)

    # Remove \n from beginning
    text = re.sub(pattern='^\n', repl='', string=text)

    # Strip , ! ?, : and remaining \n from lyrics
    text = ''.join([char.strip(",!?:") for char in text])
    text = text.replace('\n', ' ')

    # Remove contractions
    # specific
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"won\’t", "will not", text)
    text = re.sub(r"can't", "can not", text)
    text = re.sub(r"can\’t", "can not", text)
    text = re.sub(r"let's", "let us", text)
    text = re.sub(r"let\’s", "let us", text)
    text = re.sub(r"ain't", "aint", text)
    text = re.sub(r"ain\’t", "aint", text)

    # general
    text = re.sub(r"n\'t", " not", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'s", " is", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'t", " not", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'m", " am", text)
    text = re.sub(r"n\’t", " not", text)
    text = re.sub(r"\’re", " are", text)
    text = re.sub(r"\’s", " is", text)
    text = re.sub(r"\’d", " would", text)
    text = re.sub(r"\’ll", " will", text)
    text = re.sub(r"\’t", " not", text)
    text = re.sub(r"\’ve", " have", text)
    text = re.sub(r"\’m", " am", text)

    # Remove remaining punctuation
    punc = string.punctuation
    text = ''.join([char for char in text if char not in punc])

    # Remove stopwords
    if remove_stops:
        stops = stopwords.words('english')
        text = ' '.join([word for word in text.split(' ') if word not in stops])

    # Remove double spaces and beginning/trailing whitespace
    text = re.sub(pattern='( ){2,}', repl=' ', string=text)
    text = text.strip()
    text = text.replace("\n"," ")
    text = ' '.join([stemmer.stem(word) for word in text.split(' ')])
    return(text)


songs['lyrics'] = songs.apply(lambda x: preprocessText(x['lyrics']), axis=1)
print(songs)

train, test = train_test_split(songs, test_size=0.2, stratify=songs.year, random_state=1)


## Naive Bayes test results
#text_mnb = Pipeline([('vect', CountVectorizer()),
#                     ('mnb', MultinomialNB(fit_prior=False))])
#text_mnb = text_mnb.fit(train.lyrics, train.year)
#print(cross_val_score(estimator=text_mnb, X=train.lyrics, y=train.year, cv=7).mean())
#

text_mnb = Pipeline([('vect', TfidfVectorizer(ngram_range=(1,2))),
                     ('svm', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-4,
                                            random_state=123))])
text_mnb = text_mnb.fit(train.lyrics, train.year)
cross_val_score(estimator=text_mnb, X=train.lyrics, y=train.year, cv=7).mean()

print(text_mnb.score(y=test.year, X=test.lyrics))
preds = text_mnb.predict(test.lyrics)
print(classification_report(y_pred=preds, y_true=test.year))
print(pd.crosstab(preds, test.year))
