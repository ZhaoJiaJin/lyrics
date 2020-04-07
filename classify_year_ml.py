#!/usr/bin/env python

import pandas as pd
import gensim
import random
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
from xgboost import XGBClassifier

method=sys.argv[1]
vectrizer = sys.argv[2]
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




(row,col) = songs.shape

def randompick(src,l):
    #return src[:l]
    res = []
    while len(res) < l:
        got = random.choice(src)
        src.remove(got)
        res.append(got)

    return res



yearmap = {}
years = songs['year']
for i in range(0,row):
    y = years[i]
    if y not in yearmap:
        yearmap[y] = []
    yearmap[y].append(i)

need = []
targetcount = 0
for y in yearmap:
    if len(yearmap[y]) >= 800:
        targetcount += 1
        need.extend(randompick(yearmap[y],800))

print("number of decades:",targetcount)

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

train, test = train_test_split(songs, test_size=0.2, stratify=songs.year, random_state=1)


vectorizer = None
if vectrizer == "count":
    vectorizer = CountVectorizer()
else:
    vectorizer = TfidfVectorizer(ngram_range=(1,3))

if method == "bayes":
    # Naive Bayes test results
    text_mnb = Pipeline([('vect', vectorizer),
                         ('mnb', MultinomialNB(fit_prior=False))])
    text_mnb = text_mnb.fit(train.lyrics, train.year)
    cross_val_score(estimator=text_mnb, X=train.lyrics, y=train.year, cv=7).mean()

    print("accuracy:",text_mnb.score(y=test.year, X=test.lyrics))
    preds = text_mnb.predict(test.lyrics)
    print(classification_report(y_pred=preds, y_true=test.year))
    #print(pd.crosstab(preds, test.year))

elif method == "svm":
    # SVM
    text_mnb = Pipeline([('vect', vectorizer),
                         ('svm', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-4,
                                                random_state=123))])
    text_mnb = text_mnb.fit(train.lyrics, train.year)
    cross_val_score(estimator=text_mnb, X=train.lyrics, y=train.year, cv=7).mean()

    print("accuracy:",text_mnb.score(y=test.year, X=test.lyrics))
    preds = text_mnb.predict(test.lyrics)
    print(classification_report(y_pred=preds, y_true=test.year))
    #print(pd.crosstab(preds, test.year))

elif method == "xgb":
    # XGB model
    vect = vectorizer
    #vect = CountVectorizer()
    vect.fit_transform(train.lyrics)
    vect_test = vect.transform(pd.Series(test.lyrics))
    vect_train = vect.transform(pd.Series(train.lyrics))

    text_mnb = XGBClassifier(learning_rate=0.25, subsample=0.8, gamma=1, random_state=123, max_depth=6, max_delta_step=1).fit(vect_train, train.year)

    print("accuracy:",text_mnb.score(y=test.year, X=vect_test))
    preds = text_mnb.predict(vect_test)
    print(classification_report(y_pred=preds, y_true=test.year))
    #print(pd.crosstab(preds, test.year))
else:
    print("please provide method: bayes, svm, or xgb")
