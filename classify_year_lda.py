#!/usr/bin/env python

import pandas as pd
import gensim
import sys
from nltk.tokenize import word_tokenize
inputf = "./data/yearlyrics.csv"
songs = pd.read_csv(inputf)

def decades(y):
    yi = int(y)
    return y - (y%10)

(row,col) = songs.shape
lyrics = songs['lyrics']
years = songs['year']

processed_docs = []
yearmap = {}
for i in range(0,row):
    deca = decades(years[i])
    lyc = lyrics[i]
    if deca not in yearmap:
        yearmap[deca] = []
    processed_docs.append(word_tokenize(lyc))
    yearmap[deca].append(i)



# Filter out tokens that appear in
# less than 15 documents (absolute number) or
# more than 0.5 documents (fraction of total corpus size, not absolute number).

dictionary = gensim.corpora.Dictionary(processed_docs)
#count = 0
dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=1000)
#for k, v in dictionary.iteritems():
#    print(k, v)
    #count += 1
    #if count > 2250/2:
    #    break

maxid = 0
bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
print(len(bow_corpus))
for ll in bow_corpus:
    for l in ll:
        if l[0] > maxid:
            maxid = l[0]

print(maxid)

from gensim import corpora, models
tfidf = models.TfidfModel(bow_corpus)
corpus_tfidf = tfidf[bow_corpus]

lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=5, id2word=dictionary, passes=2, workers=2)


#for idx, topic in lda_model.print_topics(-1):
#    print('Topic: {} \nWords: {}'.format(idx, topic))

#print(type(lda_model))

#lda_model= gensim.models.LdaMulticore(corpus_tfidf, num_topics=5, id2word=dictionary, passes=2, workers=4)
#for idx, topic in lda_model_tfidf.print_topics(-1):
#    print('Topic: {} Word: {}'.format(idx, topic))



# I don't know how to calculate accuracy for LDA model, so I came up with this method:
# 1. Use the model to classify all the documents(lyrics)
# 2. If most the documents(lyrics) under the same category(decade) are classified as the same topic,  I will assume this model has good performance
# PS: And the performance is not very well
predictmap = {}
for ya in yearmap:
    predictmap[ya] = []
for yy in yearmap:
    for lyc in yearmap[yy]:
        bow_vector = bow_corpus[lyc]
        max_score = 0
        chooseidx = -1
        for index, score in sorted(lda_model[bow_vector], key=lambda tup: -1*tup[1]):
            if score > max_score:
                max_score = score
                chooseidx = index
        predictmap[yy].append(chooseidx)

def calcfd(src):
    res = {}
    for i in src:
        if i not in res:
            res[i] = 0
        res[i] += 1
    maxconf = 0
    maxidx = 0
    for idx in res:
        tmpconf = res[idx]*100/len(src)
        if tmpconf > maxconf:
            maxconf = tmpconf
            maxidx = idx
    return maxidx,maxconf

for a in predictmap:
    print(a,calcfd(predictmap[a]))
