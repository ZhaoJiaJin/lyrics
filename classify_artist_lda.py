#!/usr/bin/env python

import pandas as pd
import gensim
import sys
from nltk.tokenize import word_tokenize
inputf = "./data/artistlyrics.csv"
songs = pd.read_csv(inputf)

(row,col) = songs.shape
lyrics = songs['lyrics']
artists = songs['artist']

processed_docs = []
artistmap = {}
for i in range(0,row):
    art = artists[i]
    lyc = lyrics[i]
    if art not in artistmap:
        artistmap[art] = []
    processed_docs.append(word_tokenize(lyc))
    artistmap[art].append(i)


#for a in artistmap:
#    print(a,len(artistmap[a]))

# Filter out tokens that appear in
# less than 15 documents (absolute number) or
# more than 0.5 documents (fraction of total corpus size, not absolute number).

dictionary = gensim.corpora.Dictionary(processed_docs)
#count = 0
dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)
#for k, v in dictionary.iteritems():
#    print(k, v)
    #count += 1
    #if count > 2250/2:
    #    break

bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
#print(bow_corpus)


from gensim import corpora, models
tfidf = models.TfidfModel(bow_corpus)
corpus_tfidf = tfidf[bow_corpus]

#lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=9, id2word=dictionary, passes=2, workers=2)


#for idx, topic in lda_model.print_topics(-1):
#    print('Topic: {} \nWords: {}'.format(idx, topic))

#print(type(lda_model))

lda_model= gensim.models.LdaMulticore(corpus_tfidf, num_topics=9, id2word=dictionary, passes=2, workers=4)
#for idx, topic in lda_model_tfidf.print_topics(-1):
#    print('Topic: {} Word: {}'.format(idx, topic))

# I don't know how to calculate accuracy for LDA model, so I came up with this method:
# 1. Use the model to classify all the documents(lyrics)
# 2. If most the documents(lyrics) under the same category(artist) are classified as the same topic,  I will assume this model has good performance
# PS: And the performance is not very well

predictmap = {}
for artist in artistmap:
    predictmap[artist] = []
for artist in artistmap:
    for lyc in artistmap[artist]:
        bow_vector = bow_corpus[lyc]
        max_score = 0
        chooseidx = -1
        for index, score in sorted(lda_model[bow_vector], key=lambda tup: -1*tup[1]):
            if score > max_score:
                max_score = score
                chooseidx = index
        predictmap[artist].append(chooseidx)

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
