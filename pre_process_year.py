#!/usr/bin/env python

# pre processing steps
# 0. remove non-english songs
# 0.1 detect and remove all duplicate records.!!!
# 1. calcuate the number of years and the number of songs(on) of each year, same with years
# 2. replace contractions words like "won't" => "will not",
# 3. remove brackets and everything in them
# 4. remove special characters
# 5. to convert everything to lower case
# 6. tokenize
# 7. remove stop words and stemming & words less than 3 characters

import nltk
import re
from nltk.tokenize import word_tokenize
from langdetect import detect
import sys
import pandas as pd
import random
import hashlib
from nltk.stem import PorterStemmer
nltk.download('words')
nltk.download('stopwords')

import nltk
from nltk.corpus import stopwords

ps = PorterStemmer()
spwords = set(stopwords.words('english'))

def process_token(src):
    result = []
    for w in src:
        if len(w) >= 3 and (w not in spwords):
            result.append(ps.stem(w))
    return " ".join(result)


def process_lyrics(lyc):
    for ori in replaces:
        lyc = lyc.replace(ori,replaces[ori])
    lyc = rm_bracket(lyc)
    pattern = re.compile("[^a-zA-Z0-9 ]")
    lyc = pattern.sub(" ", lyc)
    lyc = lyc.lower()
    lycwords = word_tokenize(lyc)
    after = process_token(lycwords)
    return after


replaces = {
"won't": "will not",
"can't": "can not",
"n't": " not",
"'ll": " will",
"'re": " are",
"'ve": " have",
"'m": " am",
"'d": " would",
"'s": ""
}

def chooserandom(titles, lyrics, source):
    res = []
    lyricsres = []
    records = {}
    slen = len(source)
    while len(res) < 700:
        if len(source) <= 0:
            return res,lyricsres,False
        got = random.choice(source)
        source.remove(got)
        gottl = titles[got]
        gotlyc = lyrics[got]
        if gottl in records:
            continue
        else:
            records[gottl] = ""
        if gotlyc in records:
            continue
        else:
            records[gotlyc] = ""
        #print(gotlyc)
        lycafter = process_lyrics(gotlyc)
        if len(lycafter) > 50:
            res.append(got)
            lyricsres.append(lycafter)
    return res,lyricsres,True


def rm_bracket(s):
    res = ""
    inb = False
    for c in s:
        if c == '(' or c == '[' or c == '{':
            inb = True
            continue
        if c == ')' or c == ']' or c == '}':
            inb = False
            continue
        if not inb:
            res += c

    return res

def reverse(need,row):
    res = []
    for i in range(0,row):
        if i not in need:
            res.append(i)
    return res


def md5digest(s):
    m = hashlib.md5()
    m.update(s)
    return m.digest()

colname = sys.argv[1]
englishlyrics_file = "./data/english_lyrics.csv"
# 0. remove non english songs
#fname = "./data/lyrics.csv"
#
#res = pd.read_csv(fname)
#
##res = valid("./songdata.csv")
#(row,col) = res.shape
#targetcol = res[colname]
#lyrics = res['lyrics']
#
#print(res)
#droprow = []
#for i in range(0,row):
#    if i % 1000 == 0:
#        print(i,row)
#    lyr = lyrics[i]
#    try:
#        if type(lyr) != str or len(lyr) < 100 or detect(lyr) != "en":
#            droprow.append(i)
#    except:
#        droprow.append(i)
#
#
#res = res.drop(droprow)
#print(res)
#
#res.to_csv(englishlyrics_file,columns=['song','year','artist','genre','lyrics'],index=False)


step01output = "./data/english_no_dup_year_lyrics.csv"
#0.1 remve duplicate records
#res = pd.read_csv(englishlyrics_file)
#
##res = valid("./songdata.csv")
#(row,col) = res.shape
#targetcol = res[colname]
#lyrics = res['lyrics']
#titles = res['song']
#records={}
#
#droprow = []
#
#for i in range(0,row):
#    if i % 1000 == 0:
#        print(i,row)
#    tl = titles[i]
#    if tl not in records:
#        records[tl] = i
#    else:
#        if i not in droprow:
#            droprow.append(i)
#        if records[tl] not in droprow:
#            droprow.append(records[tl])
#    tl = lyrics[i]
#    if tl not in records:
#        records[tl] = i
#    else:
#        if i not in droprow:
#            droprow.append(i)
#
#res = res.drop(droprow)
#print(res)
#
#res.to_csv(step01output,columns=['song','year','artist','genre','lyrics'],index=False)
#


def decades(y):
    yi = int(y)
    return y - (y%10)


step1outfile = "./data/yearlyrics.csv"
res = pd.read_csv(step01output)

#res = valid("./songdata.csv")
(row,col) = res.shape
targetcol = res[colname]
lyrics = res['lyrics']
titles = res['song']
mapres={}

needrow = []

for i in range(0, row):
    name = decades(targetcol[i])
    if name not in mapres:
        mapres[name] = []
    mapres[name].append(i)

for k in mapres:
    print(len(mapres[k]),k)
    if len(mapres[k]) >700:
        print(len(mapres[k]),k)
        #print("{0}|{1}".format(k,len(mapres[k])))
        chooseres,lycchoose,enough = chooserandom(titles, lyrics, mapres[k])
        if enough:
            needrow.extend(chooseres)
            for lineidx in range(0,len(chooseres)):
                linen = chooseres[lineidx]
                lycc = lycchoose[lineidx]
                lyrics.update(pd.Series([lycc], index=[linen]))
        else:
            print("not enough songs!!!",len(chooseres))

res = res.drop(reverse(needrow,row))
print(res)
res.to_csv(step1outfile,columns=['song','year','artist','genre','lyrics'],index=False)


