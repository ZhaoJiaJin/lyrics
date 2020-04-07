dataset address : https://drive.google.com/drive/folders/1JE9OzwobKFnik8ybAhmCwBIgfAFdKQij?usp=sharing


# how to run

* download all the csv file from the link above and put them in the data folder.
* install dependencies:   pip install -r requirements.txt

## run the LDA model

### decade prediction

* pre_process_year.py: data preprocess
* classify_year_lda.py: classify decade

### artist prediction

* ./pre_process_artist.py: data preprocess
* ./classify_artist_lda.py: classify artist


## other models

there are two files:

* classify_year_ml.py: decade prediction
* classify_artist_ml.py: artist prediction

you can specify two arguements: model and vectorizer

* model
    * bayes
    * svm
    * xgb

* vectorizer
    * count
    * Tfidf

for example:

```python
python classify_artist_ml.py svm Tfidf
```

PS: you can also customize ngram, open corresponding python file and search ngram_range, default is (1,3) means n is from 1 to 3, you can change it to (1,2) or (1,1)


# Results

you can find accuracy result in the following files:

* ./artists_classify.result : classification results for artists.
* ./decade_classify.result : classification results for decades.

PS: I didn't really find a way to measure accuracy for LDA modelling, so I came up with my own way, which I am not sure is reasonable. As LDA accuracy is calculated differently, and it is not comparable with other accuracies, so I didn't put LDA accuracy in the results file.
