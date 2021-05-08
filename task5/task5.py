
"""
- a list of all words (vocabulary) and their frequencies,

- a list of bigram and trigram tokens that have been used at least 5 times,

- the most similar two words using Latent Semantic Analysis should be determined.

"""
import nltk
import string 
import pandas as pd
import numpy as np
from nltk.util import ngrams
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer

stop_words = set(stopwords.words('turkish'))

corpus = open('tur_wikipedia_2016_10K-sentences.txt', encoding="utf8").read()
corpus_lsa = open('tur_wikipedia_2016_10K-sentences.txt', encoding="utf8").readlines()


def word_frequency(data):
    data = nltk.word_tokenize(data)
    word = pd.Series(data)
    word = word.value_counts()
    return word


def extract_ngrams(data, num):
    n_grams = ngrams(nltk.word_tokenize(data), num)
    return [' '.join(grams) for grams in n_grams]

def print_ngrams(data, num):
    n_grams = pd.Series(extract_ngrams(data, num))
    n_grams_count = n_grams.value_counts()
    n_grams = n_grams_count.loc[lambda x: x > 5]
    return n_grams
def clean_stopword(tokens):
    tokens = nltk.word_tokenize(tokens)
    data = []
    data2 = []
    chars = ['’', 'bu', 'bir', 'Bu']
    for i in tokens:
        if i not in stop_words:
            data.append(i)
    for w in data:
        if w not in chars:
            data2.append(w)
    txt = str(data2)
    new_txt = ""
    for w in txt:
        if w not in string.punctuation:
            new_txt += w
    return new_txt

corpus = clean_stopword(corpus)
bigrams = print_ngrams(corpus, 2)
trigrams = print_ngrams(corpus, 3)
print("Bigrams in the corpus: \n")
print(bigrams)
print("Trigrams in the corpus: \n")
print(trigrams)
frequencies = word_frequency(corpus)
print("Frequency of all words: \n")
print(frequencies)
vectorizer = CountVectorizer(min_df=1)
dtm = vectorizer.fit_transform(corpus_lsa)
words = vectorizer.get_feature_names()
print(words)
words = vectorizer.fit_transform(words)
words = words.asfptype()
lsa = TruncatedSVD(2, algorithm='arpack')


dtm_lsa = lsa.fit_transform(words)
dtm_lsa = Normalizer(copy=False).fit_transform(dtm_lsa)
similarity = np.asarray(np.asmatrix(dtm_lsa) * np.asmatrix(dtm_lsa).T)
x = pd.DataFrame(similarity,index=vectorizer.get_feature_names(), columns=vectorizer.get_feature_names())
print(x)



