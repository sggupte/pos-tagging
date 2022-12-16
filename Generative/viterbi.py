# The purpose of this file is to test the HMM --> Viterbi algorithm on the real and synthetic data to get accuracy
import nltk
from hmmModel import getDataFrame, MyHMM
import numpy as np

train_corpus = nltk.corpus.brown.tagged_sents(tagset='universal')[:10]#[:16000]
val_corpus = nltk.corpus.brown.tagged_sents(tagset='universal')[16000:18000]
test_corpus = nltk.corpus.brown.tagged_sents(tagset='universal')[18000:20000]

# Convert the Data to a DataFrame
df_train = getDataFrame(train_corpus)
df_val = getDataFrame(val_corpus)
df_test = getDataFrame(test_corpus)


