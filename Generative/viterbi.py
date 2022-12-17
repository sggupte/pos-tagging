# The purpose of this file is to test the HMM --> Viterbi algorithm on the real and synthetic data to get accuracy
import nltk
from hmmModel import getDataFrame, MyHMM, getPOSMapping, getWordObs, getObsStates
import numpy as np
from tqdm import tqdm
import logging
from time import time
import pandas as pd
import ast
logging.basicConfig(level=logging.ERROR)


# Create a matrix that is 12 x number of observations
def oneHotMatrix(sentence, uniqueObs):
    myMat = np.zeros((len(sentence), len(uniqueObs)))
    for i, word in enumerate(sentence):
        for j, uniqueWord in enumerate(uniqueObs):
            if word == uniqueWord:
                myMat[i][j] = 1
    return myMat


def dfToCorpusStyle(words, pos):
    retList = []
    for wordList, posList in zip(words, pos):
        tempList = []
        for wordItem, posItem in zip(wordList, posList):
            tempList.append((wordItem, posItem))
        retList.append(tempList)
    return retList

train_corpus = nltk.corpus.brown.tagged_sents(tagset='universal')[:16000]
test_corpus = nltk.corpus.brown.tagged_sents(tagset='universal')[18000:20000]

# Convert the Data to a DataFrame
df_train = getDataFrame(train_corpus)
df_test = getDataFrame(test_corpus)

# Build the A matrix and the pi matrix from the training data...
# Build the B matrix from the observations of a testing point and loop through the train corpus

posMap = getPOSMapping()

totalCorrect = 0
totalNumWords = 0

# Run on real data
start_time = time()
for sentence, pos in tqdm(zip(df_test.Word, df_test.POS)):
    testHmm = MyHMM(train_corpus, posMap) # Inputting the training corpus
    obs = getWordObs(sentence) #obs is a set
    testHmm.loadB(obs, False)
    sentMat = oneHotMatrix(obs, testHmm.sentObs)
    predStates = testHmm.model.predict(sentMat)
    trueStates = getObsStates(pos, posMap)
    totalCorrect += np.sum(predStates == trueStates)
    totalNumWords += len(obs)

print(f"Total Time for Real Data: {time() - start_time}")
print(totalCorrect/totalNumWords)

# Run on synthesized data
df_synth = pd.read_csv("out/generatedCorpusFinal.csv", converters={"Word":ast.literal_eval,"POS":ast.literal_eval})
df_synth_train = df_synth[:16000]
df_synth_test = df_synth[18000:]

synth_train_corpus = dfToCorpusStyle(df_synth_train.Word, df_synth_train.POS)

totalCorrect = 0
totalNumWords = 0

# Run on real data
start_time = time()
for sentence, pos in tqdm(zip(df_synth_test.Word, df_synth_test.POS)):
    synthHmm = MyHMM(synth_train_corpus, posMap) # Inputting the training corpus
    obs = getWordObs(sentence) #obs is a set
    synthHmm.loadB(obs, False)
    sentMat = oneHotMatrix(obs, synthHmm.sentObs)
    predStates = synthHmm.model.predict(sentMat)
    trueStates = getObsStates(pos, posMap)
    totalCorrect += np.sum(predStates == trueStates)
    totalNumWords += len(obs)

print(f"Total Time for Synthetic Data: {time() - start_time}")
print(totalCorrect/totalNumWords)
