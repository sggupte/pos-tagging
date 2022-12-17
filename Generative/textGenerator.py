import numpy as np
import pandas as pd
import nltk
import hmmlearn.hmm as hmm
from time import time
from tqdm import tqdm
from hmmModel import getDataFrame, getPOSMapping, getWordObs, MyHMM

if __name__ == "__main__":
    # Load in the Data
    train_corpus = nltk.corpus.brown.tagged_sents(tagset='universal')[:16000]
    val_corpus = nltk.corpus.brown.tagged_sents(tagset='universal')[16000:18000]
    test_corpus = nltk.corpus.brown.tagged_sents(tagset='universal')[18000:20000]

    # Convert the Data to a DataFrame
    df_train = getDataFrame(train_corpus)
    df_val = getDataFrame(val_corpus)
    df_test = getDataFrame(test_corpus)

    # Set some variables
    np.random.seed(42)
    n_states = 12
    sentNum = 0

    # Concatenate sentences together
    trainingObservations = df_train.Word.iloc[0]
    for i, sentList in enumerate(df_train.Word):
        if i == 0:
            continue
        trainingObservations.extend(sentList)

    # Train the HMM Model on one sentence
    train_POSMap = getPOSMapping()
    sentObs = getWordObs(trainingObservations)

    # Take 20 Samples from the model
    start_time = time()
    myHmm = MyHMM(train_corpus, train_POSMap)

    # Get B from File
    isBSaved = False # True to load the B Matrix from a file
    if isBSaved:
        myHmm.loadBFromFile("out/BMatrix.npy", sentObs)
    else:
        myHmm.loadB(sentObs)

    # Print the time it takes to generate the B matrix for this model
    print(time() - start_time)

    # Generate a large number of sentences with lengths between 6 and 20
    #   Save the file as generatedCorpus.csv in the out folder
    numGeneratedSentences = 20000
    myHmm.generateCorpus(numGeneratedSentences , 6, 20, "generatedCorpus.csv",  True)
