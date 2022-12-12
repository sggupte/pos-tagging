import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk
import hmmlearn.hmm as hmm


def getDataFrame(corpus):
    dictionary = {"Word": [], "POS": []}
    for sentence in corpus:
        tempWordList = []
        tempPOSList = []
        for word in sentence:
            tempWordList.append(word[0])
            tempPOSList.append(word[1])
        dictionary["Word"].append(tempWordList)
        dictionary["POS"].append(tempPOSList)

    df = pd.DataFrame(dictionary)

    return df

def getDataFrame2(corpus):
    dictionary = {"Sentence": [], "Words": [], "POS": []}
    for i, sentence in enumerate(corpus):
        for word in sentence:
            dictionary["Words"].append(word[0])
            dictionary["POS"].append(word[1])
            dictionary["Sentence"].append(f"Sentence: {i}")

    df = pd.DataFrame(dictionary)

    return df


def getPOSMapping(corpus):
    """ The indexies of values here will be the integers for observations

    Args:
        corpus (list): list of sentences with tuples containing the word and its POS

    Returns:
        list: returns a list of parts of speech
    """
    temp = set()
    for sentence in corpus:
        for wordData in sentence:
            temp.add(wordData[1])
    
    posMap = {}
    i = 0
    for val in temp:
        posMap[val] = i
        i += 1
    return posMap


def getWordObs(sentence, posMap):
    # Take in a sentence from the corpus
    #   Example: [('The', 'DET'), ('Fulton', 'NOUN'), ('County', 'NOUN'), ('Grand', 'ADJ'), ('Jury', 'NOUN'), ('said', 'VERB')]
    # and output a list of integers
    obs = list()
    i = 0
    for wordData in sentence:
        obs.append(wordData.lower())
    return obs


def getPi(corpus, posMap):
    pi = np.zeros(len(posMap))
    for sentence in corpus:
        pi[posMap[sentence[0][1]]] += 1
    
    return pi/np.sum(pi)


def getA(corpus, posMap):
    # Create the A matrix
    n = len(posMap)
    A = np.zeros((n,n))

    # Find the probabilities of going from one state to another

    #   Initialize last and current states
    lastState = corpus[0][0][1]
    currentState = corpus[0][1][1]
    A[posMap[lastState]][posMap[currentState]] -= 1 # Don't double count the transition

    # Build A
    for sentence in corpus:
        for wordData in sentence:
            # Find the state of the last
            A[posMap[lastState]][posMap[currentState]] += 1

            # Update states
            lastState = currentState
            currentState = wordData[1]
    
    # Smoothing
    A += 1

    return (A.T/np.sum(A, axis=1)).T


def getB(corpus, obs, posMap):
    # The probability of each observation belonging to a state
    B = np.zeros((len(posMap), len(obs)))
    for i, o in enumerate(obs):
        # Search through the corpus to find all examples of it
        for sentence in corpus:
            for wordData in sentence:
                if wordData[0].lower() == o.lower():
                    B[posMap[wordData[1]]][i] += 1
    
    # Smoothing
    B += 1

    return (B.T/np.sum(B, axis=1)).T


if __name__ == "__main__":
    # Load in the Data
    train_corpus = nltk.corpus.brown.tagged_sents(tagset='universal')[:1000]#[:16000]
    #val_corpus = nltk.corpus.brown.tagged_sents(tagset='universal')[16000:18000]
    #test_corpus = nltk.corpus.brown.tagged_sents(tagset='universal')[18000:20000]

    # Convert the Data to a DataFrame
    df_train = getDataFrame(train_corpus)
    #df_val = getDataFrame(val_corpus)
    #df_test = getDataFrame(test_corpus)

    # Set some things
    np.random.seed(42)
    n_states = 12
    sentNum = 0

    print(df_train.head())

    train_POSMap = getPOSMapping(train_corpus)
    sentObs = getWordObs(df_train.Word.iloc[0], train_POSMap)
    print(df_train.Word.iloc[0])
    print(sentObs)

    # Fit the HMM
    model = hmm.MultinomialHMM(n_components=n_states, n_trials=1)

    A = getA(train_corpus, train_POSMap)
    B = getB(train_corpus, sentObs, train_POSMap)
    pi = getPi(train_corpus, train_POSMap)

    model.transmat_ = A
    model.startprob_ = pi
    model.emissionprob_ = B

    # X is a matrix that is n_rows=samples, n_columns=input sentences
    X, state_sequence = model.sample(10)
    for row in X:
        print([sentObs[i] for i, item in enumerate(row) if item][0])
    print(np.shape(X))
    print(train_POSMap)
    print(state_sequence)