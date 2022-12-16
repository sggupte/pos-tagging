import numpy as np
import pandas as pd
import nltk
import hmmlearn.hmm as hmm
from time import time
from tqdm import tqdm


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


def getPOSMapping():
    """ The indexies of values here will be the integers for observations

    Returns:
        list: returns a dictionary of parts of speech to their index mappings
    """
    posMap = {"NOUN":0,
              "VERB":1,
              "ADJ":2,
              "ADV":3,
              "PRON":4,
              "DET":5,
              "ADP":6,
              "NUM":7,
              "CONJ":8,
              "PRT":9,
              ".":10,
              "X":11}
    return posMap


def getWordObs(sentence):
    # Take in a sentence from the corpus
    #   Example: [('The', 'DET'), ('Fulton', 'NOUN'), ('County', 'NOUN'), ('Grand', 'ADJ'), ('Jury', 'NOUN'), ('said', 'VERB')]
    # and output a list of integers
    obs = list()
    for wordData in sentence:
        obs.append(wordData.lower())
    return obs


class MyHMM():
    def __init__(self, corpus, POSMap, n_states=12):
        self.corpus = corpus
        self.POSMap = POSMap
        self.sentObs = None
        self.model = hmm.MultinomialHMM(n_components=n_states, n_trials=1, algorithm="viterbi")
        A = self.getA(corpus, POSMap)
        pi = self.getPi(corpus, POSMap)
        self.model.transmat_ = A
        self.model.startprob_ = pi


    def getPi(self, corpus, posMap):
        pi = np.zeros(len(posMap))
        for sentence in corpus:
            pi[posMap[sentence[0][1]]] += 1
        return pi/np.sum(pi)


    def getA(self, corpus, posMap):
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


    def getB(self, corpus, obs, posMap, saveB):
        # The probability of each observation belonging to a state
        print(len(obs))
        B = np.zeros((len(posMap), len(obs)))
        for i, o in tqdm(enumerate(obs)):
            # Search through the corpus to find all examples of it
            for sentence in corpus:
                for wordData in sentence:
                    if wordData[0].lower() == o.lower():
                        B[posMap[wordData[1]]][i] += 1
        
            if i == 500 and saveB:
                np.save("out/BMatrix.npy", B)
    
        # Smoothing 
        B += 1

        return (B.T/np.sum(B, axis=1)).T


    def loadBFromFile(self, filename, sentObs):
        self.sentObs = sentObs
        B = np.load(f"out/{filename}")
        self.model.emissionprob_ = B


    def loadB(self, sentObs, saveB=True):
        # Always call this before samples
        self.sentObs = list(set(sentObs)) # Only use unique words
        print(len(self.sentObs))
        B = self.getB(self.corpus, self.sentObs, self.POSMap, saveB)
        self.model.emissionprob_ = B
        if saveB:
            np.save("out/BMatrix.npy", B)


    def sample(self, numSamples):
        if self.sentObs == None:
            raise Exception("Must call MyHmm.loadB() before Sampling") 
        X, state_sequence = self.model.sample(numSamples)
        return X, state_sequence


    def generateCorpus(self, numSamples, minSize, maxSize, filename, saveToCSV=False):
        numSamplesList = np.random.randint(minSize, maxSize, size=numSamples)
        dictionary = {"Word": [], "POS": []}
        for length in numSamplesList:
            X, state_sequence = self.model.sample(length)
            tempWordList = []
            tempPOSList = []
            for xRow, POSItem in zip(X, state_sequence):
                tempWordList.append([self.sentObs[i] for i, item in enumerate(xRow) if item][0])
                tempPOSList.append([i for i in self.POSMap if self.POSMap[i] == POSItem][0])
                #list(self.POSMap.values()).index(POSItem))
            dictionary["Word"].append(tempWordList)
            dictionary["POS"].append(tempPOSList)
        df = pd.DataFrame(dictionary)
        if saveToCSV:
            df.to_csv(f"out/{filename}", index=False)
        return df

    def getGeneratedSentence(self, X):
        sent = ""
        for row in X:
            sent += [self.sentObs[i] for i, item in enumerate(row) if item][0] + " "
        return sent