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
    print(len(obs))
    B = np.zeros((len(posMap), len(obs)))
    for i, o in tqdm(enumerate(obs)):
        # Search through the corpus to find all examples of it
        for sentence in corpus:
            for wordData in sentence:
                if wordData[0].lower() == o.lower():
                    B[posMap[wordData[1]]][i] += 1
        
        if i == 500:
            np.save("out/BMatrixx.npy", B)
    
    # Smoothing
    B += 1

    return (B.T/np.sum(B, axis=1)).T

class MyHMM():
    def __init__(self, corpus, POSMap, n_states=12):
        self.corpus = corpus
        self.POSMap = POSMap
        self.sentObs = None
        self.model = hmm.MultinomialHMM(n_components=n_states, n_trials=1, algorithm="viterbi")
        A = getA(corpus, POSMap)
        pi = getPi(corpus, POSMap)
        self.model.transmat_ = A
        self.model.startprob_ = pi


    def loadBFromFile(self, filename, sentObs):
        self.sentObs = sentObs
        B = np.load(f"out/{filename}")
        self.model.emissionprob_ = B


    def loadB(self, sentObs, saveB=True):
        # Always call this before samples
        self.sentObs = list(set(sentObs)) # Only use unique words
        print(len(self.sentObs))
        B = getB(self.corpus, self.sentObs, self.POSMap)
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


if __name__ == "__main__":
    # Load in the Data
    train_corpus = nltk.corpus.brown.tagged_sents(tagset='universal')[:16000]
    val_corpus = nltk.corpus.brown.tagged_sents(tagset='universal')[16000:18000]
    test_corpus = nltk.corpus.brown.tagged_sents(tagset='universal')[18000:20000]

    # Convert the Data to a DataFrame
    df_train = getDataFrame(train_corpus)
    df_val = getDataFrame(val_corpus)
    df_test = getDataFrame(test_corpus)

    # Set some things
    np.random.seed(42)
    n_states = 12
    sentNum = 0

    # Concatenate sentences together
    trainingObservations = df_train.Word.iloc[0]
    for i, sentList in enumerate(df_train.Word):
        if i == 0:
            continue
        trainingObservations.extend(sentList)
        '''if i == 1:
            break'''

    # Train the HMM Model on one sentence
    train_POSMap = getPOSMapping(train_corpus)
    sentObs = getWordObs(trainingObservations, train_POSMap)

    # Take 20 Samples from the model
    start_time = time()
    myHmm = MyHMM(train_corpus, train_POSMap)

    # Get B from File
    isBSaved = False
    if isBSaved:
        myHmm.loadBFromFile("BMatrix.npy", sentObs)
    else:
        myHmm.loadB(sentObs)

    print(time() - start_time)

    myHmm.generateCorpus(16000 , 6, 20, "myCSV.csv",  True)
