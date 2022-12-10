import numpy as np
import nltk
from typing import Sequence, Tuple, TypeVar

Q = TypeVar("Q")
V = TypeVar("V")

def getObsStates(sentence, posMap):
    # Take in a sentence from the corpus
    #   Example: [('The', 'DET'), ('Fulton', 'NOUN'), ('County', 'NOUN'), ('Grand', 'ADJ'), ('Jury', 'NOUN'), ('said', 'VERB')]
    # and output a list of integers
    obsStates = list()
    for wordData in sentence:
        obsStates.append(posMap[wordData[1]])
    return obsStates


def getWordObs(sentence, posMap):
    # Take in a sentence from the corpus
    #   Example: [('The', 'DET'), ('Fulton', 'NOUN'), ('County', 'NOUN'), ('Grand', 'ADJ'), ('Jury', 'NOUN'), ('said', 'VERB')]
    # and output a list of integers
    obs = list()
    i = 0
    for wordData in sentence:
        obs.append(wordData[0].lower())
    return obs


def getNumObs(sentence, posMap):
    # Take in a sentence from the corpus
    #   Example: [('The', 'DET'), ('Fulton', 'NOUN'), ('County', 'NOUN'), ('Grand', 'ADJ'), ('Jury', 'NOUN'), ('said', 'VERB')]
    # and output a list of integers
    obsDict = {}
    obs = list()
    i = 0
    for wordData in sentence:
        # Whenever you find a unique observation, store it in a dictionary
        if(not wordData[0].lower() in obsDict.keys()):
            obsDict[wordData[0].lower()] = i
            i += 1
        # Append the unique token integer to the list
        obs.append(obsDict[wordData[0].lower()])
    return obs, obsDict


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

    return A/np.sum(A, axis=1)
        

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

    return B/np.sum(B, axis=0)


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


def getPi(corpus, posMap):
    pi = np.zeros(len(posMap))
    for sentence in corpus:
        pi[posMap[sentence[0][1]]] += 1
    
    return pi/np.sum(pi)


def viterbi(
    obs: Sequence[int],
    pi: np.ndarray[Tuple[V], np.dtype[np.float_]],
    A: np.ndarray[Tuple[Q, Q], np.dtype[np.float_]],
    B: np.ndarray[Tuple[V, Q], np.dtype[np.float_]],
) -> tuple[list[int], float]:
    """Infer most likely state sequence using the Viterbi algorithm.

    Args:
        obs: An iterable of ints representing observations.
        pi: A 1D numpy array of floats representing initial state probabilities.
        A: A 2D numpy array of floats representing state transition probabilities.
        B: A 2D numpy array of floats representing emission probabilities.

    Returns:
        A tuple of:
        * A 1D numpy array of ints representing the most likely state sequence.
        * A float representing the probability of the most likely state sequence.
    """
    N = len(obs)
    Q, V = B.shape  # num_states, num_observations

    # d_{ti} = max prob of being in state i at step t
    #   AKA viterbi
    # \psi_{ti} = most likely state preceeding state i at step t
    #   AKA backpointer

    # initialization
    log_d = [np.log(pi) + np.log(B[:, obs[0]])]
    log_psi = [np.zeros((Q,))]

    # recursion
    for z in obs[1:]:
        log_da = np.expand_dims(log_d[-1], axis=1) + np.log(A)
        log_d.append(np.max(log_da, axis=0) + np.log(B[:, z]))
        log_psi.append(np.argmax(log_da, axis=0))

    # termination
    log_ps = np.max(log_d[-1])
    qs = [-1] * N
    qs[-1] = int(np.argmax(log_d[-1]))
    for i in range(N - 2, -1, -1):
        qs[i] = log_psi[i + 1][qs[i + 1]]

    return qs, np.exp(log_ps)

if __name__ == "__main__":
    corpus = nltk.corpus.brown.tagged_sents(tagset='universal')[:10000]
    posMap = getPOSMapping(corpus)

    A = getA(corpus, posMap)
    pi = getPi(corpus, posMap)
    totalWords = 0
    totalCorrectWords = 0

    # Test Sentence 1
    sentence = nltk.corpus.brown.tagged_sents(tagset='universal')[10150]
    obs = getWordObs(sentence, posMap)
    B = getB(corpus, obs, posMap)

    obsStates = getObsStates(sentence, posMap)
    obsNum, obsDict = getNumObs(sentence, posMap)

    predictions, p = viterbi(obsNum, pi, A, B)
    print(f"Probability: {p}")
    print(sentence)
    mask = np.array([obsStates[i] == predictions[i] for i in range(len(obsStates))])
    print(posMap)
    print(obsStates)
    print(predictions)
    print(f"Accuracy: {np.sum(mask)/len(mask)}")

    totalWords += len(mask)
    totalCorrectWords += np.sum(mask)
    
    myWords = np.array([word[0] for word in sentence])
    print(f"Missed Words: {myWords[~mask]}\n*****")

    # Test Sentence 2
    sentence = nltk.corpus.brown.tagged_sents(tagset='universal')[10151]
    obs = getWordObs(sentence, posMap)
    B = getB(corpus, obs, posMap)
    print(B)

    obsStates = getObsStates(sentence, posMap)
    obsNum, obsDict = getNumObs(sentence, posMap)

    predictions, p = viterbi(obsNum, pi, A, B)
    print(f"Probability: {p}")
    print(sentence)
    mask = np.array([obsStates[i] == predictions[i] for i in range(len(obsStates))])
    print(posMap)
    print(obsStates)
    print(predictions)
    print(f"Accuracy: {np.sum(mask)/len(mask)}")

    totalWords += len(mask)
    totalCorrectWords += np.sum(mask)

    myWords = np.array([word[0] for word in sentence])
    print(f"Missed Words: {myWords[~mask]}\n*****")

    # Test Sentence 3
    sentence = nltk.corpus.brown.tagged_sents(tagset='universal')[10152]
    obs = getWordObs(sentence, posMap)
    B = getB(corpus, obs, posMap)
    print(B)

    obsStates = getObsStates(sentence, posMap)
    obsNum, obsDict = getNumObs(sentence, posMap)

    predictions, p = viterbi(obsNum, pi, A, B)
    print(f"Probability: {p}")
    print(sentence)
    mask = np.array([obsStates[i] == predictions[i] for i in range(len(obsStates))])
    print(posMap)
    print(obsNum)
    print(obsStates)
    print(predictions)
    print(f"Accuracy: {np.sum(mask)/len(mask)}")

    totalWords += len(mask)
    totalCorrectWords += np.sum(mask)
    
    myWords = np.array([word[0] for word in sentence])
    print(f"Missed Words: {myWords[~mask]}\n*****")

    print(f"Final Accuracy: {totalCorrectWords/totalWords}")

    print(A)
    print("\n")
    print(B)