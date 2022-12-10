import nltk
import pandas as pd
import torch

data = nltk.corpus.brown.tagged_sents(tagset='universal')

dictionary = {"Word": [], "POS": []}
maxLength = 0
for sentence in data:
       tempWordList = []
       tempPOSList = []
       for word in sentence:
              tempWordList.append(word[0])
              tempPOSList.append(word[1])
       dictionary["Word"].append(tempWordList)
       dictionary["POS"].append(tempPOSList)

       if len(sentence) > maxLength:
              maxLength = len(sentence)

df = pd.DataFrame(dictionary)

print(maxLength)

print(df.head())
print(len(df))