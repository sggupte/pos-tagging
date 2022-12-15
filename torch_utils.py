import torch
from torch.utils.data import Dataset, DataLoader
import nltk
import pandas as pd
from sklearn.model_selection import train_test_split
import gensim
import gensim.downloader
#from gensim.models import KeyedVectors
from torch.nn.utils.rnn import pad_sequence


def get_df():
    data = nltk.corpus.brown.tagged_sents(tagset='universal')[:20000]

    dictionary = {"Word": [], "POS": []}
    tag2idx = {"VERB":0, "NOUN":1, "PRON":2, "ADJ":3, "ADV":4, "ADP":5, "CONJ":6, \
               "DET":7, "NUM":8, "PRT":9, "X":10, ".":11, "PAD":12}
    maxLength = 0
    for sentence in data:
           tempWordList = []
           tempPOSList = []
           for word in sentence:
                  tempWordList.append(word[0])
                  new_idx = tag2idx[word[1]]
                  tempPOSList.append(new_idx)
           dictionary["Word"].append(tempWordList)
           dictionary["POS"].append(tempPOSList)

           if len(sentence) > maxLength:
               maxLength = len(sentence)
    df = pd.DataFrame(dictionary)
    return df, maxLength
    
def get_splits(dataframe):
    # using a 80:10:10 train/val/test split
    #train, intermediate = train_test_split(dataframe, test_size=0.2)
    #val, test = train_test_split(intermediate, test_size=0.5)
    train = dataframe.iloc[:16000, :]
    val = dataframe.iloc[16000:18000, :]
    test = dataframe.iloc[18000:20000, :]
    
    
    train = train.reset_index(drop=True)
    val = val.reset_index(drop=True)
    test = test.reset_index(drop=True)
    
    return train, val, test
    
def word_embedder():
    word2vec_vectors = gensim.downloader.load('word2vec-google-news-300')
    #word2vec_vectors.save("pretrained_word2vec.model")
    #word2vec_vectors = KeyedVectors.load_word2vec_format(r"C:\Users\EricQ\gensim-data\word2vec-google-news-300\word2vec-google-news-300.gz",binary=False)
    return word2vec_vectors

class POSDataset(Dataset):
    
    def __init__(self, df, w2v):
        self.df = df
        self.w2v = w2v
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self,index):
        
        words = self.df.iloc[index].Word
        tags = self.df.iloc[index].POS
        embeddings = []
        for word in words:
            try:
                temp = torch.tensor(self.w2v[word])
            except: 
                temp = -1*torch.ones(300) # This is for out-of-vocabulary words
            
            embeddings.append(temp)
        tags = torch.tensor(tags)
        sentence_embedding = torch.vstack(embeddings)
        return sentence_embedding, tags
    
def collate_function(batch):
    tag_pad_value = 12 #This is the tag value assigned to padding
    sentences = []
    tags = []
    for sentence_embed, tag in batch:    
        sentences.append(sentence_embed)
        tags.append(tag)
    batched_sent = pad_sequence(sentences,batch_first=True)
    batched_tag = pad_sequence(tags, batch_first=True, padding_value = tag_pad_value)
    
    return batched_sent, batched_tag
    
#%%
    
if __name__ == "__main__":
    df, maxLength = get_df()
    word2vec = word_embedder()
    train, val, test = get_splits(df)
    #%%
    train_data = POSDataset(train, word2vec)
    val_data = POSDataset(val, word2vec)
    test_data = POSDataset(test, word2vec)
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True, collate_fn = collate_function)
    val_loader = DataLoader(val_data, batch_size=64, shuffle=True, collate_fn = collate_function)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)
    # Will need to use unsqueeze for test_loader samples
    counter = 0
    for batched_sent, batched_tag in val_loader:
        if counter == 5:
            break
        print(batched_sent.shape)
        print(batched_tag.shape)
        counter += 1
    