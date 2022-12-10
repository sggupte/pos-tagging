import torch
from torch.utils.data import Dataset, DataLoader
import nltk
import pandas as pd
from sklearn.model_selection import train_test_split
import gensim
import gensim.downloader
from gensim.models import KeyedVectors
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence


def get_df():
    data = nltk.corpus.brown.tagged_sents(tagset='universal')[:20000]

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
    return df
    
def get_splits(dataframe):
    # using a 80:10:10 train/val/test split
    train, intermediate = train_test_split(dataframe, test_size=0.2)
    val, test = train_test_split(intermediate, test_size=0.5)
    
    train.reset_index(drop=True)
    val.reset_index(drop=True)
    test.reset_index(drop=True)
    
    return train, val, test
    
def word_embedder():
    word2vec_vectors = gensim.downloader.load('word2vec-google-news-300')
    #word2vec_vectors.save("pretrained_word2vec.model")
    #word2vec_vectors = KeyedVectors.load_word2vec_format(r"C:\Users\EricQ\gensim-data\word2vec-google-news-300\word2vec-google-news-300.gz",binary=False)
    return word2vec_vectors

def maptag2label():
    pass

class POSDataset(Dataset):
    
    def __init__(self, df, w2v):
        self.df = df
        self.w2v = w2v
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self,index):
        
        words = self.df.Word[index]
        tags = self.df.POS[index]
        embeddings = []
        for word in tqdm(words):
            try:
                temp = torch.tensor(self.w2v[word])
            except: 
                temp = -1*torch.ones(300)
            
            embeddings.append(temp)
        
        sentence_embedding = torch.vstack(embeddings)
        return sentence_embedding, tags
    
def collate_function(batch):
    batched = pad_sequence(batch,batch_first=True)
    return batched
    
#%%
    
if __name__ == "__main__":
    df = get_df()
    word2vec = word_embedder()
    train, val, test = get_splits(df)
    #%%
    train_data = POSDataset(train, word2vec)
    train_loader = DataLoader(train_data, batch_size=1, shuffle=True,num_workers=4,\
                              collate_fn = collate_function, pin_memory=True)
    counter = 0
    #print(len(train_loader))

    
    #print(df.head())
    #train, val, test = get_splits(df)
    #print(len(train))
    #print(len(val))
    #print(len(test))