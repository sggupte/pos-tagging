# This script has miscellaneous functions and helpful things that are utilized to 
# investigate more things in the LSTM. The idea is to take functions from here and plug them into the
# main training loop to look into other approaches 

# Things we are considering: Using a nn.Embedding layer to train the embeddings instead of 
# word2vec. Also, pretraining word2vec on the brown training dataset (instead of using one pretrained
# on google news.

import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from gensim.models import Word2Vec
from torch_utils import *


class BiLSTMTagger_withEmbedding(nn.Module):

    def __init__(self, input_size, embedding_size, hidden_size, output_size, num_layers, dropout_prob=0.25):
        super(BiLSTMTagger_withEmbedding, self).__init__()        
        
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers=num_layers, bias=True, batch_first=True, bidirectional=True, dropout=dropout_prob)     
        self.fc = nn.Linear(hidden_size*2, output_size) # multiplied by 2 because bidirectional
        self.dropout = nn.Dropout(dropout_prob)
        
        
    def forward(self, x):
        x = self.dropout(self.embedding(x))
        output, (hidden, cell) = self.lstm(x)
        prediction = self.fc(self.dropout(output))
        
        return prediction
        

def map_word2idx(train_dataset:pd.DataFrame):
    data = train_dataset.Word
    mapping = {}
    for sentence in data:
        for word in sentence:
            if word not in mapping:
                mapping[word] = len(mapping)
        
    return mapping
    
    
def train_Word2Vec(train_dataset:pd.DataFrame):
    data = train_dataset.Word
    model = Word2Vec(data,vector_size=300,min_count=1, seed=42)
    model.save("word2vec_brown.model")
    return model
    
    
if __name__ == "__main__":
    
    df, _ = get_df()
    train, val, test = get_splits(df)
    mapping = map_word2idx(train)
    word2vec = train_Word2Vec(train)
    print(word2vec.wv)