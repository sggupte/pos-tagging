import torch
import torch.nn as nn

class BiLSTMTagger(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout_prob=0.25):
        super(BiLSTMTagger, self).__init__()        
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, bias=True, batch_first=True, bidirectional=True, dropout=dropout_prob)     
        self.fc = nn.Linear(hidden_size*2, output_size) # multiplied by 2 because bidirectional
        self.dropout = nn.Dropout(dropout_prob)
        
        
    def forward(self, x):
        output, (hidden, cell) = self.lstm(x)
        prediction = self.fc(self.dropout(output))
        
        return prediction
        