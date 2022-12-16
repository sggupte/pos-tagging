import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils import *
from architecture import BiLSTMTagger

import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
import time

def setSeed():
    rng = 42
    random.seed(rng)
    np.random.seed(rng)
    torch.manual_seed(rng)
    torch.backends.cudnn.deterministic = True
    
def makeNet():
    input_dim = 300
    hidden_dim = 512
    output_size = 13
    num_layers = 2
    dropout = 0.5
    
    net = BiLSTMTagger(input_dim, hidden_dim, output_size, num_layers, dropout)
    return net
    
def makeLoaders(word2vec):
    df = get_df()
    #df = get_synthetic_df() 
    # uncomment the above line if you instead want to train and test on the synthetic data! 
    # This also requires changing some of the save files btw!
    
    train, val, test = get_splits(df)
    
    train_data = POSDataset(train, word2vec)
    val_data = POSDataset(val, word2vec)
    test_data = POSDataset(test, word2vec)
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True, collate_fn = collate_function)
    val_loader = DataLoader(val_data, batch_size=64, shuffle=True, collate_fn = collate_function)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

    return train_loader, val_loader, test_loader

# This will calculate accuracy for each batch, ignores the indices with pads
def batchAccuracy(predictions, target, padding_index):
    predictions = F.softmax(predictions, dim=-1)
    predictions_max = predictions.argmax(dim=1, keepdim=False)
    nonpadded_inputs = (target != padding_index).nonzero()
    correct = predictions_max[nonpadded_inputs]
    correct = predictions_max[nonpadded_inputs].eq(target[nonpadded_inputs])
    return 100 * (correct.sum() / target[nonpadded_inputs].shape[0]).item()
    

def train(net, optimizer, criterion, dataloader, device):
    running_loss = 0
    epoch_acc = 0
    
    net.train()
    
    for x, tags in tqdm(dataloader):
        #x = x.to(device)
        #tags = tags.to(device)
        
        optimizer.zero_grad()
        output = net(x)
        
        output = output.view(-1, output.shape[-1])
        tags = tags.view(-1)
        
        loss = criterion(output, tags)
        loss.backward()
        optimizer.step()
        
        acc = batchAccuracy(output, tags, 12)

        running_loss += loss.item()
        epoch_acc += acc
    
    return running_loss, epoch_acc / len(dataloader)

def evaluate(net, dataloader, criterion, device):
    running_loss = 0
    epoch_acc = 0
    
    net.eval()
    with torch.no_grad():
        for x, tags in tqdm(dataloader):
            #x = x.to(device)
            #tags = tags.to(device)
            
            output = net(x)
            
            output = output.view(-1, output.shape[-1])
            tags = tags.view(-1)
            
            loss = criterion(output, tags)
            acc = batchAccuracy(output, tags, 12)
            running_loss += loss.item()
            epoch_acc += acc
            
    return running_loss, acc / len(dataloader)
            
def train_loop(epochs, trainloader, valloader, net, optimizer, criterion, device):
    total_train_start = time.time()
    best_val_loss = np.inf
    net.to(device)
    train_losses = []
    eval_losses = []
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}!")
        start_time = time.time()
        
        train_loss, train_acc = train(net, optimizer, criterion, trainloader, device)
        eval_loss, eval_acc = evaluate(net, valloader, criterion, device)
        
        end_time = time.time()
        elapsed_time = (end_time - start_time) / 60
        print(f"This Epoch took: {elapsed_time} minutes")
        
        print(f"This is the training loss for the Epoch: {train_loss}")
        print(f"This is the averaged training accuracy for the Epoch across batches: {train_acc:.2f}%")
        print(f"This is the evaluation loss for the Epoch: {eval_loss}")
        print(f"This is the averaged evaluation accuracy for the Epoch across batches: {eval_acc:.2f}%")
        print("\n\n")

        train_losses.append(train_loss)
        eval_losses.append(eval_loss)
        
        if eval_loss < best_val_loss:
            best_val_loss = eval_loss
            #torch.save(net.state_dict(), "Best_NN.pt")
    print("------------------------------------------------------------------------------")        
    total_train_stop = time.time()
    print(f"The best evaluation Loss is: {best_val_loss}")
    print(f"The total time to train was: {(total_train_stop - total_train_start)/60} minutes")

    return train_losses, eval_losses
    
def test_loop(testloader, net, device):
    net.to(device)
    net.eval()
    averaged_acc = 0
    full_missed_predictions = []
    full_missed_targets = []
    with torch.no_grad():
        for x, target in testloader:
            x = x.to(device)
            target = target.to(device)

            output = net(x)
            output = F.softmax(output, dim=2)
            predictions = output.squeeze(0).argmax(dim=1, keepdim=False).unsqueeze(0)

            incorrect_indices = (target != predictions)
            missed_predictions = predictions[incorrect_indices].detach().cpu().tolist()
            missed_target = target[incorrect_indices].detach().cpu().tolist()

            full_missed_predictions.extend(missed_predictions)
            full_missed_targets.extend(missed_target)

            accuracy = 100* ((target == predictions).sum() / predictions.size(-1)).item()
            averaged_acc += accuracy

    misclassification_map =  {}
    for x,y in zip(full_missed_predictions, full_missed_targets):
        if y in misclassification_map:
            misclassification_map[y].append(x)
        else:
            misclassification_map[y] = []

    return averaged_acc / len(testloader), misclassification_map
        
def plot_losses(train_loss, eval_loss):
    fig, ax = plt.subplots()
    epochs = range(1,16)
    ax.plot(epochs, train_loss, "-",c="orange", label="Train Loss")
    ax.plot(epochs, eval_loss, "b-", label="Evaluation Loss")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax.set_title("Loss as a function of Epochs")
    ax.legend(loc="best")
    #plt.savefig("Train_losses.png",dpi=300)
    
    #%%
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lr = 0.001
    EPOCHS = 15
    setSeed()
    net = makeNet()
    
    word2vec = word_embedder(saved=True)
    trainloader, valloader, testloader = makeLoaders(word2vec)
    optimizer = optim.Adam(net.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=12)
    
    train_loss, eval_loss = train_loop(EPOCHS, trainloader, valloader, net, optimizer, criterion, device)
    plot_losses(train_loss, eval_loss)
    
    best_net = makeNet()
    best_net.load_state_dict(torch.load("Best_NN.pt"))
    average_test_accuracy, misclassification_map = test_loop(testloader,best_net,device)
    print(f"The averaged_test_accuracy is: {average_test_accuracy:.2f}%")
    for key, val in misclassification_map.items():
        print(key)
        print(val)
        print("\n")
 
