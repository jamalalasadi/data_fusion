
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import torch
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
# Generate toy data that has two distinct classes and a huge gap between them
X, Y = make_blobs(n_samples=500, centers=2, random_state=0, cluster_std=0.4)  # X - features, Y - labels
# Plot the toy data
plt.scatter(x=X[:, 0], y=X[:, 1])

file_path ='/home/jamal/Documents/support vector /Re%3a_Cyberbullying_dataset/text_train_5_df.txt'


class SVM(nn.Module):
    """
    Linear Support Vector Machine
    -----------------------------
    This SVM is a subclass of the PyTorch nn module that
    implements the Linear  function. The  size  of  each 
    input sample is 2 and output sample  is 1.
    """
    def __init__(self):
        super().__init__()  # Call the init function of nn.Module
        self.fully_connected = nn.Linear(2, 1)  # Implement the Linear function
        
    def forward(self, x):
        fwd = self.fully_connected(x)  # Forward pass
        return fwd
        
    data = X  # Before feature scaling
    X = (X - X.mean())/X.std()  # Feature scaling
    Y[Y == 0] = -1  # Replace zeros with -1
    plt.scatter(x=X[:, 0], y=X[:, 1])  # After feature scaling
    plt.scatter(x=data[:, 0], y=data[:, 1], c='r')  # Before feature scalin
    
    
    
learning_rate = 0.1  # Learning rate
epoch = 10  # Number of epochs
batch_size = 1  # Batch size

X = torch.FloatTensor(X)  # Convert X and Y to FloatTensors
Y = torch.FloatTensor(Y)
N = len(Y)  # Number of samples, 500

model = SVM()  # Our model
optimizer = optim.SGD(model.parameters(), lr=learning_rate)  # Our optimizer
model.train()  # Our model, SVM is a subclass of the nn.Module, so it inherits the train method
for epoch in range(epoch):
    perm = torch.randperm(N)  # Generate a set of random numbers of length: sample size
    sum_loss = 1  # Loss for each epoch
        
    for i in range(0, N, batch_size):
        x = X[perm[i:i + batch_size]]  # Pick random samples by iterating over random permutation
        y = Y[perm[i:i + batch_size]]  # Pick the correlating class
        
        x = Variable(x)  # Convert features and classes to variables
        y = Variable(y)

        optimizer.zero_grad()  # Manually zero the gradient buffers of the optimizer
        output = model(x)  # Compute the output by doing a forward pass
        
        loss = torch.mean(torch.clamp(1 - output * y, min=0))  # hinge loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Optimize and adjust weights

        sum_loss += loss[1].data.cpu().numpy()  # Add the loss
        
    print("Epoch {}, Loss: {}".format(epoch, sum_loss[1]))
    
    
    f = open('/home/jamal/Documents/support vector /Re%3a_Cyberbullying_dataset/text_train_5_df.txt', 'r')

    rows = []
    for line in f:
        # Split on any whitespace (including tab characters)
        row = line.split()
        # Convert strings to numeric values:
        row[1] = float(row[1])
        row[2] = int(row[2])
        # Append to our list of lists:
        rows.append(row)
    
    print (rows)
    
    n = [3,5,7]
    def myFun(n):
        for i in range(0,3):
            print (n[i])
            

    