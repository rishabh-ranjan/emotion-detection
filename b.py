import numpy as np
import pandas as pd
from skimage.filters import gabor
from skimage.feature import hog
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import sys

class FaceDataset(data.Dataset):
    def __init__(self, path):
        data = torch.FloatTensor(pd.read_csv(path, header=None).values)
        y, X = data[:,0], data[:,1:]
        self.y = y.squeeze().long()
        h = hog(X[0].reshape(48,48), orientations=4, pixels_per_cell=(4,4))
        H = np.empty((X.shape[0], len(h)))
        H[0] = h
        for i in range(1, X.shape[0]):
            H[i] = hog(X[i].reshape(48,48), orientations=4, pixels_per_cell=(4,4))
        self.X = torch.cat((X, torch.FloatTensor(H)), dim=-1)

    def to(self, device):
        self.X = self.X.to(device)
        self.y = self.y.to(device)
        return self
        
    def __len__(self):
        return self.y.shape[0]
    
    def __getitem__(self, i):
        return self.X[i], self.y[i]

class NN(nn.Module):
    def __init__(self, activ=F.sigmoid, input_dim=2304):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 100)
        self.fc2 = nn.Linear(100, 7)
        self.output = nn.Softmax(dim=1)
        self.activ = activ
        self.cross_entropy = nn.CrossEntropyLoss()
    
    def forward(self, X):
        X = self.activ(self.fc1(X))
        X = self.fc2(X)
        return self.output(X)
    
    def predict(self, X):
        return torch.argmax(self(X), dim=-1)
    
    def criterion(self, y, yhat):
        return self.cross_entropy(yhat, y)

def train_epoch(net, opt, loader):
    losses = []
    for X, y in loader:
        yhat = net(X)
        loss = net.criterion(y, yhat)
        losses.append(loss.item())
        opt.zero_grad()
        loss.backward()
        opt.step()
    return losses

def train(net, opt, loader, eps=1e-6):
    losses = []
    prev_loss = float('inf')
    while True:
        new_losses = train_epoch(net, opt, loader)
        loss = sum(new_losses)/len(new_losses)
        if prev_loss - loss < eps:
            break
        prev_loss = loss
        losses += new_losses
    return losses

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_set = FaceDataset(sys.argv[1]).to(device)
test_set = FaceDataset(sys.argv[2]).to(device)

net = NN(activ=F.leaky_relu, input_dim=train_set.X.shape[1]).to(device)
opt = optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-3)
loader = data.DataLoader(train_set, 1024, shuffle=True)
train(net, opt, loader, eps=1e-8)

with open(sys.argv[3], 'w') as f:
    for x in net.predict(test_set.X):
        print(int(x), file=f)