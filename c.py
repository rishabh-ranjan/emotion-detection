import os
import time
import torch
from torch import nn, optim
import sys
from torch.utils.data import DataLoader, Dataset, random_split
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
from PIL import Image


class FaceDataset(Dataset):
    def __init__(self, path, over_sample=False, augment=False):
        data = torch.tensor(pd.read_csv(path, header=None).values)
        y, X = data[:,0], data[:,1:]
        y = y.squeeze()
        
        self.X = X.reshape(-1, 1, 48, 48).float()
        self.y = y.long()
        
    def __len__(self):
        return self.y.shape[0]
    
    def __getitem__(self, i):
        return self.X[i], self.y[i]
    
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=3, padding=0)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=2, stride=2, padding=0)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(in_features=512, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=7)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.bn1(x)
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.bn2(x)
        x = self.pool2(x)
        
        x = x.view(-1, 512)
        x = F.relu(self.fc1(x))
        x = self.bn3(x)
        
        x = self.softmax(self.fc2(x))
        
        return x 

    
def write_predictions(fname, arr):
    np.savetxt(fname, arr, fmt="%d", delimiter="\n")

if __name__ == '__main__':
    # args[1]: train_file, args[2]: test_file, args[3]: output_file
    tic = time.time()
    
    train_data = FaceDataset(sys.argv[1])
    
    batch_size = 64
    train_data_loader = DataLoader(train_data, batch_size, shuffle=True, num_workers=4)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(0)

    net = ConvNet()
    net = net.to(device)
    freq = np.bincount(train_data.y)
    inv_freq = train_data.y.shape[0] / freq
    inv_freq = 7 * inv_freq / inv_freq.sum()


    loss_function = nn.CrossEntropyLoss(weight=torch.tensor(inv_freq).float().cuda())
    optimizer = optim.Adam(net.parameters(), lr=5e-4)
    
    max_epochs = 50
    for epoch in range(max_epochs):
        
        for i, data in enumerate(train_data_loader, 0):
            images, classes = data
            images, classes = images.to(device), classes.to(device)
            
            optimizer.zero_grad()
            output = net(images)

            loss = loss_function(output, classes)
            loss.backward()
            optimizer.step()
        
        toe = time.time()
        if toe - tic > 1700:
            break
    
    
    test_dataset = FaceDataset(sys.argv[2])
    with torch.no_grad():
        prediction = []
        test_loader = torch.utils.data.DataLoader(test_dataset, 1000, shuffle=False, num_workers=4)

        for i, data in enumerate(test_loader):
            images, _ = data
            images = images.to(device)

            output = net(images)
            output = torch.argmax(output, dim=1)
            
            prediction.extend(output.cpu().tolist())

    write_predictions(sys.argv[3], np.array(prediction))
    
    
