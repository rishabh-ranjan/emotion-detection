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
        print(y.shape, X.shape)
        y = y.squeeze()
        
        self.X = X.float()
        self.y = y.long()
        
    def __len__(self):
        return self.y.shape[0]
    
    def __getitem__(self, i):
        return self.X[i], self.y[i]
    
class NNet(nn.Module):
    def __init__(self):
        super(NNet, self).__init__()
        self.fc1 = nn.Linear(2304, 100)
        self.fc2 = nn.Linear(100, 7)
        self.output = nn.Softmax(dim=1)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return self.output(x)

    
def write_predictions(fname, arr):
    np.savetxt(fname, arr, fmt="%d", delimiter="\n")

if __name__ == '__main__':
    # args[1]: train_file, args[2]: test_file, args[3]: output_file
    tic = time.time()
    
    dataset = FaceDataset(sys.argv[1])
    size = dataset.__len__()
    
    train_data, val_data = random_split(dataset, [int(0.9 * size), int(0.1 * size)]) #90-10 train-val split
    
    batch_size = 32
    train_data_loader = DataLoader(train_data, batch_size, shuffle=True, num_workers=4)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(0)

    net = NNet()
    net = net.to(device)

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=1e-4)
    
    max_epochs = 10
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
        if toe - tic > 500:
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
    
    
