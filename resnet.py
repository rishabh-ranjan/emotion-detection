import os, sys, math
import time
import torch
from torch import nn, optim
from torch import tensor
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
from facenet_pytorch import InceptionResnetV1

class FaceDataset(Dataset):
    def __init__(self, path):
        data = torch.tensor(pd.read_csv(path, header=None).values)
        y, X = data[:,0], data[:,1:]
        y = y.squeeze()
        
        self.X = X.reshape(-1, 1, 48, 48).float()
        self.y = y.long()
        self.scaler = nn.Upsample(size=160, mode='bilinear', align_corners=True)
        
    def __len__(self):
        return self.y.shape[0]
    
    def __getitem__(self, i):
        X = self.scaler(self.X[i].unsqueeze(0)).squeeze(0)    
        X = torch.cat((X, X, X))
        return X, self.y[i]

def write_predictions(fname, arr):
    np.savetxt(fname, arr, fmt="%d", delimiter="\n")
    
    
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resnet = InceptionResnetV1(classify=True, pretrained='vggface2', num_classes=7)
    resnet = resnet.to(device)
    torch.cuda.empty_cache()
    
    
    train_data = FaceDataset(sys.argv[1])
    test_data = FaceDataset(sys.argv[2])

    freq = np.bincount(train_data.y)
    inv_freq = train_data.y.shape[0] / freq
    inv_freq = 7 * inv_freq / inv_freq.sum()

    loss_function = nn.CrossEntropyLoss(weight=torch.tensor(inv_freq).float().cuda())

    lr_init = 5e-5
    loader = torch.utils.data.DataLoader(train_data, 64, shuffle=True, num_workers=8)

    for epoch in range(5):
        lr = lr_init / (epoch + 1)

        optimizer = optim.Adam(resnet.parameters(), lr=lr)
        for i, data in enumerate(loader, 0):
            images, classes = data
            images, classes = images.to(device).float(), classes.to(device).long()
            output = resnet(images)

            loss = loss_function(output, classes)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
    with torch.no_grad():
        prediction = []
        test_loader = torch.utils.data.DataLoader(test_data, 1000, shuffle=False, num_workers=8)

        for i, data in enumerate(test_loader):
            images, features = data
            images, features = images.to(device).float(), features.to(device).float()

            output = resnet(images)
            output = torch.argmax(output, dim=1)
            
            prediction.extend(output.cpu().tolist())
    
    write_predictions(sys.argv[3], np.array(prediction))

        

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
