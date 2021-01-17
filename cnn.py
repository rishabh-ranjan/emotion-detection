from imblearn.over_sampling import RandomOverSampler
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data

class FaceDataset(data.Dataset):
    def __init__(self, path, over_sample=False):
        y, X = np.split(np.genfromtxt(path, delimiter=','), [1], axis=-1)
        y = y.squeeze()
        if over_sample:
            X, y = RandomOverSampler().fit_resample(X, y)
        X = X.reshape(-1, 1, 48, 48)
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    
    def __len__(self):
        return self.y.shape[0]
    
    def __getitem__(self, i):
        return self.X[i], self.y[i]

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 64, 1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(0.25),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Dropout(0.25),

            nn.Conv2d(128, 256, 5, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),

            nn.Flatten(),

            nn.Linear(24*24*256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.25),

            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.25),

            nn.Linear(256, 7)
        )

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, X):
        return self.model(X)

    def predict(self, X):
        return torch.argmax(self.forward(X), dim=-1)

def train(model, opt, train_loader, n_epochs, test_loader=None, private_loader=None):
    losses = []
    mx_F1 = 0
    for epoch in range(n_epochs):
        model.train()
        for X, y in train_loader:
            X = X.to(device)
            y = y.to(device)
            yhat = model(X)
            loss = model.criterion(yhat, y)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
        
        torch.save(model.state_dict(), f'cnn_{lr}_{batch_size}_{epoch+1}.pt')
        
        if test_loader is not None:
            print(f'#{epoch+1}:- ', end='')
            model.eval()
            true = []
            pred = []
            for X, y in test_loader:
                X = X.to(device)
                y = y.to(device)
                yhat = model.predict(X)
                true.extend(y.cpu().tolist())
                pred.extend(yhat.cpu().tolist())
            print(f'acc: {accuracy_score(true, pred):.5f}', end=' ')
            print(f'p: {precision_score(true, pred, average="micro"):.5f}', end=' ')
            print(f'r: {recall_score(true, pred, average="micro"):.5f}', end=' ')
            print(f'f1: {f1_score(true, pred, average="micro"):.5f}', end=' ')
            print(f'P: {precision_score(true, pred, average="macro"):.5f}', end=' ')
            print(f'R: {recall_score(true, pred, average="macro"):.5f}', end=' ')
            print(f'F1: {f1_score(true, pred, average="macro"):.5f}', flush=True)
            F1 = f1_score(true, pred, average='macro')
            if F1 > mx_F1:
                mx_F1 = F1
                if private_loader is not None:
                    pred = []
                    for X, _ in private_loader:
                        X = X.to(device)
                        yhat = model.predict(X)
                        pred.extend(yhat.cpu().tolist())
                    with open(f'pred_{epoch+1}.csv', 'w') as f:
                        for i,x in enumerate(pred):
                            f.write(f'{i},{int(x)}\n')

    return losses

if __name__ == '__main__':
    import sys
    lr = float(sys.argv[1])
    batch_size = int(sys.argv[2])
    n_epochs = int(sys.argv[3])
    print('lr:', lr)
    print('batch_size:', batch_size)
    print('n_epochs:', n_epochs)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print('device:', device)
    print('loading data')
    train_set = FaceDataset('orig_train.csv', over_sample=True)
    test_set = FaceDataset('public_test.csv')
    private_set = FaceDataset('private.csv')
    print('train size:', len(train_set))
    print('test size:', len(test_set))
    print('private size:', len(private_set))
    model = Model().to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    train_loader = data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)
    private_loader = data.DataLoader(private_set, batch_size=batch_size, shuffle=False, num_workers=4)
    train_losses = train(model, opt, train_loader, n_epochs, test_loader, private_loader)
    plt.plot(train_losses)
    plt.savefig('cnn.png')
    print('training losses plotted in cnn.png')
