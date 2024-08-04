import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import KFold
from metabci.brainda.algorithms.deep_learning.fbcnet import FBCNet
from metabci.brainda.algorithms.utils.model_selection import (
    set_random_seeds,
    generate_kfold_indices, match_kfold_indices)
from metabci.brainda.datasets import AlexMI
from metabci.brainda.paradigms import MotorImagery


dataset = AlexMI()  # declare the dataset
paradigm = MotorImagery(
    channels=None,
    events=['right_hand', 'feet'],
    intervals=None,
    srate=None
)  # declare the paradigm, use recommended Options

# X,y are numpy array and meta is pandas dataFrame
X, y, meta = paradigm.get_data(
    dataset,
    subjects=[3],
    return_concat=True,
    n_jobs=None,
    verbose=False)

set_random_seeds(38)
kfold = 5
indices = generate_kfold_indices(meta, kfold=kfold)

# 创建FBCNet模型实例
filter_band = 9
estimator = FBCNet(15, 250, nBands=filter_band)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(estimator.parameters(), lr=1e-5)


def train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=25):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}')


def evaluate_model(model, test_loader):
    model.eval()
    corrects = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            corrects += torch.sum(preds == labels.data)

    accuracy = corrects.double() / len(test_loader.dataset)
    return accuracy.item()


kf = KFold(n_splits=kfold)
accs = []

for train_index, test_index in kf.split(X):
    train_data = torch.utils.data.TensorDataset(torch.Tensor(X[train_index]), torch.LongTensor(y[train_index]))
    test_data = torch.utils.data.TensorDataset(torch.Tensor(X[test_index]), torch.LongTensor(y[test_index]))

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=False)

    estimator = FBCNet(15, 250, nBands=filter_band)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(estimator.parameters(), lr=1e-5)

    train_model(estimator, criterion, optimizer, train_loader, test_loader, num_epochs=10)

    accuracy = evaluate_model(estimator, test_loader)
    accs.append(accuracy)

print(np.mean(accs))
