# -*- coding: utf-8 -*-
"""
@author: david
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import torch.nn as nn
from torchvision import datasets, transforms

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Running on {device}")
train_set = datasets.MNIST(
    "../datasets", train=True, download=True,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomPerspective(distortion_scale=0.3, p=0.8),
    ]))
test_set = datasets.MNIST(
    "../datasets", train=False, download=True,
    transform=transforms.Compose([transforms.ToTensor()]))

train_set.data[train_set.targets == 0] = 0
test_set.data[test_set.targets == 0] = 0

with open("../datasets/digits28/data_gray.npy", "rb") as handler:
    digits = np.load(handler)
    labels = np.load(handler)
    
REP = 50
train_set.data = (255 * torch.tensor(np.tile(digits, (REP, 1, 1)))).int()
train_set.targets = torch.tensor(np.tile(labels, REP))
test_set.data = (255 * torch.tensor(digits)).int()
test_set.targets = torch.tensor(labels)

train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=256, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    test_set, batch_size=512, shuffle=False)


@torch.no_grad()
def test(model, criterion, loader):

    test_loss = 0
    correct = 0
    model.eval()

    for data, target in loader:
        data, target = data.to(device), target.to(device)

        output = model(data)
        loss = criterion(output, target)
        test_loss += loss.item()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(loader.dataset)
    accuracy = 100. * correct / len(loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(loader.dataset), accuracy))

    return test_loss, accuracy


def train(model, criterion, optimizer, epoch, loader):

    total_loss = 0.0
    model.train()

    for batch_idx, (data, target) in enumerate(loader):
        optimizer.zero_grad()
        data, target = data.to(device), target.to(device)

        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(loader.dataset),
                100. * batch_idx / len(loader), loss.item()))

        total_loss += loss.item()

    return total_loss / len(loader.dataset)


model = nn.Sequential(
    # 28 x 28 x 1
    nn.Conv2d(1, 8, 3),
    # 26 x 26 x 8
    nn.MaxPool2d(2),
    # 13 x 13 x 8
    nn.ReLU(),
    # 13 x 13 x 8
    nn.Conv2d(8, 16, 4),
    # 10 x 10 x 16
    nn.MaxPool2d(2),
    # 5 x 5 x 16
    nn.ReLU(),
    # 5 x 5 x 16
    nn.Flatten(),
    # 400
    nn.Linear(400, 50),
    # 50
    nn.ReLU(),
    # 50
    nn.Linear(50, 10),
    # 10
    nn.Softmax(1),
    # 10
)
model = model.to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

losses = {"train": [], "test": []}
for epoch in range(20):
    train_loss = train(model, criterion, optimizer, epoch, train_loader)
    test_loss, test_accuracy = test(model, criterion, test_loader)
    losses["train"].append(train_loss)
    losses["test"].append(test_loss)
model.to('cpu')

plt.plot(losses["train"], label="training loss")
plt.plot(losses["test"], label="test loss")
plt.legend()
plt.title(f'Accuracy: {test_accuracy}%')
plt.show()

real = train_set.targets
prediction = torch.argmax(model(
    train_set.data.reshape((-1, 1, 28, 28)) / 255.0).detach(), axis=-1)
confusion = confusion_matrix(real, prediction)
plt.figure(figsize=(10, 8))
sns.heatmap(confusion, annot=True)
plt.title("Train confusion")
plt.show()

real = test_set.targets
prediction = torch.argmax(model(
    test_set.data.reshape((-1, 1, 28, 28)) / 255.0).detach(), axis=-1)
confusion = confusion_matrix(real, prediction)
plt.figure(figsize=(10, 8))
sns.heatmap(confusion, annot=True)
plt.title("Test confusion")
plt.show()

model_scripted = torch.jit.script(model) # Export to TorchScript
model_scripted.save('digit_classifier.pt') # Save