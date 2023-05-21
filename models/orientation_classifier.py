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
        transforms.RandomPerspective(distortion_scale=0.3, p=0.3),
    ]))
test_set = datasets.MNIST(
    "../datasets", train=False, download=True,
    transform=transforms.Compose([transforms.ToTensor()]))

direction = torch.tensor(np.random.choice(4, len(train_set)))
train_set.targets = direction
for cardinal in range(4):
    train_set.data[direction == cardinal] = torch.rot90(
        train_set.data[direction == cardinal], -cardinal, (1, 2))
    
direction = torch.tensor(np.random.choice(4, len(test_set)))
test_set.targets = direction
for cardinal in range(4):
    test_set.data[direction == cardinal] = torch.rot90(
        test_set.data[direction == cardinal], -cardinal, (1, 2))

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
    nn.Conv2d(1, 4, 3),
    # 26 x 26 x 4
    nn.MaxPool2d(2),
    # 13 x 13 x 4
    nn.ReLU(),
    # 13 x 13 x 4
    nn.Conv2d(4, 8, 4),
    # 10 x 10 x 8
    nn.MaxPool2d(2),
    # 5 x 5 x 8
    nn.ReLU(),
    # 5 x 5 x 8
    nn.Flatten(),
    # 200
    nn.Linear(200, 50),
    # 50
    nn.ReLU(),
    # 50
    nn.Linear(50, 4),
    # 4
    nn.Softmax(1),
    # 4
)
model = model.to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

losses = {"train": [], "test": []}
for epoch in range(10):
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

real = test_set.targets
prediction = torch.argmax(model(
    test_set.data.reshape((-1, 1, 28, 28)) / 255.0).detach(), axis=-1)
confusion = confusion_matrix(real, prediction)
sns.heatmap(confusion, annot=True)
plt.show()

model_scripted = torch.jit.script(model)  # Export to TorchScript
model_scripted.save('orientation_classifier.pt')  # Save
