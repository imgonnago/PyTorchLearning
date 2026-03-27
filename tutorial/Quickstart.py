#working with data
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor


#the torchvision.datasets module contains Dataset objects for many real-world vision data like CIFAR,COCO.
# In this tutorial, we use the FashinMNIST dataset.
# Evry TorchVision Dataset includes two arguments: transform and target_transform to modify the samples and  labels respectively.
def start():
    train_data, test_data = data_download()
    train_dataloader, test_dataloader = dataloader(train_data, test_data)
    model, device, NeuralNetwork= MLP()
    loss_fn, optimizer = optim_loss_fn(model)
    run(train_dataloader, test_dataloader, model, loss_fn, optimizer, device)
    Q_to_save = input('do you want to save this model?: ')
    if Q_to_save == 'y':
        saving_model(model)
    else:
        print('not save the model \n byee!')


#Download training data from open datasets.
def data_download():
    train_data = datasets.FashionMNIST(
        root='data',
        train=True,
        download=True,
        transform=ToTensor(),
    )

    test_data = datasets.FashionMNIST(
        root='data',
        train=False,
        download=True,
        transform=ToTensor(),
    )
    return train_data, test_data

def dataloader(train_data, test_data):
    batch_size = 64
    train_dataloader = DataLoader(train_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    for X, y in test_dataloader:
        print(f'shape of X [N, C, H, W]: {X.shape}')
        print(f'shape of y: {y.shape} {y.dtype}')
        break

    return train_dataloader, test_dataloader

def MLP():
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else 'cpu'
    print(f'Using {device} device')

    #Define Model
    class NeuralNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.flatten = nn.Flatten()
            self.linear_relu_stack = nn.Sequential(
                nn.Linear(28*28, 512),
                nn.ReLU(),
                nn.Linear(512,512),
                nn.ReLU(),
                nn.Linear(512,10)
            )

        def forward(self, x):
            x = self.flatten(x)
            logits = self.linear_relu_stack(x)
            return logits

    model = NeuralNetwork().to(device)

    return model, device, NeuralNetwork

def optim_loss_fn(model):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    return loss_fn, optimizer


def train(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X,y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        #BackPropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f'loss: {loss:>7f} [{current:>5d}/{size:>5d}')

def test(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f'Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n')


def run(train_dataloader, test_dataloader, model, loss_fn, optimizer, device, epochs = 5):
    for t in range(epochs):
        print(f'Epoch {t+1} \n----------------------------')
        train(train_dataloader, model, loss_fn, optimizer, device)
        test(test_dataloader, model, loss_fn, device)
    print("Done!")

def saving_model(model):
    torch.save(model.state_dict(), "model.pth")
    print("Saved PyTorch Model State to model.pth")


def loding_model(model):
    model.load_state_dict(torch.load("model.pth", weights_only=True))


if __name__ == '__main__':
    print("PyTorch Tutorial (Quickstart. but it wasn't quick lol)")
    start()