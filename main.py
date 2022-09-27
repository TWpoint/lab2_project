import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch import optim
from torchvision import datasets
from torch.autograd import Variable
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader as DataLoader


def visualize():
    # plt.imshow(train_data.data[0], cmap='gray')
    # plt.title('%i' % train_data.targets[0])
    # plt.show()
    figure = plt.figure(figsize=(10, 8))
    cols, rows = 5, 5
    for i in range(1, cols * rows + 1):
        sample_idx = torch.randint(len(train_data), size=(1,)).item()
        img, label = train_data[sample_idx]
        figure.add_subplot(rows, cols, i)
        plt.title(label)
        plt.axis('off')
        plt.imshow(img.squeeze(), cmap='gray')
    plt.savefig('./visualization.png', dpi=300)
    plt.show()


class Hyperparameters():
    def __init__(self, a, b, c, d, e, f):
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.e = e
        self.f = f
        self.g = 28
        self.g = int((self.g + 2 * self.d - self.b[0]) / self.c + 1)
        self.g = int((self.g - 2) / self.e + 1)
        self.g = int((self.g + 2 * self.d - self.b[0]) / self.c + 1)
        self.g = int((self.g - 2) / self.e + 1)
        self.g = self.g * self.g * self.f
        self.h = 10


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=hyperparameters.a, kernel_size=hyperparameters.b,
                      stride=hyperparameters.c, padding=hyperparameters.d),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=hyperparameters.e)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=hyperparameters.a, out_channels=hyperparameters.f, kernel_size=hyperparameters.b,
                      stride=hyperparameters.c, padding=hyperparameters.d),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=hyperparameters.e)
        )

        self.out = nn.Linear(in_features=hyperparameters.g, out_features=hyperparameters.h)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output, x


def train(num_epoch, cnn, loaders):
    cnn.train()
    total_step = len(loaders['train'])
    for epoch in range(num_epoch):
        for i, (images, labels) in enumerate(loaders['train']):
            b_x = Variable(images)
            b_y = Variable(labels)
            output = cnn(b_x)[0]
            loss = loss_func(output, b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i + 1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epoch, i + 1, total_step,
                                                                         loss.item()))
        test()


def test():
    cnn.eval()

    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in loaders['test']:
            test_output, last_layer = cnn(images)
            pred_y = torch.max(test_output, 1)[1].data.squeeze()
            accuracy = (pred_y == labels).sum().item() / float(labels.size(0))
    print('Test Accuracy of the model on 10000 test images: %.2f' % accuracy)


if __name__ == '__main__':
    # load data
    train_data = datasets.MNIST(root='data', train=True, transform=ToTensor(), download=True)
    test_data = datasets.MNIST(root='data', train=False, transform=ToTensor())
    loaders = {
        'train': DataLoader(train_data, batch_size=100, shuffle=True, num_workers=1),
        'test': DataLoader(test_data, batch_size=100, shuffle=True, num_workers=1)
    }

    # visualize()

    # instantiation
    hyperparameters = Hyperparameters(a=20, b=(2, 2), c=1, d=0, e=2, f=10)
    cnn = CNN()
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(cnn.parameters(), lr=0.01)

    # train & test
    num_epoch = 20
    train(num_epoch=num_epoch, cnn=cnn, loaders=loaders)
