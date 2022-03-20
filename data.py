import os
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10, CIFAR100, SVHN, Omniglot


def shuffle(x, y=None):
    index = list(range(x.shape[0]))
    np.random.shuffle(index)
    if y is not None:
        return x[index], y[index]
    return x[index]


class MyDataset(Dataset):
    def __init__(self, names, normalize=True):
        self.names = names
        if normalize:
            self.transform = transforms.Compose([

                transforms.ToTensor(),
                # transforms.Normalize(mean=[0.485, 0.456, 0.406],
                #                      std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
            ])

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        name = self.names[index]
        img = Image.open(name).convert('RGB')
        return self.transform(img), -1


def mnist_data_loader(mnist_folder='./data/mnist_data', batch_size=64):
    """ grond truth label
    """
    transform = transforms.Compose([
        transforms.Resize([32, 32]),
        transforms.ToTensor(),
    ])
    train_set = MNIST(mnist_folder, train=True, download=True, transform=transform)
    test_set = MNIST(mnist_folder, train=False, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=4)
    return train_loader, test_loader


def fashion_mnist_data_loader(folder='./data/fashion_mnist_data', batch_size=64):
    """ ground truth label
    """
    transform = transforms.Compose([
        transforms.Resize([32, 32]),
        transforms.ToTensor(),
    ])
    train_set = FashionMNIST(folder, train=True, download=True, transform=transform)
    test_set = FashionMNIST(folder, train=False, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader


def cifar10_data_loader(folder='./data/cifar10_data', batch_size=64):
    """ ground truth label
    """
    transform = transforms.Compose([
        transforms.Resize([32, 32]),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                      std=[0.229, 0.224, 0.225])
    ])
    train_set = CIFAR10(folder, train=True, download=True, transform=transform)
    test_set = CIFAR10(folder, train=False, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader


def cifar100_data_loader(folder='./data/cifar100_data', batch_size=64):
    """ ground truth label
    """
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, 4),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                      std=[0.229, 0.224, 0.225])
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                      std=[0.229, 0.224, 0.225])
    ])
    train_set = CIFAR100(folder, train=True, download=True, transform=train_transform)
    test_set = CIFAR100(folder, train=False, download=True, transform=test_transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader


def lsun_data_loader(folder='./data/lsun_data/', batch_size=64):
    """ non ground truth label
    """
    img_names = [folder + name for name in os.listdir(folder)]
    data_set = MyDataset(img_names)
    data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=True)
    return data_loader


def svhn_data_loader(folder='./data/svhn_data', batch_size=64, img_size=(32, 32)):
    """ground truth label
    """
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
    ])
    train_set = SVHN(folder, split='train', download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    val_set = SVHN(folder, split='test', download=True, transform=transform)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)

    extra_set = SVHN(folder, split='extra', download=True, transform=transform)
    extra_loader = DataLoader(extra_set, batch_size=batch_size, shuffle=True)
    return train_loader, val_loader, extra_loader


def omniglot_data_loader(folder='./data/omniglot_data', batch_size=64, img_size=(28, 28)):
    """ ground truth label
    """
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
    ])
    data_set = Omniglot(folder, download=True, transform=transform)
    data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=True, num_workers=4)
    return data_loader


def tinyImageNet_data_loader(folder='./data/tinyImageNet_data/', batch_size=64, normalize=True):
    """ ground truth label
    """
    img_names = [folder + name for name in os.listdir(folder)]
    data_set = MyDataset(img_names, normalize)
    data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=True)
    return data_loader


if __name__ == '__main__':
    
    data_train, data_test = cifar10_data_loader()  # checked
    
    for x, y in data_train:
        print(x.shape, y.shape, x.max(), x.min(), y)