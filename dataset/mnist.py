from torchvision import transforms
from torchvision.datasets import MNIST as mnist
from torch.utils.data import random_split


def MNIST(params):
    mean_val = [0.1307]
    std_val = [0.3081]
    save_path = './data'

    random_transform1 = transforms.RandomHorizontalFlip(p=0.5)
    random_transform2 = transforms.Compose([transforms.Pad(padding=4),
                                            transforms.RandomCrop((28, 28))])

    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean_val,
                             std=std_val),
        transforms.RandomChoice([random_transform1, random_transform2]),

    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean_val,
                             std=std_val),
    ])

    train_dataset = mnist(save_path,
                          train=True,
                          download=True,
                          transform=train_transform)

    test_dataset = mnist(save_path,
                         train=False,
                         download=True,
                         transform=test_transform)

    train_size = int(params['split']['train'] * len(train_dataset))
    valid_size = int(params['split']['valid'] * len(train_dataset))
    train_other_size = len(train_dataset) - train_size - valid_size

    train_datasets = random_split(train_dataset, [train_size,
                                                  valid_size,
                                                  train_other_size])

    test_size = int(params['split']['test'] * len(test_dataset))
    test_other_size = len(test_dataset) - test_size

    test_datasets = random_split(test_dataset, [test_size,
                                                test_other_size])

    train_dataset, valid_dataset, _ = train_datasets
    test_dataset, _ = test_datasets

    return train_dataset, valid_dataset, test_dataset
