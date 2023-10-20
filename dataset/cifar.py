from torchvision import transforms
from torchvision.datasets import CIFAR10 as cifar10
from torch.utils.data import random_split


def CIFAR10(params):
    mean_val = [0.4914, 0.4822, 0.4465]
    std_val = [0.2470, 0.2435, 0.2616]
    save_path = './data'

    random_transform1 = transforms.RandomHorizontalFlip(p=0.5)
    random_transform2 = transforms.Compose(
        [
            transforms.Pad(padding=4),
            transforms.RandomCrop((32, 32))
        ]
    )

    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean_val, std=std_val),
        transforms.RandomChoice([random_transform1, random_transform2]),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean_val, std=std_val),
    ])

    train_dataset = cifar10(
        save_path,
        train=True,
        download=True,
        transform=train_transform
    )

    test_dataset = cifar10(
        save_path,
        train=False,
        download=True,
        transform=test_transform
    )

    train_size = int(params['split']['train'] * len(train_dataset))
    valid_size = int(params['split']['valid'] * len(train_dataset))
    train_other_size = len(train_dataset) - train_size - valid_size

    train_datasets = random_split(
        train_dataset,
        [train_size, valid_size, train_other_size]
    )

    test_size = int(params['split']['test'] * len(test_dataset))
    test_other_size = len(test_dataset) - test_size

    test_datasets = random_split(
        test_dataset,
        [test_size, test_other_size]
    )

    train_dataset, valid_dataset, _ = train_datasets
    test_dataset, _ = test_datasets

    return train_dataset, valid_dataset, test_dataset
