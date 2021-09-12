import os

import torchvision
import torchvision.transforms as transforms
from filelock import FileLock


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


def load_data():
    data_dir = os.path.abspath("/home/ubuntu/exps/project-hpo-with-ray-tune/data")

    classes = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    with FileLock(os.path.expanduser("{}.lock".format(data_dir))):
        train_set = torchvision.datasets.CIFAR10(
            root=data_dir, train=True, download=True, transform=transform)

        test_set = torchvision.datasets.CIFAR10(
            root=data_dir, train=False, download=True, transform=transform)

    return train_set, test_set, classes
