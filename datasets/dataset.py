from datasets.cifar10 import cifar10_dataloaders
from datasets.mnist import mnist_loaders


def get_dataloaders(args):

    if args.dataset.name == "mnist":
        return mnist_loaders()
    if args.dataset.name == "cifar10":
        return cifar10_dataloaders()
    return None
