from datasets.cifar10 import cifar10_dataloaders
from datasets.fashion_mnist import fashion_mnist_loaders
from datasets.mnist import mnist_loaders


def get_dataloaders(args):
    p_args = getattr(args, args.paradigm, None)
    train_bs = getattr(p_args.train, "batch_size", 64) if p_args else 64
    eval_bs = getattr(p_args.eval, "batch_size", train_bs) if p_args else train_bs

    augment = getattr(args.dataset, "augmentation", False)
    normalize = getattr(args.dataset, "normalize", True)

    dataset_name = getattr(args.dataset, "name", "").lower().replace("-", "_")

    if dataset_name == "mnist":
        return mnist_loaders(train_batch_size=train_bs, test_batch_size=eval_bs)
    if dataset_name == "cifar10":
        return cifar10_dataloaders(
            train_batch_size=train_bs,
            test_batch_size=eval_bs,
            augment=augment,
            normalize=normalize,
        )
    if dataset_name in ("fashion_mnist", "fashionmnist"):
        return fashion_mnist_loaders(
            train_batch_size=train_bs,
            test_batch_size=eval_bs,
            augment=augment,
            normalize=normalize,
        )
    return None

