from datasets.cifar10 import cifar10_dataloaders
from datasets.mnist import mnist_loaders


def get_dataloaders(args):
    p_args = getattr(args, args.paradigm, None)
    train_bs = getattr(p_args.train, "batch_size", 64) if p_args else 64
    eval_bs = getattr(p_args.eval, "batch_size", train_bs) if p_args else train_bs

    augment = getattr(args.dataset, "augmentation", False)
    normalize = getattr(args.dataset, "normalize", True)

    if args.dataset.name == "mnist":
        return mnist_loaders(train_batch_size=train_bs, test_batch_size=eval_bs)
    if args.dataset.name == "cifar10":
        return cifar10_dataloaders(
            train_batch_size=train_bs,
            test_batch_size=eval_bs,
            augment=augment,
            normalize=normalize,
        )
    return None
