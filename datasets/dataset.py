from datasets.cifar10 import cifar10_dataloaders
from datasets.fashion_mnist import fashion_mnist_dataloaders
from datasets.mnist import mnist_dataloaders
from datasets.breast_cancer import breast_cancer_dataloaders
from datasets.adult import adult_dataloaders


def get_dataloaders(args):
    p_args = getattr(args, args.paradigm, None)
    train_bs = getattr(p_args.train, "batch_size", 64) if p_args else 64
    eval_bs = getattr(p_args.eval, "batch_size", train_bs) if p_args else train_bs

    augment = getattr(args.dataset, "augmentation", False)
    normalize = getattr(args.dataset, "normalize", True)
    num_workers = max(0, int(getattr(args.dataset, "num_workers", 0) or 0))

    dataset_name = getattr(args.dataset, "name", "").lower().replace("-", "_")

    if dataset_name == "mnist":
        return mnist_dataloaders(
            train_batch_size=train_bs,
            test_batch_size=eval_bs,
            augment=augment,
            normalize=normalize,
            num_workers=num_workers,
        )
    if dataset_name == "cifar10":
        return cifar10_dataloaders(
            train_batch_size=train_bs,
            test_batch_size=eval_bs,
            augment=augment,
            normalize=normalize,
            num_workers=num_workers,
        )
    if dataset_name in "fashion_mnist":
        return fashion_mnist_dataloaders(
            train_batch_size=train_bs,
            test_batch_size=eval_bs,
            augment=augment,
            normalize=normalize,
            num_workers=num_workers,
        )
    if dataset_name in "breast_cancer":
        test_ratio = getattr(args.dataset, "test_ratio", 0.2)
        seed = getattr(args.dataset, "seed", 42)
        stratified = getattr(args.dataset, "stratified", True)
        return breast_cancer_dataloaders(
            batch_size=train_bs,
            train_batch_size=train_bs,
            test_batch_size=eval_bs,
            test_ratio=test_ratio,
            seed=seed,
            normalize=normalize,
            stratified=stratified,
            num_workers=num_workers,
        )
    if dataset_name in "adult":
        split = getattr(args.dataset, "split", "official")
        test_ratio = getattr(args.dataset, "test_ratio", 0.2)
        seed = getattr(args.dataset, "seed", 42)
        feature_mode = getattr(args.dataset, "feature_mode", "all")
        return adult_dataloaders(
            batch_size=train_bs,
            train_batch_size=train_bs,
            test_batch_size=eval_bs,
            split=split,
            test_ratio=test_ratio,
            seed=seed,
            normalize=normalize,
            feature_mode=feature_mode,
            num_workers=num_workers,
        )
    return None

