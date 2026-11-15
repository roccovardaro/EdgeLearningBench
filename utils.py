def get_info_dataset(args, train_loader, test_loader):
    aug_flag = getattr(args.dataset, "augmentation", False)
    norm_flag = getattr(args.dataset, "normalize", True)
    n_train = len(train_loader.dataset) if hasattr(train_loader, "dataset") else len(train_loader)
    n_test = len(test_loader.dataset) if hasattr(test_loader, "dataset") else len(test_loader)
    n_total = n_train + n_test

    return aug_flag, norm_flag, n_train, n_test, n_total
