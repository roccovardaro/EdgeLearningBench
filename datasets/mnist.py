from torch.utils.data import DataLoader


def mnist_loaders(
    batch_size=128,
    train_batch_size=None,
    test_batch_size=None,
    root="./data",
    num_workers=0,
):
    from torchvision.datasets import MNIST
    from torchvision.transforms import ToTensor

    train_bs = train_batch_size or batch_size
    test_bs = test_batch_size or batch_size

    # Mantieni [B, 1, 28, 28]: le convoluzioni richiedono immagini.
    transform = ToTensor()
    train_data = MNIST(root, train=True, download=True, transform=transform)
    test_data = MNIST(root, train=False, download=True, transform=transform)
    return (
        DataLoader(train_data, batch_size=train_bs, shuffle=True, num_workers=num_workers),
        DataLoader(test_data, batch_size=test_bs, shuffle=False, num_workers=num_workers),
    )