from torch.utils.data import DataLoader


def mnist_loaders(batch_size=128, root="./data"):
    from torchvision.datasets import MNIST
    from torchvision.transforms import Compose, Normalize, ToTensor

    # Mantieni [B, 1, 28, 28]: le convoluzioni richiedono immagini.
    transform = ToTensor()
    train_data = MNIST(root, train=True, download=True, transform=transform)
    test_data = MNIST(root, train=False, download=True, transform=transform)
    return (
        DataLoader(train_data, batch_size=batch_size, shuffle=True),
        DataLoader(test_data, batch_size=batch_size, shuffle=False),
    )