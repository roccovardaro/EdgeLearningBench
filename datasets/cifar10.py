from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class CIFAR10Data:
    def __init__(
        self,
        root="./data",
        batch_size=64,
        train_batch_size=None,
        test_batch_size=None,
        num_workers=0,
        image_size=32,
        augment=True,
        normalize=True,
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2023, 0.1994, 0.2010),
    ):
        self.root = root
        self.train_batch_size = train_batch_size or batch_size
        self.test_batch_size = test_batch_size or batch_size
        self.num_workers = num_workers
        self.image_size = image_size

        resize = [transforms.Resize((self.image_size, self.image_size), antialias=True)] if self.image_size != 32 else []
        norm = [transforms.Normalize(mean=mean, std=std)] if normalize else []
        aug = [transforms.RandomCrop(self.image_size, padding=4), transforms.RandomHorizontalFlip()] if augment else []

        self.train_transform = transforms.Compose([
            *resize,
            *aug,
            transforms.ToTensor(),
            *norm,
        ])
        self.test_transform = transforms.Compose([
            *resize,
            transforms.ToTensor(),
            *norm,
        ])

        self.train_dataset = None
        self.test_dataset = None

    def setup(self, download=True):
        """Carica i dataset e associa il rispettivo preprocessing."""
        self.train_dataset = datasets.CIFAR10(
            root=self.root, train=True, download=download, transform=self.train_transform
        )
        self.test_dataset = datasets.CIFAR10(
            root=self.root, train=False, download=download, transform=self.test_transform
        )
        return self

    def _make_loader(self, dataset, shuffle, batch_size):
        if dataset is None:
            raise RuntimeError("Chiama setup() prima di creare i DataLoader.")

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def train_dataloader(self):
        return self._make_loader(self.train_dataset, shuffle=True, batch_size=self.train_batch_size)

    def test_dataloader(self):
        return self._make_loader(self.test_dataset, shuffle=False, batch_size=self.test_batch_size)


def cifar10_dataloaders(
    batch_size=64,
    train_batch_size=None,
    test_batch_size=None,
    image_size=32,
    augment=True,
    normalize=True,
    root="./data",
    num_workers=0,
):
    cifar10 = CIFAR10Data(
        root=root,
        batch_size=batch_size,
        train_batch_size=train_batch_size,
        test_batch_size=test_batch_size,
        image_size=image_size,
        num_workers=num_workers,
        augment=augment,
        normalize=normalize,
    )
    cifar10.setup()
    return cifar10.train_dataloader(), cifar10.test_dataloader()