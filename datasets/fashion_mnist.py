from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class FashionMNISTData:
    def __init__(
        self,
        root="./data",
        batch_size=64,
        train_batch_size=None,
        test_batch_size=None,
        num_workers=0,
        image_size=28,
        augment=False,
        normalize=True,
        mean=(0.2860,),
        std=(0.3530,),
    ):
        self.root = root
        self.train_batch_size = train_batch_size or batch_size
        self.test_batch_size = test_batch_size or batch_size
        self.num_workers = num_workers
        self.image_size = image_size

        norm = [transforms.Normalize(mean=mean, std=std)] if normalize else []
        aug = [
            transforms.RandomCrop(self.image_size, padding=2),
            transforms.RandomHorizontalFlip(),
        ] if augment else []

        self.train_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size), antialias=True),
            *aug,
            transforms.ToTensor(),
            *norm,
        ])
        self.test_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size), antialias=True),
            transforms.ToTensor(),
            *norm,
        ])

        self.train_dataset = None
        self.test_dataset = None

    def setup(self, download=True):
        """Carica i dataset e associa il rispettivo preprocessing."""
        self.train_dataset = datasets.FashionMNIST(
            root=self.root, train=True, download=download, transform=self.train_transform
        )
        self.test_dataset = datasets.FashionMNIST(
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
        )

    def train_dataloader(self):
        """
        Restituisce il DataLoader per il set di training.
        """
        return self._make_loader(self.train_dataset, shuffle=True, batch_size=self.train_batch_size)

    def test_dataloader(self):
        """
        Restituisce il DataLoader per il set di test.
        """
        return self._make_loader(self.test_dataset, shuffle=False, batch_size=self.test_batch_size)


def fashion_mnist_dataloaders(
    batch_size=64,
    train_batch_size=None,
    test_batch_size=None,
    image_size=28,
    augment=False,
    normalize=True,
    root="./data",
    num_workers=0,
):
    fashion_mnist = FashionMNISTData(
        root=root,
        batch_size=batch_size,
        train_batch_size=train_batch_size,
        test_batch_size=test_batch_size,
        image_size=image_size,
        num_workers=num_workers,
        augment=augment,
        normalize=normalize,
    )
    fashion_mnist.setup()
    return fashion_mnist.train_dataloader(), fashion_mnist.test_dataloader()

