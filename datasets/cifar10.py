from torch.utils.data import DataLoader
from torchvision import transforms, datasets


class CIFAR10Data:
    def __init__(
        self,
        root="./data",
        batch_size=64,
        num_workers=0,
        image_size=32,
        augment=True,
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
        train_transform=None,
        test_transform=None,
    ):
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        # Preprocessing comune a training e test.

        self.transform= transforms.Compose([transforms.Resize((self.image_size, self.image_size), antialias=True),
                            transforms.ToTensor()])

        self.train_dataset = None
        self.test_dataset = None

    def setup(self, download=True):
        """Carica i dataset e associa il rispettivo preprocessing."""
        self.train_dataset = datasets.CIFAR10(
            root=self.root,
            train=True,
            download=download,
            transform=self.transform,
        )
        self.test_dataset = datasets.CIFAR10(
            root=self.root,
            train=False,
            download=download,
            transform=self.transform,
        )
        return self

    def _make_loader(self, dataset, shuffle):
        if dataset is None:
            raise RuntimeError("Chiama setup() prima di creare i DataLoader.")

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
        )

    def train_dataloader(self):
        return self._make_loader(self.train_dataset, shuffle=True)

    def test_dataloader(self):
        return self._make_loader(self.test_dataset, shuffle=False)


def cifar10_dataloaders():
    cifar10 = CIFAR10Data()
    cifar10.setup()
    train_dataloader= cifar10.train_dataloader()
    test_dataloader= cifar10.test_dataloader()
    return train_dataloader, test_dataloader


