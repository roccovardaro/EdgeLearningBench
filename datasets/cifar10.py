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
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2023, 0.1994, 0.2010),
    ):
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size

        # ── Transform di test (sempre uguale) ────────────────────
        self.test_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

        # ── Transform di training (con o senza augmentation) ─────
        if augment:
            self.train_transform = transforms.Compose([
                transforms.RandomCrop(self.image_size, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ])
            print("[CIFAR-10] Data augmentation: ON (RandomCrop + RandomHorizontalFlip)")
        else:
            self.train_transform = self.test_transform
            print("[CIFAR-10] Data augmentation: OFF")

        self.train_dataset = None
        self.test_dataset = None

    def setup(self, download=True):
        """Carica i dataset e associa il rispettivo preprocessing."""
        self.train_dataset = datasets.CIFAR10(
            root=self.root,
            train=True,
            download=download,
            transform=self.train_transform,
        )
        self.test_dataset = datasets.CIFAR10(
            root=self.root,
            train=False,
            download=download,
            transform=self.test_transform,
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


def cifar10_dataloaders(augment=True):
    cifar10 = CIFAR10Data(augment=augment)
    cifar10.setup()
    train_dataloader = cifar10.train_dataloader()
    test_dataloader = cifar10.test_dataloader()
    return train_dataloader, test_dataloader
