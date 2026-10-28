from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from ucimlrepo import fetch_ucirepo
from paths import DATA_DIR


FEATURE_COLS = [
    "fixed_acidity",
    "volatile_acidity",
    "citric_acid",
    "residual_sugar",
    "chlorides",
    "free_sulfur_dioxide",
    "total_sulfur_dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
]
TARGET_COL = "quality"


class WineQualityData:
    """Dataset UCI Wine Quality per task di regressione caricato tramite ucimlrepo.

    Include le 11 feature chimico-fisiche in input e la variabile continua
    'quality' come target (num_classes=1).
    Supporta la suddivisione per variante ('red', 'white', 'both') e la
    standardizzazione Z-score (fit calcolato solo sul training set).
    """

    def __init__(
        self,
        root=None,
        variant="both",
        test_ratio=0.2,
        seed=42,
        normalize=True,
        batch_size=64,
        train_batch_size=None,
        test_batch_size=None,
        num_workers=0,
    ):
        self.root = Path(root) if root else (DATA_DIR / "wine_quality")
        self.variant = variant.lower() if variant else "both"
        self.test_ratio = test_ratio
        self.seed = seed
        self.normalize = normalize
        self.train_batch_size = train_batch_size or batch_size
        self.test_batch_size = test_batch_size or batch_size
        self.num_workers = num_workers

        self.train_dataset = None
        self.test_dataset = None

    def setup(self):
        """Carica il dataset da ucimlrepo (o cache locale CSV) e prepara i TensorDataset."""
        self.root.mkdir(parents=True, exist_ok=True)
        csv_path = self.root / "wine_quality.csv"

        if csv_path.exists():
            df = pd.read_csv(csv_path)
        else:
            try:
                wine = fetch_ucirepo(id=186)
                df = wine.data.original
                df.to_csv(csv_path, index=False)
            except Exception as e:
                raise RuntimeError(
                    f"Impossibile scaricare Wine Quality tramite ucimlrepo: {e}. "
                    f"Verifica la connessione di rete o posiziona il file CSV in {csv_path}."
                ) from e

        # Normalizza nomi colonne (spazi -> underscore, preserva pH)
        df.columns = [c.strip().replace(" ", "_") for c in df.columns]
        if "ph" in df.columns and "pH" not in df.columns:
            df = df.rename(columns={"ph": "pH"})

        # Filtro per variante se richiesto ('red', 'white', 'both')
        if "color" in df.columns:
            if self.variant == "red":
                df = df[df["color"].astype(str).str.lower() == "red"]
            elif self.variant == "white":
                df = df[df["color"].astype(str).str.lower() == "white"]

        X = df[FEATURE_COLS].to_numpy(dtype=np.float32)
        y = df[[TARGET_COL]].to_numpy(dtype=np.float32)

        # Split train / test con seed riproducibile
        np.random.seed(self.seed)
        indices = np.random.permutation(len(X))
        split_idx = int(len(X) * (1.0 - self.test_ratio))
        train_idx, test_idx = indices[:split_idx], indices[split_idx:]

        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        # Standardizzazione Z-score (fit solo sul train set)
        if self.normalize:
            mean = X_train.mean(axis=0, keepdims=True)
            std = X_train.std(axis=0, keepdims=True)
            std[std == 0.0] = 1.0
            X_train = (X_train - mean) / std
            X_test = (X_test - mean) / std

        self.train_dataset = TensorDataset(
            torch.from_numpy(X_train), torch.from_numpy(y_train)
        )
        self.test_dataset = TensorDataset(
            torch.from_numpy(X_test), torch.from_numpy(y_test)
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
        """Restituisce il DataLoader per il set di training."""
        return self._make_loader(self.train_dataset, shuffle=True, batch_size=self.train_batch_size)

    def test_dataloader(self):
        """Restituisce il DataLoader per il set di test."""
        return self._make_loader(self.test_dataset, shuffle=False, batch_size=self.test_batch_size)


def wine_quality_dataloaders(
    batch_size=64,
    train_batch_size=None,
    test_batch_size=None,
    variant="both",
    test_ratio=0.2,
    seed=42,
    normalize=True,
    root=None,
    num_workers=0,
):
    """Crea e restituisce i DataLoader (train, test) per il dataset Wine Quality."""
    wine = WineQualityData(
        root=root,
        variant=variant,
        test_ratio=test_ratio,
        seed=seed,
        normalize=normalize,
        batch_size=batch_size,
        train_batch_size=train_batch_size,
        test_batch_size=test_batch_size,
        num_workers=num_workers,
    )
    wine.setup()
    return wine.train_dataloader(), wine.test_dataloader()

