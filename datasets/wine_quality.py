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
    """Dataset UCI Wine Quality per task di classificazione multiclasse caricato tramite ucimlrepo.

    Include le 11 feature chimico-fisiche in input e la variabile 'quality'
    mappata su classi contigue [0, num_classes - 1] (punteggi da 3 a 9 mappati su 0-6, num_classes=7).
    Supporta la standardizzazione Z-score (fit calcolato solo sul training set)
    e split train/test stratificato o casuale.
    """

    def __init__(
        self,
        root=None,
        test_ratio=0.2,
        seed=42,
        normalize=True,
        stratified=True,
        batch_size=64,
        train_batch_size=None,
        test_batch_size=None,
        num_workers=0,
    ):
        self.root = Path(root) if root else (DATA_DIR / "wine_quality")
        self.test_ratio = test_ratio
        self.seed = seed
        self.normalize = normalize
        self.stratified = stratified
        self.train_batch_size = train_batch_size or batch_size
        self.test_batch_size = test_batch_size or batch_size
        self.num_workers = num_workers

        self.train_dataset = None
        self.test_dataset = None
        self.label_mapping = None

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

        # Mappatura multiclasse dei punteggi di qualità (es. 3-9 -> 0-6)
        unique_qualities = sorted(df[TARGET_COL].unique())
        self.label_mapping = {q: idx for idx, q in enumerate(unique_qualities)}
        y = df[TARGET_COL].map(self.label_mapping).to_numpy(dtype=np.int64)
        X = df[FEATURE_COLS].to_numpy(dtype=np.float32)

        # Split train / test con seed riproducibile (stratificato o casuale)
        if self.stratified:
            X_train, X_test, y_train, y_test = self._stratified_split(X, y)
        else:
            X_train, X_test, y_train, y_test = self._random_split(X, y)

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

    def _stratified_split(self, X, y):
        """Esegue uno split train/test stratificato preservando le proporzioni di classe."""
        rng = np.random.default_rng(self.seed)
        train_indices, test_indices = [], []

        for c in np.unique(y):
            c_idx = np.where(y == c)[0]
            rng.shuffle(c_idx)
            split_idx = int(len(c_idx) * (1.0 - self.test_ratio))
            train_indices.extend(c_idx[:split_idx])
            test_indices.extend(c_idx[split_idx:])

        train_indices = np.array(train_indices)
        test_indices = np.array(test_indices)
        rng.shuffle(train_indices)
        rng.shuffle(test_indices)

        return X[train_indices], X[test_indices], y[train_indices], y[test_indices]

    def _random_split(self, X, y):
        """Esegue uno split train/test basato su permutazione casuale."""
        np.random.seed(self.seed)
        indices = np.random.permutation(len(X))
        split_idx = int(len(X) * (1.0 - self.test_ratio))
        train_idx, test_idx = indices[:split_idx], indices[split_idx:]
        return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

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
        """Restituisce il DataLoader per il set di training."""
        return self._make_loader(self.train_dataset, shuffle=True, batch_size=self.train_batch_size)

    def test_dataloader(self):
        """Restituisce il DataLoader per il set di test."""
        return self._make_loader(self.test_dataset, shuffle=False, batch_size=self.test_batch_size)


def wine_quality_dataloaders(
    batch_size=64,
    train_batch_size=None,
    test_batch_size=None,
    test_ratio=0.2,
    seed=42,
    normalize=True,
    stratified=True,
    root=None,
    num_workers=0,
):
    """Crea e restituisce i DataLoader (train, test) per il dataset Wine Quality in classificazione multiclasse."""
    wine = WineQualityData(
        root=root,
        test_ratio=test_ratio,
        seed=seed,
        normalize=normalize,
        stratified=stratified,
        batch_size=batch_size,
        train_batch_size=train_batch_size,
        test_batch_size=test_batch_size,
        num_workers=num_workers,
    )
    wine.setup()
    return wine.train_dataloader(), wine.test_dataloader()
