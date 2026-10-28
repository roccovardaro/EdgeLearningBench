from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from ucimlrepo import fetch_ucirepo
from paths import DATA_DIR


UCI_ADULT_ID = 2

NUMERICAL_COLS = [
    "age",
    "fnlwgt",
    "education-num",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
]

CATEGORICAL_COLS = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

TARGET_COL = "income"


class AdultData:
    """Dataset UCI Adult (Census Income) per classificazione binaria caricato tramite ucimlrepo.

    Predice se il reddito annuo supera i 50K$ (>50K -> 1, <=50K -> 0).
    Supporta:
    - feature_mode: 'all' (108 feature: 6 numeriche + 102 one-hot) o 'numerical' (6 feature continue).
    - split: 'official' (benchmark UCI originale: 32.561 train, 16.281 test) oppure 'stratified'/'random'.
    - standardizzazione Z-score applicata alle feature numeriche (fit solo sul train set).
    """

    def __init__(
        self,
        root=None,
        split="official",
        test_ratio=0.2,
        seed=42,
        normalize=True,
        feature_mode="all",
        batch_size=64,
        train_batch_size=None,
        test_batch_size=None,
        num_workers=0,
    ):
        self.root = Path(root) if root else (DATA_DIR / "adult")
        self.split = split.lower() if split else "official"
        self.test_ratio = test_ratio
        self.seed = seed
        self.normalize = normalize
        self.feature_mode = feature_mode.lower() if feature_mode else "all"
        self.train_batch_size = train_batch_size or batch_size
        self.test_batch_size = test_batch_size or batch_size
        self.num_workers = num_workers

        self.train_dataset = None
        self.test_dataset = None

    def setup(self):
        """Carica il dataset da ucimlrepo (o cache locale CSV) e prepara i TensorDataset."""
        self.root.mkdir(parents=True, exist_ok=True)
        csv_path = self.root / "adult.csv"

        if csv_path.exists():
            df = pd.read_csv(csv_path)
        else:
            try:
                repo_data = fetch_ucirepo(id=UCI_ADULT_ID)
                df = repo_data.data.original
                df.to_csv(csv_path, index=False)
            except Exception as e:
                raise RuntimeError(
                    f"Impossibile scaricare Adult dataset (id={UCI_ADULT_ID}) tramite ucimlrepo: {e}. "
                    f"Verifica la connessione di rete o posiziona il file CSV in {csv_path}."
                ) from e

        # Pulizia nomi colonne e whitespace nei valori categorici
        df.columns = [c.strip() for c in df.columns]

        # Target binario: <=50K / <=50K. -> 0, >50K / >50K. -> 1
        target_series = (
            df[TARGET_COL]
            .astype(str)
            .str.strip()
            .str.rstrip(".")
        )
        y = (target_series == ">50K").astype(int).to_numpy(dtype=np.int64)

        # Feature numeriche
        for num_col in NUMERICAL_COLS:
            df[num_col] = pd.to_numeric(df[num_col], errors="coerce").fillna(0)
        X_num = df[NUMERICAL_COLS].to_numpy(dtype=np.float32)

        # Feature categoriche (One-Hot Encoding)
        if self.feature_mode == "numerical":
            X = X_num
        else:
            for cat_col in CATEGORICAL_COLS:
                df[cat_col] = df[cat_col].fillna("?").astype(str).str.strip()
            df_cat = pd.get_dummies(df[CATEGORICAL_COLS], drop_first=False, dtype=np.float32)
            X_cat = df_cat.to_numpy(dtype=np.float32)
            X = np.hstack([X_num, X_cat])

        num_feats = len(NUMERICAL_COLS)

        # Train / Test split
        if self.split == "official":
            # Split ufficiale UCI: primi 32.561 campioni train, restanti 16.281 test
            split_idx = 32561
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
        elif self.split == "stratified":
            X_train, X_test, y_train, y_test = self._stratified_split(X, y)
        else:
            X_train, X_test, y_train, y_test = self._random_split(X, y)

        # Standardizzazione Z-score delle sole colonne continue (fit calcolato solo sul train set)
        if self.normalize:
            mean = X_train[:, :num_feats].mean(axis=0, keepdims=True)
            std = X_train[:, :num_feats].std(axis=0, keepdims=True)
            std[std == 0.0] = 1.0

            X_train[:, :num_feats] = (X_train[:, :num_feats] - mean) / std
            X_test[:, :num_feats] = (X_test[:, :num_feats] - mean) / std

        self.train_dataset = TensorDataset(
            torch.from_numpy(X_train.copy()), torch.from_numpy(y_train.copy())
        )
        self.test_dataset = TensorDataset(
            torch.from_numpy(X_test.copy()), torch.from_numpy(y_test.copy())
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
        )

    def train_dataloader(self):
        """Restituisce il DataLoader per il set di training."""
        return self._make_loader(self.train_dataset, shuffle=True, batch_size=self.train_batch_size)

    def test_dataloader(self):
        """Restituisce il DataLoader per il set di test."""
        return self._make_loader(self.test_dataset, shuffle=False, batch_size=self.test_batch_size)


def adult_dataloaders(
    batch_size=64,
    train_batch_size=None,
    test_batch_size=None,
    split="official",
    test_ratio=0.2,
    seed=42,
    normalize=True,
    feature_mode="all",
    root=None,
    num_workers=0,
):
    """Crea e restituisce i DataLoader (train, test) per il dataset Adult Census Income."""
    adult = AdultData(
        root=root,
        split=split,
        test_ratio=test_ratio,
        seed=seed,
        normalize=normalize,
        feature_mode=feature_mode,
        batch_size=batch_size,
        train_batch_size=train_batch_size,
        test_batch_size=test_batch_size,
        num_workers=num_workers,
    )
    adult.setup()
    return adult.train_dataloader(), adult.test_dataloader()

