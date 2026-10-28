from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from ucimlrepo import fetch_ucirepo
from paths import DATA_DIR


DIAGNOSTIC_UCI_ID = 17
ORIGINAL_UCI_ID = 15

DIAGNOSTIC_FEATURE_COLS = [
    "radius1", "texture1", "perimeter1", "area1", "smoothness1",
    "compactness1", "concavity1", "concave_points1", "symmetry1", "fractal_dimension1",
    "radius2", "texture2", "perimeter2", "area2", "smoothness2",
    "compactness2", "concavity2", "concave_points2", "symmetry2", "fractal_dimension2",
    "radius3", "texture3", "perimeter3", "area3", "smoothness3",
    "compactness3", "concavity3", "concave_points3", "symmetry3", "fractal_dimension3"
]

ORIGINAL_FEATURE_COLS = [
    "Clump_thickness",
    "Uniformity_of_cell_size",
    "Uniformity_of_cell_shape",
    "Marginal_adhesion",
    "Single_epithelial_cell_size",
    "Bare_nuclei",
    "Bland_chromatin",
    "Normal_nucleoli",
    "Mitoses",
]


class BreastCancerData:
    """Dataset UCI Breast Cancer Wisconsin per task di classificazione binaria caricato tramite ucimlrepo.

    Supporta le varianti:
    - 'diagnostic' (default, UCI ID=17, 30 feature continue, 569 campioni):
      Target 'Diagnosis': 'B' (benigno -> 0), 'M' (maligno -> 1).
    - 'original' (UCI ID=15, 9 feature intere 1-10, 699 campioni):
      Target 'Class': 2 (benigno -> 0), 4 (maligno -> 1).

    Include standardizzazione Z-score (fit calcolato solo sul training set)
    e split riproducibile (stratificato di default).
    """

    def __init__(
        self,
        root=None,
        variant="diagnostic",
        test_ratio=0.2,
        seed=42,
        normalize=True,
        stratified=True,
        batch_size=64,
        train_batch_size=None,
        test_batch_size=None,
        num_workers=0,
    ):
        self.root = Path(root) if root else (DATA_DIR / "breast_cancer")
        self.variant = variant.lower() if variant else "diagnostic"
        self.test_ratio = test_ratio
        self.seed = seed
        self.normalize = normalize
        self.stratified = stratified
        self.train_batch_size = train_batch_size or batch_size
        self.test_batch_size = test_batch_size or batch_size
        self.num_workers = num_workers

        self.train_dataset = None
        self.test_dataset = None

    def setup(self):
        """Carica il dataset da ucimlrepo (o cache locale CSV) e prepara i TensorDataset."""
        self.root.mkdir(parents=True, exist_ok=True)

        is_diagnostic = self.variant in ("diagnostic", "wdbc")
        csv_filename = "breast_cancer_diagnostic.csv" if is_diagnostic else "breast_cancer_original.csv"
        uci_id = DIAGNOSTIC_UCI_ID if is_diagnostic else ORIGINAL_UCI_ID
        csv_path = self.root / csv_filename

        if csv_path.exists():
            df = pd.read_csv(csv_path)
        else:
            try:
                repo_data = fetch_ucirepo(id=uci_id)
                df = repo_data.data.original
                df.to_csv(csv_path, index=False)
            except Exception as e:
                raise RuntimeError(
                    f"Impossibile scaricare Breast Cancer Wisconsin (id={uci_id}) tramite ucimlrepo: {e}. "
                    f"Verifica la connessione di rete o posiziona il file CSV in {csv_path}."
                ) from e

        if is_diagnostic:
            # 30 feature continue, target Diagnosis (B -> 0, M -> 1)
            target_candidates = [c for c in df.columns if c.lower() in ("diagnosis", "target", "class")]
            target_col = target_candidates[0] if target_candidates else "Diagnosis"

            id_cols = [c for c in df.columns if c.lower() in ("id", "sample_code_number")]
            feature_cols = [c for c in df.columns if c != target_col and c not in id_cols]

            # Mappatura etichette binarie
            target_series = df[target_col].astype(str).str.strip().str.upper()
            y = target_series.map({"B": 0, "M": 1, "0": 0, "1": 1}).to_numpy(dtype=np.int64)
            X = df[feature_cols].to_numpy(dtype=np.float32)

        else:
            # 9 feature cliniche intere, target Class (2 -> 0, 4 -> 1)
            target_candidates = [c for c in df.columns if c.lower() in ("class", "diagnosis", "target")]
            target_col = target_candidates[0] if target_candidates else "Class"

            id_cols = [c for c in df.columns if "id" in c.lower() or "sample" in c.lower()]
            feature_cols = [c for c in df.columns if c != target_col and c not in id_cols]

            # Imputazione valori mancanti (es. Bare_nuclei con NaN o '?') con la mediana
            for col in feature_cols:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                df[col] = df[col].fillna(df[col].median())

            target_series = df[target_col]
            y = target_series.map({2: 0, 4: 1, "2": 0, "4": 1, 0: 0, 1: 1}).to_numpy(dtype=np.int64)
            X = df[feature_cols].to_numpy(dtype=np.float32)

        # Train / Test split riproducibile
        if self.stratified:
            X_train, X_test, y_train, y_test = self._stratified_split(X, y)
        else:
            X_train, X_test, y_train, y_test = self._random_split(X, y)

        # Standardizzazione Z-score (fit calcolato solo sul training set)
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
        )

    def train_dataloader(self):
        """Restituisce il DataLoader per il set di training."""
        return self._make_loader(self.train_dataset, shuffle=True, batch_size=self.train_batch_size)

    def test_dataloader(self):
        """Restituisce il DataLoader per il set di test."""
        return self._make_loader(self.test_dataset, shuffle=False, batch_size=self.test_batch_size)


def breast_cancer_dataloaders(
    batch_size=64,
    train_batch_size=None,
    test_batch_size=None,
    variant="diagnostic",
    test_ratio=0.2,
    seed=42,
    normalize=True,
    stratified=True,
    root=None,
    num_workers=0,
):
    """Crea e restituisce i DataLoader (train, test) per il dataset Breast Cancer Wisconsin."""
    cancer = BreastCancerData(
        root=root,
        variant=variant,
        test_ratio=test_ratio,
        seed=seed,
        normalize=normalize,
        stratified=stratified,
        batch_size=batch_size,
        train_batch_size=train_batch_size,
        test_batch_size=test_batch_size,
        num_workers=num_workers,
    )
    cancer.setup()
    return cancer.train_dataloader(), cancer.test_dataloader()

