"""
Utilizza l'implementazione efficient-kan di Blealtan
(https://github.com/Blealtan/efficient-kan) per costruire una rete KAN
basata su B-spline. Supporta sia la modalità standard (con bottleneck
di proiezione lineare) sia la modalità 'direct' (l'input flatten alimenta
direttamente i layer KAN, senza bottleneck), configurabile via parametro `direct`.
"""

import math
import torch
import torch.nn as nn
from efficient_kan import KAN

from device import get_device
from paths import CHECKPOINT_DIR
from models.metrics import accuracy, regression_metrics
from models.training_setup import get_training_components


# ── Modello ──────────────────────────────────────────────────────────────────

class KANClassifier(nn.Module):
    """Wrapper attorno a efficient-kan.KAN per classificazione di immagini.
    Supporta:
    - direct=True: l'input flatten viene passato direttamente al primo layer KAN.
    - direct=False: applica una proiezione lineare (bottleneck_dim) prima dei layer KAN.
    """

    def __init__(self, input_dim, num_classes, hidden_dims=(64, 64),
                 grid_size=5, spline_order=3, direct=False,
                 bottleneck_dim=256, dropout=0.2, use_input_norm=True):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_dims = list(hidden_dims)
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.direct = direct or (bottleneck_dim is None)
        self.bottleneck_dim = None if self.direct else bottleneck_dim
        self.use_input_norm = use_input_norm

        # ── Pre-processing ──────────────────────────────────────
        self.input_norm = nn.LayerNorm(input_dim) if use_input_norm else nn.Identity()

        if not self.direct:
            # Proiezione lineare: riduce input_dim → bottleneck_dim in modo
            # economico (parametri lineari), così i layer KAN lavorano
            # su uno spazio compatto.
            self.projection = nn.Linear(input_dim, self.bottleneck_dim)
            self.proj_norm = nn.LayerNorm(self.bottleneck_dim)
            nn.init.kaiming_normal_(self.projection.weight, a=math.sqrt(5))
            nn.init.zeros_(self.projection.bias)
            kan_in_dim = self.bottleneck_dim
        else:
            self.projection = None
            self.proj_norm = None
            kan_in_dim = input_dim

        # ── KAN layers ──────────────────────────────────────────
        layers_hidden = [kan_in_dim] + self.hidden_dims + [num_classes]
        self.kan = KAN(
            layers_hidden=layers_hidden,
            grid_size=grid_size,
            spline_order=spline_order,
        )

        # ── Normalizzazione e Dropout tra i layer KAN ───────────
        self.hidden_norms = nn.ModuleList([nn.LayerNorm(h) for h in self.hidden_dims])
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        """Forward pass con flatten automatico [B, C, H, W] → [B, D]."""
        x = x.flatten(1)
        x = self.input_norm(x)

        if not self.direct:
            x = self.projection(x)
            x = self.proj_norm(x)

        x = self.dropout(x)

        # Passaggio sequenziale attraverso i layer KAN (senza residui):
        # Per i soli layer nascosti intermedi applica LayerNorm e Dropout
        for i, layer in enumerate(self.kan.layers):
            x = layer(x)
            if i < len(self.kan.layers) - 1:
                x = self.hidden_norms[i](x)
                x = self.dropout(x)

        return x

    def regularization_loss(self):
        """Proxy per la regularizzazione L1 sugli spline weights di efficient-kan."""
        return self.kan.regularization_loss()


# ── Factory ──────────────────────────────────────────────────────────────────

def build_kan(input_dim, num_classes, hidden_dims=(64, 64),
              grid_size=5, spline_order=3, direct=False,
              bottleneck_dim=256, dropout=0.2, use_input_norm=True):
    """Costruisce un KANClassifier con i parametri specificati."""
    model = KANClassifier(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dims=hidden_dims,
        grid_size=grid_size,
        spline_order=spline_order,
        direct=direct,
        bottleneck_dim=bottleneck_dim,
        dropout=dropout,
        use_input_norm=use_input_norm,
    )

    kan_params = sum(p.numel() for p in model.kan.parameters())
    norm_params = sum(p.numel() for p in model.input_norm.parameters()) + sum(
        p.numel() for p in model.hidden_norms.parameters()
    )
    if not model.direct:
        proj_params = sum(p.numel() for p in model.projection.parameters())
        norm_params += sum(p.numel() for p in model.proj_norm.parameters())
    else:
        proj_params = 0

    total_params = sum(p.numel() for p in model.parameters())

    in_dim = input_dim if model.direct else model.bottleneck_dim
    mode_str = "Direct (senza bottleneck)" if model.direct else f"Bottleneck ({input_dim} → {model.bottleneck_dim})"
    print(f"KAN Mode: {mode_str}")
    print(f"KAN layers: {[in_dim] + list(hidden_dims) + [num_classes]}")
    print(f"Grid size: {grid_size}, Spline order: {spline_order}, Dropout: {dropout}")
    print(f"Parametri — Norm: {norm_params:,}  Proj: {proj_params:,}  "
          f"KAN: {kan_params:,}  Totale: {total_params:,}")
    return model


# ── Entry point (chiamato da main.py) ────────────────────────────────────────

def kan_run(args, train_loader, test_loader, profiler=None, device=None):
    """Addestra e valuta un KAN (direct o con bottleneck)."""
    # Parametri di training
    epochs = args.train.epochs
    reg_lambda = getattr(args.train, "regularization_lambda", 0.0)

    # Parametri del modello
    input_dim = args.model.input_dim
    num_classes = args.model.num_classes
    hidden_dims = tuple(args.model.hidden_dims)
    grid_size = getattr(args.model, "grid_size", 5)
    spline_order = getattr(args.model, "spline_order", 3)
    direct = getattr(args.model, "direct", False)
    bottleneck_dim = getattr(args.model, "bottleneck_dim", 256)
    dropout = getattr(args.model, "dropout", 0.2)
    use_input_norm = getattr(args.model, "use_input_norm", True)

    device = get_device(device)

    # Costruzione e spostamento su device
    model = build_kan(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dims=hidden_dims,
        grid_size=grid_size,
        spline_order=spline_order,
        direct=direct,
        bottleneck_dim=bottleneck_dim,
        dropout=dropout,
        use_input_norm=use_input_norm,
    ).to(device)

    optimizer, criterion = get_training_components(
        model, args, model_type="kan"
    )

    # Verifica se il task è di regressione (num_classes == 1)
    is_regression = (num_classes == 1)

    # ── Training ────────────────────────────────────────────────
    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for images, labels in train_loader:
            images = images.to(device)
            if is_regression:
                labels = labels.to(device=device, dtype=torch.float32).view(-1, 1)
            else:
                labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            if is_regression:
                outputs = outputs.view(-1, 1)

            loss = criterion(outputs, labels)

            # Regularizzazione L1 sugli spline weights (da efficient-kan)
            if reg_lambda > 0:
                loss = loss + reg_lambda * model.regularization_loss()

            loss.backward()

            # Gradient clipping: le B-spline possono generare gradienti
            # molto grandi, specialmente nelle prime epoche.
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            if not is_regression:
                correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

        epoch_loss = running_loss / total
        if is_regression:
            print(
                f"Epoch {epoch + 1:3d}/{epochs} | "
                f"loss(MSE)={epoch_loss:.4f}"
            )
            if profiler:
                profiler.log_epoch(epoch + 1, epoch_loss)
        else:
            train_acc = correct / total
            print(
                f"Epoch {epoch + 1:3d}/{epochs} | "
                f"loss={epoch_loss:.4f} | "
                f"train_acc={train_acc:.4f}"
            )
            if profiler:
                profiler.log_epoch(epoch + 1, epoch_loss, train_acc)

    # ── Valutazione ─────────────────────────────────────────────
    if is_regression:
        train_m = regression_metrics(model, train_loader, device)
        test_m = regression_metrics(model, test_loader, device)
        print(f"Train -> MSE: {train_m['mse']:.4f} | RMSE: {train_m['rmse']:.4f} | MAE: {train_m['mae']:.4f} | R²: {train_m['r2']:.4f}")
        print(f"Test  -> MSE: {test_m['mse']:.4f} | RMSE: {test_m['rmse']:.4f} | MAE: {test_m['mae']:.4f} | R²: {test_m['r2']:.4f}")
        if profiler:
            profiler.log_eval(loss=test_m["mse"], acc=test_m["r2"])
    else:
        train_acc = accuracy(model, train_loader, device)
        test_acc = accuracy(model, test_loader, device)
        print(f"Train accuracy: {train_acc:.4f}  (error: {1 - train_acc:.4f})")
        print(f"Test  accuracy: {test_acc:.4f}  (error: {1 - test_acc:.4f})")
        if profiler:
            profiler.log_eval(train_acc, test_acc)

    # ── Checkpoint ──────────────────────────────────────────────
    checkpoint_name = getattr(args.model, "checkpoint_name", "kan.pt")
    checkpoint_path = CHECKPOINT_DIR / checkpoint_name
    torch.save({
        "model_state_dict": model.state_dict(),
        "model_config": {
            "input_dim": input_dim,
            "num_classes": num_classes,
            "hidden_dims": list(hidden_dims),
            "grid_size": grid_size,
            "spline_order": spline_order,
            "direct": model.direct,
            "bottleneck_dim": model.bottleneck_dim,
            "dropout": dropout,
            "use_input_norm": use_input_norm,
        },
    }, checkpoint_path)
    print(f"Checkpoint salvato in: {checkpoint_path}")

    return model
