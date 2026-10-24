"""
Direct KAN: Modello Kolmogorov-Arnold Network che riceve direttamente
il vettore appiattito (flatten) dell'immagine senza alcuno strato lineare
intermedio per ridurre la dimensione (nessun bottleneck).
Utilizza l'implementazione efficient-kan di Blealtan.
"""

import torch
import torch.nn as nn
from efficient_kan import KAN
from device import get_device
from paths import CHECKPOINT_DIR
from models.metrics import accuracy
from models.training_setup import get_training_components


# ── Modello ──────────────────────────────────────────────────────────────────

class DirectKANClassifier(nn.Module):
    """Wrapper attorno a efficient-kan.KAN che riceve direttamente il vettore flatten.
    Il primo layer KAN si collega direttamente all'input di dimensione `input_dim`.
    """

    def __init__(self, input_dim, num_classes, hidden_dims=(64, 64),
                 grid_size=5, spline_order=3, dropout=0.2, use_input_norm=True):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_dims = list(hidden_dims)
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.use_input_norm = use_input_norm

        # ── Pre-processing ──────────────────────────────────────
        # LayerNorm normalizza il vettore flatten mantenendo i valori
        # nell'intervallo ≈ [-1, 1], supporto ideale delle B-spline
        # di efficient-kan, senza modificare la dimensionalità delle feature.
        self.input_norm = nn.LayerNorm(input_dim) if use_input_norm else nn.Identity()

        # ── KAN layers ──────────────────────────────────────────
        # Il primo layer KAN riceve direttamente input_dim feature
        layers_hidden = [input_dim] + self.hidden_dims + [num_classes]
        self.kan = KAN(
            layers_hidden=layers_hidden,
            grid_size=grid_size,
            spline_order=spline_order,
        )

        # ── Normalizzazione e Dropout tra i layer KAN ───────────
        # LayerNorm previene la deriva di scala tra layer KAN consecutivi
        self.hidden_norms = nn.ModuleList([nn.LayerNorm(h) for h in self.hidden_dims])
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        """Forward pass: riceve [B, C, H, W] o [B, D], appiattisce a [B, input_dim]
        e passa direttamente ai layer KAN."""
        x = x.flatten(1)
        x = self.input_norm(x)
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

def build_direct_kan(input_dim, num_classes, hidden_dims=(64, 64),
                     grid_size=5, spline_order=3, dropout=0.2,
                     use_input_norm=True):
    """Costruisce un DirectKANClassifier che riceve direttamente il vettore flatten."""
    model = DirectKANClassifier(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dims=hidden_dims,
        grid_size=grid_size,
        spline_order=spline_order,
        dropout=dropout,
        use_input_norm=use_input_norm,
    )

    kan_params = sum(p.numel() for p in model.kan.parameters())
    norm_params = (
        sum(p.numel() for p in model.input_norm.parameters())
        + sum(p.numel() for p in model.hidden_norms.parameters())
    )
    total_params = sum(p.numel() for p in model.parameters())

    print(f"Direct KAN layers: {[input_dim] + list(hidden_dims) + [num_classes]}")
    print(f"Grid size: {grid_size}, Spline order: {spline_order}, Dropout: {dropout}")
    print(f"Input norm: {use_input_norm} (LayerNorm({input_dim}))")
    print(f"Parametri — Norm: {norm_params:,}  KAN: {kan_params:,}  Totale: {total_params:,}")
    return model


# ── Entry point (chiamato da main.py) ────────────────────────────────────────

def direct_kan_run(args, train_loader, test_loader, profiler=None, device=None):
    """Addestra e valuta un Direct KAN con ricezione diretta del vettore flatten."""
    # Parametri di training
    epochs = args.train.epochs
    reg_lambda = getattr(args.train, "regularization_lambda", 0.0)

    # Parametri del modello
    input_dim = args.model.input_dim
    num_classes = args.model.num_classes
    hidden_dims = tuple(args.model.hidden_dims)
    grid_size = getattr(args.model, "grid_size", 5)
    spline_order = getattr(args.model, "spline_order", 3)
    dropout = getattr(args.model, "dropout", 0.2)
    use_input_norm = getattr(args.model, "use_input_norm", True)

    device = get_device(device)

    # Costruzione e spostamento su device
    model = build_direct_kan(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dims=hidden_dims,
        grid_size=grid_size,
        spline_order=spline_order,
        dropout=dropout,
        use_input_norm=use_input_norm,
    ).to(device)

    optimizer, criterion = get_training_components(
        model, args, model_type="direct_kan"
    )

    # ── Training ────────────────────────────────────────────────
    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Regularizzazione L1 sugli spline weights (da efficient-kan)
            if reg_lambda > 0:
                loss = loss + reg_lambda * model.regularization_loss()

            loss.backward()

            # Gradient clipping: stabilità gradienti B-spline
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

        print(
            f"Epoch {epoch + 1:3d}/{epochs} | "
            f"loss={running_loss / total:.4f} | "
            f"train_acc={correct / total:.4f}"
        )

        if profiler:
            profiler.log_epoch(epoch + 1, running_loss / total, correct / total)

    # ── Valutazione ─────────────────────────────────────────────
    train_acc = accuracy(model, train_loader, device)
    test_acc = accuracy(model, test_loader, device)
    print(f"Train accuracy: {train_acc:.4f}  (error: {1 - train_acc:.4f})")
    print(f"Test  accuracy: {test_acc:.4f}  (error: {1 - test_acc:.4f})")

    if profiler:
        profiler.log_eval(train_acc, test_acc)

    # ── Checkpoint ──────────────────────────────────────────────
    checkpoint_name = getattr(args.model, "checkpoint_name", "direct_kan.pt")
    checkpoint_path = CHECKPOINT_DIR / checkpoint_name
    torch.save({
        "model_state_dict": model.state_dict(),
        "model_config": {
            "input_dim": input_dim,
            "num_classes": num_classes,
            "hidden_dims": list(hidden_dims),
            "grid_size": grid_size,
            "spline_order": spline_order,
            "dropout": dropout,
            "use_input_norm": use_input_norm,
        },
    }, checkpoint_path)
    print(f"Checkpoint salvato in: {checkpoint_path}")

    return model
