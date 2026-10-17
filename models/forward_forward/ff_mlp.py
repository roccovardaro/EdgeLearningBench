"""FF-MLP: Multi-Layer Perceptron con addestramento Forward-Forward.

Implementazione dell'algoritmo Forward-Forward (Hinton, 2022) su un MLP
fully-connected.  Ogni layer viene addestrato in modo greedy senza
backpropagation end-to-end.

Caratteristiche:
- Label overlay nei primi pixel dell'input appiattito.
- Normalizzazione L2 inter-layer per impedire il passaggio diretto
  della goodness al layer successivo.
- Predizione per enumerazione di tutte le classi candidate.
"""

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn import functional as F
from device import get_device
from paths import CHECKPOINT_DIR
from models.metrics import accuracy


# ── Label overlay ────────────────────────────────────────────────────────────

def overlay_y_on_x(x, y, num_classes):
    """
    Appiattisce l'input e scrive la label nei primi *num_classes* pixel.

    x può avere forma [B, C, H, W], [B, H, W] o [B, D].
    Output: [B, D].
    """
    x = x.flatten(1).clone()

    if x.shape[1] < num_classes:
        raise ValueError(
            f"Input dimension = {x.shape[1]}, "
            f"ma servono almeno {num_classes} valori."
        )

    x[:, :num_classes] = 0.0
    label_value = x.max(dim=1).values

    if isinstance(y, int):
        x[:, y] = label_value
    else:
        batch_idx = torch.arange(x.shape[0], device=x.device)
        x[batch_idx, y] = label_value

    return x


def make_negative_labels(y, num_classes):
    """Genera per ogni esempio una label sicuramente diversa da quella corretta."""
    offset = torch.randint(low=1, high=num_classes, size=y.shape, device=y.device)
    return (y + offset) % num_classes


# ── Forward-Forward Layer ────────────────────────────────────────────────────

class FFLayer(nn.Linear):

    def __init__(self, in_features, out_features, threshold=2.0):
        super().__init__(in_features=in_features, out_features=out_features)
        self.relu = nn.ReLU()
        self.threshold = threshold

    def forward(self, x):
        """Normalizzazione L2 → Linear → ReLU.

        La normalizzazione impedisce che il layer successivo sfrutti
        direttamente la norma del vettore (che codifica la goodness).
        """
        x_direction = x / (x.norm(p=2, dim=1, keepdim=True) + 1e-4)
        return self.relu(F.linear(x_direction, self.weight, self.bias))

    def goodness(self, h):
        """Goodness = media dei quadrati delle attivazioni."""
        return h.pow(2).mean(dim=1)


# ── Forward-Forward MLP ─────────────────────────────────────────────────────

class FFNet(nn.Module):

    def __init__(self, input_dim, num_classes, hidden_dims=(500, 500), threshold=2.0):
        super().__init__()
        self.num_classes = num_classes
        self.threshold = threshold

        dims = (input_dim, *hidden_dims)
        self.layers = nn.ModuleList([
            FFLayer(dims[i], dims[i + 1], threshold=threshold)
            for i in range(len(dims) - 1)
        ])

    # ── Training ─────────────────────────────────────────────────────────

    def fit(self, train_loader, epochs_per_layer=100, lr=0.03, device=None, profiler=None):
        device = get_device(device)
        self.to(device)

        for layer_idx, layer in enumerate(self.layers):
            print(f"\nTraining layer {layer_idx + 1}/{len(self.layers)}")
            optimizer = Adam(layer.parameters(), lr=lr)

            for epoch in range(epochs_per_layer):
                total_loss, correct, total = 0.0, 0, 0

                for images, labels in train_loader:
                    images, labels = images.to(device), labels.to(device)
                    bs = labels.shape[0]

                    # Dati positivi e negativi
                    x_pos = overlay_y_on_x(images, labels, self.num_classes)
                    neg_labels = make_negative_labels(labels, self.num_classes)
                    x_neg = overlay_y_on_x(images, neg_labels, self.num_classes)

                    # Attraversa i layer già addestrati (congelati)
                    with torch.no_grad():
                        for prev in self.layers[:layer_idx]:
                            x_pos = prev(x_pos)
                            x_neg = prev(x_neg)

                    # Forward del solo layer corrente
                    h_pos = layer(x_pos)
                    h_neg = layer(x_neg)
                    g_pos = layer.goodness(h_pos)
                    g_neg = layer.goodness(h_neg)

                    # Forward-Forward loss
                    loss = torch.cat([
                        F.softplus(self.threshold - g_pos),
                        F.softplus(g_neg - self.threshold),
                    ]).mean()

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    # Statistiche
                    total_loss += loss.item() * bs
                    correct += (g_pos > g_neg).sum().item()
                    total += bs

                print(
                    f"Epoch {epoch + 1:3d}/{epochs_per_layer} | "
                    f"loss={total_loss / total:.4f} | "
                    f"train_acc={correct / total:.4f}"
                )

                if profiler:
                    profiler.log_epoch(
                        epoch=layer_idx * epochs_per_layer + epoch + 1,
                        loss=total_loss / total,
                        acc=correct / total,
                        layer=layer_idx + 1,
                    )

            # Layer completato: congela
            layer.requires_grad_(False)

        self.eval()

    # ── Score ─────────────────────────────────────────────────────────────

    def score(self, x):
        """Goodness cumulativa su tutti i layer."""
        total = torch.zeros(x.shape[0], device=x.device)
        for layer in self.layers:
            x = layer(x)
            total += layer.goodness(x)
        return total

    # ── Predict ───────────────────────────────────────────────────────────

    @torch.no_grad()
    def predict(self, images):
        """Predice la classe provando tutte le label e scegliendo la migliore."""
        goodness_per_label = []
        for label in range(self.num_classes):
            x = overlay_y_on_x(images, label, self.num_classes)
            goodness_per_label.append(self.score(x).unsqueeze(1))

        return torch.cat(goodness_per_label, dim=1).argmax(dim=1)




# ── Entry point (chiamato da main.py) ────────────────────────────────────────

def ff_mlp_run(args, train_loader, test_loader, profiler=None, device=None):
    """Addestra e valuta un FF-MLP con i parametri da config."""
    epochs_per_layer = args.train.epochs_per_layer
    lr = args.train.lr
    threshold = args.train.threshold

    input_dim = args.model.input_dim
    num_classes = args.model.num_classes
    hidden_dims = tuple(args.model.hidden_dims)

    device = get_device(device)

    model = FFNet(input_dim=input_dim, num_classes=num_classes,
                  hidden_dims=hidden_dims, threshold=threshold)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"FF-MLP layers: {[input_dim] + list(hidden_dims) + [num_classes]}")
    print(f"Threshold: {threshold}")
    print(f"Parametri totali: {total_params:,}")
    model.fit(train_loader, epochs_per_layer=epochs_per_layer, lr=lr, device=device, profiler=profiler)

    train_acc = accuracy(model, train_loader, device=device)
    test_acc = accuracy(model, test_loader, device=device)
    print(f"Train accuracy: {train_acc:.4f}  (error: {1 - train_acc:.4f})")
    print(f"Test  accuracy: {test_acc:.4f}  (error: {1 - test_acc:.4f})")

    if profiler:
        profiler.log_eval(train_acc, test_acc)

    # Salva checkpoint
    checkpoint_path = CHECKPOINT_DIR / args.model.checkpoint_name
    torch.save({
        "model_state_dict": model.state_dict(),
        "model_config": {
            "input_dim": input_dim,
            "num_classes": num_classes,
            "hidden_dims": hidden_dims,
        },
    }, checkpoint_path)
    print(f"Checkpoint salvato in: {checkpoint_path}")

    return model

