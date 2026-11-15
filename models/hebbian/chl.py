"""
CONTRASTIVE HEBBIAN LEARNING (CHL)
Paradigma di apprendimento ricorrente bio-ispirato supervisionato end-to-end (Movellan 1991, O'Reilly 1996).

Caratteristiche:
- Rete ricorrente con pesi simmetrici e feedback top-down (gamma * W^T).
- Fase Libera (Free / Anti-Hebbiana): la rete rilassa la propria dinamica senza supervisione.
- Fase Bloccata (Clamped / Hebbiana): il target y viene bloccato sull'output e il segnale retroagisce.
- Aggiornamento Locale Contrastivo: ΔW = η * ( x_clamped^T x_clamped - x_free^T x_free ).
- Completamente privo di retropropagazione del gradiente (zero autograd, zero computation graph).
"""

from typing import Sequence, Optional, List
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from device import get_device
from paths import CHECKPOINT_DIR
from models.metrics import accuracy


# ── Rete Ricorrente Contrastive Hebbian ───────────────────────────────────────

class CHLNet(nn.Module):
    """
    Rete Contrastive Hebbian Learning (CHL) multi-layer.
    I pesi non richiedono gradiente (`requires_grad=False`) e vengono aggiornati
    esclusivamente tramite la differenza tra le co-attivazioni delle fasi bloccata e libera.
    """

    def __init__(
        self,
        layer_sizes: Sequence[int],
        gamma: float = 0.1,
        lr: float = 0.05,
        t_free: int = 25,
        t_clamped: int = 25,
        tol: Optional[float] = 1e-4,
        checkpoint_name: str = "chl_model.pt",
    ):
        super().__init__()

        if len(layer_sizes) < 2 or any(s <= 0 for s in layer_sizes):
            raise ValueError("Servono almeno due layer con dimensioni positive.")
        if gamma <= 0:
            raise ValueError("gamma deve essere positivo.")

        self.layer_sizes = list(layer_sizes)
        self.gamma = gamma
        self.lr = lr
        self.t_free = t_free
        self.t_clamped = t_clamped
        self.tol = tol
        self.checkpoint_name = checkpoint_name
        self.L = len(layer_sizes)
        self.last_steps: int = 0

        # Matrici dei pesi [In, Out] e bias [Out] senza autograd
        self.W = nn.ParameterList([
            nn.Parameter(torch.randn(i, o) * (1.0 / (i ** 0.5)), requires_grad=False)
            for i, o in zip(layer_sizes[:-1], layer_sizes[1:])
        ])
        self.b = nn.ParameterList([
            nn.Parameter(torch.zeros(o), requires_grad=False)
            for o in layer_sizes[1:]
        ])

    def _prepare_input(self, x: torch.Tensor) -> torch.Tensor:
        """Verifica e appiattisce l'input se necessario."""
        x_tensor = torch.as_tensor(x, device=self.W[0].device, dtype=self.W[0].dtype)
        if x_tensor.ndim > 2:
            x_tensor = x_tensor.flatten(1)
        if x_tensor.ndim != 2 or x_tensor.shape[1] != self.layer_sizes[0]:
            raise ValueError(
                f"x deve avere forma (batch, {self.layer_sizes[0]}), ricevuta {tuple(x_tensor.shape)}."
            )
        if x_tensor.shape[0] == 0:
            raise ValueError("Il batch non può essere vuoto.")
        return x_tensor

    @torch.no_grad()
    def _settle(
        self,
        x0: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        T: Optional[int] = None,
        tol: Optional[float] = None,
    ) -> List[torch.Tensor]:
        """
        Rilassa ricorsivamente lo stato della rete verso l'equilibrio dinamico.

        Args:
            x0: Pattern di input al layer 0 [B, in_dim]
            y: Target desiderato sul layer L-1 [B, out_dim] (None per Fase Libera)
            T: Numero massimo di cicli di assestamento
            tol: Tolleranza per early stopping su stazionarietà delle attivazioni
        """
        steps = T if T is not None else (self.t_clamped if y is not None else self.t_free)
        tolerance = tol if tol is not None else self.tol

        if steps < 1:
            raise ValueError("T deve essere almeno 1.")
        if tolerance is not None and tolerance <= 0:
            raise ValueError("tol deve essere positivo.")

        x0 = self._prepare_input(x0)
        x = [x0] + [x0.new_zeros((x0.shape[0], size))for size in self.layer_sizes[1:]]
        # Es: layer_sizes [784,128,10] ->  [[Input (B,784), [B,128], [B,10]

        clamped = y is not None
        if clamped:
            y_clamped = torch.as_tensor(y, device=x0.device, dtype=x0.dtype)
            if y_clamped.shape != x[-1].shape:
                raise ValueError(
                    f"y deve avere forma {tuple(x[-1].shape)}, ricevuta {tuple(y_clamped.shape)}."
                )
            x[-1] = y_clamped.clone()

        stop = self.L - 1 if clamped else self.L
        steps_done = 0

        for _ in range(steps):
            steps_done += 1
            max_change = x0.new_zeros(())

            # Aggiornamento sequenziale layer-by-layer
            for k in range(1, stop):
                # Feedforward bottom-up: z = x_{k-1} @ W_{k-1} + b_{k-1}
                z = x[k - 1] @ self.W[k - 1] + self.b[k - 1]

                # Feedback ricorrente top-down: gamma * x_{k+1} @ W_k^T
                if k < self.L - 1:
                    z = z + self.gamma * (x[k + 1] @ self.W[k].T)

                new_x = torch.sigmoid(z)

                if tolerance is not None:
                    max_change = torch.maximum(max_change, (new_x - x[k]).abs().max())

                x[k] = new_x

            if tolerance is not None and max_change.item() < tolerance:
                break

        self.last_steps = steps_done
        return x

    def free_phase(self, x0: torch.Tensor, T: Optional[int] = None, tol: Optional[float] = None) -> List[torch.Tensor]:
        """Esegue la Fase Libera (Anti-Hebbiana / Predizione)."""
        return self._settle(x0, y=None, T=T, tol=tol)

    def clamped_phase(self, x0: torch.Tensor, y: torch.Tensor, T: Optional[int] = None, tol: Optional[float] = None) -> List[torch.Tensor]:
        """Esegue la Fase Bloccata (Hebbiana / Supervisionata con feedback target)."""
        return self._settle(x0, y=y, T=T, tol=tol)

    @torch.no_grad()
    def update(self, x_free: List[torch.Tensor], x_clamped: List[torch.Tensor], lr: Optional[float] = None) -> float:
        """
        Esegue l'aggiornamento sinaptico hebbiano contrastivo.

        ΔW_k = η * scale * ( x_clamped[k]^T x_clamped[k+1] - x_free[k]^T x_free[k+1] ) / B
        Δb_k = η * scale * mean( x_clamped[k+1] - x_free[k+1] )
        """
        batch_size = x_free[0].shape[0]
        learning_rate = lr if lr is not None else self.lr
        deltas = []

        for k in range(self.L - 1):
            scale = learning_rate * (self.gamma ** (k - (self.L - 1)))

            delta_W = (
                x_clamped[k].T @ x_clamped[k + 1]
                - x_free[k].T @ x_free[k + 1]
            ) / batch_size

            delta_b = (
                x_clamped[k + 1] - x_free[k + 1]
            ).mean(dim=0)

            self.W[k].add_(scale * delta_W)
            self.b[k].add_(scale * delta_b)
            deltas.append(delta_W.abs().mean().item())

        return sum(deltas) / max(1, len(deltas))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Passaggio forward standard per inferenza: esegue la Fase Libera e restituisce l'output finale."""
        x_free = self.free_phase(x)
        return x_free[-1]

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Restituisce la classe con massima attivazione nell'output della Fase Libera."""
        out = self.forward(x)
        return out.argmax(dim=-1)


# ── Factory ──────────────────────────────────────────────────────────────────

def build_chl(
    input_dim: int,
    num_classes: int,
    hidden_dims: Sequence[int] = (128,),
    gamma: float = 0.1,
    lr: float = 0.05,
    t_free: int = 25,
    t_clamped: int = 25,
    tol: Optional[float] = 1e-4,
    checkpoint_name: str = "chl_model.pt",
    **kwargs,
) -> CHLNet:
    """Costruisce un modello CHLNet e ne stampa la struttura dei parametri."""
    layer_sizes = [input_dim] + list(hidden_dims) + [num_classes]
    model = CHLNet(
        layer_sizes=layer_sizes,
        gamma=gamma,
        lr=lr,
        t_free=t_free,
        t_clamped=t_clamped,
        tol=tol,
        checkpoint_name=checkpoint_name,
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"CHLNet layer_sizes: {layer_sizes} | gamma={gamma} | lr={lr} | T_free={t_free} | T_clamped={t_clamped}")
    print(f"Parametri totali: {total_params:,} (Puro Contrastive Hebbian, zero autograd)")
    return model


# ── Entry point (chiamato da main.py) ────────────────────────────────────────

def chl_run(args, train_loader, test_loader, profiler=None, device=None) -> CHLNet:
    """Addestra e valuta CHLNet con il paradigma Contrastive Hebbian Learning."""
    device = get_device(device)

    # Parametri architettura
    input_dim = getattr(args.model, "input_dim", 784)
    num_classes = getattr(args.model, "num_classes", 10)
    hidden_dims = tuple(getattr(args.model, "hidden_dims", (128,)))
    gamma = getattr(args.model, "gamma", 0.1)
    t_free = getattr(args.model, "t_free", 25)
    t_clamped = getattr(args.model, "t_clamped", 25)
    tol = getattr(args.model, "tol", 1e-4)
    checkpoint_name = getattr(args.model, "checkpoint_name", "chl_model.pt")

    # Parametri training
    epochs = getattr(args.train, "epochs", 15)
    lr = getattr(args.train, "lr", 0.05)

    # Costruzione modello
    model = build_chl(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dims=hidden_dims,
        gamma=gamma,
        lr=lr,
        t_free=t_free,
        t_clamped=t_clamped,
        tol=tol,
        checkpoint_name=checkpoint_name,
    ).to(device)

    print("\n" + "=" * 65)
    print(f"ADDESTRAMENTO CONTRASTIVE HEBBIAN LEARNING {epochs} epoche")
    print("=" * 65)

    budget_tracker = getattr(args, "budget_tracker", getattr(profiler, "budget_tracker", None))
    budget_stop = False

    for epoch in range(epochs):
        t0 = time.time()
        running_loss, correct, total = 0.0, 0, 0
        total_dw = 0.0
        steps = 0

        for images, labels in train_loader:
            x0 = images.to(device)
            if x0.ndim > 2:
                x0 = x0.flatten(1)
            y = F.one_hot(labels.to(device), num_classes=num_classes).float()

            if budget_tracker and budget_tracker.static_fwd_macs == 0:
                budget_tracker.attach_model(model, sample_input=x0[:1], device=device)

            # 1. Fase Libera (Negative Phase / Predizione)
            x_free = model.free_phase(x0)
            free_steps = model.last_steps

            # Calcolo metriche batch
            preds = x_free[-1].argmax(dim=-1) #Predizione
            correct += (preds == labels.to(device)).sum().item()
            total += labels.size(0)

            # Loss MSE locale tra predizione libera e target
            loss = F.mse_loss(x_free[-1], y)
            running_loss += loss.item() * labels.size(0)

            # 2. Fase Bloccata (Positive Phase / Conduzione Target)
            x_clamped = model.clamped_phase(x0, y=y)
            clamped_steps = model.last_steps

            # 3. Aggiornamento Locale Contrastivo Hebbiano
            dw = model.update(x_free, x_clamped, lr=lr)
            total_dw += dw
            steps += 1

            if budget_tracker and budget_tracker.step(
                batch_size=labels.size(0),
                chl_free_steps=free_steps,
                chl_clamped_steps=clamped_steps,
                epoch=epoch + 1,
            ):
                budget_stop = True
                break

        elapsed = time.time() - t0
        epoch_loss = running_loss / max(1, total)
        train_acc = correct / max(1, total)
        avg_dw = total_dw / max(1, steps)

        print(
            f"CHL Epoch {epoch + 1:2d}/{epochs} | "
            f"Loss (MSE): {epoch_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Δw medio: {avg_dw:.6f} | "
            f"Tempo: {elapsed:.2f}s"
        )

        profiler.log_epoch(epoch + 1, loss=epoch_loss, acc=train_acc, time_sec=elapsed)

        if budget_stop:
            break

    # ── Valutazione Finale ───────────────────────────────────────────────
    args.edge_profiler.stop()

    print("\n" + "=" * 65)
    print("VALUTAZIONE FINALE CONTRASTIVE HEBBIAN LEARNING")
    print("=" * 65)

    test_acc = accuracy(model, test_loader, device=device)
    train_acc = accuracy(model, train_loader, device=device)
    print(f"Train accuracy: {train_acc:.4f}  (error: {1 - train_acc:.4f})")
    print(f"Test  accuracy: {test_acc:.4f}  (error: {1 - test_acc:.4f})")

    profiler.log_eval(train_acc, test_acc)

    # Salvataggio checkpoint
    ckpt_path = CHECKPOINT_DIR / checkpoint_name
    torch.save({
        "model_state_dict": model.state_dict(),
        "model_config": {
            "paradigm": "chl",
            "layer_sizes": model.layer_sizes,
            "gamma": gamma,
            "t_free": t_free,
            "t_clamped": t_clamped,
        },
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
    }, ckpt_path)
    print(f"Checkpoint salvato in: {ckpt_path}")

    return model

