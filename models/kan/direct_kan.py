"""
Direct KAN: modulo di retrocompatibilità.
Reindirizza all'implementazione unificata in `models.kan.kan` impostando `direct=True`.
"""

from models.kan.kan import KANClassifier, build_kan, kan_run

# Alias di compatibilità
DirectKANClassifier = KANClassifier


def build_direct_kan(input_dim, num_classes, hidden_dims=(64, 64),
                     grid_size=5, spline_order=3, dropout=0.2,
                     use_input_norm=True):
    """Costruisce un Direct KAN inoltrando a build_kan con direct=True."""
    return build_kan(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dims=hidden_dims,
        grid_size=grid_size,
        spline_order=spline_order,
        direct=True,
        bottleneck_dim=None,
        dropout=dropout,
        use_input_norm=use_input_norm,
    )


def direct_kan_run(args, train_loader, test_loader, profiler=None, device=None):
    """Inoltra l'addestramento e valutazione a kan_run garantendo direct=True."""
    if hasattr(args, "model"):
        args.model.direct = True
    return kan_run(args, train_loader, test_loader, profiler=profiler, device=device)

