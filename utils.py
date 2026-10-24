def save_loss_graph(loss, model, output_dir, filename="snn_mlp_backprop"):
    """Salva il grafo autograd della loss in formato SVG.

    Da chiamare dopo il calcolo della loss, prima di loss.backward().
    """
    from pathlib import Path
    from torchviz import make_dot

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dot = make_dot(
        loss,
        params=dict(model.named_parameters()),
    )

    path = dot.render(
        filename=filename,
        directory=str(output_dir),
        format="svg",
        cleanup=True,
    )

    print(f"Grafo salvato in: {path}")
    return path