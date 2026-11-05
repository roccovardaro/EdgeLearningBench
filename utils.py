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


def get_info_dataset(args, train_loader, test_loader):
    aug_flag = getattr(args.dataset, "augmentation", False)
    norm_flag = getattr(args.dataset, "normalize", True)
    n_train = len(train_loader.dataset) if hasattr(train_loader, "dataset") else len(train_loader)
    n_test = len(test_loader.dataset) if hasattr(test_loader, "dataset") else len(test_loader)
    n_total = n_train + n_test

    return aug_flag,norm_flag,n_train,n_test,n_total