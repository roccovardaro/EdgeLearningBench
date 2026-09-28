from argument import get_args
from datasets.dataset import get_dataloaders
from models.forward_forward.ff_mlp import ff_mlp_run
from models.forward_forward.ff_resnet import ff_resnet_run
from models.backpropagation.resnet import resnet_run
from models.backpropagation.kan import kan_run
from models.backpropagation.cnn_kan import cnn_kan_run
from models.backpropagation.mlp import mlp_run

_PARADIGMS = {
    "ff_mlp": ff_mlp_run,
    "ff_resnet": ff_resnet_run,
    "resnet": resnet_run,
    "kan": kan_run,
    "cnn_kan": cnn_kan_run,
    "mlp": mlp_run,
}


def main(args):
    paradigm = args.paradigm
    paradigm_args = getattr(args, paradigm, None)

    if paradigm_args is None:
        raise ValueError(f"Sezione '{paradigm}' mancante nel config.")

    run_fn = _PARADIGMS.get(paradigm)
    if run_fn is None:
        raise ValueError(f"Paradigma '{paradigm}' non riconosciuto. "
                         f"Opzioni: {', '.join(_PARADIGMS)}")

    train_loader, test_loader = get_dataloaders(args)
    run_fn(paradigm_args, train_loader, test_loader)


if __name__ == "__main__":
    main(get_args())
