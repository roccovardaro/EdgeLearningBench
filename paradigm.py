from models.forward_forward.ff_mlp import ff_mlp_run
from models.forward_forward.ff_resnet import ff_resnet_run
from models.baseline.resnet import resnet_run
from models.baseline.cnn import cnn_run
from models.kan.kan import kan_run
from models.kan.direct_kan import direct_kan_run
from models.kan.cnn_kan import cnn_kan_run
from models.baseline.mlp import mlp_run
from models.spiking_neural_network.snn_mlp import snn_mlp_run
from models.spiking_neural_network.snn_cnn import snn_cnn_run
from models.spiking_neural_network.snn_resnet import snn_resnet_run
from models.spiking_neural_network.snn_stdp import snn_stdp_run
from models.spiking_neural_network.ann_snn_conversion.mlp_snn import mlp_snn_run
from models.spiking_neural_network.ann_snn_conversion.cnn_snn import cnn_snn_run
from models.ltc.ltc import ltc_run
from models.ltc.cnn_ltc import cnn_ltc_run
from models.hebbian.chl import chl_run

_PARADIGMS = {
    "ff_mlp": ff_mlp_run,
    "ff_resnet": ff_resnet_run,
    "resnet": resnet_run,
    "cnn": cnn_run,
    "kan": kan_run,
    "direct_kan": direct_kan_run,
    "cnn_kan": cnn_kan_run,
    "mlp": mlp_run,
    "snn_mlp": snn_mlp_run,
    "snn_cnn": snn_cnn_run,
    #"snn_resnet": snn_resnet_run,
    "snn_stdp": snn_stdp_run,
    "mlp_snn": mlp_snn_run,
    "cnn_snn": cnn_snn_run,
    "ltc": ltc_run,
    "cnn_ltc": cnn_ltc_run,
    "chl": chl_run,
}

def get_paradigm(paradigm_name):
    if paradigm_name not in _PARADIGMS:
        raise ValueError(f"Paradigma {paradigm_name} non trovato")
    return _PARADIGMS.get(paradigm_name)

def get_paradigm_list():
    return list(_PARADIGMS.keys())