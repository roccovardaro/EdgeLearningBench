# EdgeLearningBench

**EdgeLearningBench** è un framework di benchmarking sviluppato in **Python** e **PyTorch** nell'ambito della tesi dal titolo *"Esplorazione di Paradigmi Neurali per l'Edge AI: uno studio comparativo"*. Il progetto nasce per confrontare sperimentalmente paradigmi di apprendimento e inferenza convenzionali, neuromorfici, a tempo continuo e locali in scenari di **Edge AI**.

Il progetto integra architetture eterogenee all'interno di una pipeline sperimentale comune, consentendo sia l'addestramento canonico per epoche sia la valutazione in regime **Iso-Compute** tramite il tracciamento e l'arresto a budget prefissati di operazioni sinaptiche equivalenti ($\text{MAC}_{\text{eq}}$).

---

## Paradigmi e Modelli Supportati

I paradigmi registrati in [`paradigm.py`](paradigm.py) coprono cinque famiglie principali:

| Famiglia | Identificativo (`--paradigm`) | Descrizione | Implementazione |
| :--- | :--- | :--- | :--- |
| **Baseline (BP)** | `mlp`, `cnn`, `resnet` | Architetture standard addestrate con *backpropagation* e `AdamW` | [`models/baseline/`](models/baseline/) |
| **Kolmogorov-Arnold Networks** | `kan`, `direct_kan`, `cnn_kan` | Reti basate su funzioni di attivazione B-spline (`efficient-kan`) | [`models/kan/`](models/kan/) |
| **Forward-Forward** | `ff_mlp`, `ff_resnet` | Apprendimento locale greedy layer-by-layer/block-by-block senza BP globale | [`models/forward_forward/`](models/forward_forward/) |
| **Contrastive Hebbian** | `chl` | Rete ricorrente simmetrica con fase libera/bloccata e aggiornamento locale | [`models/hebbian/chl.py`](models/hebbian/chl.py) |
| **Liquid Time-Constant** | `ltc`, `cnn_ltc` | Reti ricorrenti a tempo continuo basate su ODE (`ncps`) | [`models/ltc/`](models/ltc/) |
| **Spiking (BPTT)** | `snn_mlp`, `snn_cnn` | SNN addestrate direttamente con *surrogate gradient* (`SpikingJelly`) | [`models/spiking_neural_network/`](models/spiking_neural_network/) |
| **Spiking (ANN2SNN)** | `mlp_snn`, `cnn_snn` | Conversione post-addestramento ANN $\to$ SNN con calibrazione a *Rate Coding* | [`models/spiking_neural_network/ann_snn_conversion/`](models/spiking_neural_network/ann_snn_conversion/) |
| **Spiking (STDP)** | `snn_stdp` | Apprendimento non supervisionato bio-ispirato con inibizione laterale e voting | [`models/spiking_neural_network/snn_stdp.py`](models/spiking_neural_network/snn_stdp.py) |

> **Nota su Apple Silicon / SpikingJelly:** su macOS con backend `mps`, l'esecuzione dei modelli `SpikingJelly` commuta automaticamente su `cpu` per compatibilità con i moduli neuromorfici.

---

## Dataset Supportati

Le configurazioni e i dataloader sono gestiti tramite [`argument.py`](argument.py), [`configs/`](configs/) e [`datasets/`](datasets/):

* **Visione artificiale:** `mnist`, `fashion_mnist`, `cifar10`
* **Dati tabellari:** `adult`, `breast_cancer`

---

## Profilazione Hardware e Valutazione Iso-Compute

Il modulo [`profiler/`](profiler/) raccoglie metriche di sistema e computazionali durante addestramento e valutazione, esportando i risultati in formato CSV e JSON nella cartella `results/`:

* **`EdgeProfiler`** ([`profiler/edge_profiler.py`](profiler/edge_profiler.py)): monitora durata, utilizzo CPU, memoria RAM (RSS e picco di processo), memoria GPU (`cuda` / `mps`) e consumo energetico stimato tramite `CodeCarbon`.
* **`TrainEvalProfiler`** ([`profiler/train_eval_profiler.py`](profiler/train_eval_profiler.py)): registra perdita e accuratezza per epoca e in fase di test.
* **`SNNProfiler`** ([`profiler/snn_profiler.py`](profiler/snn_profiler.py)): traccia tasso medio di scarica (*firing rate*), sparsità temporale, operazioni sinaptiche (SOP) e stima energetica neuromorfica.
* **`BudgetTracker`** ([`profiler/budget_tracker.py`](profiler/budget_tracker.py)): stima in tempo reale le operazioni sinaptiche teoriche in $\text{MAC}_{\text{eq}}$ ($1\,\text{MAC} = 2\,\text{FLOP}$, $1\,\text{AC} = 0.50\,\text{MAC}$) tenendo conto della specificità di ciascun paradigma e interrompendo il training al raggiungimento della soglia `--mac`.

---

## Installazione

Si consiglia l'utilizzo di un ambiente virtuale Python:

```bash
python3 -m venv .venv
source .venv/bin/activate        # Su Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Le dipendenze principali includono `torch`, `torchvision`, `spikingjelly`, `efficient-kan`, `ncps`, `psutil`, `codecarbon`, `PyYAML`, `ucimlrepo` e `pandas` (vedi [`requirements.txt`](requirements.txt)).

---

## Utilizzo

### Esecuzione di un singolo paradigma

Tramite [`main.py`](main.py) è possibile addestrare e valutare un singolo modello su un dataset specifico:

```bash
# Addestramento standard (budget MAC disattivo)
python main.py --dataset mnist --paradigm mlp --device auto

# Addestramento in regime Iso-Compute con budget fissato (es. 1e11 MAC)
python main.py --dataset cifar10 --paradigm snn_cnn --device auto --mac 1e11

# Conversione ANN -> SNN (richiede il checkpoint della relativa baseline in checkpoint/)
python main.py --dataset mnist --paradigm mlp_snn
```

Principali argomenti da riga di comando (definiti in [`argument.py`](argument.py)):
* `--dataset`: nome del dataset (`mnist`, `fashion_mnist`, `cifar10`, `adult`, `breast_cancer`) o percorso a un file `.yaml`.
* `--paradigm`: nome del modello/paradigma tra quelli elencati sopra.
* `--device`: dispositivo di calcolo (`auto`, `cpu`, `cuda`, `mps`).
* `--mac` / `--max_macs`: budget massimo di operazioni MAC equivalenti (`0` per disattivare l'arresto a budget).
* `--profiler_config`: percorso al file di configurazione del profiler (default: `configs/profiler.yaml`).

### Esecuzione dell'intera suite su un dataset

Lo script [`run_all.py`](run_all.py) avvia in sequenza tutti i paradigmi registrati su un dato dataset e ne riepiloga i tempi di esecuzione:

```bash
python run_all.py --dataset mnist --device auto --mac 0
```

Per eseguire campagne multi-budget automatizzate in ambiente PowerShell sono disponibili gli script dedicati nella directory [`powershell/`](powershell/) (es. `run_mnist.ps1`, `run_cifar10.ps1`, `run_fashion_mnist.ps1`, `run_adult.ps1`, `run_breast_cancer.ps1`).

---

## Struttura del Repository

```text
EdgeLearningBench/
├── main.py                 # Entry point per l'esecuzione di un singolo paradigma
├── run_all.py              # Esecuzione sequenziale di tutti i paradigmi su un dataset
├── paradigm.py             # Registro centrale dei paradigmi supportati
├── argument.py             # Parsing CLI e caricamento configurazioni YAML
├── device.py               # Selezione automatica del dispositivo (CUDA / MPS / CPU)
├── paths.py                # Gestione percorsi (data/, checkpoint/, results/)
├── utils.py                # Utility di ispezione dataset e logging
├── requirements.txt        # Dipendenze Python
├── LICENSE                 # Licenza MIT
├── README.md               # Documentazione del repository
├── configs/                # Configurazioni per dataset e profiler (profiler.yaml)
├── datasets/               # Caricamento e preprocessing dei dataset visivi e tabellari
├── models/                 # Implementazioni delle architetture e regole di apprendimento
│   ├── baseline/           # mlp.py, cnn.py, resnet.py
│   ├── kan/                # kan.py, direct_kan.py, cnn_kan.py
│   ├── forward_forward/    # ff_mlp.py, ff_resnet.py
│   ├── hebbian/            # chl.py
│   ├── ltc/                # ltc.py, cnn_ltc.py
│   └── spiking_neural_network/
│       ├── snn_mlp.py, snn_cnn.py, snn_resnet.py, snn_stdp.py
│       └── ann_snn_conversion/   # mlp_snn.py, cnn_snn.py, costs.py
├── profiler/               # EdgeProfiler, TrainEvalProfiler, SNNProfiler, BudgetTracker
├── powershell/             # Script PowerShell per campagne sperimentali multi-budget
├── checkpoint/             # Pesi salvati dai modelli addestrati
└── results/                # Log CSV/JSON generati dai profiler
```

---

## Licenza

Questo progetto è distribuito sotto licenza **MIT** — vedi il file [LICENSE](LICENSE) per i dettagli.
