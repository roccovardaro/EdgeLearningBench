"""
EdgeProfiler: Profiler leggero per benchmark Deep Learning su dispositivi Edge.

Monitora in tempo reale (in background):
- TEMPO: durata complessiva in secondi
- ENERGIA: consumo in Joules, kWh, potenza media in Watt ed emissioni CO2 (via CodeCarbon)
- RAM: memoria RSS, picco di allocazione di memoria di processo (kernel getrusage)
- CPU: utilizzo percentuale del processo
- GPU: memoria allocata (PyTorch CUDA o Apple Silicon MPS)
"""

import csv
import json
import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import psutil

# Rilevamento PyTorch (opzionale per GPU tracking)
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

# Rilevamento CodeCarbon per tracking energetico
try:
    from codecarbon import EmissionsTracker
    CODECARBON_AVAILABLE = True
except ImportError:
    CODECARBON_AVAILABLE = False


def _get_peak_ram_mb() -> float:
    """Restituisce il picco assoluto di RAM (RSS) registrato dal kernel del SO."""
    try:
        import resource
        ru_maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Su macOS ru_maxrss è in byte, su Linux è in kilobyte
        if sys.platform == "darwin":
            return ru_maxrss / (1024.0 * 1024.0)
        return ru_maxrss / 1024.0
    except Exception:
        return 0.0


def _get_gpu_snapshot() -> Dict[str, Any]:
    """Campiona le metriche GPU attuali (CUDA o MPS)."""
    snapshot = {
        "available": False,
        "type": "none",
        "device_name": None,
        "allocated_mb": 0.0,
    }

    if not TORCH_AVAILABLE:
        return snapshot

    if torch.cuda.is_available():
        snapshot["available"] = True
        snapshot["type"] = "cuda"
        snapshot["device_name"] = torch.cuda.get_device_name(0)
        snapshot["allocated_mb"] = torch.cuda.memory_allocated(0) / (1024.0 * 1024.0)
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        snapshot["available"] = True
        snapshot["type"] = "mps"
        snapshot["device_name"] = "Apple Silicon MPS"
        if hasattr(torch.mps, "current_allocated_memory"):
            snapshot["allocated_mb"] = torch.mps.current_allocated_memory() / (1024.0 * 1024.0)

    return snapshot


class EdgeProfiler:
    """Profiler per l'analisi prestazionale di modelli Deep Learning su dispositivi Edge."""

    def __init__(
        self,
        interval_sec: float = 1.0,
        track_energy: bool = True,
        tag: Optional[str] = None,
        output_dir: Optional[Union[str, Path]] = None,
    ):
        self.interval_sec = max(0.1, interval_sec)
        self.track_energy = track_energy
        self.tag = tag or "benchmark"
        self.output_dir = Path(output_dir) if output_dir else Path.cwd() / "results"

        self._process = psutil.Process(os.getpid())
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        self._start_time: float = 0.0
        self._end_time: float = 0.0
        self._peak_kernel_rss: Optional[float] = None

        self.samples: List[Dict[str, Any]] = []
        self._initial_rss_mb: float = 0.0

        # Inizializzazione CodeCarbon
        self._energy_tracker = None
        self._energy_data: Dict[str, Any] = {
            "tracked": False,
            "energy_kwh": None,
            "energy_joules": None,
            "emissions_kg": None,
            "avg_power_watts": None,
        }

        if self.track_energy:
            if CODECARBON_AVAILABLE:
                try:
                    self._energy_tracker = EmissionsTracker(
                        save_to_file=False,
                        log_level="error"
                    )
                except Exception as e:
                    print(f"[EdgeProfiler] Avviso: inizializzazione CodeCarbon fallita ({e}).")
                    self._energy_tracker = None
            else:
                print(
                    "[EdgeProfiler] Avviso: 'codecarbon' non è installato. "
                    "Per monitorare l'energia: pip install codecarbon"
                )

    def _sample(self) -> Dict[str, Any]:
        """Esegue un singolo campionamento delle metriche (RAM, CPU, GPU)."""
        now = time.time()
        elapsed = now - self._start_time if self._start_time > 0 else 0.0

        # CPU
        proc_cpu = self._process.cpu_percent(interval=None)

        # RAM
        mem_info = self._process.memory_info()
        proc_rss_mb = mem_info.rss / (1024.0 * 1024.0)
        proc_vms_mb = mem_info.vms / (1024.0 * 1024.0)

        # GPU
        gpu_info = _get_gpu_snapshot()

        return {
            "timestamp": now,
            "elapsed_sec": round(elapsed, 2),
            "cpu_process_pct": proc_cpu,
            "ram_process_rss_mb": round(proc_rss_mb, 2),
            "ram_process_vms_mb": round(proc_vms_mb, 2),
            "gpu_available": gpu_info["available"],
            "gpu_allocated_mb": round(gpu_info["allocated_mb"], 2),
        }

    def _monitor_loop(self):
        """Loop eseguito in un thread in background per campionare periodicamente."""
        self._process.cpu_percent(interval=None)

        while self._running:
            time.sleep(self.interval_sec)
            if not self._running:
                break
            try:
                sample = self._sample()
                with self._lock:
                    self.samples.append(sample)
            except Exception:
                pass

    def start(self):
        """Avvia il monitoraggio in background e il tracker energetico."""
        if self._running:
            return

        with self._lock:
            self.samples.clear()
            self._start_time = time.time()
            self._end_time = 0.0
            self._peak_kernel_rss = None
            self._initial_rss_mb = self._process.memory_info().rss / (1024.0 * 1024.0)
            self._running = True

        if self._energy_tracker:
            try:
                self._energy_tracker.start()
            except Exception as e:
                print(f"[EdgeProfiler] Errore nell'avvio di CodeCarbon: {e}")

        # Campione iniziale immediato
        initial_sample = self._sample()
        with self._lock:
            self.samples.append(initial_sample)

        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()

    def stop(self) -> Dict[str, Any]:
        """Interrompe il monitoraggio e congela il riepilogo finale."""
        if not self._running:
            return self.summary()

        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None

        self._end_time = time.time()
        self._peak_kernel_rss = _get_peak_ram_mb()

        # Campione finale
        final_sample = self._sample()
        with self._lock:
            self.samples.append(final_sample)

        # Stop CodeCarbon
        if self._energy_tracker:
            try:
                emissions = self._energy_tracker.stop()
                data = getattr(self._energy_tracker, "final_emissions_data", None)
                energy_kwh = getattr(data, "energy_consumed", 0.0) if data else 0.0
                energy_joules = energy_kwh * 3.6e6
                duration = self.duration_sec
                avg_power_watts = (energy_joules / duration) if duration > 0 else 0.0

                self._energy_data = {
                    "tracked": True,
                    "energy_kwh": energy_kwh,
                    "energy_joules": energy_joules,
                    "emissions_kg": emissions,
                    "avg_power_watts": avg_power_watts,
                }
            except Exception as e:
                print(f"[EdgeProfiler] Errore nel fermare CodeCarbon: {e}")

        return self.summary()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    @property
    def duration_sec(self) -> float:
        """Durata complessiva del monitoraggio in secondi."""
        if self._start_time == 0.0:
            return 0.0
        end = self._end_time if self._end_time > 0.0 else time.time()
        return max(0.0, end - self._start_time)

    def summary(self) -> Dict[str, Any]:
        """Calcola e restituisce il report sintetico delle sole metriche: tempo, energia, RAM, CPU, GPU."""
        with self._lock:
            samples = list(self.samples)

        duration = self.duration_sec
        peak_kernel_rss = self._peak_kernel_rss if self._peak_kernel_rss is not None else _get_peak_ram_mb()

        # Aggregazioni RAM
        rss_values = [s["ram_process_rss_mb"] for s in samples] if samples else [0.0]
        max_sampled_rss = max(rss_values)
        mean_rss = sum(rss_values) / len(rss_values)
        final_rss = rss_values[-1] if rss_values else 0.0

        # Aggregazioni CPU
        cpu_proc_values = [s["cpu_process_pct"] for s in samples] if samples else [0.0]
        mean_cpu_proc = sum(cpu_proc_values) / len(cpu_proc_values)
        max_cpu_proc = max(cpu_proc_values)

        # Aggregazioni GPU
        gpu_info = _get_gpu_snapshot()
        gpu_alloc_values = [s["gpu_allocated_mb"] for s in samples if s["gpu_available"]]
        peak_gpu_mb = max(gpu_alloc_values) if gpu_alloc_values else 0.0

        return {
            "tag": self.tag,
            "duration_sec": round(duration, 2),
            "samples_count": len(samples),
            "ram": {
                "peak_process_rss_mb": round(max(peak_kernel_rss, max_sampled_rss), 2),
                "mean_process_rss_mb": round(mean_rss, 2),
                "initial_rss_mb": round(self._initial_rss_mb, 2),
                "final_rss_mb": round(final_rss, 2),
            },
            "cpu": {
                "mean_process_pct": round(mean_cpu_proc, 1),
                "max_process_pct": round(max_cpu_proc, 1),
            },
            "gpu": {
                "available": gpu_info["available"],
                "type": gpu_info["type"],
                "device_name": gpu_info["device_name"],
                "peak_allocated_mb": round(peak_gpu_mb, 2),
            },
            "energy": self._energy_data,
        }

    def print_summary(self):
        """Stampa un report visuale pulito con tempo, RAM, CPU, GPU ed energia."""
        rep = self.summary()
        ram = rep["ram"]
        cpu = rep["cpu"]
        gpu = rep["gpu"]
        energy = rep["energy"]

        print("\n" + "=" * 62)
        print(f"       EDGE PROFILER REPORT: {rep['tag'].upper()} (TRAINING)")
        print("=" * 62)
        print(f"  Durata:             {rep['duration_sec']:.2f} s  ({rep['samples_count']} campioni)")

        # RAM
        print("-" * 62)
        print("  [RAM MEMORY]")
        print(f"  Picco RSS (Peak):   {ram['peak_process_rss_mb']:.2f} MB")
        print(f"  Media Processo:     {ram['mean_process_rss_mb']:.2f} MB")
        print(f"  Inizio -> Fine:     {ram['initial_rss_mb']:.2f} MB -> {ram['final_rss_mb']:.2f} MB")

        # CPU
        print("-" * 62)
        print("  [CPU]")
        print(f"  Carico Medio Proc:  {cpu['mean_process_pct']:.1f}%  (Max: {cpu['max_process_pct']:.1f}%)")

        # GPU
        print("-" * 62)
        print("  [GPU]")
        if gpu["available"]:
            print(f"  Dispositivo:        {gpu['device_name']} ({gpu['type'].upper()})")
            print(f"  Picco Memoria GPU:  {gpu['peak_allocated_mb']:.2f} MB")
        else:
            print("  Dispositivo:        Nessuna GPU rilevata (Esecuzione CPU / Edge)")

        # Energia
        print("-" * 62)
        print("  [ENERGIA & CONSUMI]")
        if energy["tracked"] and energy["energy_joules"] is not None:
            print(f"  Energia Consumata:  {energy['energy_joules']:.2f} J  ({energy['energy_kwh']:.6f} kWh)")
            print(f"  Potenza Media:      {energy['avg_power_watts']:.2f} W")
            if energy["emissions_kg"] is not None:
                print(f"  Emissioni CO2:      {energy['emissions_kg'] * 1000.0:.3f} g CO2eq")
        else:
            print("  Stato:              CodeCarbon non attivo o stima non disponibile.")

        print("=" * 62 + "\n")

    def _resolve_export_path(self, filepath: Optional[Union[str, Path]], suffix: str, ext: str) -> Path:
        """Risolve e crea la cartella di destinazione per il file di export (CSV o JSON)."""
        if filepath is not None:
            path = Path(filepath)
        else:
            now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            target_dir = self.output_dir or (Path.cwd() / "results")
            target_dir = Path(target_dir)
            path = target_dir / f"{self.tag}_{suffix}_{now_str}.{ext}"

        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def export_csv(self, filepath: Optional[Union[str, Path]] = None) -> Path:
        """Esporta la serie temporale dei campioni in formato CSV."""
        target_path = self._resolve_export_path(filepath, "samples", "csv")

        with self._lock:
            samples = list(self.samples)

        if not samples:
            print("[EdgeProfiler] Nessun campione da esportare.")
            return target_path

        fieldnames = list(samples[0].keys())
        with open(target_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(samples)

        print(f"[EdgeProfiler] Campioni CSV esportati in: {target_path}")
        return target_path

    def export_json(self, filepath: Optional[Union[str, Path]] = None) -> Path:
        """Esporta energia e consumi (e riepilogo hardware) in formato JSON."""
        target_path = self._resolve_export_path(filepath, "energy", "json")

        summary = self.summary()
        energy = summary["energy"]
        emissions_kg = energy.get("emissions_kg")

        data = {
            "tag": self.tag,
            "duration_sec": summary["duration_sec"],
            "energy_tracked": energy.get("tracked", False),
            "energy_joules": round(energy["energy_joules"], 4) if energy.get("energy_joules") is not None else None,
            "energy_kwh": round(energy["energy_kwh"], 8) if energy.get("energy_kwh") is not None else None,
            "avg_power_watts": round(energy["avg_power_watts"], 4) if energy.get("avg_power_watts") is not None else None,
            "emissions_kg": emissions_kg,
            "emissions_g_co2eq": round(emissions_kg * 1000.0, 4) if emissions_kg is not None else None,
            "hardware_summary": {
                "ram": summary["ram"],
                "cpu": summary["cpu"],
                "gpu": summary["gpu"],
            },
        }

        with open(target_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

        print(f"[EdgeProfiler] Dati energetici JSON esportati in: {target_path}")
        return target_path
