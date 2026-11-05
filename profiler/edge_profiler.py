"""
EdgeProfiler: Profiler leggero per benchmark su sistemi embedded ed Edge (Raspberry Pi, CPU, GPU, Apple Silicon).

Monitora in tempo reale (in background):
- CPU: utilizzo percentuale del processo, frequenza, temperatura e throttling (Raspberry Pi)
- RAM: memoria RSS e VMS (processo), picco assoluto (kernel getrusage)
- GPU: memoria allocata (PyTorch CUDA o Apple Silicon MPS)
- ENERGIA: consumo in kWh, Joules, potenza media in Watt ed emissioni CO2 (via CodeCarbon se installato)
"""

import csv
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import psutil

# Rilevamento presenza vcgencmd (Raspberry Pi OS) per evitare chiamate subprocess a vuoto
_HAS_VCGENCMD = shutil.which("vcgencmd") is not None

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
        # Su macOS ru_maxrss è in byte, su Linux (incluso Raspberry Pi) è in kilobyte
        if sys.platform == "darwin":
            return ru_maxrss / (1024.0 * 1024.0)
        return ru_maxrss / 1024.0
    except Exception:
        return 0.0


def _get_rpi_temperature() -> Optional[float]:
    """Legge la temperatura del SoC (funziona nativamente su Raspberry Pi / Linux)."""
    # 1. Linux sysfs (standard Raspberry Pi)
    thermal_path = "/sys/class/thermal/thermal_zone0/temp"
    if os.path.exists(thermal_path):
        try:
            with open(thermal_path, "r", encoding="utf-8") as f:
                return float(f.read().strip()) / 1000.0
        except Exception:
            pass

    # 2. psutil sensors
    try:
        if hasattr(psutil, "sensors_temperatures"):
            temps = psutil.sensors_temperatures()
            if temps:
                for name, entries in temps.items():
                    if entries:
                        return float(entries[0].current)
    except Exception:
        pass

    # 3. vcgencmd (utility ufficiale Raspberry Pi OS)
    if _HAS_VCGENCMD:
        try:
            out = subprocess.check_output(
                ["vcgencmd", "measure_temp"],
                stderr=subprocess.DEVNULL,
                timeout=0.5
            ).decode().strip()
            # Output tipico: temp=48.2'C
            val = out.replace("temp=", "").replace("'C", "")
            return float(val)
        except Exception:
            pass

    return None


def _check_rpi_throttling() -> Optional[Dict[str, Any]]:
    """Controlla lo stato di thermal/voltage throttling su Raspberry Pi."""
    if not _HAS_VCGENCMD:
        return None
    try:
        out = subprocess.check_output(
            ["vcgencmd", "get_throttled"],
            stderr=subprocess.DEVNULL,
            timeout=0.5
        ).decode().strip()
        val = int(out.split("=")[1], 16)
        return {
            "raw_hex": hex(val),
            "under_voltage": bool(val & 0x1),
            "freq_capped": bool(val & 0x2),
            "currently_throttled": bool(val & 0x4),
            "temp_limit_active": bool(val & 0x8),
            "has_throttled_past": bool(val & 0x40000),
        }
    except Exception:
        return None


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
        """
        Args:
            interval_sec: Intervallo di campionamento delle metriche in secondi.
            track_energy: Se True, attiva CodeCarbon per il tracking energetico.
            tag: Nome identificativo dell'esperimento (es. 'snn_mlp_cifar10').
            output_dir: Directory di output per salvataggio automatico di report/CSV.
        """
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
        self._current_phase: str = "main"

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
        """Esegue un singolo campionamento delle metriche di sistema."""
        now = time.time()
        elapsed = now - self._start_time if self._start_time > 0 else 0.0

        # CPU
        proc_cpu = self._process.cpu_percent(interval=None)

        # RAM
        mem_info = self._process.memory_info()
        proc_rss_mb = mem_info.rss / (1024.0 * 1024.0)
        proc_vms_mb = mem_info.vms / (1024.0 * 1024.0)

        # Temperatura (RPi / Linux)
        temp_c = _get_rpi_temperature()

        # GPU
        gpu_info = _get_gpu_snapshot()

        return {
            "timestamp": now,
            "elapsed_sec": round(elapsed, 2),
            "phase": self._current_phase,
            "cpu_process_pct": proc_cpu,
            "ram_process_rss_mb": round(proc_rss_mb, 2),
            "ram_process_vms_mb": round(proc_vms_mb, 2),
            "temp_c": round(temp_c, 1) if temp_c is not None else None,
            "gpu_available": gpu_info["available"],
            "gpu_allocated_mb": round(gpu_info["allocated_mb"], 2),
        }

    def _monitor_loop(self):
        """Loop eseguito in un thread in background per campionare periodicamente."""
        # Primo campionamento per inizializzare il calcolo di cpu_percent
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
        """Interrompe il monitoraggio e calcola il riepilogo finale."""
        if not self._running:
            return self.summary()

        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None

        self._end_time = time.time()

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

    def set_phase(self, phase_name: str):
        """Imposta l'etichetta della fase corrente (es. 'train', 'epoch_3', 'eval')."""
        with self._lock:
            self._current_phase = phase_name

    @contextmanager
    def phase(self, phase_name: str):
        """Context manager per taggare un blocco di esecuzione specifico."""
        prev_phase = self._current_phase
        self.set_phase(phase_name)
        try:
            yield
        finally:
            self.set_phase(prev_phase)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    @property
    def duration_sec(self) -> float:
        """Durata complessiva dell'esperimento in secondi."""
        if self._start_time == 0.0:
            return 0.0
        end = self._end_time if self._end_time > 0.0 else time.time()
        return max(0.0, end - self._start_time)

    def summary(self) -> Dict[str, Any]:
        """Calcola e restituisce il report sintetico delle metriche."""
        with self._lock:
            samples = list(self.samples)

        duration = self.duration_sec
        peak_kernel_rss = _get_peak_ram_mb()

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

        # Aggregazioni Temperatura
        temp_values = [s["temp_c"] for s in samples if s["temp_c"] is not None]
        mean_temp = (sum(temp_values) / len(temp_values)) if temp_values else None
        max_temp = max(temp_values) if temp_values else None
        throttling_info = _check_rpi_throttling()

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
            "temperature": {
                "available": len(temp_values) > 0,
                "mean_c": round(mean_temp, 1) if mean_temp is not None else None,
                "max_c": round(max_temp, 1) if max_temp is not None else None,
                "throttling": throttling_info,
            },
            "energy": self._energy_data,
        }

    def print_summary(self):
        """Stampa un report visuale pulito a console."""
        rep = self.summary()
        ram = rep["ram"]
        cpu = rep["cpu"]
        gpu = rep["gpu"]
        temp = rep["temperature"]
        energy = rep["energy"]

        print("\n" + "=" * 62)
        print(f"       EDGE PROFILER REPORT: {rep['tag'].upper()}")
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

        # Temperatura / Raspberry Pi
        if temp["available"]:
            print("-" * 62)
            print("  [TEMPERATURA & THERMAL THROTTLING]")
            print(f"  Temperatura Media:  {temp['mean_c']} °C  (Max: {temp['max_c']} °C)")
            if temp["throttling"]:
                thr = temp["throttling"]
                warns = []
                if thr.get("currently_throttled"):
                    warns.append("IN CORSO!")
                if thr.get("has_throttled_past"):
                    warns.append("AVVENUTO IN PASSATO")
                status = ", ".join(warns) if warns else "OK (Nessun throttling)"
                print(f"  Stato Throttling:   {status}")

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
                "temperature": summary["temperature"],
            },
        }

        with open(target_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

        print(f"[EdgeProfiler] Dati energetici JSON esportati in: {target_path}")
        return target_path
