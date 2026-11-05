import argparse
import subprocess
import sys
import time
from device import get_device
from paradigm import get_paradigm_list

def main():
    parser = argparse.ArgumentParser(description="Esegue tutti i paradigmi in sequenza su un dataset.")
    parser.add_argument(
        "--dataset",
        "-d",
        default="mnist",
        help="Nome del dataset (default: mnist). Opzioni: mnist, cifar10, fashion_mnist, adult, breast_cancer, wine_quality",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Dispositivo di calcolo (cpu, mps, cuda, auto; default: auto)",
    )
    args = parser.parse_args()

    device = str(get_device(args.device))
    print(f"Dataset: {args.dataset} | Dispositivo: {device}")
    results = {}

    for paradigm in get_paradigm_list():
        print(f"\n>>> Esecuzione: {paradigm}")
        cmd = [
            sys.executable,
            "main.py",
            "--dataset", args.dataset,
            "--paradigm", paradigm,
            "--device", device,
        ]

        start = time.time()
        subprocess.run(cmd)
        elapsed = time.time() - start

        results[paradigm] = elapsed
        print(f">>> {paradigm} completato in {elapsed:.2f} s")

    print("\n" + "=" * 45)
    print(f"TEMPI DI ESECUZIONE SU {args.dataset.upper()}")
    print("=" * 45)
    for paradigm, elapsed in results.items():
        print(f"{paradigm:<15} : {elapsed:>7.2f} s ({elapsed / 60:>5.2f} min)")
    print("-" * 45)
    total_time = sum(results.values())
    print(f"{'TOTALE':<15} : {total_time:>7.2f} s ({total_time / 60:>5.2f} min)")
    print("=" * 45)


if __name__ == "__main__":
    main()
