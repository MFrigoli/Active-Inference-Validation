"""
run_thesis.py — Master Workflow
Esegue l'intero sistema e organizza i file generati in outputs/.
"""

import os
import subprocess
import sys
from pathlib import Path


def setup_wandb() -> None:
    """Chiede se usare wandb. Se no, imposta WANDB_MODE=disabled (ereditato dai sottoprocessi)."""
    print("\n" + "-" * 80)
    print("  WEIGHTS & BIASES")
    answer = input("  Collegare wandb per il logging degli esperimenti? (y/n): ").strip().lower()
    if answer != "y":
        os.environ["WANDB_MODE"] = "disabled"
        print("  [wandb] skipped")
    else:
        os.environ["WANDB_MODE"] = "online"
        print("  [wandb] attivo — usa credenziali salvate (wandb login)")


def create_output_dirs():
    for sub in ["ablation1", "ablation2", "stress_test", "decision"]:
        Path(f"outputs/json/{sub}").mkdir(parents=True, exist_ok=True)
        Path(f"outputs/graphs/{sub}").mkdir(parents=True, exist_ok=True)


def organize():
    """Sposta i file JSON generati nelle sottocartelle outputs/json/."""
    moved = 0
    mapping = {
        lambda n: n.startswith("trace_") and n.endswith(".json"):
            Path("outputs/json/ablation2"),
        lambda n: n == "ablation_comparison.json":
            Path("outputs/json/ablation2"),
        lambda n: n.startswith("ablation1_") and n.endswith(".json"):
            Path("outputs/json/ablation1"),
        lambda n: n.startswith("fallacious_") and n.endswith(".json"):
            Path("outputs/json/stress_test"),
    }
    for f in Path(".").glob("*"):
        if not f.is_file() or f.suffix == ".py":
            continue
        for pred, dest_dir in mapping.items():
            if pred(f.name):
                try:
                    f.replace(dest_dir / f.name)
                    moved += 1
                except Exception:
                    pass
                break

    return moved


def run_step(i, total, script):
    print(f"\n{'='*80}")
    print(f"  STEP {i}/{total}: {script}")
    print(f"{'='*80}")

    result = subprocess.run(
        [sys.executable, script],
        cwd=Path(__file__).parent,
    )

    if result.returncode != 0:
        answer = input("\nErrore. Continua comunque? (y/n): ")
        if answer.lower() != "y":
            sys.exit(1)

    moved = organize()
    print(f"\n[OK] {moved} file spostati in outputs/")


def main():
    print("\n" + "=" * 80)
    print("  RAILWAY ACTIVE INFERENCE — COMPLETE WORKFLOW")
    print("=" * 80)

    create_output_dirs()
    setup_wandb()

    steps = [
        "ablation2.py",
        "ablation1.py",
        "stress_test.py",
        "visualize_ablation.py",
        "visualize_stress_test.py",
    ]

    for i, script in enumerate(steps, 1):
        run_step(i, len(steps), script)

    print("\n" + "=" * 80)
    print("  COMPLETATO")
    print("  outputs/json/ablation2/     <- trace_*.json, ablation_comparison.json")
    print("  outputs/json/ablation1/     <- ablation1_*.json")
    print("  outputs/json/stress_test/   <- fallacious_*.json")
    print("  outputs/graphs/decision/    <- decision_pathway_*.png")
    print("  outputs/graphs/ablation1/   <- ablation1.png")
    print("  outputs/graphs/ablation2/   <- ablation2.png")
    print("  outputs/graphs/stress_test/ <- stress_pathway_*.png, stress_test_comparison.png")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
