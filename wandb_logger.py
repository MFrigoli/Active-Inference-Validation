"""
wandb_logger.py
Weights & Biases logging — unico punto di accesso per tutti i run.

Disabilitare dall'esterno (ereditato dai sottoprocessi):
    os.environ["WANDB_MODE"] = "disabled"
"""

import os
from pathlib import Path

# Reindirizza i file di run wandb in outputs/ invece di creare wandb/ nella cwd.
os.environ.setdefault("WANDB_DIR", str(Path(__file__).parent / "outputs"))

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

PROJECT = "railway-active-inference"

_SIM_COLUMNS = [
    "t", "switch_real", "switch_observed", "belief_estimate",
    "uncertainty", "anomaly", "velocity", "policy", "attack",
    "G_maintain", "G_slow", "G_stop", "hazard", "cost", "neg_pv",
    "epistemic_value",
]


def log_run(
    group: str,
    name: str,
    config: dict,
    metrics: dict,
    simulation_log: list,
) -> None:
    if os.environ.get("WANDB_MODE") == "disabled":
        return
    if not _WANDB_AVAILABLE:
        print("=" * 60)
        print("  ATTENZIONE: wandb non trovato.")
        print("  Installa con:  pip install wandb")
        print("  Poi esegui:    wandb login")
        print("=" * 60)
        return

    try:
        run = wandb.init(
            project=PROJECT,
            group=group,
            name=name,
            config=config,
            reinit="finish_previous",
        )
        run.log({f"metrics/{k}": v for k, v in metrics.items()})
        rows = [[entry.get(col) for col in _SIM_COLUMNS] for entry in simulation_log]
        run.log({"simulation": wandb.Table(columns=_SIM_COLUMNS, data=rows)})
        run.finish()
        print(f"[wandb] logged {group}/{name}")
    except Exception as e:
        print(f"[wandb] errore {group}/{name}: {e}")
