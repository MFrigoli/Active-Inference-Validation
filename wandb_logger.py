"""
wandb_logger.py
Weights & Biases logging — unico punto di accesso per tutti i run.
"""

import os
import sys
from pathlib import Path

# Wandb scrive una cartella wandb/ nella cwd che shadowa il pacchetto stesso.
# La reindirizziamo e rimuoviamo temporaneamente la cwd da sys.path all'import.
os.environ.setdefault("WANDB_DIR", str(Path(__file__).parent / "outputs"))

_cwd = os.getcwd()
_removed = False
if _cwd in sys.path:
    sys.path.remove(_cwd)
    _removed = True

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False
finally:
    if _removed:
        sys.path.insert(0, _cwd)

PROJECT = "railway-active-inference"

_SIM_COLUMNS = [
    "t", "switch_real", "switch_observed", "belief_estimate",
    "uncertainty", "anomaly", "velocity", "policy", "attack",
    "G_maintain", "G_slow", "G_stop", "hazard", "cost", "risk",
    "epistemic_value",
]


def log_run(
    group: str,
    name: str,
    config: dict,
    metrics: dict,
    simulation_log: list,
) -> None:
    """Log un singolo run wandb. No-op se wandb non installato."""
    if not _WANDB_AVAILABLE:
        print(f"[wandb] not installed — skip {group}/{name}")
        return

    run = wandb.init(
        project=PROJECT,
        group=group,
        name=name,
        config=config,
        reinit=True,
    )

    run.log({f"metrics/{k}": v for k, v in metrics.items()})

    rows = [[entry.get(col) for col in _SIM_COLUMNS] for entry in simulation_log]
    run.log({"simulation": wandb.Table(columns=_SIM_COLUMNS, data=rows)})

    run.finish()
    print(f"[wandb] logged {group}/{name}")
