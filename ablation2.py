"""
ablation.py
Ablation study — 8 combinazioni dei termini EFE.

Esegui:  python ablation.py
Output:  trace_<variante>.json  +  ablation_comparison.json
"""

import json

from constants    import ATTACK_START, ATTACK_END
from belief_state import BeliefState
from controller   import EFEController
from simulation   import simulate, compute_metrics
from wandb_logger import log_run


ABLATION_CONFIGS = {
    "full": {
        "hazard": True,  "cost": True,  "epistemic": True,
        "label":       "FULL — Hazard + Cost + Epistemic  (baseline)",
        "description": "Sistema completo, tutti i termini EFE attivi",
    },
    "no_hazard": {
        "hazard": False, "cost": True,  "epistemic": True,
        "label":       "NO_HAZARD — Cost + Epistemic",
        "description": "Ablato: nessun termine di pericolo/prossimità",
    },
    "no_cost": {
        "hazard": True,  "cost": False, "epistemic": True,
        "label":       "NO_COST — Hazard + Epistemic",
        "description": "Ablato: nessun costo operativo",
    },
    "no_epistemic": {
        "hazard": True,  "cost": True,  "epistemic": False,
        "label":       "NO_EPISTEMIC — Hazard + Cost",
        "description": "Ablato: nessun drive epistemico (ricerca di informazioni)",
    },
    "only_hazard": {
        "hazard": True,  "cost": False, "epistemic": False,
        "label":       "ONLY_HAZARD — solo Hazard",
        "description": "Solo termine di pericolo attivo",
    },
    "only_cost": {
        "hazard": False, "cost": True,  "epistemic": False,
        "label":       "ONLY_COST — solo Cost",
        "description": "Solo costo operativo attivo",
    },
    "only_epistemic": {
        "hazard": False, "cost": False, "epistemic": True,
        "label":       "ONLY_EPISTEMIC — solo Epistemic",
        "description": "Solo drive epistemico attivo",
    },
    "none": {
        "hazard": False, "cost": False, "epistemic": False,
        "label":       "NONE — nessun termine  (G = 0 sempre)",
        "description": "Tutti i termini disabilitati; scelta azione arbitraria",
    },
}


def run_ablation_study(attack_start: int = ATTACK_START, attack_end: int = ATTACK_END) -> dict:
    """Esegui tutte le 8 varianti e salva i trace JSON."""
    print("=" * 80)
    print("ABLATION STUDY — 8 VARIANTI EFE")
    print(f"Finestra attacco: t={attack_start}..{attack_end}")
    print("=" * 80)

    all_results = {}
    baseline_f1 = None

    for i, (variant, cfg) in enumerate(ABLATION_CONFIGS.items(), 1):
        print(f"\n[{i}/8] {cfg['label']}")
        print("-" * 80)

        belief     = BeliefState()
        controller = EFEController(
            enable_hazard    = cfg["hazard"],
            enable_cost      = cfg["cost"],
            enable_epistemic = cfg["epistemic"],
        )
        log     = simulate(attack_start, attack_end, belief=belief, controller=controller)
        metrics = compute_metrics(log, baseline_f1=baseline_f1)

        if variant == "full":
            baseline_f1 = metrics["f1"]
            metrics.pop("delta_f1_vs_baseline", None)

        result = {
            "variant":          variant,
            "config":           cfg,
            "metrics":          metrics,
            "simulation_log":   log,
            "belief_trace":     belief.trace,
            "controller_trace": controller.trace,
        }
        all_results[variant] = result

        log_run(
            group="ablation2",
            name=variant,
            config={
                "hazard":    cfg["hazard"],
                "cost":      cfg["cost"],
                "epistemic": cfg["epistemic"],
            },
            metrics=metrics,
            simulation_log=log,
        )

        fname = f"trace_{variant}.json"
        with open(fname, "w") as f:
            json.dump(result, f, indent=2)
        print(
            f"[OK] {fname}"
            f"  |  F1={metrics['f1']:.3f}"
            f"  precision={metrics['precision']:.3f}"
            f"  recall={metrics['recall']:.3f}"
        )

    summary = {
        "description": "Ablation study — 8 combinazioni componenti EFE",
        "variants": {
            k: {"config": v["config"], "metrics": v["metrics"]}
            for k, v in all_results.items()
        },
    }
    with open("ablation_comparison.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 80)
    print("[OK] COMPLETATO")
    for v in ABLATION_CONFIGS:
        print(f"  trace_{v}.json")
    print("  ablation_comparison.json")
    print("=" * 80)

    return all_results


if __name__ == "__main__":
    run_ablation_study()
