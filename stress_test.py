"""
stress_test.py  —  folder 23
Sistemi fallaci: output corretto, processo sbagliato.

Ogni sistema raggiunge il comportamento superficialmente corretto
(rallenta durante l'attacco) ma per ragioni sbagliate.
Questo dimostra perché la valutazione black-box non basta:
bisogna ispezionare il processo di ragionamento (white-box).

Esegui:  python stress_test.py
Output:  fallacious_<sistema>.json  +  fallacious_comparison.json

Per dimostrare il fallimento del sistema Lucky su una finestra diversa:
    Cambia attack_start/attack_end in __main__ e riesegui. Sono in linea 70
"""

import json

from constants    import ATTACK_START, ATTACK_END
from belief_state import BeliefState
from controller   import EFEController
from simulation   import simulate, compute_metrics
from wandb_logger import log_run

from fallacious import (
    LuckyBeliefState,
    ParanoidBeliefState,
    OverCautiousBeliefState,
    RiggedEFEController,
)


SYSTEMS = {
    "baseline": {
        "label":       "BASELINE — processo corretto",
        "description": "Active Inference standard con EFE calibrata correttamente",
        "flaw":        "nessuno",
        "belief_cls":  BeliefState,
        "ctrl_cls":    EFEController,
    },
    "paranoid": {
        "label":       "PARANOID — soglia anomalia troppo bassa (0.01)",
        "description": "Il rumore del sensore supera la soglia → falsi allarmi continui",
        "flaw":        "threshold=0.01 << SENSOR_NOISE=0.05; alta FP rate",
        "belief_cls":  ParanoidBeliefState,
        "ctrl_cls":    EFEController,
    },
    "rigged_cost": {
        "label":       "RIGGED_COST — costo maintain gonfiato (0.9 vs 0.1)",
        "description": "Rallenta ma non per drive epistemico genuino",
        "flaw":        "costo distorto; epistemic_slow vince per costo, non per valore informativo",
        "belief_cls":  BeliefState,
        "ctrl_cls":    RiggedEFEController,
    },
    "over_cautious": {
        "label":       "OVER_CAUTIOUS — incertezza sempre 0.9",
        "description": "Non converge mai → rallenta sempre, attacco o no",
        "flaw":        "uncertainty non si abbassa mai; drive epistemico non si spegne mai",
        "belief_cls":  OverCautiousBeliefState,
        "ctrl_cls":    EFEController,
    },
    "lucky": {
        "label":       "LUCKY — timing hardcoded a t=22..28",
        "description": "Funziona solo per questa finestra storica; fragile",
        "flaw":        "prior hardcoded; cambiare la finestra di attacco → sistema cieco",
        "belief_cls":  LuckyBeliefState,
        "ctrl_cls":    EFEController,
    },
}


def run_stress_test(attack_start: int = ATTACK_START, attack_end: int = ATTACK_END) -> dict:
    """Esegui tutti i sistemi e salva i trace JSON."""
    n = len(SYSTEMS)
    print("=" * 80)
    print("STRESS TEST — output corretto, processo sbagliato")
    print(f"Finestra attacco: t={attack_start}..{attack_end}")
    print("=" * 80)

    all_results = {}
    baseline_f1 = None

    for i, (key, cfg) in enumerate(SYSTEMS.items(), 1):
        print(f"\n[{i}/{n}] {cfg['label']}")
        print("-" * 80)

        belief     = cfg["belief_cls"]()
        controller = cfg["ctrl_cls"]()
        log        = simulate(attack_start, attack_end, belief=belief, controller=controller)
        metrics    = compute_metrics(log, baseline_f1=baseline_f1)

        if key == "baseline":
            baseline_f1 = metrics["f1"]
            metrics.pop("delta_f1_vs_baseline", None)

        result = {
            "system":           key,
            "label":            cfg["label"],
            "description":      cfg["description"],
            "flaw":             cfg["flaw"],
            "metrics":          metrics,
            "simulation_log":   log,
            "belief_trace":     belief.trace,
            "controller_trace": controller.trace,
        }
        all_results[key] = result

        log_run(
            group="stress_test",
            name=key,
            config={
                "belief_cls": cfg["belief_cls"].__name__,
                "ctrl_cls":   cfg["ctrl_cls"].__name__,
                "flaw":       cfg["flaw"],
            },
            metrics=metrics,
            simulation_log=log,
        )

        fname = f"fallacious_{key}.json"
        with open(fname, "w") as f:
            json.dump(result, f, indent=2)

        print(
            f"[OK] {fname}"
            f"  |  F1={metrics['f1']:.3f}"
            f"  precision={metrics['precision']:.3f}"
            f"  fp={metrics['fp']}  fn={metrics['fn']}"
        )

    summary = {
        "description":   "Stress test — sistemi fallaci: output corretto, processo sbagliato",
        "attack_window": {"start": attack_start, "end": attack_end},
        "systems": {
            k: {"label": v["label"], "flaw": v["flaw"], "metrics": v["metrics"]}
            for k, v in all_results.items()
        },
    }
    with open("fallacious_comparison.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 80)
    print("[OK] COMPLETATO")
    for k in SYSTEMS:
        print(f"  fallacious_{k}.json")
    print("  fallacious_comparison.json")
    print("=" * 80)

    return all_results


if __name__ == "__main__":
    run_stress_test()
