"""
ablation1.py
Ablation studio — 5 varianti componenti architetturali.

Esegui:  python ablation1.py
Output:  ablation1_<variant>.json
"""

import json

from constants    import ATTACK_START, ATTACK_END
from belief_state import BeliefState
from controller   import EFEController
from simulation   import simulate, compute_metrics
from wandb_logger import log_run

from fallacious import (
    NoThresholdBeliefState,
    NoUncertaintyBeliefState,
    NoModelBeliefState,
)


ABLATION1_CONFIGS = {
    "baseline": {
        "label":       "BASELINE (sistema completo)",
        "description": "Tutti i componenti attivi — comportamento corretto",
        "flaw":        "nessuno",
        "belief_cls":  BeliefState,
        "epistemic":   True,
    },
    "no_epistemic": {
        "label":       "SENZA VALORE EPISTEMICO",
        "description": "EFE senza termine epistemico — mantieni vince sempre",
        "flaw":        "EFE(rallenta)=−PV+0.4 > EFE(mantieni)=−PV+0.1 sempre",
        "belief_cls":  BeliefState,
        "epistemic":   False,
    },
    "no_threshold": {
        "label":       "SENZA SOGLIA ANOMALIA",
        "description": "Belief sempre TRUST_SENSOR — anomalia mai rilevata",
        "flaw":        "soglia disabilitata; incertezza bassa; valore epistemico≈0",
        "belief_cls":  NoThresholdBeliefState,
        "epistemic":   True,
    },
    "no_uncertainty": {
        "label":       "SENZA INCERTEZZA DINAMICA",
        "description": "Incertezza sempre 0.1 — valore epistemico troppo basso",
        "flaw":        "incertezza=0.1 → epistemico=0.1 → mantieni vince sempre",
        "belief_cls":  NoUncertaintyBeliefState,
        "epistemic":   True,
    },
    "no_model": {
        "label":       "SENZA MODELLO INTERNO",
        "description": "Modello prevede sempre 0 — FP fuori attacco, FN durante",
        "flaw":        "expected=0 → durante attacco errore_predizione≈0 → nessuna anomalia",
        "belief_cls":  NoModelBeliefState,
        "epistemic":   True,
    },
}


def run_ablation1(attack_start: int = ATTACK_START, attack_end: int = ATTACK_END) -> dict:
    """Esegui le 5 varianti architetturali e salva i trace JSON."""
    n = len(ABLATION1_CONFIGS)
    print("=" * 80)
    print("ABLATION 1 — VARIANTI COMPONENTI ARCHITETTURALI")
    print(f"Finestra attacco: t={attack_start}..{attack_end}")
    print("=" * 80)

    all_results = {}
    baseline_f1 = None

    for i, (key, cfg) in enumerate(ABLATION1_CONFIGS.items(), 1):
        print(f"\n[{i}/{n}] {cfg['label']}")
        print("-" * 80)

        belief     = cfg["belief_cls"]()
        controller = EFEController(
            enable_risk      = True,
            enable_cost      = True,
            enable_epistemic = cfg["epistemic"],
        )
        log     = simulate(attack_start, attack_end, belief=belief, controller=controller)
        metrics = compute_metrics(log, baseline_f1=baseline_f1)

        if key == "baseline":
            baseline_f1 = metrics["f1"]
            metrics.pop("delta_f1_vs_baseline", None)

        result = {
            "key":              key,
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
            group="ablation1",
            name=key,
            config={
                "belief_cls": cfg["belief_cls"].__name__,
                "epistemic":  cfg["epistemic"],
            },
            metrics=metrics,
            simulation_log=log,
        )

        fname = f"ablation1_{key}.json"
        with open(fname, "w") as f:
            json.dump(result, f, indent=2)
        print(
            f"[OK] {fname}"
            f"  |  F1={metrics['f1']:.3f}"
            f"  precision={metrics['precision']:.3f}"
            f"  tp={metrics['tp']}  fp={metrics['fp']}"
        )

    print("\n" + "=" * 80)
    print("[OK] COMPLETATO")
    for k in ABLATION1_CONFIGS:
        print(f"  ablation1_{k}.json")
    print("=" * 80)

    return all_results


if __name__ == "__main__":
    run_ablation1()
