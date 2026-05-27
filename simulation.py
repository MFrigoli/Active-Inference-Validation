"""
simulation.py
Loop di simulazione e calcolo delle metriche.
"""

import random

from constants import (
    TIME_STEPS, V_NOMINAL,
    TRANSITION_START, TRANSITION_END,
    ATTACK_START, ATTACK_END, DEFAULT_SEED,
)
from environment     import Switch, Sensor
from generative_model import GenerativeModel
from belief_state    import BeliefState
from controller      import EFEController
from train           import Train


def simulate(
    attack_start: int = ATTACK_START,
    attack_end:   int = ATTACK_END,
    *,
    model:      GenerativeModel = None,
    belief:     BeliefState     = None,
    controller: EFEController   = None,
    seed:       int             = DEFAULT_SEED,
    verbose:    bool            = True,
) -> list:
    """Esegui un episodio di simulazione e restituisce il log passo per passo.

    Args:
        attack_start, attack_end:
            Finestra di attacco inclusiva.
            Condizione stretta: attack_start >= attack_end → nessun attacco
            (utile per il preset "zero attacchi").
        model / belief / controller:
            Inietta istanze personalizzate per testare sistemi fallaci.
            Se None vengono usate le implementazioni corrette.
        seed: seed RNG per riproducibilità.
        verbose: stampa un riassunto per ogni timestep.

    Returns:
        Lista di dict — uno per timestep, con tutte le quantità osservabili.
    """
    random.seed(seed)

    switch     = Switch()
    sensor     = Sensor()
    belief     = belief     or BeliefState()
    controller = controller or EFEController()
    train      = Train()

    if model is not None:
        belief.model = model

    log = []

    for t in range(TIME_STEPS):
        switch.update(t)

        # Condizione stretta: start == end → nessun attacco
        attack = attack_start < attack_end and attack_start <= t <= attack_end
        obs    = sensor.read(switch.state, attack)

        belief.update(obs, t)
        policy, efe = controller.decide(belief, t)
        train.apply(policy)

        entry = {
            "t":               t,
            "switch_real":     switch.state,
            "switch_observed": obs,
            "belief_estimate": belief.estimate,
            "uncertainty":     belief.uncertainty,
            "anomaly":         int(belief.anomaly),
            "velocity":        train.velocity,
            "policy":          policy,
            "attack":          attack,
            "G_maintain":      efe["maintain"]["G"],
            "G_slow":          efe["epistemic_slow"]["G"],
            "G_stop":          efe["pragmatic_stop"]["G"],
            "risk":            efe[policy]["risk"],
            "cost":            efe[policy]["cost"],
            "neg_pv":          efe[policy]["neg_pv"],
            "epistemic_value": efe[policy]["epistemic_value"],
        }
        log.append(entry)

        if verbose:
            tag  = "[ATTACK]" if attack       else "  SAFE  "
            anom = "[ANOMALY]" if belief.anomaly else ""
            print(
                f"[t={t:02d}] {tag}"
                f"  real={switch.state:.1f}"
                f"  obs={obs:.2f}"
                f"  est={belief.estimate:.2f}"
                f"  unc={belief.uncertainty:.2f}"
                f"  G=[{efe['maintain']['G']:+.2f}"
                f" {efe['epistemic_slow']['G']:+.2f}"
                f" {efe['pragmatic_stop']['G']:+.2f}]"
                f"  -> {policy}  {anom}"
            )

    return log


def compute_metrics(log: list, baseline_f1: float = None) -> dict:
    """Calcola le metriche di rilevamento rispetto alla FINESTRA DI TRANSIZIONE.

    Ground truth: l'anomalia deve scattare quando (a) l'attacco è attivo E
    (b) siamo nella finestra di transizione [TRANSITION_START, TRANSITION_END].
    Fuori da quella finestra lo scambio è stabile e l'iniezione è banalmente
    rilevabile da qualsiasi soglia — contiamo solo i casi difficili.

        TP = anomaly=1  AND  attack=True  AND  t in [20, 30]
        FP = anomaly=1  AND  attack=False  (falso allarme, qualsiasi t)
        FN = anomaly=0  AND  attack=True   AND  t in [20, 30]
    """
    tp = fp = fn = 0
    v_under_attack = []

    for row in log:
        t      = row["t"]
        anom   = row["anomaly"]
        attack = row["attack"]
        in_win = TRANSITION_START <= t <= TRANSITION_END

        if anom and attack and in_win:
            tp += 1
        elif anom and not attack:
            fp += 1
        elif not anom and attack and in_win:
            fn += 1

        if attack:
            v_under_attack.append(row["velocity"])

    precision   = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall      = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1          = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0.0
    )
    avg_v_atk   = sum(v_under_attack) / len(v_under_attack) if v_under_attack else V_NOMINAL
    degradation = (V_NOMINAL - avg_v_atk) / V_NOMINAL * 100

    result = {
        "tp": tp, "fp": fp, "fn": fn,
        "precision":                 round(precision,   4),
        "recall":                    round(recall,      4),
        "f1":                        round(f1,          4),
        "avg_velocity_under_attack": round(avg_v_atk,   2),
        "degradation_pct":           round(degradation, 2),
    }
    if baseline_f1 is not None:
        result["delta_f1_vs_baseline"] = round(f1 - baseline_f1, 4)

    return result
