"""
fallacious/rigged_cost.py
Sistema Rigged-Cost: costo di maintain gonfiato (0.9 vs corretto 0.1).

DIFETTO: il controller sceglie epistemic_slow non perché voglia ridurre
l'incertezza, ma perché maintain è artificialmente troppo costoso.
L'EpistemicValue non gioca alcun ruolo reale nella decisione.
"""

from constants    import TRANSITION, BASE_COSTS

RIGGED_COSTS = {**BASE_COSTS, "maintain": 0.9}   # DIFETTO: maintain 9× il valore corretto
from belief_state import BeliefState
from controller   import EFEController


class RiggedEFEController(EFEController):
    """Controller EFE con costo di maintain gonfiato a 0.9 (corretto: 0.1).

    DIFETTO: il costo distorto forza la scelta di epistemic_slow
    indipendentemente dal valore epistemico reale. Superficie comportamentale
    corretta, ragione sbagliata.
    """

    def _compute_efe(self, policy: str, belief: BeliefState) -> dict:
        if self.enable_hazard:
            proximity = max(0.0, 1.0 - 2.0 * abs(belief.estimate - TRANSITION))
            unc_term  = belief.uncertainty * 0.5
            hazard    = min(1.5, proximity + unc_term)
        else:
            proximity = unc_term = hazard = 0.0

        cost   = RIGGED_COSTS[policy] if self.enable_cost else 0.0
        neg_pv = hazard + cost

        epistemic_value = (
            belief.uncertainty
            if self.enable_epistemic and policy == "epistemic_slow"
            else 0.0
        )

        G = neg_pv - epistemic_value

        return {
            "policy": policy,
            "hazard": hazard,
            "hazard_components": {"proximity": proximity, "unc_term": unc_term},
            "cost":            cost,
            "neg_pv":          neg_pv,
            "epistemic_value": epistemic_value,
            "G":               G,
            "note": (
                "RIGGED: maintain cost=0.9 → scelta forzata, non epistemica"
                if policy == "maintain" else ""
            ),
            "belief": {
                "estimate":    belief.estimate,
                "uncertainty": belief.uncertainty,
                "anomaly":     belief.anomaly,
            },
        }
