"""
controller.py
EFE Controller — selezione della policy via G(π).
"""

from constants import BASE_COSTS, TRANSITION
from belief_state import BeliefState


class EFEController:
    """Seleziona le azioni minimizzando la Expected Free Energy G(π).

    Decomposizione canonica:

        G(π) = Risk(π) − EpistemicValue(π)

    dove:
        Risk(π)           = Hazard(π) + OperationalCost(π)   [termine pragmatico]
        EpistemicValue(π) ≈ H[Q(s_t)]  se π=epistemic_slow, altrimenti 0

    Hazard codifica la vicinanza alla zona di pericolo (TRANSITION),
    pesata dall'incertezza corrente della belief:
        proximity = max(0,   1 − 2·|E_Q[s_t] − TRANSITION|)
        Hazard    = min(1.5, proximity + 0.5·uncertainty)

    Policy π ∈ {maintain, epistemic_slow, pragmatic_stop}.
    L'agente esegue  π* = argmin_π G(π).

    Ogni flag disabilita indipendentemente un termine della EFE (ablation study).
    """

    def __init__(
        self,
        enable_hazard:    bool = True,
        enable_cost:      bool = True,
        enable_epistemic: bool = True,
    ):
        self.enable_hazard    = enable_hazard
        self.enable_cost      = enable_cost
        self.enable_epistemic = enable_epistemic
        self.trace: list = []

    def _compute_efe(self, policy: str, belief: BeliefState) -> dict:
        """Calcola il breakdown completo della EFE per una singola policy."""

        # ── Hazard ────────────────────────────────────────────────────────────
        if self.enable_hazard:
            proximity = max(0.0, 1.0 - 2.0 * abs(belief.estimate - TRANSITION))
            unc_term  = belief.uncertainty * 0.5
            hazard    = min(1.5, proximity + unc_term)
        else:
            proximity = unc_term = hazard = 0.0

        # ── Costo operativo ───────────────────────────────────────────────────
        cost = BASE_COSTS[policy] if self.enable_cost else 0.0

        # ── Risk = Hazard + Cost  (termine pragmatico) ────────────────────────
        risk = hazard + cost

        # ── EpistemicValue  (guadagno informativo) ────────────────────────────
        # Rallentare permette di osservare lo scambio più attentamente,
        # riducendo l'entropia posteriore H[Q(s_t)] ≈ uncertainty.
        epistemic_value = (
            belief.uncertainty
            if self.enable_epistemic and policy == "epistemic_slow"
            else 0.0
        )

        # ── Expected Free Energy ──────────────────────────────────────────────
        G = risk - epistemic_value

        return {
            "policy": policy,
            "hazard": hazard,
            "hazard_components": {
                "proximity": proximity,
                "unc_term":  unc_term,
            },
            "cost":            cost,
            "risk":            risk,
            "epistemic_value": epistemic_value,
            "G":               G,
            "belief": {
                "estimate":    belief.estimate,
                "uncertainty": belief.uncertainty,
                "anomaly":     belief.anomaly,
            },
        }

    def decide(self, belief: BeliefState, t: int) -> tuple:
        """Restituisce (policy_migliore, breakdown EFE per ogni policy)."""
        policies = ["maintain", "epistemic_slow", "pragmatic_stop"]
        efe      = {p: self._compute_efe(p, belief) for p in policies}

        best     = min(efe, key=lambda p: efe[p]["G"])
        g_sorted = sorted(efe[p]["G"] for p in policies)
        margin   = g_sorted[1] - efe[best]["G"]   # distanza dalla seconda scelta

        self.trace.append({
            "t":        t,
            "G":        {p: efe[p]["G"] for p in policies},
            "chosen":   best,
            "margin":   margin,
            "full_efe": efe,
        })
        return best, efe
