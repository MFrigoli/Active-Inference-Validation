"""
controller.py
EFE Controller — selezione della policy via EFE(π).

Seleziona le azioni minimizzando la Expected Free Energy EFE(π).

Decomposizione canonica:

    EFE(π) = −PragmaticValue(π) − EpistemicValue(π)

dove:
    PragmaticValue(π) = −Risk(π) − Cost(π)
    Risk              = min(1.5, proximity + 0.5·uncertainty)
    proximity         = max(0,   1 − 2·|E_Q[s_t] − TRANSITION|)
    EpistemicValue    = uncertainty  se π=epistemic_slow, altrimenti 0

Nota: nel codice −PragmaticValue è calcolato direttamente come
    neg_pv = Risk + Cost
in modo che EFE = neg_pv − EpistemicValue.

Policy π ∈ {maintain, epistemic_slow, pragmatic_stop}.
L'agente esegue  π* = argmin_π EFE(π).

Ogni flag disabilita indipendentemente un termine della EFE (ablation study).
"""

from constants import BASE_COSTS, TRANSITION
from belief_state import BeliefState


class EFEController:

    def __init__(
        self,
        enable_risk:      bool = True,
        enable_cost:      bool = True,
        enable_epistemic: bool = True,
    ):
        self.enable_risk      = enable_risk
        self.enable_cost      = enable_cost
        self.enable_epistemic = enable_epistemic
        self.trace: list = []

    def _compute_efe(self, policy: str, belief: BeliefState) -> dict:
        """Calcola il breakdown completo della EFE per una singola policy."""

        # ── Risk (prossimità alla zona di pericolo) ──────────────────────────
        if self.enable_risk:
            proximity = max(0.0, 1.0 - 2.0 * abs(belief.estimate - TRANSITION))
            unc_term  = belief.uncertainty * 0.5
            risk      = min(1.5, proximity + unc_term)
        else:
            proximity = unc_term = risk = 0.0

        # ── Cost (costo operativo) ────────────────────────────────────────────
        cost = BASE_COSTS[policy] if self.enable_cost else 0.0

        # ── −PragmaticValue = Risk + Cost  (termine pragmatico totale) ───────
        neg_pv = risk + cost

        # ── EpistemicValue  (guadagno informativo) ────────────────────────────
        # Rallentare permette di osservare lo scambio più attentamente,
        # riducendo l'entropia posteriore H[Q(s_t)] ≈ uncertainty.
        epistemic_value = (
            belief.uncertainty
            if self.enable_epistemic and policy == "epistemic_slow"
            else 0.0
        )

        # ── Expected Free Energy ──────────────────────────────────────────────
        # EFE(π) = −PragmaticValue(π) − EpistemicValue(π)
        G = neg_pv - epistemic_value

        return {
            "policy": policy,
            "risk": risk,
            "risk_components": {
                "proximity": proximity,
                "unc_term":  unc_term,
            },
            "cost":            cost,
            "neg_pv":          neg_pv,
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
