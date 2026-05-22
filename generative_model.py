"""
generative_model.py
Modello generativo interno dell'agente — prior P(s_t).

Codifica la conoscenza a priori sulla meccanica: lo scambio transita
tra TRANSITION_START e TRANSITION_END, poi si stabilizza a STABLE_B.

In termini FEP questo è il prior P(s_t) nel modello generativo P(o, s).
Il metodo predict() restituisce la media del prior E_P[s_t].
"""

from constants import STABLE_A, STABLE_B, TRANSITION, TRANSITION_START, TRANSITION_END


class GenerativeModel:

    def predict(self, t: int) -> float:
        """Restituisce lo stato atteso dello scambio al timestep t."""
        if TRANSITION_START <= t <= TRANSITION_END:
            return TRANSITION
        if t > TRANSITION_END:
            return STABLE_B
        return STABLE_A
