"""
fallacious/no_uncertainty.py
Sistema senza uncertainty dynamics.

DIFETTO: l'incertezza è sempre fissa a 0.1 — il valore minimo.
Epistemic value = uncertainty = 0.1 costante.
EFE(slow) = Risk + 0.4 − 0.1 = Risk + 0.3 > EFE(maintain) = Risk + 0.1
→ maintain vince sempre → velocità sempre 10, TP = 0.
"""

from constants    import ANOMALY_THRESHOLD, TRANSITION
from belief_state import BeliefState


class NoUncertaintyBeliefState(BeliefState):
    """Belief con incertezza sempre fissa al minimo (0.1)."""

    def update(self, observation: float, t: int) -> None:
        prior            = self.model.predict(t)
        prediction_error = abs(observation - prior)

        step = {
            "t":                 t,
            "observation":       observation,
            "prior":             prior,
            "prediction_error":  prediction_error,
            "threshold":         ANOMALY_THRESHOLD,
            "pre_estimate":      self.estimate,
            "pre_uncertainty":   self.uncertainty,
            "note":              "NO_UNCERTAINTY: uncertainty fissa a 0.1",
        }

        if prediction_error > ANOMALY_THRESHOLD:
            self.anomaly     = True
            self.estimate    = prior
            self.uncertainty = 0.1   # DIFETTO: non sale a 1.0
            step["decision"] = "TRUST_MODEL"
            step["reason"]   = "PROBLEMA: anomalia rilevata ma uncertainty resta 0.1"
        else:
            self.anomaly     = False
            self.estimate    = observation
            self.uncertainty = 0.1   # DIFETTO: non calcolata dinamicamente
            step["decision"] = "TRUST_SENSOR"
            step["reason"]   = "PROBLEMA: uncertainty bloccata a 0.1"

        step["post_estimate"]    = self.estimate
        step["post_uncertainty"] = self.uncertainty
        step["anomaly"]          = self.anomaly
        self.trace.append(step)
