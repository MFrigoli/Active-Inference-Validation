"""
fallacious/no_threshold.py
Sistema senza soglia di anomalia.

DIFETTO: il belief non usa soglia → ogni osservazione è sempre assorbita
(TRUST_SENSOR). Anomalia mai rilevata → uncertainty non sale durante
l'attacco → epistemic value non cresce → l'agente non rallenta.
"""

from constants    import ANOMALY_THRESHOLD, TRANSITION
from belief_state import BeliefState


class NoThresholdBeliefState(BeliefState):
    """Belief senza anomaly threshold: sempre TRUST_SENSOR."""

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
            "note":              "NO_THRESHOLD: sempre TRUST_SENSOR",
        }

        self.anomaly     = False
        self.estimate    = observation
        self.uncertainty = max(0.1, 1.0 - abs(self.estimate - TRANSITION) * 2)
        step["decision"] = "TRUST_SENSOR"
        step["reason"]   = "PROBLEMA: threshold disabilitata → anomalia mai rilevata"

        step["post_estimate"]    = self.estimate
        step["post_uncertainty"] = self.uncertainty
        step["anomaly"]          = self.anomaly
        self.trace.append(step)
