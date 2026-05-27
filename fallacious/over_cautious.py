"""
fallacious/over_cautious.py
Sistema Over-Cautious: incertezza sempre alta (0.9).

DIFETTO: nel ramo TRUST_SENSOR l'incertezza viene fissata a 0.9 invece di
essere calcolata dalla distanza tra stima e TRANSITION. L'alta incertezza
gonfia sia il termine Risk (via unc_term) sia l'EpistemicValue, quindi
l'agente rallenta sempre — non per l'attacco, ma perché non riesce mai a
costruire una belief confidente.
"""

from constants    import ANOMALY_THRESHOLD, TRANSITION
from belief_state import BeliefState


class OverCautiousBeliefState(BeliefState):
    """Belief con incertezza inchiodata a 0.9 nel ramo normale.

    DIFETTO: l'agente non distingue mai tra stati certi e incerti.
    L'EpistemicValue rimane elevato anche quando lo stato dello scambio
    è completamente noto, causando rallentamenti non giustificati.
    """

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
            "note":              "OVER_CAUTIOUS: uncertainty inchiodata a 0.9",
        }

        if prediction_error > ANOMALY_THRESHOLD:
            self.anomaly     = True
            self.uncertainty = 1.0
            self.estimate    = prior
            step["decision"] = "TRUST_MODEL"
            step["reason"]   = "Anomalia genuina rilevata"
        else:
            self.anomaly     = False
            self.estimate    = observation
            self.uncertainty = 0.9   # DIFETTO: non converge mai a belief fiduciosa
            step["decision"] = "TRUST_SENSOR"
            step["reason"]   = "PROBLEMA: uncertainty alta senza motivo → rallenta sempre"

        step["post_estimate"]    = self.estimate
        step["post_uncertainty"] = self.uncertainty
        step["anomaly"]          = self.anomaly
        self.trace.append(step)
