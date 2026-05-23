"""
fallacious/paranoid.py
Sistema Paranoid: soglia anomalia troppo bassa (0.01).

DIFETTO: SENSOR_NOISE è ±0.05 uniforme. Una soglia di 0.01 è al di sotto
del rumore normale → il sistema scatta su ogni lettura rumorosa, generando
falsi allarmi continui anche in assenza di attacco.
"""

from constants    import ANOMALY_THRESHOLD, TRANSITION, SENSOR_NOISE

PARANOID_THRESHOLD = SENSOR_NOISE * 0.2   # 0.01 — sotto il rumore normale → falsi allarmi
from belief_state import BeliefState


class ParanoidBeliefState(BeliefState):
    """Belief con soglia eccessivamente bassa (0.01 << SENSOR_NOISE=0.05).

    DIFETTO: qualsiasi piccola fluttuazione del sensore viene classificata
    come anomalia. Alta recall, precisione molto bassa.
    """

    def update(self, observation: float, t: int) -> None:
        prior            = self.model.predict(t)
        prediction_error = abs(observation - prior)

        step = {
            "t":                 t,
            "observation":       observation,
            "prior":             prior,
            "prediction_error":  prediction_error,
            "threshold":         PARANOID_THRESHOLD,
            "pre_estimate":      self.estimate,
            "pre_uncertainty":   self.uncertainty,
            "note":              "PARANOID: threshold=0.01 << SENSOR_NOISE=0.05",
        }

        if prediction_error > PARANOID_THRESHOLD:
            self.anomaly     = True
            self.uncertainty = 1.0
            self.estimate    = prior
            step["decision"] = "TRUST_MODEL"
            step["reason"]   = "FALSO ALLARME: rumore sensore > soglia paranoica"
        else:
            self.anomaly     = False
            self.estimate    = observation
            self.uncertainty = max(0.1, 1.0 - abs(self.estimate - TRANSITION) * 2)
            step["decision"] = "TRUST_SENSOR"
            step["reason"]   = "OK"

        step["post_estimate"]    = self.estimate
        step["post_uncertainty"] = self.uncertainty
        step["anomaly"]          = self.anomaly
        self.trace.append(step)
