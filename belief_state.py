"""
belief_state.py
Belief posteriore Q(s_t) — aggiornata via predictive coding.
"""

from constants import ANOMALY_THRESHOLD, STABLE_A, TRANSITION
from generative_model import GenerativeModel


class BeliefState:
    """Posterior Q(s_t) sullo stato dello scambio, aggiornata via predictive coding.

    Regola di aggiornamento (inferenza Bayesiana semplificata):
        prediction_error = |osservazione − prior P(s_t)|

        prediction_error > ANOMALY_THRESHOLD  >>>  TRUST_MODEL
            L'osservazione è troppo lontana dal prior >>> probabile FDIA.
            Il posterior torna alla media del prior; incertezza al massimo.

        prediction_error <= ANOMALY_THRESHOLD  >>>  TRUST_SENSOR
            L'osservazione è coerente col prior >>> assorbi nel belief.
            L'incertezza è alta vicino a TRANSITION, bassa vicino a STABLE_A/B.

    L'incertezza è un proxy scalare per l'entropia posteriore H[Q(s_t)].
    White-box: ogni quantità intermedia viene salvata in self.trace.
    """

    def __init__(self, model: GenerativeModel = None):
        self.model       = model or GenerativeModel()
        self.estimate    = STABLE_A   # media posteriore E_Q[s_t]
        self.uncertainty = 0.1        # proxy per H[Q(s_t)]
        self.anomaly     = False      # True quando si sospetta FDIA
        self.trace: list = []         # log white-box passo per passo

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
        }

        if prediction_error > ANOMALY_THRESHOLD:
            self.anomaly     = True
            self.uncertainty = 1.0      # entropia massima
            self.estimate    = prior    # torna al modello generativo
            step["decision"] = "TRUST_MODEL"
            step["reason"]   = (
                f"prediction_error {prediction_error:.3f} > "
                f"threshold {ANOMALY_THRESHOLD}"
            )
        else:
            self.anomaly     = False
            self.estimate    = observation
            # H[Q] è massima vicino a TRANSITION, minima agli estremi
            self.uncertainty = max(
                0.1,
                1.0 - abs(self.estimate - TRANSITION) * 2
            )
            step["decision"] = "TRUST_SENSOR"
            step["reason"]   = (
                f"prediction_error {prediction_error:.3f} <= "
                f"threshold {ANOMALY_THRESHOLD}"
            )

        step["post_estimate"]    = self.estimate
        step["post_uncertainty"] = self.uncertainty
        step["anomaly"]          = self.anomaly
        self.trace.append(step)
