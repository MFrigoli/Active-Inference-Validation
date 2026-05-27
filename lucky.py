"""
fallacious/lucky.py
Sistema Lucky: timing hardcoded a t=22..28.

DIFETTO: il modello generativo conosce a priori la finestra dell'attacco storico.
Fuori da quella finestra il prior viene sostituito dall'osservazione stessa
(prediction_error = 0), rendendo il sistema cieco a qualsiasi attacco fuori da t=22..28.
"""

from constants        import STABLE_A, STABLE_B, TRANSITION, TRANSITION_END, ANOMALY_THRESHOLD, ATTACK_START, ATTACK_END
from generative_model import GenerativeModel
from belief_state     import BeliefState


class LuckyGenerativeModel(GenerativeModel):
    """Prior hardcoded sulla finestra storica dell'attacco.

    DIFETTO: predict() restituisce TRANSITION solo per t in [22, 28].
    Fuori da quella finestra il prior è STABLE_A, ignorando la fisica reale
    della transizione meccanica [TRANSITION_START, TRANSITION_END].
    """

    def predict(self, t: int) -> float:
        if ATTACK_START <= t <= ATTACK_END:
            return TRANSITION
        if t > TRANSITION_END:
            return STABLE_B
        return STABLE_A


class LuckyBeliefState(BeliefState):
    """Belief accoppiato a LuckyGenerativeModel.

    DIFETTO: fuori dalla finestra hardcoded, il prior viene forzato uguale
    all'osservazione (prediction_error = 0). Il sistema è strutturalmente
    incapace di rilevare attacchi fuori da t=22..28.
    """

    def __init__(self):
        super().__init__(model=LuckyGenerativeModel())

    def update(self, observation: float, t: int) -> None:
        prior     = self.model.predict(t)
        in_window = ATTACK_START <= t <= ATTACK_END

        # Fuori finestra: fede cieca nel sensore → prediction_error = 0
        if not in_window:
            prior = observation

        prediction_error = abs(observation - prior)

        step = {
            "t":                 t,
            "observation":       observation,
            "prior":             prior,
            "prediction_error":  prediction_error,
            "threshold":         ANOMALY_THRESHOLD,
            "pre_estimate":      self.estimate,
            "pre_uncertainty":   self.uncertainty,
            "note": (
                "LUCKY: finestra hardcoded attiva"
                if in_window else
                "LUCKY: prior = observation (cieco fuori finestra)"
            ),
        }

        if prediction_error > ANOMALY_THRESHOLD:
            self.anomaly     = True
            self.uncertainty = 1.0
            self.estimate    = prior
            step["decision"] = "TRUST_MODEL"
            step["reason"]   = "FRAGILE: funziona solo perché il timing è hardcoded"
        else:
            self.anomaly     = False
            self.estimate    = observation
            self.uncertainty = max(0.1, 1.0 - abs(self.estimate - TRANSITION) * 2)
            step["decision"] = "TRUST_SENSOR"
            step["reason"]   = (
                "OK (finestra hardcoded)" if in_window
                else "CIECO: prior forzato = observation, attacco invisibile"
            )

        step["post_estimate"]    = self.estimate
        step["post_uncertainty"] = self.uncertainty
        step["anomaly"]          = self.anomaly
        self.trace.append(step)
