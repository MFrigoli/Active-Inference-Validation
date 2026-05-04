"""
environment.py
Ambiente fisico: Switch (scambio) e Sensor (sensore con possibile FDIA).
"""

import random

from constants import (
    STABLE_A, STABLE_B, TRANSITION,
    TRANSITION_START, TRANSITION_END,
    SENSOR_NOISE,
)


class Switch:
    """Scambio ferroviario fisico.

    Stati: {STABLE_A=0.0, TRANSITION=0.5, STABLE_B=1.0}
    La transizione è deterministica e guidata dal tempo —
    l'agente non può controllarla, solo osservarla e ragionarci sopra.
    """

    def __init__(self):
        self.state: float = STABLE_A

    def update(self, t: int) -> None:
        if TRANSITION_START <= t <= TRANSITION_END:
            self.state = TRANSITION
        elif t > TRANSITION_END:
            self.state = STABLE_B
        else:
            self.state = STABLE_A


class Sensor:
    """Modello di osservazione P(o_t | s_t).

    Operazione normale:
        Restituisce true_state + rumore uniforme in [-SENSOR_NOISE, +SENSOR_NOISE].

    FDIA (False Data Injection Attack):
        L'attaccante inietta lo stato OPPOSTO per massimizzare la discrepanza:
            true_state < TRANSITION  >>>  inietta STABLE_B (=1.0)
            true_state >= TRANSITION >>>  inietta STABLE_A (=0.0)
        Risultato: prediction_error supera sempre ANOMALY_THRESHOLD,
        rendendo l'attacco sempre rilevabile da un modello generativo calibrato.
    """

    def read(self, true_state: float, attack: bool = False) -> float:
        if attack:
            return STABLE_B if true_state < TRANSITION else STABLE_A
        noise = random.uniform(-SENSOR_NOISE, SENSOR_NOISE)
        return max(0.0, min(1.0, true_state + noise))
