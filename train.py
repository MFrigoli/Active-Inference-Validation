"""
train.py
Effettore: mappa la policy alla velocità del treno.
"""

from constants import V_NOMINAL, V_SLOW, V_STOP


class Train:
    """Effettore che mappa policy → velocità."""

    def __init__(self):
        self.velocity: float = V_NOMINAL

    def apply(self, policy: str) -> None:
        if policy == "maintain":
            self.velocity = V_NOMINAL
        elif policy == "epistemic_slow":
            self.velocity = V_SLOW
        elif policy == "pragmatic_stop":
            self.velocity = V_STOP
