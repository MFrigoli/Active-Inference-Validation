"""
fallacious/no_model.py
Sistema senza modello interno.

DIFETTO: il modello generativo prevede sempre STABLE_A=0 invece di
conoscere la dinamica dello scambio ferroviario. Conseguenze:
  t=0..19:        switch=0, sensor≈0, expected=0  → pred_error≈rumore  (OK)
  t=20..21,29..30: switch=0.5, expected=0          → pred_error≈0.5 > soglia → FP
  t=22..28 (attacco): sensor inietta opposto di switch (=0), expected=0
                      → pred_error≈0 < soglia → FN durante attacco!
  t=31+:          switch=1.0, expected=0            → pred_error≈1.0 > soglia → FP
"""

from constants        import STABLE_A
from generative_model import GenerativeModel
from belief_state     import BeliefState


class NoModelGenerativeModel(GenerativeModel):
    """Modello generativo cieco: prevede sempre STABLE_A=0."""

    def predict(self, t: int) -> float:
        return STABLE_A


class NoModelBeliefState(BeliefState):
    """BeliefState con modello interno disabilitato (prevede sempre 0)."""

    def __init__(self):
        super().__init__(model=NoModelGenerativeModel())
