"""
constants.py
Costanti fisiche e parametri del sistema.
"""

# Timestep totali della simulazione
TIME_STEPS = 50

# Stati dello scambio ferroviario
STABLE_A         = 0.0   # scambio allineato al binario A
TRANSITION       = 0.5   # scambio in transizione (zona di pericolo)
STABLE_B         = 1.0   # scambio allineato al binario B

# Finestra temporale della transizione meccanica
TRANSITION_START = 20
TRANSITION_END   = 30

# Velocità del treno (km/h)
V_NOMINAL = 10   # velocità di crociera normale
V_SLOW    = 4    # velocità ridotta per raccolta informazioni
V_STOP    = 0    # arresto di emergenza

# Sensore
SENSOR_NOISE      = 0.05   # rumore uniforme ±SENSOR_NOISE su letture normali
ANOMALY_THRESHOLD = 0.30   # soglia prediction_error: TRUST_MODEL vs TRUST_SENSOR

# Finestra di attacco FDIA di default
ATTACK_START = 22
ATTACK_END   = 28

# Seed RNG di default per riproducibilità
DEFAULT_SEED = 42

# Costi operativi per policy — termine OperationalCost nella EFE
# Riflettono il trade-off reale: fermarsi è costoso ma massimamente sicuro
BASE_COSTS = {
    "maintain":       0.1,   # economico: mantieni velocità
    "epistemic_slow": 0.4,   # moderato: rallenta per raccogliere informazioni
    "pragmatic_stop": 0.8,   # costoso: fermata completa, ritardo orario
}
