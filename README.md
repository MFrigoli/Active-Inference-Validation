# Railway Active Inference: Rilevamento Anomalie FDIA

Implementazione Python di un agente **Active Inference** per il rilevamento e la
mitigazione di attacchi **False Data Injection Attack (FDIA)** su uno scambio
ferroviario, basata sul Free Energy Principle (FEP).

---

## Panoramica

Il sistema dimostra come un agente intelligente può:
- **Rilevare anomalie** nei dati sensoriali tramite predictive coding
- **Minimizzare la Free Energy** (EFE) attraverso azioni epistemiche e pragmatiche
- **Rallentare e fermarsi** in modo razionale quando rileva minacce
- **Distinguere** tra output superficialmente corretto e processo decisionale genuino

---

## Architettura del sistema

```
┌─────────────────────────────────────────────────────────────┐
│              RAILWAY ACTIVE INFERENCE SYSTEM                │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Environment:  Switch  (transizione stato scambio)          │
│       ↓                                                     │
│  Sensor:       Sensor  (lettura con possibile FDIA)         │
│       ↓                                                     │
│  Belief:       BeliefState  (predictive coding bayesiano)   │
│       ↓                                                     │
│  Decision:     EFEController  (minimizzazione EFE)          │
│       ↓                                                     │
│  Action:       Train  (maintain / epistemic_slow / stop)    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Struttura dei file

```
.
├── constants.py             Costanti fisiche e parametri globali
├── environment.py           Switch (scambio) e Sensor (sensore + FDIA)
├── generative_model.py      Modello generativo interno P(s_t)
├── belief_state.py          Belief posteriore Q(s_t) via predictive coding
├── controller.py            EFE Controller — selezione policy π*
├── train.py                 Effettore: policy → velocità treno
├── simulation.py            Loop di simulazione e calcolo metriche
│
├── ablation2.py             Ablation 2: 8 combinazioni termini EFE
├── ablation1.py             Ablation 1: 5 varianti componenti architetturali
├── stress_test.py           Stress test: sistemi fallaci
│
├── visualize_ablation.py    Grafici ablation 1 e 2
├── visualize_stress_test.py Grafici stress test
├── plot_style.py            Stile matplotlib condiviso
├── wandb_logger.py          Integrazione opzionale Weights & Biases
│
├── run_thesis.py            Master workflow — esegue tutto in sequenza
│
└── fallacious/              Sistemi fallaci (varianti di BeliefState e Controller)
    ├── paranoid.py          Soglia anomalia troppo bassa → falsi allarmi continui
    ├── rigged_cost.py       Costo maintain gonfiato → rallenta per ragione sbagliata
    ├── over_cautious.py     Incertezza sempre alta → rallenta anche senza attacco
    ├── lucky.py             Timing hardcoded → funziona solo su finestra nota
    ├── no_threshold.py      Soglia disabilitata → anomalia mai rilevata
    ├── no_uncertainty.py    Incertezza fissa a 0.1 → valore epistemico nullo
    └── no_model.py          Modello prevede sempre 0 → FN durante attacco
```

---

## Quick Start

### Prerequisiti

```bash
pip install numpy matplotlib
pip install wandb   # opzionale — logging esperimenti su Weights & Biases
```

### Workflow completo

```bash
python run_thesis.py
```

Esegue in sequenza: ablation2 → ablation1 → stress_test →
visualize_ablation → visualize_stress_test.
I file JSON e PNG vengono organizzati automaticamente in `outputs/`.

### Esperimenti singoli

```bash
python ablation2.py           # 8 combinazioni termini EFE
python ablation1.py           # 5 varianti componenti architetturali
python stress_test.py         # sistemi fallaci
python visualize_ablation.py  # grafici (richiede JSON ablation1/2)
```

---

## Modello teorico

### Formula canonica EFE

```
EFE(π) = −PragmaticValue(π) − EpistemicValue(π)

PragmaticValue(π) = −Hazard(π) − Cost(π)
```

Nel codice `neg_pv = Hazard + Cost` (= −PragmaticValue), quindi:

```
G(π) = neg_pv(π) − epistemic_value(π)
π*   = argmin_π  G(π)
```

### Componenti della EFE

#### 1. Hazard (rischio pragmatico)
```
proximity = max(0,   1 − 2·|E_Q[s_t] − TRANSITION|)
unc_term  = uncertainty · 0.5
Hazard    = min(1.5, proximity + unc_term)
```
Aumenta quando la stima si avvicina al punto di transizione critico
e con l'incertezza sul belief state.

#### 2. Cost (costo operativo)
```
Cost(maintain)       = 0.1   # economico: mantieni velocità
Cost(epistemic_slow) = 0.4   # moderato: rallenta per raccogliere info
Cost(pragmatic_stop) = 0.8   # costoso: fermata completa
```

#### 3. Epistemic Value (valore informativo)
```
EpistemicValue(epistemic_slow) = uncertainty
EpistemicValue(altri)          = 0
```
Solo l'azione `epistemic_slow` genera informazione riducendo
l'entropia posteriore H[Q(s_t)].

### Belief update (predictive coding)

```
prediction_error = |osservazione − prior P(s_t)|

prediction_error > ANOMALY_THRESHOLD  →  TRUST_MODEL
    anomaly = True;  uncertainty = 1.0;  estimate = prior

prediction_error ≤ ANOMALY_THRESHOLD  →  TRUST_SENSOR
    anomaly = False; estimate = observation
    uncertainty = max(0.1,  1 − 2·|estimate − TRANSITION|)
```

---

## Ablation Study 2 — 8 combinazioni EFE

Valuta l'importanza di ogni termine della formula EFE
disabilitandolo indipendentemente.

| Variante | Formula | Descrizione |
|---|---|---|
| `full` | (H+C) − E | ✓ Baseline — sistema completo |
| `no_hazard` | C − E | Senza termine di pericolo/prossimità |
| `no_cost` | H − E | Senza costi operativi |
| `no_epistemic` | H + C | Senza drive epistemico |
| `only_hazard` | H | Solo pericolo |
| `only_cost` | C | Solo costo operativo |
| `only_epistemic` | −E | Solo drive epistemico |
| `none` | 0 | Nessun termine — scelta arbitraria |

**Output:** `outputs/json/ablation2/trace_<variante>.json`

---

## Ablation Study 1 — 5 varianti architetturali

Valuta i componenti del belief state ablando singoli meccanismi.

| Chiave | Componente rimosso | Difetto |
|---|---|---|
| `baseline` | nessuno | sistema corretto |
| `no_epistemic` | termine epistemico EFE | EFE(slow) > EFE(maintain) sempre → maintain vince sempre |
| `no_threshold` | soglia anomalia | belief sempre TRUST_SENSOR → valore epistemico ≈ 0 |
| `no_uncertainty` | incertezza dinamica | uncertainty = 0.1 fissa → valore epistemico troppo basso |
| `no_model` | modello generativo | expected = 0 sempre → prediction_error ≈ 0 → nessuna anomalia |

**Output:** `outputs/json/ablation1/ablation1_<variante>.json`

---

## Stress Test — output corretto, processo sbagliato

Cinque sistemi che rallentano durante l'attacco ma per ragioni sbagliate.
Dimostra perché la valutazione black-box non è sufficiente.

| Sistema | Difetto | Perché sbagliato |
|---|---|---|
| `baseline` | nessuno | riferimento corretto |
| `paranoid` | soglia = 0.01 << SENSOR_NOISE | FP continui — il rumore supera sempre la soglia |
| `rigged_cost` | cost(maintain) = 0.9 | rallenta per costo distorto, non per drive epistemico |
| `over_cautious` | uncertainty = 0.9 fissa | drive epistemico non si spegne mai → rallenta sempre |
| `lucky` | timing hardcoded t=22–28 | fragile — cambia finestra attacco e il sistema è cieco |

**Output:** `outputs/json/stress_test/fallacious_<sistema>.json`

---

## Metriche di rilevamento

Le metriche sono calcolate sulla **finestra di transizione** [20, 30]
(i casi difficili, dove lo stato reale è ambiguo).

```
TP = anomaly=True  AND  attack=True  AND  t ∈ [20, 30]
FP = anomaly=True  AND  attack=False  (qualsiasi t)
FN = anomaly=False AND  attack=True   AND  t ∈ [20, 30]

Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
F1        = 2 · Precision · Recall / (Precision + Recall)
```

Le metriche misurano il **layer di inferenza** (belief state), non
l'azione finale — coerente con l'obiettivo di valutare il processo
decisionale e non solo l'output.

---

## Parametri principali (`constants.py`)

| Costante | Valore | Descrizione |
|---|---|---|
| `TIME_STEPS` | 50 | Durata simulazione |
| `TRANSITION_START` | 20 | Inizio transizione scambio |
| `TRANSITION_END` | 30 | Fine transizione scambio |
| `ATTACK_START` | 22 | Inizio attacco FDIA (default) |
| `ATTACK_END` | 28 | Fine attacco FDIA (default) |
| `SENSOR_NOISE` | ±0.05 | Rumore uniforme sensore |
| `ANOMALY_THRESHOLD` | 0.30 | Soglia prediction error |
| `V_NOMINAL / V_SLOW / V_STOP` | 10 / 4 / 0 | Velocità treno (km/h) |

---

## Interpretazione dello scenario

| Fase | t | Cosa succede |
|---|---|---|
| Stabile A | 0–19 | Switch = 0.0, nessun attacco, agente mantiene v=10 |
| Transizione | 20–30 | Switch = 0.5 (zona critica) |
| Attacco FDIA | 22–28 | Sensore inietta stato opposto — max discrepanza |
| Rilevamento atteso | 22–28 | `anomaly=True` → EFE favorisce `epistemic_slow` o `stop` |
| Recupero | 29–49 | Switch = 1.0, nessun attacco, incertezza cala, v→10 |

---

## Output generati

```
outputs/
├── json/
│   ├── ablation1/      ablation1_baseline.json  …  ablation1_no_model.json
│   ├── ablation2/      trace_full.json  …  trace_none.json
│   │                   ablation_comparison.json
│   └── stress_test/    fallacious_baseline.json  …  fallacious_lucky.json
│                       fallacious_comparison.json
└── graphs/
    ├── ablation1/      ablation1.png
    ├── ablation2/      ablation2.png
    ├── decision/       decision_pathway_<variante>.png  (× 8)
    └── stress_test/    stress_pathway_*.png
                        stress_test_comparison.png
```

---

## Visualizzazioni generate

**`visualize_ablation.py`** produce tre tipi di grafici:

1. **Percorso decisionale a 8 layer** (`decision/`) — per ogni variante ablation2:
   stato reale vs sensore, belief + incertezza, prediction error vs soglia,
   EFE per azione, confronto EFE, velocità risultante.

2. **Griglia comparativa ablation2** (`ablation2/`) — velocità del treno
   per tutte le 8 varianti con metriche TP/FP/Precision.

3. **Griglia ablation1** (`ablation1/`) — Baseline vs varianti architetturali:
   velocità, prediction error & incertezza, valore epistemico, −PragmaticValue.

---

## Dipendenze

```bash
pip install numpy matplotlib
pip install wandb   # opzionale
```

---

## Riferimenti teorici

- Friston, K. (2010). *The free-energy principle: a unified brain theory?* Nature Reviews Neuroscience.
- Parr, T., & Friston, K. J. (2019). *Generalised free energy and active inference.* Biological Cybernetics.
- Da Costa, L., et al. (2020). *Active inference on discrete state-spaces.* Journal of Mathematical Psychology.
