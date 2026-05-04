# Railway Active Inference: Rilevamento Anomalie FDIA

## 📋 Panoramica

Questo progetto implementa un sistema di **Active Inference** basato su principi di Bayesian decision-making per il rilevamento e la mitigazione di attacchi **False Data Injection Attack (FDIA)** in sistemi ferroviari. 

Il sistema dimostra come un agente intelligente può:
- **Rilevare anomalie** nei dati sensoriali
- **Minimizzare il Free Energy** (incertezza) attraverso azioni epistemiche
- **Rallentare e fermarsi** in modo intelligente quando rileva minacce
- **Apprendere dalle fallaci** e identificare processi decisionali sbagliati

---

## 🎯 Architettura del Sistema

```
┌─────────────────────────────────────────────────────────────┐
│              RAILWAY ACTIVE INFERENCE SYSTEM                │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Environment:  Switch (state changes)                       │
│  ↓                                                          │
│  Sensor:       SwitchSensor (subject to FDIA attacks)       │
│  ↓                                                          │
│  Belief:       BeliefEstimator (Bayesian state update)      │
│  ↓                                                          │
│  Decision:     EFE Minimization (epistemic + pragmatic)     │
│  ↓                                                          │
│  Action:       Train controller (maintain/slow/stop)        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Componenti principali

1. **BeliefEstimator**: Aggiorna una stima bayesiana dello stato
2. **Controller (Active Inference)**: Minimizza la Free Energy tramite la formula:
   ```
   EFE(π) = Hazard(π) + Cost(π) - EpistemicValue(π)
   ```
3. **Ambiente simulato**: Switch ferroviario che cambia stato nel tempo

---

## 📁 Struttura dei File

### File di simulazione

| File | Scopo |
|------|-------|
| `railway_active_inference_enhanced.py` | **Core**: 8 varianti ablation study della EFE |
| `fallacious_systems.py` | Sistemi "fallaci" che ottengono risultati corretti con processi sbagliati |
| `visualize_analysis.py` | Genera grafici 2D delle simulazioni |
| `visualize_fallacious_pathways.py` | Visualizza percorsi decisionali dei sistemi fallaci |
| `analyze_fallacious.py` | Analisi comparativa dei sistemi fallaci |
| `validate_necessity.py` | Valida che ogni componente della EFE sia necessario |
| `validate_thesis.py` | Verifiche di correttezza del sistema |
| `run_thesis.py` | Master workflow per eseguire tutto |

### Output generati

```
outputs/
├── data/                      # JSON con risultati simulazioni
│   ├── trace_full.json
│   ├── trace_no_hazard.json
│   ├── trace_no_epistemic.json
│   ├── ablation_comparison.json
│   └── ...
├── graphs/                    # Grafici visualizzazioni
│   ├── full_behavior.png
│   ├── ablation_comparison.png
│   └── ...
├── analysis/                  # Testi di analisi
│   └── validation_report.txt
└── fallacious/                # Risultati sistemi fallaci
    ├── fallacious_normal_baseline.json
    ├── fallacious_paranoid_threshold_troppo_basso.json
    └── ...
```

---

## 🚀 Quick Start

### Prerequisiti

```bash
pip install numpy matplotlib scipy
```

### Esecuzione completa (tutti gli step)

```bash
python run_thesis.py
```

Questo esegue automaticamente in sequenza:
1. Simulazione con 8 varianti ablation
2. Visualizzazioni 2D
3. Validazione della tesi
4. Analisi sistemi fallaci
5. Visualizzazione percorsi fallaci
6. Validazione della necessità dei componenti

### Esecuzione singoli moduli

```bash
# Solo simulazioni ablation
python railway_active_inference_enhanced.py

# Solo validazioni
python validate_thesis.py

# Solo sistemi fallaci
python fallacious_systems.py
python analyze_fallacious.py
```

---

## 🧠 Modello Teorico

### Free Energy Principle

Il sistema minimizza la **Free Energy** (EFE) attraverso tre componenti:

#### 1. **Hazard** (Rischio pragmatico)
```
Hazard(π) = Distance(estimate, transition_point) + Uncertainty(belief)
```
- Quantifica il rischio di essere nello stato critico
- Aumenta quando l'agente si avvicina al punto di transizione
- Aumenta con l'incertezza epistemica

#### 2. **Cost** (Costo operativo)
```
Cost(maintain) = 0.1      # Azione più economica
Cost(slow)     = 0.4      # Azione intermedia
Cost(stop)     = 0.8      # Azione più costosa
```
- Rappresenta il costo energetico/operativo
- Bilancia tra efficienza e sicurezza

#### 3. **Epistemic Value** (Valore informativo)
```
EpistemicValue(slow) = Uncertainty(belief)
EpistemicValue(other) = 0
```
- Solo l'azione "epistemic_slow" fornisce informazioni
- Quantifica quanto l'azione riduce l'incertezza

### Formula EFE canonica

```
EFE(π) =      Risk(π)        -   EpistemicValue(π)
         └──────┬─────────┘     └────────┬────────┘
            Pragmatic term        Epistemic term

Risk(π) = Hazard(π) + Cost(π)
```

---

## 🔬 Ablation Study (8 Varianti)

Il sistema valuta l'importanza di ogni componente:

| Variante | Formula | Descrizione |
|----------|---------|-------------|
| `full` | H + C - E | ✅ Baseline - Sistema completo |
| `no_hazard` | C - E | Senza consapevolezza del pericolo |
| `no_cost` | H - E | Senza costi operativi |
| `no_epistemic` | H + C | Senza valore informativo |
| `only_hazard` | H | Solo consapevolezza del pericolo |
| `only_cost` | C | Solo costi operativi |
| `only_epistemic` | -E | Solo valore informativo |
| `none` | 0 | Nessun componente (baseline fallace) |

### Ipotesi sperimentali

- **H1**: La variante `full` ha la migliore performance
- **H2**: Rimuovere qualsiasi componente degrada la qualità
- **H3**: Il `no_epistemic` ha performance peggiore di `full`
- **H4**: `only_hazard` rallenta troppo
- **H5**: `none` fallisce completamente

---

## ⚠️ Sistemi Fallaci: Output Giusto, Processo Sbagliato

Una categoria speciale di esperimenti che dimostra quando un sistema otiene il **risultato corretto** (rallenta durante l'attacco) ma per **ragioni sbagliate**:

### 1. **ParanoidBelief** (Threshold troppo basso)
- **Problema**: Rileva "anomalie" ovunque (false allarmi)
- **Output**: Rallenta ✓
- **Processo**: Sbagliato ✗ (paranoico, non selettivo)
- **Lesson**: Un threshold corretto è cruciale

### 2. **RiggedCostEFE** (Costi truccati)
- **Problema**: "maintain" ha costo altissimo (0.9 invece di 0.1)
- **Output**: Rallenta ✓
- **Processo**: Sbagliato ✗ (scelta forzata, non epistemica)
- **Lesson**: I costi devono riflettere la realtà operativa

### 3. **OverCautiousBelief** (Incertezza sempre alta)
- **Problema**: Mantiene incertezza artificialmente alta
- **Output**: Rallenta ✓
- **Processo**: Sbagliato ✗ (cautela ingiustificata)
- **Lesson**: L'incertezza deve essere calcolata correttamente

### 4. **LuckyBelief** (Timing hardcoded)
- **Problema**: Modello interno hardcoded per lo scenario specifico
- **Output**: Rallenta ✓
- **Processo**: Sbagliato ✗ (fragile, non generalizza)
- **Lesson**: Il modello deve essere generale e apprendibile

---

## 📊 Metriche di Valutazione

### 1. **Anomaly Detection Performance**
```python
- Precision: Anomalie vere / Anomalie rilevate
- Recall: Anomalie rilevate / Anomalie totali
- F1-score: Media armonica di precision e recall
```

### 2. **Action Quality**
```python
- EFE minimization: La scelta minimizza davvero l'EFE?
- Decision coherence: Le decisioni sono coerenti nel tempo?
- Timing accuracy: L'agente agisce al momento giusto?
```

### 3. **System Robustness**
```python
- Generalization: Funziona in scenari diversi?
- Stability: Mantiene coerenza con parametri variabili?
- Resilience: Resiste a variazioni nei parametri?
```

---

## ✅ Validazioni Implementate

### `validate_thesis.py` - Correttezza del sistema
- ✓ Anomaly Detection: L'agente rileva l'attacco?
- ✓ EFE Minimization: Le azioni minimizzano l'EFE?
- ✓ Epistemic Value Impact: Il componente epistemico funziona?
- ✓ Belief Coherence: Il belief update è consistente?
- ✓ Uncertainty Dynamics: L'incertezza aumenta correttamente?

### `validate_necessity.py` - Necessità dei componenti
- ✓ Component Interaction: I componenti interagiscono correttamente?
- ✓ Redundancy Check: Ogni componente è necessario?
- ✓ Failure Analysis: Cosa succede senza ogni componente?
- ✓ Optimization Path: Il percorso è ottimale?

---

## 📈 Visualizzazioni Generate

### 1. **Comportamento temporale** (`visualize_analysis.py`)
```
Time series per ogni variante ablation:
- Real state vs Estimated state
- Sensor readings (con attacco)
- Train velocity (azioni)
- Uncertainty evolution
- Anomaly detections
```

### 2. **Confronto ablation** (`visualize_analysis.py`)
```
Confronto matriciale delle 8 varianti:
- Velocity profile
- Uncertainty timeline
- Action distribution
- Performance metrics
```

### 3. **Percorsi fallaci** (`visualize_fallacious_pathways.py`)
```
Analisi dei sistemi fallaci:
- Threshold effects
- Cost distortions
- Parameter sensitivity
- Decision trees
```

---

## 🔧 Configurazione dei Parametri

Tutti i parametri del sistema sono definiti in costanti:

```python
# Tempi
TIME_STEPS = 50          # Durata simulazione
ATTACK_START = 22        # Inizio attacco FDIA
ATTACK_END = 28          # Fine attacco FDIA

# Stati
STABLE_A = 0.0           # Stato stabile iniziale
TRANSITION = 0.5         # Punto critico (transizione)
STABLE_B = 1.0           # Stato stabile finale

# Velocità treno
V_NOMINAL = 10           # Velocità normale
V_SLOW = 4               # Velocità ridotta
V_STOP = 0               # Velocità ferma

# Sensore
SENSOR_NOISE = 0.05      # Rumore gaussiano

# Belief
ANOMALY_THRESHOLD = 0.3  # Soglia prediction error
```

---

## 📚 Interpretazione dei Risultati

### Scenario di attacco FDIA

1. **t=0-21**: Sistema stabile, agente mantiene velocità nominale
2. **t=22-28**: Attacco attivo
   - Sensore legge sempre 0.0 (bugia)
   - Vero stato cambia a 0.5 (transizione critica)
   - Agente dovrebbe rilevare la discrepanza
   - Agente dovrebbe rallentare/fermarsi
3. **t=29-49**: Ritorno allo stato stabile, recupero

### Output atteso da variante `full`

```
t=22: Anomalia rilevata → Azione: epistemic_slow
t=23: Incertezza aumenta → Azione: epistemic_slow
t=24-28: Continua monitoraggio → Possibile: pragmatic_stop
t=29+: Incertezza diminuisce → Ritorno a: maintain
```

---

## 🎓 Per Tesi/Paper

### Sezioni consigliate

1. **Introduzione**: Il problema FDIA nei sistemi ferroviari
2. **Modello Teorico**: Free Energy Principle e EFE
3. **Architettura**: Componenti del sistema (Belief, Controller)
4. **Ablation Study**: Importanza di ogni componente
5. **Sistemi Fallaci**: Output giusto ≠ Processo giusto
6. **Risultati**: Grafici e tabelle comparativi
7. **Validazione**: Verifiche di correttezza implementate
8. **Conclusioni**: Generalizzazione e applicazioni future

### Figure raccomandate

1. Diagramma architettura sistema
2. Time series simulazioni (full vs no_epistemic)
3. Comparison plot delle 8 varianti
4. Heatmap ablation study
5. Decision tree fallaci sistemi
6. Validation results table

---

## 🐛 Troubleshooting

### Problema: "trace_full.json not found"
**Soluzione**: Eseguire prima `railway_active_inference_enhanced.py`

### Problema: Grafici non si generano
**Soluzione**: Controllare che matplotlib sia installato:
```bash
pip install matplotlib
```

### Problema: JSON decode error
**Soluzione**: I file JSON potrebbero essere stati spostati in cartelle. Controllare `/outputs/data/`

### Problema: Validazione fallisce
**Soluzione**: Controllare i parametri in `railway_active_inference_enhanced.py`. Il threshold deve essere 0.3.

---

## 🔗 Dipendenze

```
numpy >= 1.20        # Operazioni numeriche
matplotlib >= 3.4    # Visualizzazioni
scipy >= 1.7         # Funzioni matematiche
json (builtin)       # Serializzazione
pathlib (builtin)    # File system
```

---

## 📝 Citazioni teoriche

- Friston, K. (2010). "The Free Energy Principle: A Unified Brain Theory"
- Parr, T., & Friston, K. J. (2018). "The Active Inference Principle"
- Smith, R., et al. (2022). "Active Inference: State of the Art"

---

## 📧 Note di sviluppo

Questo progetto è stato sviluppato come framework completo per:
- ✅ Simulare agenti con Active Inference
- ✅ Testare l'importanza di componenti tramite ablation
- ✅ Identificare fallacies nei processi decisionali
- ✅ Validare la correttezza teorica
- ✅ Generare visualizzazioni per presentazioni

---

## 📄 Licenza

Academic use - Destinato a ricerca universitaria e tesi

---

## ✨ Autore

Framework sviluppato per ricerca in Active Inference applicato a sistemi critici ferroviari.

**Ultimo aggiornamento**: 2026

---

## 🎯 Prossimi passi

1. Estendere a scenari multi-sensore
2. Implementare apprendimento del modello generativo
3. Testare su dati ferroviari reali
4. Aggiungere attack patterns diversi (non solo FDIA)
5. Parallelizzare le simulazioni per scaling

