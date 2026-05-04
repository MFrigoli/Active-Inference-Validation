"""
visualize_stress_test.py
Genera due tipi di grafici per lo stress test:
  1) Griglia comparativa 3×2 + tabella metriche  (stress_test_comparison.png)
  2) Percorso decisionale a 8 layer per ogni sistema fallace  (stress_pathway_<system>.png)

Legge i JSON da outputs/stress_test/.
Salva i PNG in outputs/graphs/<style>/.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path

from plot_style import (
    apply_style, shade_attack, shade_transition, clean_axis,
    ATTACK_START, ATTACK_END, TRANSITION_START, TRANSITION_END,
    ANOMALY_THRESHOLD,
)

STYLE    = "academic"
DATA_DIR = Path("outputs/json/stress_test")

SYSTEMS = [
    ("baseline",      "NORMALE",        "Processo corretto"),
    ("paranoid",      "PARANOICO",      "Soglia = 0.01 (troppo bassa)"),
    ("rigged_cost",   "COSTO TRUCCATO", "Maintain costa 0.9 (truccato)"),
    ("over_cautious", "OVER-CAUTIOUS",  "Uncertainty = 0.9 fissa"),
    ("lucky",         "LUCKY",          "Timing hardcoded t=22–28"),
]

SYSTEM_COLORS = {
    "baseline":      "#2e7d32",
    "paranoid":      "#e67e22",
    "rigged_cost":   "#c0392b",
    "over_cautious": "#8e44ad",
    "lucky":         "#2c3e50",
}

SYSTEM_IS_RIGGED = {
    "baseline":      False,
    "paranoid":      False,
    "rigged_cost":   True,
    "over_cautious": False,
    "lucky":         False,
}

SYSTEM_THRESHOLD = {
    "baseline":      ANOMALY_THRESHOLD,
    "paranoid":      0.01,
    "rigged_cost":   ANOMALY_THRESHOLD,
    "over_cautious": ANOMALY_THRESHOLD,
    "lucky":         ANOMALY_THRESHOLD,
}


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def load_system(key: str) -> dict | None:
    path = DATA_DIR / f"fallacious_{key}.json"
    if not path.exists():
        print(f"  [WARN] {path} non trovato")
        return None
    with open(path) as f:
        return json.load(f)


def ensure_dir() -> Path:
    out = Path("outputs/graphs/stress_test")
    out.mkdir(parents=True, exist_ok=True)
    return out


def compute_metrics(data: dict) -> tuple:
    log = data["simulation_log"]
    tp = fp = 0
    fp_times = []
    for e in log:
        in_win = TRANSITION_START <= e["t"] <= TRANSITION_END
        slowed = e["policy"] != "maintain"
        if in_win and slowed:
            tp += 1
        elif not in_win and slowed:
            fp += 1
            fp_times.append(e["t"])
    prec = tp / (tp + fp) if (tp + fp) > 0 else None
    return tp, fp, prec, fp_times


def extract_arrays(data: dict) -> dict:
    log  = data["simulation_log"]
    ctrl = data["controller_trace"]
    btr  = data.get("belief_trace", [])

    t        = np.array([e["t"]               for e in log])
    real     = np.array([e["switch_real"]     for e in log])
    sensor   = np.array([e["switch_observed"] for e in log])
    belief   = np.array([e["belief_estimate"] for e in log])
    unc      = np.array([e["uncertainty"]     for e in log])
    anomaly  = np.array([e["anomaly"]         for e in log], dtype=float)
    velocity = np.array([e["velocity"]        for e in log])
    policy   = [e["policy"]                   for e in log]
    efe_m    = np.array([e["G_maintain"]      for e in log])
    efe_s    = np.array([e["G_slow"]          for e in log])
    efe_st   = np.array([e["G_stop"]          for e in log])

    pred_err = (
        np.array([e["prediction_error"] for e in btr])
        if btr and "prediction_error" in btr[0]
        else np.abs(sensor - real)
    )

    def comp(action_key, field):
        return np.array([d["full_efe"][action_key][field] for d in ctrl])

    ep_a = np.array([e["epistemic_value"] for e in log])

    return dict(
        t=t, real=real, sensor=sensor, belief=belief, unc=unc,
        anomaly=anomaly, velocity=velocity, policy=policy,
        efe_m=efe_m, efe_s=efe_s, efe_st=efe_st,
        pred_err=pred_err, ep_a=ep_a,
        hazard_m =comp("maintain",       "hazard"),
        risk_m   =comp("maintain",       "risk"),
        hazard_s =comp("epistemic_slow", "hazard"),
        risk_s   =comp("epistemic_slow", "risk"),
        ep_s     =comp("epistemic_slow", "epistemic_value"),
        hazard_st=comp("pragmatic_stop", "hazard"),
        risk_st  =comp("pragmatic_stop", "risk"),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Grafico 1 — Griglia comparativa 3×2 + tabella
# ─────────────────────────────────────────────────────────────────────────────

def plot_stress_comparison(all_data: dict, P: dict, outdir: Path):
    fig = plt.figure(figsize=(12, 12))
    gs  = GridSpec(3, 2, hspace=0.40, wspace=0.12, height_ratios=[1, 1, 1.2])
    positions = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0)]

    metrics_rows = []

    for idx, (key, label, note) in enumerate(SYSTEMS):
        row, col = positions[idx]
        ax   = fig.add_subplot(gs[row, col])
        data = all_data.get(key)

        if data is None:
            ax.set_visible(False)
            continue

        log = data["simulation_log"]
        t   = np.array([e["t"]        for e in log])
        vel = np.array([e["velocity"] for e in log])

        shade_attack(ax, P, alpha=0.18)
        shade_transition(ax, P, alpha=0.08)

        color = P["ok"] if key == "baseline" else P["fail"]
        ax.plot(t, vel, color=color, lw=1.8)
        ax.fill_between(t, 0, vel, color=color, alpha=0.10)

        tp, fp, prec, fp_times = compute_metrics(data)
        prec_str = f"{prec:.2f}" if prec is not None else "--"

        for fp_t in fp_times:
            ax.axvline(fp_t, color=P["attack"], lw=0.5, alpha=0.35, zorder=1)

        status       = "OK" if key == "baseline" else "FALLACE"
        status_color = P["ok"] if key == "baseline" else P["fail"]
        ax.set_title(f"[{status}] {label}\nTP={tp}  FP={fp}  Prec={prec_str}",
                     loc="left", fontsize=10, fontweight="bold", color=status_color)
        ax.text(0.98, 0.08, note, transform=ax.transAxes, fontsize=8,
                ha="right", va="bottom", style="italic", color="#444444",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="#cccccc", linewidth=0.6, alpha=0.92))

        ax.set_ylim(-0.5, 12)
        ax.set_xlim(0, 50)
        ax.set_ylabel("Velocità")
        ax.set_xlabel("Tempo (timestep)")
        clean_axis(ax)

        metrics_rows.append([label, tp, fp, prec_str, note])

    # Tabella riepilogativa in cella (2,1)
    ax_table = fig.add_subplot(gs[2, 1])
    ax_table.axis("off")

    col_labels = ["Sistema", "TP", "FP", "Prec.", "Caratteristica"]
    table = ax_table.table(
        cellText=metrics_rows, colLabels=col_labels,
        cellLoc="left",
        bbox=[0.0, 0.15, 1.0, 0.85],
        colWidths=[0.30, 0.08, 0.08, 0.10, 0.44],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.6)

    for j in range(len(col_labels)):
        cell = table[(0, j)]
        cell.set_facecolor(P["belief"])
        cell.set_text_props(weight="bold", color="white")
        cell.set_edgecolor("white")
    for i in range(1, len(metrics_rows) + 1):
        row_color = "#e8f5e9" if i == 1 else "#fdecea"
        for j in range(len(col_labels)):
            table[(i, j)].set_facecolor(row_color)
            table[(i, j)].set_edgecolor("white")

    ax_table.set_title("Metriche comparative", loc="left", fontsize=11, fontweight="bold", pad=12)

    # Legenda incollata sotto la tabella, stessa cella axes
    ax_table.text(
        #0.5, -0.04, 
        0.5, 0.08,
        "Legenda: rallentamento in t [20,30] (transizione reale + attacco FDIA)\n"
        "TP = rilevazioni corrette in [20,30]  ·  FP = rilevazioni fuori [20,30]",
        transform=ax_table.transAxes,
        ha="center", va="top", fontsize=8.5, style="italic", color="#1a3a5c",
        clip_on=False,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#dce8f7",
                  edgecolor="#5b82b8", linewidth=0.8, alpha=0.95),
    )

    fig.suptitle("Stress Test — Output corretto non implica processo corretto",
                 fontsize=13, fontweight="bold", y=0.975)
    fig.subplots_adjust(left=0.06, right=0.995, top=0.90, bottom=0.14,
                        hspace=0.40, wspace=0.12)

    out = outdir / "stress_test_comparison.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  [OK] {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Grafico 2 — Percorso decisionale a 8 layer per sistema fallace
# ─────────────────────────────────────────────────────────────────────────────

def plot_stress_pathway(key: str, data: dict, P: dict, outdir: Path):
    d         = extract_arrays(data)
    color     = SYSTEM_COLORS[key]
    threshold = SYSTEM_THRESHOLD[key]
    label     = data["label"]

    fig, axes = plt.subplots(8, 1, figsize=(9, 16), sharex=False)
    fig.subplots_adjust(hspace=0.45, top=0.95, bottom=0.04)
    for ax in axes:
        ax.set_xlim(0, 50)

    # Layer 1 — Ground truth vs Sensore
    ax = axes[0]
    shade_attack(ax, P, label="Periodo FDIA Attack")
    ax.plot(d["t"], d["real"],   color=P["ground_truth"], lw=1.6, label="Ground Truth (stato reale)")
    ax.plot(d["t"], d["sensor"], color=P["sensor"],       lw=1.2, ls="--",
            label="Sensore (può essere compromesso)")
    ax.set_ylabel("Stato dello\nscambio")
    ax.set_ylim(-0.1, 1.15)
    ax.set_title("1. Stato fisico: realtà vs sensore", loc="left")
    ax.legend(loc="upper left", ncol=1, fontsize=7)
    clean_axis(ax)

    # Layer 2 — Belief + incertezza + anomalie
    ax = axes[1]
    shade_attack(ax, P)
    ax.fill_between(d["t"], 0, d["unc"], color=P["uncertainty"], alpha=0.3, label="Incertezza")
    ax.plot(d["t"], d["belief"], color=P["belief"], lw=1.6, label="Belief (stima interna)")
    ax.plot(d["t"], d["unc"],    color=P["uncertainty"], lw=1.0, ls="--")
    anom_mask = d["anomaly"] > 0
    if anom_mask.any():
        ax.scatter(d["t"][anom_mask], d["belief"][anom_mask],
                   color=P["attack"], s=25, zorder=5, marker="x", label="Anomalia rilevata")
    ax.set_ylabel("Belief State")
    ax.set_ylim(-0.1, 1.15)
    ax.set_title("2. Inference Layer: come interpreta i dati?", loc="left")
    ax.legend(loc="upper left", ncol=1, fontsize=7)
    clean_axis(ax)

    # Layer 3 — Prediction error vs soglia
    ax = axes[2]
    shade_attack(ax, P)
    ax.plot(d["t"], d["pred_err"], color=P["hazard"], lw=1.4,
            label="Prediction Error |sensore - modello|")
    ax.axhline(threshold, color=P["fail"], lw=1.0, ls="--",
               label=f"Soglia anomalia = {threshold}")
    ax.fill_between(d["t"], threshold, d["pred_err"],
                    where=d["pred_err"] > threshold,
                    color=P["attack"], alpha=0.25, label="Anomalia (error > soglia)")
    ax.set_ylabel("Prediction Error")
    ax.set_ylim(-0.05, max(0.75, d["pred_err"].max() * 1.15))
    ax.set_title("3. Prediction Error vs Soglia di Anomalia", loc="left")
    ax.legend(loc="upper left", ncol=1, fontsize=7)
    clean_axis(ax)

    # Layers 4a/4b/4c — EFE per azione
    efe_data = [
        ("maintain",       d["efe_m"],  d["risk_m"],  d["hazard_m"],  P["maintain"],
         "4a. EFE Maintain  (v=10)"),
        ("epistemic_slow", d["efe_s"],  d["risk_s"],  d["hazard_s"],  P["epistemic_slow"],
         "4b. EFE Slow      (v=4)"),
        ("pragmatic_stop", d["efe_st"], d["risk_st"], d["hazard_st"], P["pragmatic_stop"],
         "4c. EFE Stop      (v=0)"),
    ]
    for i, (a_name, efe_arr, risk_arr, haz_arr, c, title) in enumerate(efe_data):
        ax = axes[3 + i]
        shade_attack(ax, P)
        ax.plot(d["t"], risk_arr,  color=P["fail"],      lw=1.2, label="Risk")
        ax.plot(d["t"], d["ep_a"], color=P["epistemic"], lw=1.2, label="Epistemic Value")
        ax.plot(d["t"], efe_arr,   color=c,              lw=1.6, ls="--",
                label=f"EFE {a_name.split('_')[0]} (= R−E)")
        chosen = np.array([p == a_name for p in d["policy"]])
        if chosen.any():
            ax.scatter(d["t"][chosen], efe_arr[chosen], color=c, s=20,
                       zorder=5, edgecolors="white", linewidths=0.4, label="Azione scelta")
        ax.set_ylabel("Valore EFE")
        ax.set_title(title, loc="left")
        ax.legend(loc="upper left", ncol=1, fontsize=7)
        clean_axis(ax)

    # Layer 5 — Confronto EFE delle 3 azioni
    ax = axes[6]
    shade_attack(ax, P)
    ax.plot(d["t"], d["efe_m"],  color=P["maintain"],       lw=1.4, label="Maintain (v=10)")
    ax.plot(d["t"], d["efe_s"],  color=P["epistemic_slow"], lw=1.4, label="Epistemic Slow (v=4)")
    ax.plot(d["t"], d["efe_st"], color=P["pragmatic_stop"], lw=1.4, label="Pragmatic Stop (v=0)")
    policy_arr = np.array(d["policy"])
    efe_chosen = np.where(policy_arr == "maintain", d["efe_m"],
                 np.where(policy_arr == "epistemic_slow", d["efe_s"], d["efe_st"]))
    c_chosen = [
        P["maintain"] if p == "maintain" else
        P["epistemic_slow"] if p == "epistemic_slow" else
        P["pragmatic_stop"] for p in d["policy"]
    ]
    ax.scatter(d["t"], efe_chosen, c=c_chosen, s=20, zorder=5,
               edgecolors="white", linewidths=0.4, label="Azione scelta")
    ax.set_ylabel("Expected Free\nEnergy")
    ax.set_title("5. Decision Layer: quale azione ha EFE minima?", loc="left")
    ax.legend(loc="upper left", ncol=1, fontsize=7)
    clean_axis(ax)

    # Layer 6 — Velocità risultante
    ax = axes[7]
    shade_attack(ax, P)
    for j in range(len(d["t"]) - 1):
        p = d["policy"][j]
        c = "#c8e6c9" if p == "maintain" else "#bbdefb" if p == "epistemic_slow" else "#ffcdd2"
        ax.axvspan(d["t"][j], d["t"][j + 1], color=c, alpha=0.5)
    ax.plot(d["t"], d["velocity"], color=color, lw=2.2)
    ax.set_ylabel("Velocità")
    ax.set_ylim(-0.5, 12)
    ax.set_xlabel("Tempo (steps)")
    ax.set_title("6. Action Layer: cosa fa l'agente?", loc="left")
    clean_axis(ax)

    fig.suptitle(
        f"Percorso Decisionale — {label}\n"
        r"$\mathrm{EFE} = \mathrm{Risk} - \mathrm{Epistemic}$,  "
        r"$\mathrm{Risk} = \mathrm{Hazard} + \mathrm{Cost}$",
        fontsize=11, fontweight="bold",
    )

    out = outdir / f"stress_pathway_{key}.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  [OK] {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print(f"  VISUALIZE STRESS TEST — stile: {STYLE}")
    print("=" * 70)

    P      = apply_style(STYLE)
    outdir = ensure_dir()

    all_data = {}
    for key, _, _ in SYSTEMS:
        d = load_system(key)
        if d is not None:
            all_data[key] = d

    if not all_data:
        print("  [ERROR] Nessun dato trovato. Esegui prima: python stress_test.py")
        return

    print(f"\n[1/2] Griglia comparativa stress test:")
    plot_stress_comparison(all_data, P, outdir)

    print(f"\n[2/2] Pathway a 8 layer per sistema ({len(all_data)} sistemi):")
    for key, data in all_data.items():
        plot_stress_pathway(key, data, P, outdir)

    print(f"\n  Output in: {outdir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
