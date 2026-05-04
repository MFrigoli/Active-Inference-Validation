"""
visualize_ablation.py
Genera due tipi di grafici per l'ablation study:
  1) Percorso decisionale a 8 layer per ciascuna variante  (decision_pathway_<variant>.png)
  2) Griglia comparativa velocita' per tutte le 8 varianti  (ablation_comparison.png)

Legge i JSON da outputs/ablation/.
Salva i PNG in outputs/graphs/<style>/.
"""

import json
import textwrap
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path

from plot_style import (
    apply_style, shade_attack, shade_transition, clean_axis,
    ATTACK_START, ATTACK_END, TRANSITION_START, TRANSITION_END,
    ANOMALY_THRESHOLD,
)

STYLE         = "academic"
DATA_DIR      = Path("outputs/json/ablation2")
DATA_DIR_ABL1 = Path("outputs/json/ablation1")

VARIANTS = ["full", "no_hazard", "no_cost", "no_epistemic",
            "only_hazard", "only_cost", "only_epistemic", "none"]

ABLATION1_KEYS = [
    "baseline", "no_epistemic", "no_threshold", "no_uncertainty", "no_model",
]

_ABLATION1_COLORS = {
    "baseline":       "#27ae60",
    "no_epistemic":   "#e67e22",
    "no_threshold":   "#e74c3c",
    "no_uncertainty": "#8e44ad",
    "no_model":       "#2c3e50",
}

_ABLATION1_FAILURES = {
    "baseline":       "N/A - sistema corretto",
    "no_epistemic":   "EFE(slow) = Risk + 0.4 > EFE(maintain) = Risk + 0.1 sempre -> maintain vince sempre",
    "no_threshold":   "Nessuna anomalia rilevata -> uncertainty rimane 0.1 -> epistemic=0 -> maintain vince sempre",
    "no_uncertainty": "epistemic_value = uncertainty * ... = 0 -> epistemic azzerato -> maintain vince sempre",
    "no_model":       "prediction_error = |sensor - 0| sempre ~ 0 in fase stabile -> nessuna anomalia rilevata durante attacco",
}

LABELS = {
    "full":           "FULL\n(H+C) - E",
    "no_hazard":      "NO_HAZARD\nC - E",
    "no_cost":        "NO_COST\nH - E",
    "no_epistemic":   "NO_EPISTEMIC\nH + C",
    "only_hazard":    "ONLY_HAZARD\nH",
    "only_cost":      "ONLY_COST\nC",
    "only_epistemic": "ONLY_EPISTEMIC\n-E",
    "none":           "NONE\n0",
}


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def load_trace(variant: str) -> dict | None:
    path = DATA_DIR / f"trace_{variant}.json"
    if not path.exists():
        print(f"  [WARN] {path} non trovato")
        return None
    with open(path) as f:
        return json.load(f)


def ensure_dirs() -> dict:
    paths = {
        "decision":  Path("outputs/graphs/decision"),
        "ablation1": Path("outputs/graphs/ablation1"),
        "ablation2": Path("outputs/graphs/ablation2"),
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def extract_arrays(trace: dict) -> dict:
    """Estrae array numpy dal simulation_log e controller_trace."""
    log  = trace["simulation_log"]
    ctrl = trace["controller_trace"]

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
    hazard   = np.array([e["hazard"]          for e in log])
    cost_a   = np.array([e["cost"]            for e in log])
    risk_a   = np.array([e["risk"]            for e in log])
    ep_a     = np.array([e["epistemic_value"] for e in log])
    pred_err = np.abs(sensor - real)

    # Per-action hazard/risk/epistemic dalle decisioni del controller
    def comp(action_key, field):
        return np.array([d["full_efe"][action_key][field] for d in ctrl])

    return dict(
        t=t, real=real, sensor=sensor, belief=belief, unc=unc,
        anomaly=anomaly, velocity=velocity, policy=policy,
        efe_m=efe_m, efe_s=efe_s, efe_st=efe_st,
        hazard=hazard, cost_a=cost_a, risk_a=risk_a, ep_a=ep_a,
        pred_err=pred_err,
        # componenti per-azione (per i layer EFE)
        hazard_m=comp("maintain",       "hazard"),
        risk_m  =comp("maintain",       "risk"),
        hazard_s=comp("epistemic_slow", "hazard"),
        risk_s  =comp("epistemic_slow", "risk"),
        ep_s    =comp("epistemic_slow", "epistemic_value"),
        hazard_st=comp("pragmatic_stop","hazard"),
        risk_st  =comp("pragmatic_stop","risk"),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Grafico 1 — Percorso decisionale a 8 layer
# ─────────────────────────────────────────────────────────────────────────────

def plot_pathway(variant: str, trace: dict, P: dict, outdir: Path):
    d = extract_arrays(trace)

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
    ax.axhline(ANOMALY_THRESHOLD, color=P["fail"], lw=1.0, ls="--",
               label=f"Soglia anomalia = {ANOMALY_THRESHOLD}")
    ax.fill_between(d["t"], ANOMALY_THRESHOLD, d["pred_err"],
                    where=d["pred_err"] > ANOMALY_THRESHOLD,
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
    for i, (a_name, efe_arr, risk_arr, haz_arr, color, title) in enumerate(efe_data):
        ax = axes[3 + i]
        shade_attack(ax, P)
        ax.plot(d["t"], risk_arr,     color=P["fail"],     lw=1.2, label="Risk")
        ax.plot(d["t"], d["ep_a"],    color=P["epistemic"],lw=1.2, label="Epistemic Value")
        ax.plot(d["t"], efe_arr,      color=color,         lw=1.6, ls="--",
                label=f"EFE {a_name.split('_')[0]} (= R−E)")
        chosen = np.array([p == a_name for p in d["policy"]])
        if chosen.any():
            ax.scatter(d["t"][chosen], efe_arr[chosen], color=color, s=20,
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
    color_chosen = [
        P["maintain"] if p == "maintain" else
        P["epistemic_slow"] if p == "epistemic_slow" else
        P["pragmatic_stop"] for p in d["policy"]
    ]
    ax.scatter(d["t"], efe_chosen, c=color_chosen, s=20, zorder=5,
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
    ax.plot(d["t"], d["velocity"], color="#1a237e", lw=2.2)
    ax.set_ylabel("Velocità")
    ax.set_ylim(-0.5, 12)
    ax.set_xlabel("Tempo (steps)")
    ax.set_title("6. Action Layer: cosa fa l'agente?", loc="left")
    clean_axis(ax)

    fig.suptitle(
        f"Percorso Decisionale — {variant}\n"
        r"$\mathrm{EFE} = \mathrm{Risk} - \mathrm{Epistemic}$,  "
        r"$\mathrm{Risk} = \mathrm{Hazard} + \mathrm{Cost}$",
        fontsize=11, fontweight="bold",
    )

    out = outdir / f"decision_pathway_{variant}.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  [OK] {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Grafico 2 — Griglia comparativa 4×2
# ─────────────────────────────────────────────────────────────────────────────

def plot_ablation2(traces: dict, P: dict, outdir: Path):
    fig, axes = plt.subplots(4, 2, figsize=(9.5, 11), sharex=True, sharey=True)
    axes_flat = axes.flatten()

    for i, variant in enumerate(VARIANTS):
        ax    = axes_flat[i]
        trace = traces.get(variant)

        if trace is None:
            ax.text(0.5, 0.5, f"{LABELS.get(variant, variant)}\n(dati non disponibili)",
                    ha="center", va="center", transform=ax.transAxes, fontsize=9)
            clean_axis(ax)
            continue

        d = extract_arrays(trace)
        shade_attack(ax, P, alpha=0.18)
        shade_transition(ax, P, alpha=0.08)
        ax.plot(d["t"], d["velocity"], color=P["belief"], lw=1.6)
        ax.fill_between(d["t"], 0, d["velocity"], color=P["belief"], alpha=0.12)

        # Metriche: TP/FP basati sulla finestra di transizione
        expected = (d["t"] >= TRANSITION_START) & (d["t"] <= TRANSITION_END)
        slowed   = np.array([p != "maintain" for p in d["policy"]])
        tp = int((expected & slowed).sum())
        fp = int((~expected & slowed).sum())
        prec = f"{tp/(tp+fp):.2f}" if (tp + fp) > 0 else "--"

        ax.set_title(f"{LABELS.get(variant, variant)}\nTP={tp}  FP={fp}  Prec={prec}",
                     loc="left", fontsize=9)
        ax.set_ylim(-0.5, 12)
        clean_axis(ax)

    for ax in axes[-1, :]:
        ax.set_xlabel("Tempo (timestep)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Velocità (km/h)")

    fig.suptitle(
        "Ablation Study — Velocità del treno per ogni variante EFE\n"
        r"$\mathrm{EFE} = \mathrm{Risk} - \mathrm{Epistemic}$,  "
        r"$\mathrm{Risk} = \mathrm{Hazard} + \mathrm{Cost}$",
        fontsize=12, fontweight="bold", y=0.98,
    )
    fig.text(
        0.5, 0.01,
        "Zona rossa: attacco FDIA (t=22–28). Zona gialla: transizione reale (t=20–30).",
        ha="center", fontsize=8, style="italic", color="#555555",
    )
    plt.tight_layout(rect=[0, 0.02, 1, 0.97])

    out = outdir / "ablation2.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  [OK] {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Grafico 3 — Ablation1: griglia 5×3 componenti architetturali
# ─────────────────────────────────────────────────────────────────────────────

def load_ablation1(key: str) -> dict | None:
    path = DATA_DIR_ABL1 / f"ablation1_{key}.json"
    if not path.exists():
        print(f"  [WARN] {path} non trovato")
        return None
    with open(path) as f:
        return json.load(f)


def plot_ablation1(data_map: dict, P: dict, outdir: Path):
    n_rows = len(ABLATION1_KEYS)
    ATTACK_COLOR = "#ffcccc"

    fig = plt.figure(figsize=(20, 4 * n_rows))
    fig.suptitle(
        "Ablation 1 — Effetto della rimozione di ciascun componente",
        fontsize=15, fontweight="bold", y=0.995,
    )
    gs = GridSpec(n_rows, 3, figure=fig,
                  hspace=0.7, wspace=0.35,
                  top=0.96, bottom=0.04, left=0.08, right=0.97)

    for row_i, key in enumerate(ABLATION1_KEYS):
        data = data_map.get(key)
        color   = _ABLATION1_COLORS[key]
        failure = _ABLATION1_FAILURES[key]
        is_ok   = (key == "baseline")

        if data is None:
            for col in range(3):
                fig.add_subplot(gs[row_i, col]).text(
                    0.5, 0.5, "Dati non disponibili",
                    ha="center", va="center", transform=plt.gca().transAxes)
            continue

        log     = data["simulation_log"]
        btr     = data.get("belief_trace", [])
        label   = data["label"]
        metrics = data["metrics"]

        t        = np.array([e["t"]               for e in log])
        vel      = np.array([e["velocity"]        for e in log])
        policy   = np.array([e["policy"]          for e in log])
        unc      = np.array([e["uncertainty"]     for e in log])
        ep_a     = np.array([e["epistemic_value"] for e in log])
        pred_err = (
            np.array([e["prediction_error"] for e in btr])
            if btr and "prediction_error" in btr[0]
            else np.abs(
                np.array([e["switch_observed"] for e in log]) -
                np.array([e["switch_real"]     for e in log])
            )
        )

        tp   = metrics["tp"]
        fp   = metrics["fp"]
        prec = metrics["precision"]
        status = "OK" if is_ok else "FAIL"
        title_color = "green" if is_ok else "red"
        box_color   = "#d5f5e3" if is_ok else "#fadbd8"

        # ── Col 0: Velocità ──────────────────────────────────────────────────
        ax0 = fig.add_subplot(gs[row_i, 0])
        ax0.axvspan(ATTACK_START, ATTACK_END, color=ATTACK_COLOR, alpha=0.5, zorder=0)
        ax0.plot(t, vel, color=color, lw=2.0)
        slow_mask = policy == "epistemic_slow"
        if slow_mask.any():
            ax0.scatter(t[slow_mask], vel[slow_mask],
                        color="blue", s=20, zorder=5)
        ax0.set_title(f"[{status}] {label}", fontsize=10, fontweight="bold",
                      color=title_color)
        ax0.set_ylabel("Velocità", fontsize=9)
        ax0.set_xlim(0, 50)
        ax0.set_ylim(-0.5, 12)
        ax0.grid(True, alpha=0.3)
        ax0.spines["top"].set_visible(False)
        ax0.spines["right"].set_visible(False)
        ax0.text(0.02, 0.05,
                 f"TP={tp}  FP={fp}  Prec={prec}",
                 transform=ax0.transAxes, fontsize=8, va="bottom",
                 bbox=dict(boxstyle="round", facecolor=box_color, alpha=0.9))
        if row_i == n_rows - 1:
            ax0.set_xlabel("Tempo (timestep)", fontsize=9)

        # ── Col 1: Prediction Error & Uncertainty ────────────────────────────
        ax1 = fig.add_subplot(gs[row_i, 1])
        ax1.axvspan(ATTACK_START, ATTACK_END, color=ATTACK_COLOR, alpha=0.5, zorder=0)
        ax1.plot(t, pred_err, color=color,       lw=1.8, label="Prediction error")
        ax1.plot(t, unc,      color="steelblue", lw=1.5, ls="--", label="Uncertainty")
        ax1.axhline(ANOMALY_THRESHOLD, color="red", lw=1.2, ls=":",
                    label=f"Soglia ({ANOMALY_THRESHOLD})")
        ax1.set_title("Prediction Error & Uncertainty", fontsize=10)
        ax1.set_ylabel("Valore", fontsize=9)
        ax1.set_xlim(0, 50)
        ax1.set_ylim(-0.05, 1.1)
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=8, loc="upper left")
        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)
        if row_i == n_rows - 1:
            ax1.set_xlabel("Tempo (timestep)", fontsize=9)

        # ── Col 2: Epistemic Value ────────────────────────────────────────────
        ax2 = fig.add_subplot(gs[row_i, 2])
        ax2.axvspan(ATTACK_START, ATTACK_END, color=ATTACK_COLOR, alpha=0.5, zorder=0)
        ax2.plot(t, ep_a, color="purple", lw=2.0, label="Epistemic value")
        ax2.set_title("Epistemic Value", fontsize=10)
        ax2.set_ylabel("Valore", fontsize=9)
        ax2.set_xlim(0, 50)
        ax2.set_ylim(-0.05, 1.2)
        ax2.grid(True, alpha=0.3)
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)
        wrapped = textwrap.fill(failure, width=30)
        ax2.text(0.03, 0.95, wrapped,
                 transform=ax2.transAxes, fontsize=8, va="top", ha="left",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor=box_color, alpha=0.9),
                 linespacing=1.3)
        if row_i == n_rows - 1:
            ax2.set_xlabel("Tempo (timestep)", fontsize=9)

    # Separatori tra le righe
    for i in range(1, n_rows):
        y_pos = 1.0 - (i / n_rows)
        fig.add_artist(plt.Line2D(
            [0.02, 0.98], [y_pos, y_pos],
            transform=fig.transFigure,
            color="black", linewidth=1.5, alpha=0.4,
        ))

    out = outdir / "ablation1.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print(f"  VISUALIZE ABLATION — stile: {STYLE}")
    print("=" * 70)

    P    = apply_style(STYLE)
    dirs = ensure_dirs()

    traces = {}
    for v in VARIANTS:
        t = load_trace(v)
        if t is not None:
            traces[v] = t

    if not traces:
        print("  [ERROR] Nessun trace trovato. Esegui prima: python ablation.py")
        return

    print(f"\n[1/3] Decision pathway a 8 layer ({len(traces)} varianti):")
    for variant, trace in traces.items():
        plot_pathway(variant, trace, P, dirs["decision"])

    print(f"\n[2/3] Griglia ablation2:")
    plot_ablation2(traces, P, dirs["ablation2"])

    data_abl1 = {}
    for k in ABLATION1_KEYS:
        d = load_ablation1(k)
        if d is not None:
            data_abl1[k] = d

    if data_abl1:
        print(f"\n[3/3] Griglia ablation1 ({len(data_abl1)} varianti):")
        plot_ablation1(data_abl1, P, dirs["ablation1"])
    else:
        print(f"\n[3/3] SKIP ablation1 — esegui prima: python ablation1.py")

    print(f"\n  Output in: outputs/graphs/")
    print("=" * 70)


if __name__ == "__main__":
    main()
