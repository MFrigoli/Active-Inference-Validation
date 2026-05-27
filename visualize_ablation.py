"""
visualize_ablation.py
Genera due tipi di grafici per l'ablation study:
  1) Percorso decisionale a 8 layer per ciascuna variante  (decision_pathway_<variant>.png)
  2) Griglia comparativa velocita' per tutte le 8 varianti  (ablation_comparison.png)

Legge i JSON da outputs/ablation/.
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

STYLE         = "academic"
DATA_DIR      = Path("outputs/json/ablation2")
DATA_DIR_ABL1 = Path("outputs/json/ablation1")

VARIANTS = ["full", "no_risk", "no_cost", "no_epistemic",
            "only_risk", "only_cost", "only_epistemic", "none"]

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
    "no_epistemic":   "EFE(slow) = −PV + 0.4 > EFE(maintain) = −PV + 0.1 sempre -> maintain vince sempre",
    "no_threshold":   "Nessuna anomalia rilevata -> uncertainty rimane 0.1 -> epistemic=0 -> maintain vince sempre",
    "no_uncertainty": "epistemic_value = uncertainty * ... = 0 -> epistemic azzerato -> maintain vince sempre",
    "no_model":       "prediction_error = |sensor - 0| sempre ~ 0 in fase stabile -> nessuna anomalia rilevata durante attacco",
}

LABELS = {
    "full":           "FULL\n(R+C) - E",
    "no_risk":        "NO_RISK\nC - E",
    "no_cost":        "NO_COST\nR - E",
    "no_epistemic":   "NO_EPISTEMIC\nR + C",
    "only_risk":      "ONLY_RISK\nR",
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
    risk     = np.array([e["risk"]             for e in log])
    cost_a   = np.array([e["cost"]            for e in log])
    neg_pv_a = np.array([e["neg_pv"]          for e in log])
    ep_a     = np.array([e["epistemic_value"] for e in log])
    pred_err = np.abs(sensor - real)

    # Per-action risk/epistemic dalle decisioni del controller
    def comp(action_key, field):
        return np.array([d["full_efe"][action_key][field] for d in ctrl])

    return dict(
        t=t, real=real, sensor=sensor, belief=belief, unc=unc,
        anomaly=anomaly, velocity=velocity, policy=policy,
        efe_m=efe_m, efe_s=efe_s, efe_st=efe_st,
        risk=risk, cost_a=cost_a, neg_pv_a=neg_pv_a, ep_a=ep_a,
        pred_err=pred_err,
        # componenti per-azione (per i layer EFE)
        risk_m  =comp("maintain",       "risk"),
        neg_pv_m=comp("maintain",       "neg_pv"),
        risk_s  =comp("epistemic_slow", "risk"),
        neg_pv_s=comp("epistemic_slow", "neg_pv"),
        ep_s    =comp("epistemic_slow", "epistemic_value"),
        risk_st =comp("pragmatic_stop", "risk"),
        neg_pv_st=comp("pragmatic_stop", "neg_pv"),
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
    ax.plot(d["t"], d["pred_err"], color=P["risk"], lw=1.4,
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
        ("maintain",       d["efe_m"],  d["neg_pv_m"],  d["risk_m"],  P["maintain"],
         "4a. EFE Maintain  (v=10)"),
        ("epistemic_slow", d["efe_s"],  d["neg_pv_s"],  d["risk_s"],  P["epistemic_slow"],
         "4b. EFE Slow      (v=4)"),
        ("pragmatic_stop", d["efe_st"], d["neg_pv_st"], d["risk_st"], P["pragmatic_stop"],
         "4c. EFE Stop      (v=0)"),
    ]
    for i, (a_name, efe_arr, neg_pv_arr, haz_arr, color, title) in enumerate(efe_data):
        ax = axes[3 + i]
        shade_attack(ax, P)
        ax.plot(d["t"], neg_pv_arr,   color=P["fail"],     lw=1.2, label="−PragmaticValue")
        ax.plot(d["t"], d["ep_a"],    color=P["epistemic"],lw=1.2, label="Valore epistemico")
        ax.plot(d["t"], efe_arr,      color=color,         lw=1.6, ls="--",
                label=f"EFE {a_name.split('_')[0]} (= −PV−E)")
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
        r"$\mathrm{EFE}(\pi) = -\mathrm{PragmaticValue}(\pi) - \mathrm{EpistemicValue}(\pi)$,  "
        r"$\mathrm{PragmaticValue}(\pi) = -\mathrm{Risk}(\pi) - \mathrm{Cost}(\pi)$",
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
        r"$\mathrm{EFE}(\pi) = -\mathrm{PragmaticValue}(\pi) - \mathrm{EpistemicValue}(\pi)$,  "
        r"$\mathrm{PragmaticValue}(\pi) = -\mathrm{Risk}(\pi) - \mathrm{Cost}(\pi)$",
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
    CASES = [
        ("baseline",       _ABLATION1_COLORS["baseline"],       "Baseline"),
        ("no_uncertainty", _ABLATION1_COLORS["no_uncertainty"], "Senza Uncertainty"),
    ]

    # 5 righe × 2 colonne: righe 0-1 = Baseline, riga 2 = gap, righe 3-4 = Senza Uncertainty
    fig = plt.figure(figsize=(13, 14))
    gs  = GridSpec(5, 2, figure=fig,
                   height_ratios=[1, 1, 0.15, 1, 1],
                   hspace=0.55, wspace=0.32,
                   top=0.93, bottom=0.05, left=0.08, right=0.97)

    fig.suptitle("Ablation 1 — Baseline vs Senza Uncertainty",
                 fontsize=14, fontweight="bold")

    # Pre-crea tutti gli 8 assi
    axes = {
        "baseline": [
            fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]),
            fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]),
        ],
        "no_uncertainty": [
            fig.add_subplot(gs[3, 0]), fig.add_subplot(gs[3, 1]),
            fig.add_subplot(gs[4, 0]), fig.add_subplot(gs[4, 1]),
        ],
    }

    # Linea separatrice tra i due blocchi
    fig.add_artist(plt.Line2D(
        [0.04, 0.96], [0.505, 0.505],
        transform=fig.transFigure, color="#aaaaaa", lw=1.2, ls="--",
    ))

    for key, color, _ in CASES:
        axs      = axes[key]
        data     = data_map.get(key)
        box_color = "#fef3e2"

        if data is None:
            for ax in axs:
                ax.text(0.5, 0.5, "Dati non disponibili",
                        ha="center", va="center", transform=ax.transAxes)
                clean_axis(ax)
            continue

        log  = data["simulation_log"]
        btr  = data.get("belief_trace", [])
        mets = data["metrics"]

        t      = np.array([e["t"]               for e in log])
        vel    = np.array([e["velocity"]        for e in log])
        unc    = np.array([e["uncertainty"]     for e in log])
        ep_a     = np.array([e["epistemic_value"] for e in log])
        neg_pv_a = np.array([e["neg_pv"]          for e in log])
        pred_err = (
            np.array([e["prediction_error"] for e in btr])
            if btr and "prediction_error" in btr[0]
            else np.abs(
                np.array([e["switch_observed"] for e in log]) -
                np.array([e["switch_real"]     for e in log])
            )
        )

        tp   = mets["tp"]
        fp   = mets["fp"]
        prec = mets["precision"]

        status      = "OK"  if key == "baseline" else "FAIL"
        title_color = color if key == "baseline" else "#e74c3c"
        full_label  = data["label"]

        _leg_kw = dict(fontsize=7, framealpha=0.9,
                       facecolor="#fef3e2", edgecolor="#f39c12")

        # [0] Azione — titolo colorato (verde/rosso)
        ax = axs[0]
        ax.plot(t, vel, color=color, lw=2.0)
        ax.set_ylim(-0.5, 12)
        ax.set_ylabel("Velocità (km/h)", fontsize=9)
        ax.set_title(f"[{status}] {full_label}", fontsize=10,
                     fontweight="bold", color=title_color)
        ax.text(0.02, 0.05, f"TP={tp}  FP={fp}  Prec={prec:.2f}",
                transform=ax.transAxes, fontsize=8, va="bottom",
                bbox=dict(boxstyle="round", facecolor=box_color, alpha=0.9))

        # [1] Prediction Error & Uncertainty — titolo nero
        ax = axs[1]
        ax.plot(t, pred_err, color=color,       lw=1.6, label="Prediction error")
        ax.plot(t, unc,      color="steelblue", lw=1.5, ls="--", label="Uncertainty")
        ax.axhline(ANOMALY_THRESHOLD, color="red", lw=1.0, ls=":",
                   label=f"Soglia ({ANOMALY_THRESHOLD})")
        ax.set_ylim(-0.05, 1.15)
        ax.set_ylabel("Valore", fontsize=9)
        ax.set_title("Prediction Error & Uncertainty", fontsize=10, fontweight="bold")
        ax.legend(loc="upper left", **_leg_kw)

        # [2] Epistemic Value — titolo nero
        ax = axs[2]
        ax.plot(t, ep_a, color="purple", lw=1.8)
        ax.set_ylim(-0.05, 1.2)
        ax.set_ylabel("Valore", fontsize=9)
        ax.set_xlabel("Tempo (timestep)", fontsize=9)
        ax.set_title("Valore epistemico", fontsize=10, fontweight="bold")

        # [3] Pragmatic Value — titolo nero
        ax = axs[3]
        ax.plot(t, neg_pv_a, color="#e67e22", lw=1.8)
        ax.set_ylabel("Valore", fontsize=9)
        ax.set_xlabel("Tempo (timestep)", fontsize=9)
        ax.set_title("Pragmatic Value", fontsize=10, fontweight="bold")

        for ax in axs:
            shade_attack(ax, P, alpha=0.2)
            ax.set_xlim(0, 50)
            clean_axis(ax)

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
