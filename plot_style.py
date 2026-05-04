"""
plot_style.py
Modulo centralizzato di stile per tutti i grafici.

Due stili disponibili:
- "academic":  IEEE/Nature style, serif, palette sobria, griglia minima
- "didactic":  sans-serif moderno, palette colorblind-friendly, annotazioni pulite

Uso:
    from plot_style import apply_style, shade_attack, shade_transition, clean_axis
    P = apply_style("academic")
"""

import matplotlib.pyplot as plt
import matplotlib as mpl

_PALETTE_ACADEMIC = {
    "hazard":         "#1f3a5f",
    "cost":           "#8c6d31",
    "epistemic":      "#5b2c6f",
    "efe":            "#2d2d2d",
    "maintain":       "#2e7d32",
    "epistemic_slow": "#1565c0",
    "pragmatic_stop": "#c62828",
    "ground_truth":   "#2e7d32",
    "sensor":         "#c62828",
    "belief":         "#1565c0",
    "uncertainty":    "#90a4ae",
    "attack":         "#d32f2f",
    "transition":     "#f9a825",
    "ok":             "#2e7d32",
    "fail":           "#c62828",
    "v1": "#1f3a5f", "v2": "#8c6d31", "v3": "#5b2c6f", "v4": "#2e7d32",
    "v5": "#c62828", "v6": "#37474f", "v7": "#6a1b9a", "v8": "#bf360c",
}

_PALETTE_DIDACTIC = {
    "hazard":         "#d62728",
    "cost":           "#ff7f0e",
    "epistemic":      "#9467bd",
    "efe":            "#1f1f1f",
    "maintain":       "#2ca02c",
    "epistemic_slow": "#1f77b4",
    "pragmatic_stop": "#d62728",
    "ground_truth":   "#2ca02c",
    "sensor":         "#d62728",
    "belief":         "#1f77b4",
    "uncertainty":    "#bdbdbd",
    "attack":         "#e53935",
    "transition":     "#fbc02d",
    "ok":             "#2ca02c",
    "fail":           "#d62728",
    "v1": "#1f77b4", "v2": "#ff7f0e", "v3": "#2ca02c", "v4": "#d62728",
    "v5": "#9467bd", "v6": "#8c564b", "v7": "#e377c2", "v8": "#17becf",
}

ATTACK_START     = 22
ATTACK_END       = 28
TRANSITION_START = 20
TRANSITION_END   = 30
ANOMALY_THRESHOLD = 0.30


def apply_style(style="academic"):
    if style == "academic":
        mpl.rcParams.update({
            "font.family":        "serif",
            "font.serif":         ["Times New Roman", "DejaVu Serif", "Liberation Serif"],
            "font.size":           10,
            "axes.titlesize":      11,
            "axes.titleweight":    "bold",
            "axes.labelsize":      10,
            "axes.linewidth":      0.8,
            "axes.spines.top":     False,
            "axes.spines.right":   False,
            "axes.grid":           True,
            "grid.color":          "#dddddd",
            "grid.linewidth":      0.4,
            "grid.linestyle":      "-",
            "xtick.labelsize":     9,
            "ytick.labelsize":     9,
            "xtick.direction":     "in",
            "ytick.direction":     "in",
            "xtick.major.size":    3,
            "ytick.major.size":    3,
            "legend.fontsize":     9,
            "legend.frameon":      False,
            "legend.handlelength": 2.0,
            "lines.linewidth":     1.4,
            "figure.dpi":          120,
            "savefig.dpi":         300,
            "savefig.bbox":        "tight",
            "savefig.pad_inches":  0.05,
        })
        return _PALETTE_ACADEMIC

    elif style == "didactic":
        mpl.rcParams.update({
            "font.family":        "sans-serif",
            "font.sans-serif":    ["Helvetica", "Arial", "DejaVu Sans"],
            "font.size":           11,
            "axes.titlesize":      13,
            "axes.titleweight":    "bold",
            "axes.labelsize":      11,
            "axes.labelweight":    "bold",
            "axes.linewidth":      1.0,
            "axes.spines.top":     False,
            "axes.spines.right":   False,
            "axes.grid":           True,
            "grid.color":          "#e8e8e8",
            "grid.linewidth":      0.6,
            "axes.axisbelow":      True,
            "xtick.labelsize":     10,
            "ytick.labelsize":     10,
            "legend.fontsize":     10,
            "legend.frameon":      True,
            "legend.framealpha":   0.95,
            "legend.edgecolor":    "#bbbbbb",
            "lines.linewidth":     2.0,
            "figure.dpi":          120,
            "savefig.dpi":         300,
            "savefig.bbox":        "tight",
            "savefig.pad_inches":  0.15,
        })
        return _PALETTE_DIDACTIC

    else:
        raise ValueError(f"Stile non riconosciuto: {style!r}. Usa 'academic' o 'didactic'.")


def shade_attack(ax, P, label=None, alpha=None):
    a = alpha if alpha is not None else 0.18
    ax.axvspan(ATTACK_START - 0.5, ATTACK_END + 0.5,
               color=P["attack"], alpha=a, zorder=0, label=label)


def shade_transition(ax, P, label=None, alpha=None):
    a = alpha if alpha is not None else 0.08
    ax.axvspan(TRANSITION_START - 0.5, TRANSITION_END + 0.5,
               color=P["transition"], alpha=a, zorder=0, label=label)


def clean_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
