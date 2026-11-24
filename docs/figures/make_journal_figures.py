"""Figures du journal de projet (docs/JOURNAL_PROJET.md).

Usage : .venv/bin/python docs/figures/make_journal_figures.py
Écrit chaque figure en PNG (pour le .md) et en PDF (pour le rapport LaTeX)
dans docs/figures/. Ne lit que les données déjà préparées par run_data.py."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from Code.build_model import TEST_WEEKS, load_inputs
from Function.functions_model import forecast_rows, model_data

OUT = Path(__file__).resolve().parent
# Palette catégorielle de référence (ordre fixe : bleu, orange, aqua)
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "savefig.bbox": "tight",
    "figure.dpi": 150,
})


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=200)
    fig.savefig(OUT / f"{name}.pdf")
    plt.close(fig)


def split_boundaries():
    """Bornes exactes entraînement initial / validation / test (mêmes règles que build_model)."""
    features, targets, _ = load_inputs()
    dates = model_data(features["EURUSD"], targets["EURUSD"]).index
    n = len(dates)
    train_ratio = (n - TEST_WEEKS + 0.5) / n
    val = forecast_rows(n, "validation", train_ratio)
    test = forecast_rows(n, "test", train_ratio)
    return dates, val, test


def fig_split():
    """Figure 1 : découpage chronologique et principe du walk-forward."""
    dates, val, test = split_boundaries()
    week = pd.Timedelta(days=7)
    segments = [
        ("Entraînement initial", dates[0], dates[val[0] - 1], val[0], BLUE),
        ("Validation", dates[val[0]], dates[val[-1]], len(val), ORANGE),
        ("Test (hors échantillon)", dates[test[0]], dates[test[-1]], len(test), AQUA),
    ]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.2, 3.9), height_ratios=[1, 1.35],
                                   sharex=True)

    # Panneau A : les trois périodes
    for label, start, end, n_weeks, color in segments:
        ax1.barh(0, (end + week) - start, left=start, height=0.5, color=color,
                 edgecolor="white", linewidth=2)
        mid = start + ((end + week) - start) / 2
        ax1.text(mid, 0.42, f"{label}\n{n_weeks} semaines", ha="center", va="bottom",
                 color=INK, fontsize=8)
        ax1.text(mid, -0.42, f"{start:%Y-%m-%d} → {end:%Y-%m-%d}", ha="center", va="top",
                 color=INK2, fontsize=7)
    ax1.set_ylim(-1.0, 1.25)
    ax1.set_yticks([])
    ax1.spines["left"].set_visible(False)
    ax1.set_title("A. Découpage chronologique (aucun mélange dans le temps)", loc="left",
                  color=INK)

    # Panneau B : fenêtre croissante, ré-estimée chaque semaine
    examples = [val[0], val[len(val) // 2], test[0], test[len(test) // 2], test[-1]]
    for row, i in enumerate(examples):
        y = len(examples) - row
        ax2.barh(y, dates[i] - dates[0], left=dates[0], height=0.45, color=BLUE)
        ax2.plot(dates[i] + week / 2, y, "o", color=ORANGE, ms=6,
                 markeredgecolor="white", markeredgewidth=1.5)
    ax2.set_yticks(range(1, len(examples) + 1),
                   [f"semaine prévue n° {i + 1}" for i in reversed(examples)], fontsize=7)
    ax2.axvline(dates[test[0]], color=INK2, lw=0.8, ls="--")
    ax2.text(dates[test[0]], len(examples) + 0.75, " début du test", color=INK2, fontsize=7,
             va="center")
    ax2.set_ylim(0.4, len(examples) + 1.1)
    ax2.set_title("B. Walk-forward : modèle ré-estimé sur tout le passé, puis prévision de "
                  "la semaine suivante", loc="left", color=INK)
    ax2.set_xlabel("Date (semaines du jeudi)")
    ax2.xaxis.set_major_locator(mdates.YearLocator(2))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax2.grid(axis="x", color=GRID, lw=0.5)
    handles = [plt.Rectangle((0, 0), 1, 1, color=BLUE),
               plt.Line2D([], [], marker="o", color=ORANGE, ls="", ms=6)]
    ax2.legend(handles, ["Données d'entraînement de cette semaine-là", "Semaine prévue"],
               frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=2,
               fontsize=7)
    fig.tight_layout()
    save(fig, "fig01_decoupage_walk_forward")


if __name__ == "__main__":
    fig_split()
    print("Figures écrites dans", OUT)
