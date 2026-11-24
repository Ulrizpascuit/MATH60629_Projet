"""E1.5 — Stabilité de la sélection MIXT (bootstrap par blocs de la validation).

MIXT choisit, pour chaque devise, le modèle au meilleur R² sur les 354 semaines de
validation. On rééchantillonne ces semaines par blocs de 8 semaines consécutives
(2 000 tirages, Künsch 1989) et on regarde à quelle fréquence chaque modèle serait
choisi. Une sélection stable donnerait une fréquence proche de 100 % pour le
modèle retenu. Lit les prévisions de la référence E0 (aucune ré-estimation).

Usage : .venv/bin/python Code/experiments/e1_5_stabilite_mixt.py
Sorties : experiments/E1_5_stabilite_mixt/"""
import numpy as np
import pandas as pd

from common import (INK, LABELS, METHODS, PAIR_LABELS, PAIRS, load_reference, out_dir, plt,
                    reference_predictions, save_fig, write_environment)

DEST = out_dir("E1_5_stabilite_mixt")
BLOCK, B, SEED = 8, 2000, 123


def block_indices(n, rng):
    starts = rng.integers(0, n - BLOCK + 1, int(np.ceil(n / BLOCK)))
    return np.concatenate([np.arange(s, s + BLOCK) for s in starts])[:n]


def r2(y, pred):
    return 1 - np.sum((y - pred) ** 2, axis=0) / np.sum((y - y.mean()) ** 2)


def main():
    ref = load_reference()
    rng = np.random.default_rng(SEED)
    rows, chosen = [], {}
    for pair in PAIRS:
        preds = np.column_stack([reference_predictions(ref, m, pair, "validation")[0]
                                 for m in METHODS])
        y = reference_predictions(ref, METHODS[0], pair, "validation")[1]
        original = METHODS[int(np.argmax(r2(y[:, None], preds)))]
        assert original == ref["mixt_selection"].set_index("Currency").loc[pair, "Selected_Model"]
        chosen[pair] = original
        counts = np.zeros(len(METHODS))
        for _ in range(B):
            idx = block_indices(len(y), rng)
            counts[np.argmax(r2(y[idx, None], preds[idx]))] += 1
        for m, c in zip(METHODS, counts):
            rows.append({"Pair": pair, "Method": m, "Frequence": c / B, "Choix_MIXT": m == original})
    freq = pd.DataFrame(rows)
    freq.to_csv(DEST / "frequences_selection.csv", index=False)
    summary = freq[freq.Choix_MIXT][["Pair", "Method", "Frequence"]].rename(
        columns={"Method": "Modele_MIXT", "Frequence": "Frequence_choix_MIXT"})
    summary["Modele_le_plus_frequent"] = [freq[freq.Pair == p].sort_values("Frequence").Method.iloc[-1]
                                          for p in summary.Pair]
    summary.to_csv(DEST / "resume_stabilite.csv", index=False)
    print(summary.round(3).to_string(index=False))
    fig_freq(freq, chosen)
    write_environment(DEST, block=BLOCK, B=B, seed=SEED,
                      source="experiments/E0_reference_mac/models/models.pkl")


def fig_freq(freq, chosen):
    mat = freq.pivot(index="Method", columns="Pair", values="Frequence").loc[METHODS, PAIRS]
    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    im = ax.imshow(100 * mat.values, cmap="Blues", vmin=0, vmax=100, aspect="auto")
    for i, m in enumerate(METHODS):
        for j, p in enumerate(PAIRS):
            v = 100 * mat.values[i, j]
            ax.text(j, i, f"{v:.0f}" + (" ◆" if chosen[p] == m else ""), ha="center", va="center",
                    fontsize=7.5, color="white" if v > 60 else INK)
    ax.set_xticks(range(len(PAIRS)), [PAIR_LABELS[p] for p in PAIRS], fontsize=7.5)
    ax.set_yticks(range(len(METHODS)), [LABELS[m] for m in METHODS])
    ax.spines[:].set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    cb.set_label("Fréquence de sélection (%)")
    ax.set_title(f"◆ modèle choisi par MIXT ; {B} tirages, blocs de {BLOCK} semaines",
                 loc="left", fontsize=8)
    save_fig(fig, DEST, "fig_stabilite_mixt")


if __name__ == "__main__":
    main()
