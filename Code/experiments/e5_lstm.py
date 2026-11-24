"""E5 — Complément : LSTM à 5 sorties pour les rendements (apprentissage multitâche).

Entrée : séquence des L dernières semaines des 55 variables distinctes des 5 paires
(10 propres à chaque paire + 5 communes). Sortie : les 5 rendements de la semaine
suivante. Réestimation toutes les 52 semaines (fenêtre croissante) ; une Ridge de
référence (par paire, ses 15 variables, λ par la CV retenue à l'étape 1) est
réestimée aux mêmes dates.

Grille : unités cachées {8, 32} × L {4, 12} ; 5 graines par réglage (prévision =
moyenne des 5 réseaux). Réglage retenu = meilleur R² de validation moyen sur les
5 devises ; seul ce réglage est évalué, une fois, sur le test.

Usage : .venv/bin/python Code/experiments/e5_lstm.py
Sorties : experiments/E5_lstm/"""
import itertools
import json
import pickle
import time

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed

from common import (INK, INK2, PAIR_LABELS, PAIRS, SERIES, accuracy, historical_mean,
                    load_frames, load_reference, out_dir, plt, rank_ic, save_fig,
                    write_environment)
from e4_ablation import block_bootstrap_sharpe
from e3_meta import sharpe
from Function.functions_strategy import model_matrix, strategy_returns
from Function.glmnet import cv_glmnet

DEST = out_dir("E5_lstm")
SHARED = ["dgs2_diff", "dgs10_diff", "gold_return", "sp500_return", "oil_return"]
HIDDEN, SEQ = [8, 32], [4, 12]
GRID = list(itertools.product(HIDDEN, SEQ))
SEEDS = [0, 1, 2, 3, 4]
REFIT = 52
LR, BATCH, MAX_EPOCHS, PATIENCE, ES_FRAC = 1e-3, 64, 200, 20, 0.2


## ---- données ----
def union_inputs(frames):
    """Matrice semaines × 55 (10 variables propres par paire + 5 communes) et cibles semaines × 5."""
    cols, blocks = [], []
    for pair in PAIRS:
        fr = frames[pair]
        own = [i for i, c in enumerate(fr["columns"]) if c not in SHARED]
        blocks.append(fr["X"][:, own])
        cols += [f"{pair}:{fr['columns'][i]}" for i in own]
    fr0 = frames[PAIRS[0]]
    shared_idx = [fr0["columns"].index(c) for c in SHARED]
    blocks.append(fr0["X"][:, shared_idx])
    cols += SHARED
    Z = np.column_stack(blocks)
    Y = np.column_stack([frames[p]["y"] for p in PAIRS])
    assert Z.shape[1] == 55 and all(frames[p]["dates"].equals(fr0["dates"]) for p in PAIRS)
    return Z, Y, cols


def blocks_of(rows):
    """Découpe des semaines à prévoir en blocs de REFIT semaines (début de bloc = date de réestimation)."""
    return [rows[k:k + REFIT] for k in range(0, len(rows), REFIT)]


## ---- LSTM ----
class Net(torch.nn.Module):
    def __init__(self, n_in, hidden, n_out):
        super().__init__()
        self.lstm = torch.nn.LSTM(n_in, hidden, batch_first=True)
        self.head = torch.nn.Linear(hidden, n_out)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1])


def sequences(Z, idx, L):
    return np.stack([Z[i - L + 1:i + 1] for i in idx])


def fit_predict_lstm(Z, Y, train_end, predict_rows, hidden, L, seed):
    """Entraîne sur les semaines j < train_end (avec L - 1 semaines d'historique), arrêt précoce sur
    les derniers 20 % chronologiques, puis prévoit predict_rows. Tout est standardisé sur l'entraînement."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(1)
    idx = np.arange(L - 1, train_end)
    mu, sd = Z[:train_end].mean(0), Z[:train_end].std(0) + 1e-12
    ym, ys = Y[:train_end].mean(0), Y[:train_end].std(0) + 1e-12
    Zs, Ys = (Z - mu) / sd, (Y - ym) / ys
    n_es = int(np.ceil(ES_FRAC * len(idx)))
    tr, es = idx[:-n_es], idx[-n_es:]
    Xtr = torch.tensor(sequences(Zs, tr, L), dtype=torch.float32)
    Ytr = torch.tensor(Ys[tr], dtype=torch.float32)
    Xes = torch.tensor(sequences(Zs, es, L), dtype=torch.float32)
    Yes = torch.tensor(Ys[es], dtype=torch.float32)
    net = Net(Z.shape[1], hidden, Y.shape[1])
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    loss_fn = torch.nn.MSELoss()
    gen = torch.Generator().manual_seed(seed)
    best, best_state, wait, epochs = np.inf, None, 0, 0
    for epoch in range(MAX_EPOCHS):
        net.train()
        perm = torch.randperm(len(Xtr), generator=gen)
        for b in range(0, len(Xtr), BATCH):
            sel = perm[b:b + BATCH]
            opt.zero_grad()
            loss_fn(net(Xtr[sel]), Ytr[sel]).backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            v = loss_fn(net(Xes), Yes).item()
        if v < best - 1e-9:
            best, wait, epochs = v, 0, epoch + 1
            best_state = {k: t.clone() for k, t in net.state_dict().items()}
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        pred = net(torch.tensor(sequences(Zs, predict_rows, L), dtype=torch.float32)).numpy()
    return pred * ys + ym, epochs


def lstm_period(Z, Y, rows, hidden, L, seed):
    preds, epochs = [], []
    for block in blocks_of(rows):
        p, e = fit_predict_lstm(Z, Y, block[0], block, hidden, L, seed)
        preds.append(p)
        epochs.append(e)
    return np.vstack(preds), epochs


## ---- Ridge de référence, même fréquence ----
def ridge_period(frames, rows):
    out = np.empty((len(rows), len(PAIRS)))
    for j, pair in enumerate(PAIRS):
        X, y = frames[pair]["X"], frames[pair]["y"]
        k = 0
        for block in blocks_of(rows):
            a0, beta, _ = cv_glmnet(X[:block[0]], y[:block[0]], 0.0, nfolds=5, seed=123)
            out[k:k + len(block), j] = a0 + X[block] @ beta
            k += len(block)
    return out


def scores(frames, rows, P, period):
    rec = []
    for j, pair in enumerate(PAIRS):
        fr = frames[pair]
        y = fr["y"][rows]
        rec.append({"Pair": pair, "Period": period,
                    **accuracy(y, P[:, j], historical_mean(fr["y"], rows))})
    return rec


def main():
    t0 = time.time()
    frames, ref = load_frames(), load_reference()
    Z, Y, cols = union_inputs(frames)
    rows = {p: frames[PAIRS[0]][p] for p in ["validation", "test"]}

    # 1) Grille sur la validation (5 graines par réglage)
    tasks = [(h, L, s) for (h, L) in GRID for s in SEEDS]
    res = Parallel(n_jobs=-1)(delayed(lstm_period)(Z, Y, rows["validation"], h, L, s)
                              for h, L, s in tasks)
    val = {t: r for t, r in zip(tasks, res)}
    grid_rows = []
    for h, L in GRID:
        ens = np.mean([val[(h, L, s)][0] for s in SEEDS], axis=0)
        sc = pd.DataFrame(scores(frames, rows["validation"], ens, "validation"))
        per_seed = [pd.DataFrame(scores(frames, rows["validation"], val[(h, L, s)][0],
                                        "validation")).R2.mean() for s in SEEDS]
        grid_rows.append({"Hidden": h, "L": L, "R2_validation_moyen": sc.R2.mean(),
                          "R2_OS_validation_moyen": sc.R2_OS.mean(),
                          "R2_par_graine_min": min(per_seed), "R2_par_graine_max": max(per_seed),
                          "Epoques_moyennes": np.mean([np.mean(val[(h, L, s)][1]) for s in SEEDS])})
    grid = pd.DataFrame(grid_rows)
    best = grid.loc[grid.R2_validation_moyen.idxmax()]
    h, L = int(best.Hidden), int(best.L)
    print(grid.round(5).to_string(index=False))
    print(f"Réglage retenu (validation) : {h} unités, L = {L}")

    # 2) Test, une seule fois, pour le réglage retenu
    tst = Parallel(n_jobs=-1)(delayed(lstm_period)(Z, Y, rows["test"], h, L, s) for s in SEEDS)
    P_test = np.mean([r[0] for r in tst], axis=0)
    P_val = np.mean([val[(h, L, s)][0] for s in SEEDS], axis=0)

    # 3) Ridge de même fréquence
    R_val, R_test = ridge_period(frames, rows["validation"]), ridge_period(frames, rows["test"])

    acc = []
    for name, Pv, Pt in [("LSTM", P_val, P_test), ("Ridge_52", R_val, R_test)]:
        for period, P in [("validation", Pv), ("test", Pt)]:
            acc += [{"Model": name, **r} for r in scores(frames, rows[period], P, period)]
    seed_spread = [pd.DataFrame(scores(frames, rows["test"], r[0], "test")).R2_OS.mean() for r in tst]
    acc = pd.DataFrame(acc)
    acc.to_csv(DEST / "precision_lstm_ridge.csv", index=False)
    grid.to_csv(DEST / "grille_validation.csv", index=False)

    summary = []
    actual = model_matrix(ref["sharpe_models"]["MIXT"], "return_actual")
    for name, Pv, Pt in [("LSTM", P_val, P_test), ("Ridge_52", R_val, R_test)]:
        row = {"Model": name}
        for period, P in [("validation", Pv), ("test", Pt)]:
            A = np.column_stack([frames[p]["y"][rows[period]] for p in PAIRS])
            d = acc[(acc.Model == name) & (acc.Period == period)]
            row[f"R2_OS_{period}"] = d.R2_OS.mean()
            row[f"IC_{period}"], row[f"IC_t_{period}"] = rank_ic(P, A)
        r = strategy_returns(pd.DataFrame(Pt, index=actual.index, columns=PAIRS), actual)
        lo, hi = block_bootstrap_sharpe(r.values[:, None], np.random.default_rng(123))
        row.update({"Sharpe_A1_test": sharpe(r.values), "Sharpe_IC95_bas": lo[0],
                    "Sharpe_IC95_haut": hi[0]})
        summary.append(row)
    summary = pd.DataFrame(summary)
    summary.to_csv(DEST / "resume.csv", index=False)
    print(summary.round(4).to_string(index=False))
    print("R2_OS de test par graine :", np.round(seed_spread, 4))

    (DEST / "choix.json").write_text(json.dumps(
        {"hidden": h, "L": L, "regle": "meilleur R2 de validation moyen sur 5 devises",
         "R2_OS_test_par_graine": seed_spread}, indent=1, default=float), encoding="utf-8")
    (DEST / "models").mkdir(exist_ok=True)
    with open(DEST / "models" / "predictions.pkl", "wb") as f:
        pickle.dump({"P_val": P_val, "P_test": P_test, "R_val": R_val, "R_test": R_test,
                     "val_grid": val, "test_seeds": tst, "choix": (h, L)}, f)
    fig_compare(acc, seed_spread)
    write_environment(DEST, seconds=round(time.time() - t0, 1), torch=torch.__version__,
                      hidden=h, L=L, refit=REFIT, seeds=SEEDS)


def fig_compare(acc, seed_spread):
    t = acc[acc.Period == "test"]
    fig, ax = plt.subplots(figsize=(6.2, 2.9))
    w = 0.38
    for k, (name, lab) in enumerate([("Ridge_52", "Ridge (réestimée toutes les 52 sem.)"),
                                     ("LSTM", "LSTM à 5 sorties (moyenne de 5 graines)")]):
        v = 100 * t[t.Model == name].set_index("Pair").loc[PAIRS].R2_OS.values
        ax.bar(np.arange(len(PAIRS)) + (k - 0.5) * w, v, width=w, color=SERIES[k],
               edgecolor="white", linewidth=1, label=lab)
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_xticks(range(len(PAIRS)), [PAIR_LABELS[p] for p in PAIRS])
    ax.set_ylabel("$R^2$ hors échantillon, test (%)")
    lo, hi = 100 * min(seed_spread), 100 * max(seed_spread)
    ax.set_title(f"LSTM, moyenne des 5 devises selon la graine : de {lo:.1f} à {hi:.1f} %"
                 .replace(".", ","), loc="left", fontsize=8, color=INK2)
    ax.legend(frameon=False, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    save_fig(fig, DEST, "fig_lstm_contre_ridge")


if __name__ == "__main__":
    main()
