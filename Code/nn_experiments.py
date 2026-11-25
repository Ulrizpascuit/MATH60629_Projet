"""Expériences avec les réseaux de neurones (après Code/build_panel.py).

Étapes (résultats en cache dans Data/Models/nn/, une étape déjà faite est relue) :
  1. tune     : grille d'hyperparamètres évaluée sur la validation seulement
                (354 semaines, réestimation tous les 13 semaines, 5 graines) ;
  2. final    : la meilleure configuration de chaque famille (MLP/GRU x MSE/rang),
                10 graines, sur la validation et sur le test ;
  3. ridge    : référence linéaire sur exactement les mêmes entrées et la même cible ;
  4. ablation : mêmes réseaux avec les 15 features d'origine seulement, et MLP
                entraîné séparément pour chaque paire (sans mise en commun) ;
  5. curves   : courbes d'apprentissage ; importance des variables par permutation
                (modèle estimé sur l'estimation initiale, mesurée sur la validation).

Usage : python Code/nn_experiments.py [étape ...]   (par défaut : toutes)"""
import itertools
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import RidgeCV

from Function.functions import FEATURES, MODELS
from Function.functions_nn import (
    add_pair_onehot, fit_model, predict, refit_blocks, standardize, walk_forward_task,
)

OUT = MODELS / "nn"
INITIAL, TRAIN = 529, 883  # fin de l'estimation initiale, fin de l'entraînement
REFIT_EVERY = 13
N_TUNE_SEEDS, N_FINAL_SEEDS = 5, 10
FAMILIES = [("mlp", "mse"), ("mlp", "rank"), ("gru", "mse"), ("gru", "rank")]


def load_panel():
    d = np.load(FEATURES / "panel.npz", allow_pickle=True)
    return {k: d[k] for k in d.files}


def weekly_ic(scores, ret):
    """Corrélation de Spearman, chaque semaine, entre scores et rendements des 5 paires."""
    rs = scores.argsort(1).argsort(1).astype(float)
    rr = ret.argsort(1).argsort(1).astype(float)
    rs -= rs.mean(1, keepdims=True)
    rr -= rr.mean(1, keepdims=True)
    return (rs * rr).sum(1) / np.sqrt((rs ** 2).sum(1) * (rr ** 2).sum(1))


def grid():
    cfgs = []
    for loss in ("mse", "rank"):
        for hidden, drop, wd, lr in itertools.product(
                [(32,), (32, 16), (32, 16, 8), (64, 32, 16)], [0.1, 0.3], [1e-4, 1e-2], [1e-3, 1e-4]):
            cfgs.append(dict(arch="mlp", loss=loss, hidden=hidden, dropout=drop,
                             weight_decay=wd, lr=lr))
        for hidden, seq, wd, lr in itertools.product([8, 16, 32], [4, 12], [1e-4, 1e-2], [1e-3, 1e-4]):
            cfgs.append(dict(arch="gru", loss=loss, hidden=hidden, seq_len=seq, dropout=0.2,
                             weight_decay=wd, lr=lr))
    return cfgs


def run_walk_forward(X, z, cfgs, start, end, seeds, n_jobs=-1):
    """Prévisions en fenêtre croissante : preds[c][s] de forme (end - start, P)."""
    blocks = refit_blocks(start, end, REFIT_EVERY)
    tasks = [(c, s, b) for c in range(len(cfgs)) for s in seeds for b in blocks]
    # Les GRU (plus lents) d'abord, pour mieux répartir le travail
    tasks.sort(key=lambda t: cfgs[t[0]]["arch"] != "gru")
    res = Parallel(n_jobs=n_jobs, batch_size=4)(
        delayed(walk_forward_task)(X, z, b, cfgs[c], s) for c, s, b in tasks)
    preds = {c: {s: np.zeros((end - start, X.shape[1])) for s in seeds} for c in range(len(cfgs))}
    for (c, s, _), (block, p) in zip(tasks, res):
        preds[c][s][block - start] = p
    return preds


def cached(name, fn):
    path = OUT / f"{name}.pkl"
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    t = time.time()
    out = fn()
    OUT.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(out, f)
    print(f"[{name}] {time.time() - t:.0f} s")
    return out


## ---- 1. tune ----
def stage_tune(d):
    X, z, ret = d["X"], d["z"], d["ret"]
    cfgs = grid()
    preds = run_walk_forward(X, z, cfgs, INITIAL, TRAIN, list(range(N_TUNE_SEEDS)))
    rows = []
    for c, cfg in enumerate(cfgs):
        ens = np.mean([preds[c][s] for s in preds[c]], axis=0)
        ic = weekly_ic(ens, ret[INITIAL:TRAIN])
        rows.append(dict(cfg, ic=float(np.nanmean(ic)),
                         ic_t=float(np.nanmean(ic) / np.nanstd(ic) * np.sqrt(np.sum(~np.isnan(ic))))))
    return {"cfgs": cfgs, "results": rows}


def best_configs(tune):
    best = {}
    for arch, loss in FAMILIES:
        rows = [r for r in tune["results"] if r["arch"] == arch and r["loss"] == loss]
        r = max(rows, key=lambda r: r["ic"])
        best[f"{arch}_{loss}"] = {k: v for k, v in r.items() if k not in ("ic", "ic_t")}
    return best


## ---- 2. final ----
def stage_final(d, best, X=None, suffix=""):
    X = d["X"] if X is None else X
    cfgs = list(best.values())
    seeds = list(range(N_FINAL_SEEDS))
    val = run_walk_forward(X, d["z"], cfgs, INITIAL, TRAIN, seeds)
    test = run_walk_forward(X, d["z"], cfgs, TRAIN, len(X), seeds)
    return {name + suffix: {"cfg": cfgs[i], "val": val[i], "test": test[i]}
            for i, name in enumerate(best)}


## ---- 3. ridge ----
def stage_ridge(d):
    X, z = d["X"], d["z"]
    out = {}
    for period, (start, end) in {"val": (INITIAL, TRAIN), "test": (TRAIN, len(X))}.items():
        pred = np.zeros((end - start, X.shape[1]))
        alphas = []
        for block in refit_blocks(start, end, REFIT_EVERY):
            Xs = add_pair_onehot(standardize(X, block[0]))
            n, P, F = Xs[:block[0]].shape
            m = RidgeCV(alphas=np.logspace(-1, 5, 25)).fit(
                Xs[:block[0]].reshape(-1, F), np.clip(z[:block[0]], -4, 4).reshape(-1))
            pred[block - start] = m.predict(Xs[block].reshape(-1, F)).reshape(len(block), P)
            alphas.append(m.alpha_)
        out[period] = pred
        out[f"{period}_alphas"] = alphas
    return out


## ---- 4. ablation : un MLP par paire (sans mise en commun) ----
def stage_per_pair(d, cfg):
    X, z = d["X"], d["z"]
    out = {}
    for period, (start, end) in {"val": (INITIAL, TRAIN), "test": (TRAIN, len(X))}.items():
        blocks = refit_blocks(start, end, REFIT_EVERY)
        tasks = [(p, s, b) for p in range(X.shape[1]) for s in range(N_FINAL_SEEDS) for b in blocks]
        res = Parallel(n_jobs=-1, batch_size=4)(
            delayed(walk_forward_task)(X[:, [p]], z[:, [p]], b, cfg, s) for p, s, b in tasks)
        pred = np.zeros((N_FINAL_SEEDS, end - start, X.shape[1]))
        for (p, s, _), (block, pr) in zip(tasks, res):
            pred[s, block - start, p] = pr[:, 0]
        out[period] = pred.mean(0)
    return out


## ---- 5. curves / importance ----
def stage_importance(d, best):
    """Importance par permutation : baisse de l'IC de validation quand on mélange
    une feature (entre semaines et paires), modèle estimé sur l'estimation initiale."""
    X, z, ret = d["X"], d["z"], d["ret"]
    weeks = np.arange(INITIAL, TRAIN)
    out = {}
    for name in ("mlp_mse", "mlp_rank"):
        cfg = best[name]

        def one(seed):
            model, Xs, hist, best_epoch = fit_model(X, z, INITIAL, cfg, seed)
            rng = np.random.default_rng(seed)
            base = predict(model, Xs, weeks)
            perm_preds = []
            for f in range(X.shape[2]):
                reps = []
                for _ in range(5):
                    Xp = Xs.copy()
                    block = Xp[weeks, :, f]
                    Xp[weeks, :, f] = rng.permutation(block.reshape(-1)).reshape(block.shape)
                    reps.append(predict(model, Xp, weeks))
                perm_preds.append(reps)
            return base, np.array(perm_preds), hist, best_epoch

        res = Parallel(n_jobs=N_FINAL_SEEDS)(delayed(one)(s) for s in range(N_FINAL_SEEDS))
        base = np.mean([r[0] for r in res], axis=0)
        ic0 = np.nanmean(weekly_ic(base, ret[weeks]))
        drops = []
        for f in range(X.shape[2]):
            ics = [np.nanmean(weekly_ic(np.mean([r[1][f][k] for r in res], axis=0), ret[weeks]))
                   for k in range(5)]
            drops.append(ic0 - np.mean(ics))
        out[name] = {"ic": ic0, "drop": np.array(drops),
                     "histories": [r[2] for r in res], "best_epochs": [r[3] for r in res]}
    return out


def main(stages):
    d = load_panel()
    print(f"Panel {d['X'].shape}, validation {INITIAL}-{TRAIN}, test {TRAIN}-{len(d['X'])}")
    tune = cached("tune", lambda: stage_tune(d))
    best = best_configs(tune)
    for k, v in best.items():
        print(k, v)
    if "final" in stages:
        cached("final", lambda: stage_final(d, best))
    if "ridge" in stages:
        cached("ridge", lambda: stage_ridge(d))
    if "ablation" in stages:
        base15 = d["X"][:, :, :15]
        cached("ablation_base15", lambda: stage_final(d, best, X=base15, suffix="_base15"))
        cached("ablation_per_pair", lambda: stage_per_pair(d, best["mlp_mse"]))
    if "curves" in stages:
        cached("importance", lambda: stage_importance(d, best))


if __name__ == "__main__":
    args = sys.argv[1:] or ["final", "ridge", "ablation", "curves"]
    main(args)
