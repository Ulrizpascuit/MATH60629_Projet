"""Réseau factorisé : amplitude et direction conditionnelle, puis sélection de variables.

Pour un seuil k (en écarts-types de z = r(t+1) / vol_12(t)) :
  * tête amplitude : p = P(|z| > k), entraînée sur toutes les observations ;
  * tête direction conditionnelle : q = P(z > 0 | |z| > k), entraînée seulement
    sur les grands mouvements.
Alors P(forte hausse) = p q et P(forte baisse) = p (1 - q).

Étapes (cache dans Data/Models/nn/) :
  cond_tune       : grille évaluée sur la validation (log-loss de la tête de direction) ;
  cond_importance : importance par permutation sur la validation, réseau estimé sur
                    l'estimation initiale seulement (aucune information du test) ;
  cond_final      : meilleures configurations, 10 graines, toutes les variables et
                    variables retenues ;
  cond_logit      : mêmes probabilités avec des régressions logistiques."""
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from joblib import Parallel, delayed
from scipy.special import expit
from sklearn.linear_model import LogisticRegression

from Code.nn_experiments import INITIAL, REFIT_EVERY, TRAIN, cached, load_panel
from Function.functions_nn import (
    add_pair_onehot, fit_model, predict, refit_blocks, standardize, walk_forward_task,
)

KS = [0.5, 1.0]
N_TUNE_SEEDS, N_FINAL_SEEDS, N_PERM = 5, 10, 5
EPS = 1e-12


## ---- metrics ----
def cond_log_losses(p_big, q, z, k):
    """Log-loss de l'amplitude (toutes les obs.) et de la direction (grands mouvements)."""
    big = np.abs(z) > k
    up = z > 0
    ll_amp = -np.mean(big * np.log(p_big + EPS) + (~big) * np.log(1 - p_big + EPS))
    ll_dir = -np.mean(up[big] * np.log(q[big] + EPS) + (~up[big]) * np.log(1 - q[big] + EPS))
    return ll_amp, ll_dir


def base_rates(z, start, end, k):
    P = z.shape[1]
    p_big, q = np.zeros((end - start, P)), np.zeros((end - start, P))
    for block in refit_blocks(start, end, REFIT_EVERY):
        zz = z[:block[0]]
        big = np.abs(zz) > k
        p_big[block - start] = big.mean()
        q[block - start] = (zz[big] > 0).mean()
    return p_big, q


def to_probs(logits):
    return expit(logits[..., 0]), expit(logits[..., 1])


def ensemble(seed_logits):
    pb, q = zip(*[to_probs(v) for v in seed_logits.values()])
    return np.mean(pb, 0), np.mean(q, 0)


## ---- grid / runner ----
def grid():
    cfgs = []
    for k in KS:
        for hidden, drop, wd, lr in itertools.product([(32, 16), (64, 32, 16)], [0.1, 0.3],
                                                      [1e-4, 1e-2], [1e-3, 1e-4]):
            cfgs.append(dict(arch="mlp", loss="cond", k=k, hidden=hidden, dropout=drop,
                             weight_decay=wd, lr=lr))
        for hidden, seq, wd in itertools.product([16, 32], [4, 12], [1e-4, 1e-2]):
            cfgs.append(dict(arch="gru", loss="cond", k=k, hidden=hidden, seq_len=seq, dropout=0.2,
                             weight_decay=wd, lr=1e-3))
    return cfgs


def run(X, z, cfgs, start, end, seeds):
    blocks = refit_blocks(start, end, REFIT_EVERY)
    tasks = [(c, s, b) for c in range(len(cfgs)) for s in seeds for b in blocks]
    tasks.sort(key=lambda t: cfgs[t[0]]["arch"] != "gru")
    res = Parallel(n_jobs=-1, batch_size=4)(
        delayed(walk_forward_task)(X, z, b, cfgs[c], s) for c, s, b in tasks)
    out = {c: {s: np.zeros((end - start, X.shape[1], 2)) for s in seeds} for c in range(len(cfgs))}
    for (c, s, _), (block, p) in zip(tasks, res):
        out[c][s][block - start] = p
    return out


def skills(p_big, q, z, k, start, end, z_all):
    bp, bq = base_rates(z_all, start, end, k)
    ll = cond_log_losses(p_big, q, z, k)
    llb = cond_log_losses(bp, bq, z, k)
    return 1 - ll[0] / llb[0], 1 - ll[1] / llb[1]


## ---- 1. tune ----
def stage_tune(d):
    X, z = d["X"], d["z"]
    cfgs = grid()
    preds = run(X, z, cfgs, INITIAL, TRAIN, list(range(N_TUNE_SEEDS)))
    rows = []
    for c, cfg in enumerate(cfgs):
        pb, q = ensemble(preds[c])
        s_amp, s_dir = skills(pb, q, z[INITIAL:TRAIN], cfg["k"], INITIAL, TRAIN, z)
        rows.append(dict(cfg, skill_amp=s_amp, skill_dir=s_dir))
    return {"cfgs": cfgs, "results": rows}


def best_configs(tune):
    """Meilleure configuration par (architecture, k) selon la tête de direction."""
    best = {}
    for arch, k in itertools.product(["mlp", "gru"], KS):
        rows = [r for r in tune["results"] if r["arch"] == arch and r["k"] == k]
        r = max(rows, key=lambda r: r["skill_dir"])
        best[f"{arch}_k{k}"] = {key: v for key, v in r.items() if not key.startswith("skill")}
    return best


## ---- 2. importance ----
def _importance_seed(X, z, cfg, seed):
    k = cfg["k"]
    model, Xs, _, _ = fit_model(X, z, INITIAL, cfg, seed)
    weeks = np.arange(INITIAL, TRAIN)
    zv = z[weeks]
    base = cond_log_losses(*to_probs(predict(model, Xs, weeks)), zv, k)
    rng = np.random.default_rng(seed)
    F = X.shape[2]
    out = np.zeros((F, 2))
    for f in range(F):
        for _ in range(N_PERM):
            Xp = Xs.copy()
            col = Xp[weeks, ..., f]
            Xp[weeks, ..., f] = rng.permutation(col.reshape(-1)).reshape(col.shape)
            ll = cond_log_losses(*to_probs(predict(model, Xp, weeks)), zv, k)
            out[f] += (np.array(ll) - np.array(base)) / N_PERM  # hausse de la log-loss
    return out


def stage_importance(d, cfg):
    """Hausse de la log-loss de validation quand une variable est mélangée
    (positive = variable utile), moyenne sur 10 réseaux estimés sur les 529
    premières semaines."""
    res = Parallel(n_jobs=N_FINAL_SEEDS)(
        delayed(_importance_seed)(d["X"], d["z"], cfg, s) for s in range(N_FINAL_SEEDS))
    res = np.array(res)  # (seeds, F, 2)
    return {"cfg": cfg, "mean": res.mean(0), "se": res.std(0, ddof=1) / np.sqrt(len(res)),
            "features": list(d["features"])}


def selected_features(imp):
    """On garde une variable si elle améliore la log-loss de validation d'au moins une tête."""
    return [f for f, m in zip(imp["features"], imp["mean"]) if (m > 0).any()]


## ---- 3. final ----
def stage_final(d, best, keep):
    idx = [list(d["features"]).index(f) for f in keep]
    out = {}
    seeds = list(range(N_FINAL_SEEDS))
    for label, X in [("all", d["X"]), ("sel", d["X"][:, :, idx])]:
        cfgs = list(best.values())
        val = run(X, d["z"], cfgs, INITIAL, TRAIN, seeds)
        test = run(X, d["z"], cfgs, TRAIN, len(X), seeds)
        for i, name in enumerate(best):
            out[f"{name}_{label}"] = {"cfg": cfgs[i], "val": val[i], "test": test[i]}
    return out


## ---- 4. logistic ----
def _logit_block(X, z, block, k, C):
    Xs = add_pair_onehot(standardize(X, block[0]))
    F = Xs.shape[-1]
    Xtr, Xte = Xs[:block[0]].reshape(-1, F), Xs[block].reshape(-1, F)
    ztr = z[:block[0]].reshape(-1)
    big = np.abs(ztr) > k
    m_amp = LogisticRegression(C=C, max_iter=2000).fit(Xtr, big)
    m_dir = LogisticRegression(C=C, max_iter=2000).fit(Xtr[big], ztr[big] > 0)
    P = X.shape[1]
    return block, m_amp.predict_proba(Xte)[:, 1].reshape(len(block), P), \
        m_dir.predict_proba(Xte)[:, 1].reshape(len(block), P)


def stage_logit(d, keep):
    idx = [list(d["features"]).index(f) for f in keep]
    out = {}
    for label, X in [("all", d["X"]), ("sel", d["X"][:, :, idx])]:
        for k in KS:
            best = (None, np.inf, None)
            for C in [0.001, 0.01, 0.1, 1.0]:
                res = Parallel(n_jobs=-1)(delayed(_logit_block)(X, d["z"], b, k, C)
                                          for b in refit_blocks(INITIAL, TRAIN, REFIT_EVERY))
                pb, q = np.concatenate([r[1] for r in res]), np.concatenate([r[2] for r in res])
                ll = cond_log_losses(pb, q, d["z"][INITIAL:TRAIN], k)[1]  # tête de direction
                if ll < best[1]:
                    best = (C, ll, (pb, q))
            res = Parallel(n_jobs=-1)(delayed(_logit_block)(X, d["z"], b, k, best[0])
                                      for b in refit_blocks(TRAIN, len(X), REFIT_EVERY))
            out[f"logit_k{k}_{label}"] = {"C": best[0], "val": best[2],
                                          "test": (np.concatenate([r[1] for r in res]),
                                                   np.concatenate([r[2] for r in res]))}
    return out


def main():
    d = load_panel()
    tune = cached("cond_tune", lambda: stage_tune(d))
    best = best_configs(tune)
    for name, cfg in best.items():
        r = [x for x in tune["results"] if all(x[key] == cfg[key] for key in cfg)][0]
        print(name, cfg, {key: round(v, 4) for key, v in r.items() if key.startswith("skill")})
    # Importance mesurée sur la famille dont la tête de direction est la meilleure en validation
    top = max(best, key=lambda n: [r for r in tune["results"]
                                   if all(r[key] == best[n][key] for key in best[n])][0]["skill_dir"])
    imp = cached("cond_importance", lambda: stage_importance(d, best[top]))
    keep = selected_features(imp)
    print(f"Importance mesurée avec {top} ; {len(keep)} variables gardées sur {len(imp['features'])} :")
    print("  retirées :", [f for f in imp["features"] if f not in keep])
    cached("cond_final", lambda: stage_final(d, best, keep))
    cached("cond_logit", lambda: stage_logit(d, keep))


if __name__ == "__main__":
    main()
