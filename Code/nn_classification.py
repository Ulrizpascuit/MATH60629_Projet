"""Classification des grands mouvements avec un réseau à deux têtes.

Pour chaque (semaine, paire), avec z = r(t+1) / vol_12(t) et un seuil k :
  * tête « direction » : 3 classes, forte baisse (z < -k), neutre, forte hausse (z > k) ;
  * tête « amplitude » : grand mouvement dans un sens ou dans l'autre (|z| > k).
Décision : acheter la paire de plus grande P(forte hausse), vendre celle de plus
grande P(forte baisse) (paires différentes), avec une variante qui s'abstient
quand la probabilité est faible.

Étapes (cache dans Data/Models/nn/) :
  cls_tune  : grille évaluée sur la validation (log-loss), k = 0,5 et 1 ;
  cls_final : meilleure configuration par (architecture, k), 10 graines ;
  cls_logit : régressions logistiques sur les mêmes entrées (référence linéaire)."""
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from joblib import Parallel, delayed
from scipy.special import expit, softmax
from sklearn.linear_model import LogisticRegression

from Code.nn_experiments import INITIAL, REFIT_EVERY, TRAIN, cached, load_panel
from Function.functions_nn import add_pair_onehot, refit_blocks, standardize, walk_forward_task

KS = [0.5, 1.0]
N_TUNE_SEEDS, N_FINAL_SEEDS = 5, 10


def labels(z, k):
    return np.where(z > k, 2, np.where(z < -k, 0, 1)), (np.abs(z) > k).astype(int)


def to_probs(logits):
    """(…, 4) logits -> probabilités (…, 3) de direction et (…) d'amplitude."""
    return softmax(logits[..., :3], -1), expit(logits[..., 3])


def ensemble_probs(seed_logits):
    pd_, pm = zip(*[to_probs(v) for v in seed_logits.values()])
    return np.mean(pd_, 0), np.mean(pm, 0)


def log_losses(p_dir, p_mag, z, k):
    y_dir, y_mag = labels(z, k)
    eps = 1e-12
    ll_dir = -np.mean(np.log(np.take_along_axis(p_dir, y_dir[..., None], -1)[..., 0] + eps))
    ll_mag = -np.mean(y_mag * np.log(p_mag + eps) + (1 - y_mag) * np.log(1 - p_mag + eps))
    return ll_dir, ll_mag


def base_rates(z, start, end, k):
    """Fréquences de base en fenêtre croissante (même calendrier de réestimation)."""
    P = z.shape[1]
    p_dir = np.zeros((end - start, P, 3))
    p_mag = np.zeros((end - start, P))
    for block in refit_blocks(start, end, REFIT_EVERY):
        y_dir, y_mag = labels(z[:block[0]], k)
        p_dir[block - start] = np.bincount(y_dir.ravel(), minlength=3) / y_dir.size
        p_mag[block - start] = y_mag.mean()
    return p_dir, p_mag


def grid():
    cfgs = []
    for k in KS:
        for hidden, drop, wd, lr in itertools.product([(32, 16), (64, 32, 16)], [0.1, 0.3],
                                                      [1e-4, 1e-2], [1e-3, 1e-4]):
            cfgs.append(dict(arch="mlp", loss="cls", k=k, hidden=hidden, dropout=drop,
                             weight_decay=wd, lr=lr))
        for hidden, seq, wd in itertools.product([16, 32], [4, 12], [1e-4, 1e-2]):
            cfgs.append(dict(arch="gru", loss="cls", k=k, hidden=hidden, seq_len=seq, dropout=0.2,
                             weight_decay=wd, lr=1e-3))
    return cfgs


def run(X, z, cfgs, start, end, seeds):
    blocks = refit_blocks(start, end, REFIT_EVERY)
    tasks = [(c, s, b) for c in range(len(cfgs)) for s in seeds for b in blocks]
    tasks.sort(key=lambda t: cfgs[t[0]]["arch"] != "gru")
    res = Parallel(n_jobs=-1, batch_size=4)(
        delayed(walk_forward_task)(X, z, b, cfgs[c], s) for c, s, b in tasks)
    out = {c: {s: np.zeros((end - start, X.shape[1], 4)) for s in seeds} for c in range(len(cfgs))}
    for (c, s, _), (block, p) in zip(tasks, res):
        out[c][s][block - start] = p
    return out


def stage_tune(d):
    X, z = d["X"], d["z"]
    cfgs = grid()
    preds = run(X, z, cfgs, INITIAL, TRAIN, list(range(N_TUNE_SEEDS)))
    zv = z[INITIAL:TRAIN]
    rows = []
    for c, cfg in enumerate(cfgs):
        p_dir, p_mag = ensemble_probs(preds[c])
        b_dir, b_mag = base_rates(z, INITIAL, TRAIN, cfg["k"])
        ll = log_losses(p_dir, p_mag, zv, cfg["k"])
        llb = log_losses(b_dir, b_mag, zv, cfg["k"])
        rows.append(dict(cfg, skill_dir=1 - ll[0] / llb[0], skill_mag=1 - ll[1] / llb[1],
                         skill=1 - (ll[0] + ll[1]) / (llb[0] + llb[1])))
    return {"cfgs": cfgs, "results": rows}


def best_configs(tune):
    best = {}
    for arch, k in itertools.product(["mlp", "gru"], KS):
        rows = [r for r in tune["results"] if r["arch"] == arch and r["k"] == k]
        r = max(rows, key=lambda r: r["skill"])
        best[f"{arch}_k{k}"] = {key: v for key, v in r.items() if not key.startswith("skill")}
    return best


def stage_final(d, best):
    cfgs = list(best.values())
    seeds = list(range(N_FINAL_SEEDS))
    val = run(d["X"], d["z"], cfgs, INITIAL, TRAIN, seeds)
    test = run(d["X"], d["z"], cfgs, TRAIN, len(d["X"]), seeds)
    return {name: {"cfg": cfgs[i], "val": val[i], "test": test[i]} for i, name in enumerate(best)}


def _logit_block(X, z, block, k, C):
    Xs = add_pair_onehot(standardize(X, block[0]))
    F = Xs.shape[-1]
    Xtr, Xte = Xs[:block[0]].reshape(-1, F), Xs[block].reshape(-1, F)
    y_dir, y_mag = labels(z[:block[0]], k)
    m_dir = LogisticRegression(C=C, max_iter=2000).fit(Xtr, y_dir.ravel())
    m_mag = LogisticRegression(C=C, max_iter=2000).fit(Xtr, y_mag.ravel())
    P = X.shape[1]
    return block, m_dir.predict_proba(Xte).reshape(len(block), P, 3), \
        m_mag.predict_proba(Xte)[:, 1].reshape(len(block), P)


def stage_logit(d):
    """Régressions logistiques (multinomiale et binaire) ; C choisi sur la validation."""
    X, z = d["X"], d["z"]
    out = {}
    for k in KS:
        best_C, best_ll, res_val = None, np.inf, None
        for C in [0.001, 0.01, 0.1, 1.0]:
            res = Parallel(n_jobs=-1)(delayed(_logit_block)(X, z, b, k, C)
                                      for b in refit_blocks(INITIAL, TRAIN, REFIT_EVERY))
            p_dir = np.concatenate([r[1] for r in res])
            p_mag = np.concatenate([r[2] for r in res])
            ll = sum(log_losses(p_dir, p_mag, z[INITIAL:TRAIN], k))
            if ll < best_ll:
                best_C, best_ll, res_val = C, ll, (p_dir, p_mag)
        res = Parallel(n_jobs=-1)(delayed(_logit_block)(X, z, b, k, best_C)
                                  for b in refit_blocks(TRAIN, len(X), REFIT_EVERY))
        out[f"logit_k{k}"] = {"C": best_C, "val": res_val,
                              "test": (np.concatenate([r[1] for r in res]),
                                       np.concatenate([r[2] for r in res]))}
    return out


def main():
    d = load_panel()
    tune = cached("cls_tune", lambda: stage_tune(d))
    best = best_configs(tune)
    for name, cfg in best.items():
        r = [x for x in tune["results"] if all(x[key] == cfg[key] for key in cfg)][0]
        print(name, cfg, {key: round(v, 4) for key, v in r.items() if key.startswith("skill")})
    cached("cls_final", lambda: stage_final(d, best))
    cached("cls_logit", lambda: stage_logit(d))


if __name__ == "__main__":
    main()
