"""Port de glmnet (R, famille gaussienne) et de cv.glmnet.

Même algorithme que la version C++ de glmnet (descente de coordonnées en mode
« covariance », utilisé par R quand il y a moins de 500 variables) : même
standardisation, même suite de lambdas, mêmes critères d'arrêt (thresh = 1e-7,
fdev = 1e-5, devmax = 0.999, mnlam = 5) et mêmes plis de validation croisée
(set.seed(123) de R). Compilé avec numba pour la vitesse."""
import numpy as np
from numba import njit

from Function.r_random import cv_folds

# glmnet.control() par défaut
_EPS, _BIG, _MNLAM, _FDEV, _DEVMAX = 1e-6, 9.9e35, 5, 1e-5, 0.999


@njit(cache=True)
def _update(k, full, X, a, g, xv, vp, ju, mm, ia, c, state, ab, dem, nx):
    """Mise à jour de la coordonnée k. state = [nin, dlx, rsq]."""
    ni = a.shape[0]
    old = a[k]
    gk = g[k]
    u = gk + old * xv[k]
    v = abs(u) - vp[k] * ab
    new = 0.0
    if v > 0.0:
        new = np.copysign(v, u) / (xv[k] + vp[k] * dem)
    a[k] = new
    if new == old:
        return 0
    if full and mm[k] == 0:
        nin = int(state[0]) + 1
        if nin > nx:
            return -1
        state[0] = nin
        mm[k] = nin
        ia[nin - 1] = k
        for j in range(ni):
            if not ju[j]:
                continue
            if j == k:
                c[j, nin - 1] = xv[j]
            elif mm[j] != 0:
                c[j, nin - 1] = c[k, mm[j] - 1]
            else:
                c[j, nin - 1] = np.dot(X[:, j], X[:, k])
    diff = new - old
    d2 = xv[k] * diff * diff
    if d2 > state[1]:
        state[1] = d2
    state[2] += diff * (2.0 * gk - diff * xv[k])
    col = mm[k] - 1
    if full:
        for j in range(ni):
            if ju[j]:
                g[j] -= c[j, col] * diff
    else:
        for l in range(int(state[0])):
            j = ia[l]
            g[j] -= c[j, col] * diff
    return 0


@njit(cache=True)
def _elnet_cov_path(X, g, xv, vp, ju, beta, nlam, flmin, ulam, thr, maxit, ne, nx):
    ni = X.shape[1]
    omb = 1.0 - beta
    alf = 1.0
    if flmin < 1.0:
        alf = max(_EPS, flmin) ** (1.0 / (nlam - 1.0))
    mnl = min(_MNLAM, nlam)
    a = np.zeros(ni)
    mm = np.zeros(ni, np.int64)
    ia = np.zeros(max(nx, 1), np.int64)
    c = np.zeros((ni, max(nx, 1)))
    da = np.zeros(ni)
    state = np.zeros(3)  # nin, dlx, rsq
    ao = np.zeros((ni, nlam))
    almo = np.zeros(nlam)
    lmu = 0
    nlp = 0
    iz = False
    lmda_curr = 0.0
    for m in range(nlam):
        alm0 = lmda_curr
        alm = alm0
        if flmin >= 1.0:
            alm = ulam[m]
        elif m > 1:
            alm *= alf
        elif m == 0:
            alm = _BIG
        else:
            alm0 = 0.0
            for j in range(ni):
                if ju[j] and vp[j] > 0.0:
                    alm0 = max(alm0, abs(g[j]) / vp[j])
            alm0 /= max(beta, 1e-3)
            alm = alm0 * alf
        lmda_curr = alm
        dem = alm * omb
        ab = alm * beta
        rsq_prev = state[2]

        do_partial = iz
        while True:
            if do_partial:
                # Passes sur les variables actives seulement
                nin = int(state[0])
                for l in range(nin):
                    da[l] = a[ia[l]]
                iz = True
                while True:
                    nlp += 1
                    state[1] = 0.0
                    for l in range(int(state[0])):
                        _update(ia[l], False, X, a, g, xv, vp, ju, mm, ia, c, state, ab, dem, nx)
                    if state[1] < thr:
                        break
                    if nlp > maxit:
                        return ao, almo, -1, nlp
                nin = int(state[0])
                for l in range(nin):
                    da[l] -= a[ia[l]]
                for j in range(ni):
                    if mm[j] != 0 or not ju[j]:
                        continue
                    s = 0.0
                    for l in range(nin):
                        s += da[l] * c[j, l]
                    g[j] += s
            # Passe complète sur toutes les variables
            nlp += 1
            state[1] = 0.0
            for k in range(ni):
                if ju[k]:
                    if _update(k, True, X, a, g, xv, vp, ju, mm, ia, c, state, ab, dem, nx) < 0:
                        return ao, almo, -2, nlp
            if state[1] < thr:
                break
            if nlp > maxit:
                return ao, almo, -1, nlp
            do_partial = True

        rsq = state[2]
        me = 0
        for k in range(ni):
            ao[k, m] = a[k]
            if a[k] != 0.0:
                me += 1
        almo[m] = alm
        lmu = m + 1
        if lmu < mnl or flmin >= 1.0:
            continue
        prop = np.inf if rsq == 0.0 else (rsq - rsq_prev) / rsq
        if me > ne or prop < _FDEV or rsq > _DEVMAX:
            break
    return ao, almo, lmu, nlp


def glmnet_gaussian(x, y, alpha, lambdas=None, nlambda=100, thresh=1e-7, maxit=100000):
    """glmnet(x, y, alpha = alpha, standardize = TRUE, intercept = TRUE).

    Retourne (lambda, a0, beta) avec beta de dimension (p, nombre de lambdas)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n, p = x.shape
    w = np.ones(n) / n
    v = np.sqrt(w)
    ju = np.array([np.any(x[:, j] != x[0, j]) for j in range(p)])
    xm = x.T @ w
    X = v[:, None] * (x - xm)
    xs = np.sqrt((X ** 2).sum(axis=0))
    xs[~ju] = 1.0
    X = np.asfortranarray(X / xs)
    xv = np.ones(p)
    ym = y @ w
    Y = v * (y - ym)
    ys = np.sqrt(Y @ Y)
    Y = Y / ys
    g = X.T @ Y
    vp = np.ones(p)
    ne = p + 1
    nx = min(ne * 2 + 20, p)
    if lambdas is None:
        flmin = 1e-4 if n > p else 1e-2
        ulam = np.zeros(1)
        nlam = nlambda
    else:
        flmin = 1.0
        ulam = np.sort(np.asarray(lambdas, dtype=float))[::-1] / ys
        nlam = len(ulam)
    ao, almo, lmu, _ = _elnet_cov_path(X, g, xv, vp, ju, float(alpha), nlam, flmin,
                                       np.ascontiguousarray(ulam), thresh, maxit, ne, nx)
    if lmu < 0:
        raise RuntimeError("glmnet: convergence failure")
    lam = almo[:lmu] * ys
    beta = ao[:, :lmu] * (ys / xs)[:, None]
    a0 = ym - xm @ beta
    if lambdas is None and len(lam) > 2:
        # fix.lam : premier lambda extrapolé (sinon il vaudrait « big »)
        llam = np.log(lam)
        lam[0] = np.exp(2 * llam[1] - llam[2])
    return lam, a0, beta


def lambda_interp(lam, s):
    """lambda.interp de glmnet : interpolation linéaire des coefficients en lambda.
    Retourne (left, right, frac), indices base 0."""
    s = np.atleast_1d(np.asarray(s, dtype=float))
    k = len(lam)
    if k == 1:
        z = np.zeros(len(s), dtype=int)
        return z, z, np.ones(len(s))
    sfrac = (lam[0] - s) / (lam[0] - lam[-1])
    lnorm = (lam[0] - lam) / (lam[0] - lam[-1])
    sfrac = np.clip(sfrac, lnorm.min(), lnorm.max())
    coord = np.interp(sfrac, lnorm, np.arange(1, k + 1))
    left = np.floor(coord).astype(int) - 1
    right = np.ceil(coord).astype(int) - 1
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = (sfrac - lnorm[right]) / (lnorm[left] - lnorm[right])
    frac[left == right] = 1.0
    frac[np.abs(lnorm[left] - lnorm[right]) < np.finfo(float).eps] = 1.0
    return left, right, frac


def predict_glmnet(lam, a0, beta, newx, s):
    """predict.glmnet(fit, newx, s) : coefficients interpolés aux lambdas s."""
    left, right, frac = lambda_interp(lam, s)
    b = beta[:, left] * frac + beta[:, right] * (1 - frac)
    b0 = a0[left] * frac + a0[right] * (1 - frac)
    return b0 + newx @ b


def cv_glmnet(x, y, alpha, nfolds=5, seed=123):
    """cv.glmnet(x, y, alpha, nfolds) après set.seed(seed), type.measure = "mse".

    Comme dans R, chaque pli calcule son propre chemin de lambdas ; ses
    prévisions sont interpolées aux lambdas du modèle complet.
    Retourne (a0, beta) au lambda.min, ainsi que le chemin complet."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    foldid = cv_folds(len(y), nfolds, seed)
    lam, a0, beta = glmnet_gaussian(x, y, alpha)
    cvm = np.zeros(len(lam))
    for i in range(1, nfolds + 1):
        test = foldid == i
        lam_i, a0_i, beta_i = glmnet_gaussian(x[~test], y[~test], alpha)
        pred = predict_glmnet(lam_i, a0_i, beta_i, x[test], lam)
        cvm += ((y[test, None] - pred) ** 2).mean(axis=0) * test.sum()
    cvm /= len(y)
    # lambda.min : le plus grand lambda qui atteint le minimum
    idx = int(np.flatnonzero(cvm <= cvm.min())[0])
    return a0[idx], beta[:, idx], {"lambda": lam, "cvm": cvm, "index": idx}
