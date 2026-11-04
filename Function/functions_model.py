"""Modèles de prévision des returns, walk-forward en fenêtre croissante et GARCH(1,1)."""
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from numba import njit
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestRegressor

from Function.glmnet import cv_glmnet

MODEL_TYPES = ["Naive", "Linear", "Stepwise", "Lasso", "Ridge", "Elastic_Net",
               "Random_Forest", "PCA"]


## ---- regression_metrics ----
def regression_metrics(actual, predicted):
    actual, predicted = np.asarray(actual, float), np.asarray(predicted, float)
    err = actual - predicted
    return {
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mae": float(np.mean(np.abs(err))),
        "r2": float(1 - np.sum(err ** 2) / np.sum((actual - actual.mean()) ** 2)),
    }


## ---- ols ----
def ols_fit(X, y):
    """lm(y ~ X) : coefficients (intercept en premier)."""
    A = np.column_stack([np.ones(len(y)), X])
    return np.linalg.lstsq(A, y, rcond=None)[0]


def ols_predict(coef, X):
    return coef[0] + np.atleast_2d(X) @ coef[1:]


## ---- forward_stepwise_path ----
@njit(cache=True)
def _forward_path(Z, y):
    """Forward stepwise : part du modèle nul et ajoute, à chaque étape, la
    variable qui donne le plus petit RSS. Orthogonalisation de Gram-Schmidt
    (même RSS que lm, beaucoup plus vite). Retourne l'ordre d'entrée et le RSS
    de M0, M1, ..., Mp."""
    n, p = Z.shape
    r = y - y.mean()
    order = np.empty(p, np.int64)
    rss = np.empty(p + 1)
    rss[0] = r @ r
    used = np.zeros(p, np.bool_)
    for k in range(p):
        best, best_rss = -1, np.inf
        for j in range(p):
            if used[j]:
                continue
            zz = Z[:, j] @ Z[:, j]
            if zz <= 1e-300:
                continue
            zr = Z[:, j] @ r
            cand = rss[k] - zr * zr / zz
            if cand < best_rss:
                best, best_rss = j, cand
        used[best] = True
        order[k] = best
        q = Z[:, best] / np.sqrt(Z[:, best] @ Z[:, best])
        r = r - (q @ r) * q
        rss[k + 1] = r @ r
        for j in range(p):
            if not used[j]:
                Z[:, j] = Z[:, j] - (q @ Z[:, j]) * q
    return order, rss


def select_stepwise_model(X, y, criterion="BIC"):
    """Meilleur modèle de chaque taille (M0..Mp), puis choix selon le BIC
    (BIC de lm : -2 logLik + log(n) * (nombre de coefficients + 1))."""
    n, p = X.shape
    # Colonnes centrées (orthogonales à l'intercept), en ordre Fortran pour numba
    Z = np.asfortranarray(X - X.mean(axis=0))
    order, rss = _forward_path(Z, np.asarray(y, float))
    k = np.arange(p + 1)
    loglik = -0.5 * n * (np.log(2 * np.pi) + 1 - np.log(n) + np.log(rss))
    n_par = k + 2  # intercept + k variables + sigma
    score = -2 * loglik + (np.log(n) if criterion == "BIC" else 2.0) * n_par
    best = int(np.argmin(score))
    return list(order[:best])


## ---- pca ----
def pca_regression(X_train, y_train, X_test):
    """prcomp(center, scale.) sur l'entraînement, composantes de valeur propre > 1."""
    center = X_train.mean(axis=0)
    scale = X_train.std(axis=0, ddof=1)
    Z = (X_train - center) / scale
    _, d, vt = np.linalg.svd(Z, full_matrices=False)
    eig = d ** 2 / (len(Z) - 1)
    k = max(1, int(np.sum(eig > 1)))
    rot = vt[:k].T
    coef = ols_fit(Z @ rot, y_train)
    return ols_predict(coef, ((X_test - center) / scale) @ rot)


## ---- fit_predict_model ----
def fit_predict_model(X_train, y_train, X_test, model_type, columns, nfolds=5, num_trees=500):
    """Estime une méthode sur l'entraînement et prédit la semaine de test."""
    if model_type == "Naive":
        # Return de la semaine qui vient de se terminer
        return float(X_test[0, columns.index("lag_return")])
    if model_type == "Linear":
        return float(ols_predict(ols_fit(X_train, y_train), X_test)[0])
    if model_type == "Stepwise":
        selected = select_stepwise_model(X_train, y_train)
        coef = ols_fit(X_train[:, selected], y_train)
        return float(ols_predict(coef, X_test[:, selected])[0])
    if model_type in ("Lasso", "Ridge", "Elastic_Net"):
        alpha = {"Lasso": 1.0, "Ridge": 0.0, "Elastic_Net": 0.5}[model_type]
        a0, beta, _ = cv_glmnet(X_train, y_train, alpha, nfolds=nfolds, seed=123)
        return float(a0 + X_test[0] @ beta)
    if model_type == "Random_Forest":
        # Mêmes réglages que ranger : 500 arbres, mtry = floor(sqrt(p)),
        # nœuds de 5 observations ou moins non divisés, tirage avec remise
        model = RandomForestRegressor(
            n_estimators=num_trees, max_features=int(np.floor(np.sqrt(X_train.shape[1]))),
            min_samples_split=6, bootstrap=True, random_state=123, n_jobs=1,
        )
        model.fit(X_train, y_train)
        return float(model.predict(X_test)[0])
    if model_type == "PCA":
        return float(pca_regression(X_train, y_train, X_test)[0])
    raise ValueError(f"Méthode inconnue : {model_type}")


## ---- model_data ----
def model_data(features, target):
    """target (déjà décalé : r(t -> t+1)) + features, lignes complètes seulement."""
    data = pd.concat([target.rename("target"), features], axis=1, join="inner").dropna()
    return data


def forecast_rows(n, period, train_ratio, initial_train_ratio=0.60):
    """Semaines à prédire (indices base 0).
    test : après les premiers train_ratio ; validation : fin de l'entraînement,
    après les premiers initial_train_ratio (le test n'est jamais utilisé)."""
    train_size = int(np.floor(train_ratio * n))
    if period == "test":
        return np.arange(train_size, n)
    return np.arange(int(np.floor(initial_train_ratio * train_size)), train_size)


## ---- build_walk_forward_model ----
def build_walk_forward_model(features, target, model_type, period, train_ratio):
    """Walk-forward : la méthode est ré-estimée chaque semaine sur les semaines
    1 à i-1, puis prédit la semaine i."""
    data = model_data(features, target)
    y = data["target"].values
    X = np.ascontiguousarray(data.drop(columns="target").values)
    columns = list(data.columns[1:])
    rows = forecast_rows(len(data), period, train_ratio)
    predictions = np.array([
        fit_predict_model(X[:i], y[:i], X[i:i + 1], model_type, columns) for i in rows
    ])
    actual = y[rows]
    return {"model_type": model_type, "predictions": predictions, "actual": actual,
            "dates": data.index[rows], **regression_metrics(actual, predictions)}


def build_walk_forward_models(features_by_pair, targets, period, train_ratio,
                              model_types=MODEL_TYPES, n_jobs=-1):
    """Toutes les méthodes x paires, en parallèle (une tâche par méthode x paire).
    Résultats identiques avec ou sans parallélisme (graines fixées)."""
    tasks = [(m, pair) for m in model_types for pair in features_by_pair]
    # Les tâches lentes d'abord (meilleur équilibrage entre processus)
    slow = {"Random_Forest": 0, "Lasso": 1, "Elastic_Net": 1, "Ridge": 1}
    tasks.sort(key=lambda t: slow.get(t[0], 2))
    results = Parallel(n_jobs=n_jobs)(
        delayed(build_walk_forward_model)(
            features_by_pair[pair], targets[pair], m, period, train_ratio)
        for m, pair in tasks
    )
    out = {m: {} for m in model_types}
    for (m, pair), res in zip(tasks, results):
        out[m][pair] = res
    return {m: {pair: out[m][pair] for pair in features_by_pair} for m in model_types}


def build_mixt_models(models, best_models):
    """Modèle mixte : pour chaque paire, la méthode choisie parmi models."""
    mixt = {}
    for pair, selected in best_models.items():
        mixt[pair] = dict(models[selected][pair], selected_model=selected)
    return mixt


## ---- garch ----
@njit(cache=True)
def _garch_nll(theta, x):
    """-log-vraisemblance du GARCH(1,1) normal et son gradient.
    Initialisation de rugarch : h_1 = moyenne des résidus au carré."""
    mu, omega, alpha, beta = theta[0], theta[1], theta[2], theta[3]
    n = len(x)
    e = x - mu
    h = np.mean(e * e)
    dh = np.zeros(4)
    dh[0] = -2.0 * np.mean(e)
    ll = 0.0
    grad = np.zeros(4)
    for t in range(n):
        if t > 0:
            ep = e[t - 1]
            d0 = -2.0 * alpha * ep + beta * dh[0]
            d1 = 1.0 + beta * dh[1]
            d2 = ep * ep + beta * dh[2]
            d3 = h + beta * dh[3]
            h = omega + alpha * ep * ep + beta * h
            dh[0], dh[1], dh[2], dh[3] = d0, d1, d2, d3
        if h <= 0.0:
            return 1e100, grad
        et = e[t]
        ll += np.log(2 * np.pi) + np.log(h) + et * et / h
        c = 1.0 / h - et * et / (h * h)
        for k in range(4):
            grad[k] += c * dh[k]
        grad[0] += -2.0 * et / h
    return 0.5 * ll, 0.5 * grad


@njit(cache=True)
def garch_filter(theta, x):
    """Variance conditionnelle filtrée de la dernière observation."""
    mu, omega, alpha, beta = theta[0], theta[1], theta[2], theta[3]
    e = x - mu
    h = np.mean(e * e)
    for t in range(1, len(x)):
        h = omega + alpha * e[t - 1] * e[t - 1] + beta * h
    return h


def fit_garch(x, start=None):
    """MLE du GARCH(1,1) (mêmes contraintes que rugarch : alpha + beta < 1).
    Plusieurs points de départ la première fois, puis départ à la semaine d'avant."""
    var = np.var(x)
    eps = 1e-8
    bounds = [(-100 * abs(np.mean(x)) - 1e-6, 100 * abs(np.mean(x)) + 1e-6),
              (eps, 100 * var), (eps, 1 - eps), (eps, 1 - eps)]
    starts = [start] if start is not None else [
        np.array([np.mean(x), var * (1 - a - b), a, b])
        for a, b in [(0.05, 0.90), (0.10, 0.80), (0.10, 0.88), (0.03, 0.95), (0.20, 0.70)]
    ]
    cons = [{"type": "ineq", "fun": lambda th: 1 - eps - th[2] - th[3],
             "jac": lambda th: np.array([0.0, 0.0, -1.0, -1.0])}]
    best = None
    for s in starts:
        res = minimize(_garch_nll, s, args=(x,), jac=True, method="L-BFGS-B", bounds=bounds,
                       options={"ftol": 1e-15, "gtol": 1e-10, "maxiter": 2000})
        if res.x[2] + res.x[3] >= 1 - eps:
            res = minimize(_garch_nll, res.x, args=(x,), jac=True, method="SLSQP",
                           bounds=bounds, constraints=cons,
                           options={"ftol": 1e-15, "maxiter": 2000})
        if best is None or res.fun < best.fun:
            best = res
    ok = np.isfinite(best.fun) and best.x[2] + best.x[3] < 1
    return best.x, ok


def build_walk_forward_garch_model(target, realized_vol, dates):
    """GARCH(1,1) ré-estimé chaque semaine sur tous les returns connus à la date
    du forecast (fenêtre croissante). Si une estimation échoue, on garde les
    paramètres de la semaine d'avant."""
    returns = target.dropna()
    return_dates = returns.index.values
    x_all = returns.values * 100
    n = len(dates)
    sigma = np.zeros(n)
    params = np.full((n, 4), np.nan)
    failed = np.zeros(n, bool)
    theta = None
    for k, date in enumerate(dates):
        # target(t) = return(t -> t+1) n'est connu qu'à t+1 : strictement avant la date prédite
        last = int(np.sum(return_dates < np.datetime64(date)))
        train = x_all[:last]
        new_theta, ok = fit_garch(train, start=theta)
        if ok:
            theta = new_theta
        else:
            if theta is None:
                raise RuntimeError("GARCH(1,1) did not converge.")
            failed[k] = True
        last_sigma2 = garch_filter(theta, train)
        resid = train[-1] - theta[0]
        next_sigma2 = theta[1] + theta[2] * resid ** 2 + theta[3] * last_sigma2
        sigma[k] = np.sqrt(next_sigma2) / 100
        params[k] = theta
    actual = realized_vol.reindex(dates).values
    assert not np.isnan(actual).any()
    return {"predictions": sigma, "actual": actual, "dates": dates,
            "params": pd.DataFrame(params, index=dates, columns=["mu", "omega", "alpha", "beta"]),
            "refit_failed": failed, **regression_metrics(actual, sigma)}


def build_walk_forward_garch_models(targets, realized_vols, models, n_jobs=-1):
    pairs = list(models)
    results = Parallel(n_jobs=n_jobs)(
        delayed(build_walk_forward_garch_model)(targets[p], realized_vols[p], models[p]["dates"])
        for p in pairs
    )
    return dict(zip(pairs, results))


## ---- build_sharpe_model ----
def build_sharpe_model(model, garch_model):
    """Sharpe prévu = return prévu / volatilité prévue (GARCH) ;
    réel = return réalisé / volatilité réalisée."""
    assert model["dates"].equals(garch_model["dates"])
    predictions = model["predictions"] / garch_model["predictions"]
    actual = model["actual"] / garch_model["actual"]
    return {"predictions": predictions, "actual": actual, "dates": model["dates"],
            **regression_metrics(actual, predictions),
            "return_predictions": model["predictions"], "return_actual": model["actual"],
            "return_rmse": model["rmse"],
            "vol_predictions": garch_model["predictions"], "vol_actual": garch_model["actual"]}


def build_sharpe_models(models, garch_models):
    return {pair: build_sharpe_model(models[pair], garch_models[pair]) for pair in models}


## ---- metric_table ----
def metric_table(groups, metric="rmse"):
    """Une ligne par paire, une colonne par méthode."""
    first = next(iter(groups.values()))
    out = pd.DataFrame({"Currency": list(first)})
    for name, models in groups.items():
        out[f"{name}_{metric.upper()}"] = [models[p][metric] for p in first]
    return out
