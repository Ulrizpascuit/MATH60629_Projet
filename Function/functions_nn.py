"""Réseaux de neurones pour classer les 5 paires de devises chaque semaine.

Deux architectures partagées entre les paires (un seul jeu de poids, la paire
est donnée en entrée par un encodage one-hot) :
  * MLP : features de la semaine t ;
  * GRU : séquence des L dernières semaines de features de la paire.
Deux fonctions de perte :
  * "mse"  : erreur quadratique sur la cible z = r(t+1) / vol_12(t) ;
  * "rank" : perte de classement par liste (ListNet) bidirectionnelle, qui
    compare la distribution softmax des scores des 5 paires à celle des cibles,
    pour la meilleure paire (softmax de z) et la pire (softmax de -z).
Chaque modèle est réestimé en fenêtre croissante, avec arrêt précoce sur les
20 % de semaines les plus récentes de la fenêtre d'entraînement."""
import numpy as np
import torch
from torch import nn

N_PAIRS = 5


## ---- preprocessing ----
def standardize(X, train_weeks):
    """Centre-réduit chaque feature avec les semaines d'entraînement seulement,
    puis limite à +/- 5 écarts-types."""
    flat = X[:train_weeks].reshape(-1, X.shape[-1])
    mu, sd = flat.mean(axis=0), flat.std(axis=0)
    sd[sd == 0] = 1.0
    return np.clip((X - mu) / sd, -5, 5).astype(np.float32)


def add_pair_onehot(X):
    T, P, _ = X.shape
    eye = np.broadcast_to(np.eye(P, dtype=np.float32), (T, P, P))
    return np.concatenate([X, eye], axis=-1)


def sequences(X, seq_len):
    """(T, P, F) -> (T, P, L, F) : les L dernières semaines (répète la première au début)."""
    T = X.shape[0]
    idx = np.arange(T)[:, None] - np.arange(seq_len - 1, -1, -1)[None, :]
    return X[np.clip(idx, 0, None)].transpose(0, 2, 1, 3)


## ---- models ----
class MLP(nn.Module):
    def __init__(self, n_in, hidden, dropout, n_out=1):
        super().__init__()
        self.n_out = n_out
        layers, d = [], n_in
        for h in hidden:
            layers += [nn.Linear(d, h), nn.SiLU(), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, n_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x):  # x : (B, P, F) -> (B, P) ou (B, P, n_out)
        out = self.net(x)
        return out.squeeze(-1) if self.n_out == 1 else out


class GRUNet(nn.Module):
    def __init__(self, n_in, hidden, dropout, n_out=1):
        super().__init__()
        self.n_out = n_out
        self.gru = nn.GRU(n_in, hidden, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(hidden, n_out)

    def forward(self, x):  # x : (B, P, L, F)
        B, P, L, F = x.shape
        _, h = self.gru(x.reshape(B * P, L, F))
        out = self.out(self.drop(h[-1])).reshape(B, P, self.n_out)
        return out.squeeze(-1) if self.n_out == 1 else out


## ---- losses ----
def rank_loss(scores, z):
    """ListNet « top-1 » dans les deux sens : meilleure paire et pire paire."""
    top = -(torch.softmax(z, -1) * torch.log_softmax(scores, -1)).sum(-1)
    bottom = -(torch.softmax(-z, -1) * torch.log_softmax(-scores, -1)).sum(-1)
    return (top + bottom).mean()


def mse_loss(scores, z):
    return ((scores - z) ** 2).mean()


def class_labels(z, k):
    """0 = forte baisse (z < -k), 1 = neutre, 2 = forte hausse (z > k)."""
    return (z > k).long() * 2 + ((z >= -k) & (z <= k)).long()


def cls_loss(out, z, k):
    """Deux têtes : direction en 3 classes (entropie croisée) et amplitude
    |z| > k (entropie croisée binaire). out : (B, P, 4) = 3 logits + 1 logit."""
    direction = nn.functional.cross_entropy(out[..., :3].reshape(-1, 3), class_labels(z, k).reshape(-1))
    big = (z.abs() > k).float()
    magnitude = nn.functional.binary_cross_entropy_with_logits(out[..., 3], big)
    return direction + magnitude


LOSSES = {"mse": mse_loss, "rank": rank_loss}


## ---- training ----
def prepare_inputs(X, train_weeks, cfg):
    """Entrées du réseau : standardisation (entraînement seulement), one-hot de
    la paire et, pour le GRU, séquences des seq_len dernières semaines."""
    Xs = add_pair_onehot(standardize(X, train_weeks))
    return sequences(Xs, cfg["seq_len"]) if cfg["arch"] == "gru" else Xs


def fit_model(X, z, train_weeks, cfg, seed, max_epochs=150, patience=15, batch_weeks=64):
    """Entraîne un réseau sur les semaines [0, train_weeks).

    X : (T, P, F) features brutes ; z : (T, P) cibles (connues jusqu'à train_weeks - 1).
    cfg : {"arch": "mlp"|"gru", "loss": "mse"|"rank", "hidden", "dropout",
           "weight_decay", "lr", "seq_len" (GRU)}.
    Retourne (modèle, entrées préparées, historique des pertes, meilleure époque)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(1)
    Xs = prepare_inputs(X, train_weeks, cfg)
    n_out = 4 if cfg["loss"] == "cls" else 1
    if cfg["arch"] == "gru":
        model = GRUNet(Xs.shape[-1], cfg["hidden"], cfg["dropout"], n_out)
    else:
        model = MLP(Xs.shape[-1], cfg["hidden"], cfg["dropout"], n_out)
    # Cible limitée à +/- 4 pour l'entraînement (quelques semaines extrêmes)
    zt = np.clip(z[:train_weeks], -4, 4).astype(np.float32)
    n_fit = int(round(0.8 * train_weeks))  # arrêt précoce : 20 % les plus récentes
    Xf, zf = torch.from_numpy(Xs[:n_fit]), torch.from_numpy(zt[:n_fit])
    Xv, zv = torch.from_numpy(Xs[n_fit:train_weeks]), torch.from_numpy(zt[n_fit:train_weeks])
    if cfg["loss"] == "cls":
        k = cfg["k"]
        loss_fn = lambda out, target, k=k: cls_loss(out, target, k)  # noqa: E731
        # Biais de sortie initialisés aux fréquences de base de l'entraînement :
        # le réseau part du prédicteur « taux de base »
        zfit = zt[:n_fit]
        freq = np.array([(zfit < -k).mean(), (np.abs(zfit) <= k).mean(), (zfit > k).mean()])
        last = model.out if cfg["arch"] == "gru" else model.net[-1]
        with torch.no_grad():
            last.bias[:3] = torch.tensor(np.log(freq), dtype=torch.float32)
            p_big = 1 - freq[1]
            last.bias[3] = float(np.log(p_big / (1 - p_big)))
    else:
        loss_fn = LOSSES[cfg["loss"]]
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.get("lr", 1e-3),
                            weight_decay=cfg["weight_decay"])
    gen = torch.Generator().manual_seed(seed)

    best, best_state, best_epoch, wait, history = np.inf, None, 0, 0, []
    for epoch in range(max_epochs):
        model.train()
        perm = torch.randperm(n_fit, generator=gen)
        train_loss = 0.0
        for start in range(0, n_fit, batch_weeks):
            b = perm[start:start + batch_weeks]
            opt.zero_grad()
            loss = loss_fn(model(Xf[b]), zf[b])
            loss.backward()
            opt.step()
            train_loss += loss.item() * len(b)
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(Xv), zv).item()
        history.append((train_loss / n_fit, val_loss))
        if val_loss < best - 1e-6:
            best, best_epoch, wait = val_loss, epoch, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, Xs, np.array(history), best_epoch


def predict(model, Xs, weeks):
    with torch.no_grad():
        return model(torch.from_numpy(np.ascontiguousarray(Xs[weeks]))).numpy()


def train_predict(X, z, train_weeks, predict_weeks, cfg, seed):
    model, Xs, _, _ = fit_model(X, z, train_weeks, cfg, seed)
    return predict(model, Xs, predict_weeks)


def refit_blocks(start, end, every):
    """Blocs de semaines [start, end) : réestimation au début de chaque bloc."""
    return [np.arange(s, min(s + every, end)) for s in range(start, end, every)]


def walk_forward_task(X, z, block, cfg, seed):
    """Une réestimation : entraînement sur les semaines < block[0], prévision du bloc."""
    return block, train_predict(X, z, int(block[0]), block, cfg, seed)
