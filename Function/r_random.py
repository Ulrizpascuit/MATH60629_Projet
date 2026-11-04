"""Générateur aléatoire de R (Mersenne-Twister, sample.kind = "Rejection").

Reproduit set.seed(), runif() et sample() de R au bit près : les plis de
validation croisée de glmnet et les stratégies au hasard sont donc les mêmes
qu'en R avec set.seed(123)."""
import math

import numpy as np

_N, _M = 624, 397
_UPPER, _LOWER = 0x80000000, 0x7FFFFFFF
_MAG01 = (0, 0x9908B0DF)
_I2_32M1 = 2.328306437080797e-10  # 1 / (2^32 - 1)


class RRandom:
    def __init__(self, seed):
        self.set_seed(seed)

    def set_seed(self, seed):
        # Initial scrambling (RNG.c, Randomize -> Init_RNG)
        s = seed & 0xFFFFFFFF
        for _ in range(50):
            s = (69069 * s + 1) & 0xFFFFFFFF
        dummy = []
        for _ in range(_N + 1):
            s = (69069 * s + 1) & 0xFFFFFFFF
            dummy.append(s)
        self.mti = _N  # FixupSeeds : dummy[0] = mti = N
        self.mt = dummy[1:]

    def _genrand(self):
        mt = self.mt
        if self.mti >= _N:
            for kk in range(_N - _M):
                y = (mt[kk] & _UPPER) | (mt[kk + 1] & _LOWER)
                mt[kk] = mt[kk + _M] ^ (y >> 1) ^ _MAG01[y & 1]
            for kk in range(_N - _M, _N - 1):
                y = (mt[kk] & _UPPER) | (mt[kk + 1] & _LOWER)
                mt[kk] = mt[kk + (_M - _N)] ^ (y >> 1) ^ _MAG01[y & 1]
            y = (mt[_N - 1] & _UPPER) | (mt[0] & _LOWER)
            mt[_N - 1] = mt[_M - 1] ^ (y >> 1) ^ _MAG01[y & 1]
            self.mti = 0
        y = mt[self.mti]
        self.mti += 1
        y ^= y >> 11
        y ^= (y << 7) & 0x9D2C5680
        y ^= (y << 15) & 0xEFC60000
        y ^= y >> 18
        return y * 2.3283064365386963e-10

    def unif_rand(self):
        value = self._genrand()
        if value <= 0.0:
            return 0.5 * _I2_32M1
        if 1.0 - value <= 0.0:
            return 1.0 - 0.5 * _I2_32M1
        return value

    def runif(self, n=1, low=0.0, high=1.0):
        return np.array([low + (high - low) * self.unif_rand() for _ in range(n)])

    def _rbits(self, bits):
        v = 0
        for _ in range(0, bits + 1, 16):
            v1 = int(math.floor(self.unif_rand() * 65536))
            v = 65536 * v + v1
        return v & ((1 << bits) - 1)

    def unif_index(self, dn):
        """R_unif_index : entier uniforme dans [0, dn) par rejet."""
        if dn <= 0:
            return 0
        bits = int(math.ceil(math.log2(dn)))
        while True:
            dv = self._rbits(bits)
            if dn > dv:
                return dv

    def sample_int(self, n, size=None):
        """sample.int(n, size) sans remise ; renvoie des entiers de 1 à n."""
        size = n if size is None else size
        if size < 2:
            return np.array([self.unif_index(n) + 1 for _ in range(size)])
        x = list(range(n))
        out = []
        for _ in range(size):
            j = self.unif_index(n)
            out.append(x[j] + 1)
            n -= 1
            x[j] = x[n]
        return np.array(out)


def cv_folds(n_obs, nfolds=5, seed=123):
    """Plis de cv.glmnet après set.seed(seed) : sample(rep(1:nfolds, length = N))."""
    rng = RRandom(seed)
    base = np.resize(np.arange(1, nfolds + 1), n_obs)
    return base[rng.sample_int(n_obs) - 1]
