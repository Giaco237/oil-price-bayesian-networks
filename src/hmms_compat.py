"""A pure NumPy stand-in for the parts of `hmms` this project uses.

`hmms` is a Cython package last released in 2017. It does not build against
NumPy 2 and it needs a C toolchain, so on several of our machines it cannot be
installed at all. This module reimplements the same four algorithms with the
same call signatures, so a notebook can run either way:

    try:
        import hmms
    except ImportError:
        import hmms_compat as hmms

Numerical agreement with the Cython library was checked on the project panel:
identical parameters give identical forward and backward matrices, identical
log likelihoods and identical Viterbi paths to within floating point, and one
hundred Baum-Welch iterations from identical starting parameters reproduce the
same decoded regimes.

Three implementation points are deliberate. The forward and backward recursions
are scaled rather than run on raw probabilities, and Viterbi runs in log space,
so sequences of several hundred months do not underflow; `hmms` returns the log
of alpha and beta, so the public interface is unchanged. And `random` draws from
the legacy global generator, in the same order and shapes as the Cython library,
so an upstream `np.random.seed(k)` fixes the starting parameters and fixes them
to the same values either backend would have used.
"""
from __future__ import annotations

import numpy as np

__all__ = ["DtHMM", "print_parameters"]


def _as_seq(e) -> np.ndarray:
    return np.asarray(e, dtype=np.int64).ravel()


class DtHMM:
    """Discrete-time hidden Markov model.

    Attributes a, b and pi carry the transition matrix, the emission matrix and
    the initial distribution, named as in `hmms`.
    """

    def __init__(self, a, b, pi):
        self.a = np.asarray(a, dtype=float)
        self.b = np.asarray(b, dtype=float)
        self.pi = np.asarray(pi, dtype=float).ravel()
        self._check()

    # ---------------------------------------------------------------- construction
    @classmethod
    def random(cls, n_states: int, n_symbols: int, seed: int | None = None):
        # With no explicit seed, draw from the legacy global generator, in the same
        # order and shapes as the Cython library (A, then B, then Pi). Two
        # consequences, both required: np.random.seed(k) upstream makes the draw
        # reproducible, and the starting parameters are bit for bit those `hmms`
        # would have produced from the same seed, so either backend decodes the
        # same regimes. A default_rng here would ignore np.random.seed entirely.
        rng = np.random if seed is None else np.random.default_rng(seed)
        a = rng.random((n_states, n_states))
        b = rng.random((n_states, n_symbols))
        pi = rng.random(n_states)
        a /= a.sum(1, keepdims=True)
        b /= b.sum(1, keepdims=True)
        pi /= pi.sum()
        return cls(a, b, pi)

    @classmethod
    def from_file(cls, path: str):
        d = np.load(path)
        return cls(d["a"], d["b"], d["pi"])

    def save_params(self, path: str):
        if not path.endswith(".npz"):
            path += ".npz"
        np.savez(path, a=self.a, b=self.b, pi=self.pi)

    def _check(self):
        n, m = self.b.shape
        assert self.a.shape == (n, n), "A must be square and match B's rows"
        assert self.pi.shape == (n,), "Pi must have one entry per state"

    # ---------------------------------------------------------------- recursions
    def _forward_scaled(self, e):
        """Scaled forward pass. Returns alpha-hat and the scaling factors c."""
        n = len(self.pi)
        T = len(e)
        alpha = np.zeros((T, n))
        c = np.zeros(T)
        alpha[0] = self.pi * self.b[:, e[0]]
        c[0] = alpha[0].sum()
        if c[0] > 0:
            alpha[0] /= c[0]
        for t in range(1, T):
            alpha[t] = self.b[:, e[t]] * (alpha[t - 1] @ self.a)
            c[t] = alpha[t].sum()
            if c[t] > 0:
                alpha[t] /= c[t]
        return alpha, c

    def _backward_scaled(self, e, c):
        n = len(self.pi)
        T = len(e)
        beta = np.zeros((T, n))
        beta[-1] = 1.0
        for t in range(T - 2, -1, -1):
            beta[t] = self.a @ (self.b[:, e[t + 1]] * beta[t + 1])
            if c[t + 1] > 0:
                beta[t] /= c[t + 1]
        return beta

    def forward(self, emissions):
        """log alpha, unscaled, as `hmms` returns it."""
        e = _as_seq(emissions)
        alpha, c = self._forward_scaled(e)
        with np.errstate(divide="ignore"):
            return np.log(alpha) + np.cumsum(np.log(c))[:, None]

    def backward(self, emissions):
        """log beta, unscaled, as `hmms` returns it."""
        e = _as_seq(emissions)
        _, c = self._forward_scaled(e)
        beta = self._backward_scaled(e, c)
        logc = np.log(c)
        tail = np.concatenate([np.cumsum(logc[::-1])[::-1][1:], [0.0]])
        with np.errstate(divide="ignore"):
            return np.log(beta) + tail[:, None]

    def emission_estimate(self, emissions) -> float:
        """log P(O | lambda)."""
        e = _as_seq(emissions)
        _, c = self._forward_scaled(e)
        with np.errstate(divide="ignore"):
            return float(np.sum(np.log(c)))

    def viterbi(self, emissions):
        """Most likely state path. Returns (log probability, path)."""
        e = _as_seq(emissions)
        n, T = len(self.pi), len(e)
        with np.errstate(divide="ignore"):
            la, lb, lpi = np.log(self.a), np.log(self.b), np.log(self.pi)
        delta = np.zeros((T, n))
        psi = np.zeros((T, n), dtype=np.int64)
        delta[0] = lpi + lb[:, e[0]]
        for t in range(1, T):
            cand = delta[t - 1][:, None] + la          # (from, to)
            psi[t] = np.argmax(cand, axis=0)
            delta[t] = cand[psi[t], np.arange(n)] + lb[:, e[t]]
        path = np.zeros(T, dtype=np.int64)
        path[-1] = int(np.argmax(delta[-1]))
        for t in range(T - 1, 0, -1):
            path[t - 1] = psi[t, path[t]]
        return float(delta[-1].max()), path

    # ---------------------------------------------------------------- training
    def baum_welch(self, data, iterations: int = 10, est=None):
        """Re-estimate Pi, A and B in place from one or more emission sequences.

        `data` is a sequence of sequences, as `hmms` expects.
        """
        seqs = [_as_seq(s) for s in (data if isinstance(data, (list, tuple))
                                     else [data])]
        seqs = [s for s in seqs if len(s) >= 2]
        n, m = self.b.shape
        for _ in range(int(iterations)):
            pi_acc = np.zeros(n)
            a_num = np.zeros((n, n))
            a_den = np.zeros(n)
            b_num = np.zeros((n, m))
            b_den = np.zeros(n)
            for e in seqs:
                T = len(e)
                alpha, c = self._forward_scaled(e)
                beta = self._backward_scaled(e, c)
                gamma = alpha * beta
                s = gamma.sum(1, keepdims=True)
                s[s == 0] = 1.0
                gamma = gamma / s
                pi_acc += gamma[0]
                for t in range(T - 1):
                    denom = c[t + 1] if c[t + 1] > 0 else 1.0
                    xi = (alpha[t][:, None] * self.a
                          * (self.b[:, e[t + 1]] * beta[t + 1])[None, :]) / denom
                    a_num += xi
                a_den += gamma[:-1].sum(0)
                for k in range(m):
                    b_num[:, k] += gamma[e == k].sum(0)
                b_den += gamma.sum(0)
            a_den[a_den == 0] = 1.0
            b_den[b_den == 0] = 1.0
            self.pi = pi_acc / max(len(seqs), 1)
            self.pi /= self.pi.sum()
            self.a = a_num / a_den[:, None]
            self.a /= self.a.sum(1, keepdims=True)
            self.b = b_num / b_den[:, None]
            self.b /= self.b.sum(1, keepdims=True)
        return self


def print_parameters(model: DtHMM):
    import pandas as pd
    print("Initial probabilities (Pi):")
    print(pd.DataFrame(model.pi.reshape(-1, 1)).round(6).to_string(header=False))
    print("\nTransition probabilities matrix (A):")
    print(pd.DataFrame(model.a).round(6).to_string())
    print("\nEmission probabilities matrix (B):")
    print(pd.DataFrame(model.b).round(6).to_string())
