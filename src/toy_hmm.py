"""The two-regime worked example of Step 1, with the four algorithms written out.

These implementations exist so the pseudocode in the report can be checked line by
line against numbers, and so the `hmms` library can be cross-checked against something
independent. Production runs use `hmms`.
"""
from __future__ import annotations

import numpy as np

STATES = ["Bull", "Bear"]
SYMBOLS = {0: "down", 1: "up"}

PI = np.array([0.6, 0.4])
A = np.array([[0.7, 0.3],
              [0.4, 0.6]])
B = np.array([[0.2, 0.8],      # Bull: down 0.2, up 0.8
              [0.7, 0.3]])     # Bear: down 0.7, up 0.3
O = np.array([1, 0, 1])        # up, down, up


def forward(pi, A, B, O):
    T, N = len(O), len(pi)
    alpha = np.zeros((T, N))
    alpha[0] = pi * B[:, O[0]]
    for t in range(1, T):
        alpha[t] = B[:, O[t]] * (alpha[t - 1] @ A)
    return alpha


def backward(A, B, O):
    T, N = len(O), A.shape[0]
    beta = np.zeros((T, N))
    beta[-1] = 1.0
    for t in range(T - 2, -1, -1):
        beta[t] = A @ (B[:, O[t + 1]] * beta[t + 1])
    return beta


def viterbi(pi, A, B, O):
    T, N = len(O), len(pi)
    delta = np.zeros((T, N))
    psi = np.zeros((T, N), dtype=int)
    delta[0] = pi * B[:, O[0]]
    for t in range(1, T):
        for i in range(N):
            cand = delta[t - 1] * A[:, i]
            psi[t, i] = int(np.argmax(cand))
            delta[t, i] = cand[psi[t, i]] * B[i, O[t]]
    path = [int(np.argmax(delta[-1]))]
    for t in range(T - 1, 0, -1):
        path.insert(0, int(psi[t, path[0]]))
    return delta, psi, path


def baum_welch_step(pi, A, B, O):
    alpha, beta = forward(pi, A, B, O), backward(A, B, O)
    T, N = len(O), len(pi)
    p_O = alpha[-1].sum()
    gamma = alpha * beta / p_O
    xi = np.zeros((T - 1, N, N))
    for t in range(T - 1):
        for i in range(N):
            for j in range(N):
                xi[t, i, j] = (alpha[t, i] * A[i, j] * B[j, O[t + 1]]
                               * beta[t + 1, j] / p_O)
    pi_new = gamma[0]
    A_new = xi.sum(0) / gamma[:-1].sum(0)[:, None]
    B_new = np.zeros_like(B)
    for k in range(B.shape[1]):
        B_new[:, k] = gamma[O == k].sum(0) / gamma.sum(0)
    return gamma, xi, pi_new, A_new, B_new, p_O
