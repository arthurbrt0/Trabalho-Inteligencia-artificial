"""Pré-processamento: transforma um tabuleiro em vetor numérico.

Abordagem 1 (A1): somente o tabuleiro atual, codificado numericamente
    x -> +1, o -> -1, b (vazio) -> 0           (9 features)

Abordagem 2 (A2): features agregadas pedidas no enunciado
    n_x, n_o, occ_0..occ_8 (posições ocupadas, 0/1), lines_2x, lines_2o,
    n_vazias, vez (1 = vez do X, 0 = vez do O)                (15 features)
    - "linha com 2 X" = linha com exatamente 2 X e a 3ª casa vazia (ameaça de vitória)
    - "linha com 2 O" idem para O.
    - vez: X sempre começa, logo é a vez de X quando n_x == n_o.

Abordagem 2+ (A2P, EXTRA, fora do enunciado): A2 + lines_3x + lines_3o
    (nº de linhas completas de X / O). Serve para testar a hipótese de que a
    A2 falha porque suas features não "enxergam" a linha completa.
"""
from __future__ import annotations

import numpy as np

from .game import B, LINES, O, X

ENC = {X: 1, O: -1, B: 0}

A1_COLS = [f"c{i}" for i in range(9)]
A2_COLS = (
    ["n_x", "n_o"]
    + [f"occ_{i}" for i in range(9)]
    + ["lines_2x", "lines_2o", "n_vazias", "vez"]
)
A2P_COLS = A2_COLS + ["lines_3x", "lines_3o"]


def feats_a1(board) -> list[int]:
    return [ENC[c] for c in board]


def _line_counts(board, player: str):
    two = three = 0
    for line in LINES:
        cells = [board[i] for i in line]
        k = cells.count(player)
        if k == 3:
            three += 1
        elif k == 2 and cells.count(B) == 1:
            two += 1
    return two, three


def feats_a2(board) -> list[int]:
    nx, no = board.count(X), board.count(O)
    two_x, _ = _line_counts(board, X)
    two_o, _ = _line_counts(board, O)
    occ = [0 if c == B else 1 for c in board]
    vez = 1 if nx == no else 0
    return [nx, no, *occ, two_x, two_o, board.count(B), vez]


def feats_a2p(board) -> list[int]:
    _, three_x = _line_counts(board, X)
    _, three_o = _line_counts(board, O)
    return feats_a2(board) + [three_x, three_o]


APPROACHES = {
    "A1": (feats_a1, A1_COLS, "A1: tabuleiro (9 features)"),
    "A2": (feats_a2, A2_COLS, "A2: features agregadas (15 features)"),
    "A2P": (feats_a2p, A2P_COLS, "A2+: A2 + linhas completas (extra, 17 features)"),
}


def to_matrix(boards, approach: str) -> np.ndarray:
    fn = APPROACHES[approach][0]
    return np.array([fn(tuple(b)) for b in boards], dtype=float)
