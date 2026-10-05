"""Regras do jogo da velha e "juiz" (ground truth) usado para rotular tabuleiros.

Um tabuleiro é uma tupla de 9 caracteres, na ordem do dataset da UCI:
    (tl, tm, tr, ml, mm, mr, bl, bm, br)  ->  índices 0..8 (linha a linha)
Cada célula é 'x', 'o' ou 'b' (blank/vazio).

IMPORTANTE: este módulo é o *juiz de regras*. Ele só é usado para
  (1) rotular o dataset de treino e
  (2) medir se a IA acertou/errou durante o front end.
A decisão do jogo no front end vem do modelo de IA, não deste módulo.
"""
from __future__ import annotations

from collections import deque

LINES = (
    (0, 1, 2), (3, 4, 5), (6, 7, 8),
    (0, 3, 6), (1, 4, 7), (2, 5, 8),
    (0, 4, 8), (2, 4, 6),
)

X, O, B = "x", "o", "b"

CLASSES = ("tem_jogo", "x_venceu", "o_venceu", "empate")
CLASS_TO_ID = {c: i for i, c in enumerate(CLASSES)}
CLASS_TEXT = {
    "tem_jogo": "Tem jogo (partida em andamento)",
    "x_venceu": "Jogador X venceu",
    "o_venceu": "Jogador O venceu",
    "empate": "Empate (deu velha)",
}

EMPTY_BOARD = (B,) * 9


def has_won(board, player: str) -> bool:
    return any(all(board[i] == player for i in line) for line in LINES)


def true_state(board) -> str:
    """Classe correta do tabuleiro segundo as regras do jogo."""
    xw, ow = has_won(board, X), has_won(board, O)
    if xw and not ow:
        return "x_venceu"
    if ow and not xw:
        return "o_venceu"
    if xw and ow:
        raise ValueError(f"Tabuleiro ilegal (dois vencedores): {board}")
    if B not in board:
        return "empate"
    return "tem_jogo"


def next_player(board) -> str:
    """X sempre começa => se há o mesmo nº de X e O, é a vez de X."""
    return X if board.count(X) == board.count(O) else O


def is_legal(board) -> bool:
    """Tabuleiro alcançável numa partida em que X começa e as jogadas alternam."""
    nx, no = board.count(X), board.count(O)
    if nx not in (no, no + 1):
        return False
    xw, ow = has_won(board, X), has_won(board, O)
    if xw and ow:
        return False
    if xw and nx != no + 1:
        return False
    if ow and nx != no:
        return False
    return True


def legal_moves(board):
    return [i for i, c in enumerate(board) if c == B]


def play(board, pos: int, player: str):
    b = list(board)
    b[pos] = player
    return tuple(b)


def enumerate_reachable():
    """Todos os tabuleiros alcançáveis a partir do vazio (X começa).

    Retorna dict {tabuleiro: ply}, onde ply = nº de jogadas já feitas.
    Não expande estados terminais (a partida acaba ali). Total esperado: 5478.
    """
    seen = {EMPTY_BOARD: 0}
    queue = deque([EMPTY_BOARD])
    while queue:
        board = queue.popleft()
        if true_state(board) != "tem_jogo":
            continue
        player = next_player(board)
        for pos in legal_moves(board):
            nxt = play(board, pos, player)
            if nxt not in seen:
                seen[nxt] = seen[board] + 1
                queue.append(nxt)
    return seen


def render(board) -> str:
    sym = {X: "X", O: "O", B: " "}
    rows = []
    for r in range(3):
        cells = [sym[board[3 * r + c]] for c in range(3)]
        rows.append(" " + " | ".join(cells))
    return "\n---+---+---\n".join(rows)
