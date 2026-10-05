"""Front end (terminal) do jogo da velha: humano x máquina aleatória.

A cada jogada (humana ou da máquina) o modelo de IA classifica o tabuleiro em
  Tem jogo / X venceu / O venceu / Empate
e o jogo segue (ou termina) a partir da saída da IA, com estas regras do enunciado:
  * IA diz "tem jogo" mas o jogo JÁ ACABOU (não detectou o fim)  -> o jogo é encerrado
    (não há como continuar) e conta como ERRO da IA.
  * IA diz que o jogo acabou mas AINDA HÁ JOGO (detecção incorreta) -> o jogo CONTINUA
    e conta como ERRO da IA.
  * Caso a IA acerte a classe, conta como ACERTO.
O "juiz de regras" (ttt.game.true_state) é usado apenas para medir acertos/erros.

Uso:
  python src/frontend.py
  python src/frontend.py --play o
  python src/frontend.py --auto 500
  python src/frontend.py --stats
  python src/frontend.py --model models/A1_SVM.joblib
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
import time
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from ttt.features import APPROACHES
from ttt.game import (B, CLASS_TEXT, CLASSES, O, X, legal_moves, next_player, play, true_state)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = ROOT / "models" / "best_model.joblib"
LOG = ROOT / "results" / "frontend_log.csv"
LOG_FIELDS = ["timestamp", "mode", "model", "game_id", "turn", "board", "truth", "pred", "correct"]


class AIClassifier:
    """Envolve o modelo treinado: tabuleiro -> classe."""

    def __init__(self, path: Path):
        bundle = joblib.load(path)
        self.model = bundle["model"]
        self.approach = bundle["approach"]
        self.algo = bundle["algo"]
        self.name = f"{self.approach}|{self.algo}"
        self._feat = APPROACHES[self.approach][0]

    def predict(self, board) -> str:
        x = np.array([self._feat(board)], dtype=float)
        return CLASSES[int(self.model.predict(x)[0])]


class Score:
    def __init__(self):
        self.hits = 0
        self.errors = 0
        self.human = self.machine = self.draws = 0

    @property
    def total(self):
        return self.hits + self.errors

    @property
    def acc(self):
        return self.hits / self.total if self.total else float("nan")

    def line(self):
        return f"IA: {self.hits} acertos, {self.errors} erros -> acurácia {100 * self.acc:.1f}% ({self.total} classificações)"


def draw_board(board) -> str:
    sym = {X: "X", O: "O"}
    rows = []
    for r in range(3):
        cells = [sym.get(board[3 * r + c], str(3 * r + c + 1)) for c in range(3)]
        rows.append(" " + " | ".join(cells))
    return "\n---+---+---\n".join(rows)


def append_log(rows):
    """Anexa ao log CSV. Tenta de novo se o arquivo estiver momentaneamente bloqueado
    (ex.: sincronização do OneDrive) e nunca derruba o jogo por falha de log."""
    LOG.parent.mkdir(exist_ok=True)
    for attempt in range(10):
        try:
            new = not LOG.exists()
            with open(LOG, "a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
                if new:
                    w.writeheader()
                w.writerows(rows)
            return
        except PermissionError:
            time.sleep(0.5)
    print("[aviso] não foi possível gravar o log desta partida (arquivo bloqueado).")


def evaluate_turn(ai, board, score, log_rows, ctx):
    """Roda a IA no tabuleiro, atualiza o placar e registra no log.

    Retorna (pred, truth, ok, game_over). Regra do enunciado: se o jogo realmente
    acabou (truth != tem_jogo) ele é encerrado mesmo que a IA não tenha detectado;
    se ainda há jogo, ele continua mesmo que a IA tenha dito que acabou.
    Compartilhada entre o front end de terminal e o gráfico.
    """
    pred = ai.predict(board)
    truth = true_state(board)
    ok = pred == truth
    score.hits += ok
    score.errors += not ok
    log_rows.append(dict(timestamp=time.strftime("%Y-%m-%d %H:%M:%S"), mode=ctx["mode"], model=ai.name,
                         game_id=ctx["game_id"], turn=ctx["turn"], board="".join(board), truth=truth,
                         pred=pred, correct=int(ok)))
    return pred, truth, ok, truth != "tem_jogo"


def classify_turn(ai, board, score, log_rows, ctx, verbose):
    """Versão de terminal: avalia o turno e imprime as mensagens."""
    pred, truth, ok, game_over = evaluate_turn(ai, board, score, log_rows, ctx)
    if verbose:
        print(f"  >> IA: {CLASS_TEXT[pred]}")
        if ok:
            print("  >> (a IA acertou)")
        else:
            print(f"  >> [ERRO DA IA] o correto era: {CLASS_TEXT[truth]}")
            if game_over:
                print("  >> A IA não detectou o fim da partida: o jogo é ENCERRADO.")
            else:
                print("  >> A IA detectou um fim de jogo inexistente: o jogo CONTINUA.")
        print("  >> " + score.line())
    return game_over, truth


def play_game(ai, score, human_symbol, mode, game_id, rng, verbose=True, human_input=None):
    board = (B,) * 9
    machine_symbol = O if human_symbol == X else X
    log_rows: list[dict] = []
    turn = 0
    if verbose:
        print("\n=== Nova partida: você é", human_symbol.upper(), "| X começa ===")
        print(draw_board(board))
    while True:
        player = next_player(board)
        moves = legal_moves(board)
        if player == human_symbol:
            if mode == "auto":
                pos = rng.choice(moves)
            else:
                pos = human_input(board, moves)
                if pos is None:
                    append_log(log_rows)
                    return None
            if verbose:
                print(f"\nVocê ({player.upper()}) jogou na casa {pos + 1}.")
        else:
            pos = rng.choice(moves)
            if verbose:
                print(f"\nMáquina ({player.upper()}) jogou na casa {pos + 1}.")
        board = play(board, pos, player)
        turn += 1
        if verbose:
            print(draw_board(board))
        game_over, truth = classify_turn(ai, board, score, log_rows, dict(mode=mode, game_id=game_id, turn=turn), verbose)
        if game_over:
            break
    append_log(log_rows)
    if truth == "empate":
        score.draws += 1
    elif (truth == "x_venceu") == (human_symbol == X):
        score.human += 1
    else:
        score.machine += 1
    if verbose:
        print("\n*** Fim de jogo:", CLASS_TEXT[truth], "***")
    return truth


def ask_move(board, moves):
    while True:
        s = input(f"\nSua jogada (1-9, 'q' para sair) {[m + 1 for m in moves]}: ").strip().lower()
        if s in ("q", "sair"):
            return None
        if s.isdigit() and int(s) - 1 in moves:
            return int(s) - 1
        print("Jogada inválida, tente de novo.")


def print_stats():
    if not LOG.exists():
        print("Ainda não há interações registradas.")
        return
    import pandas as pd
    df = pd.read_csv(LOG)
    for (mode, model), g in df.groupby(["mode", "model"]):
        games = g["game_id"].nunique()
        print(f"[{mode:5s}] modelo={model:22s} partidas={games:5d} classificações={len(g):6d} "
              f"acertos={int(g.correct.sum()):6d} erros={int((1 - g.correct).sum()):5d} "
              f"acurácia={100 * g.correct.mean():.2f}%")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--play", choices=["x", "o"], default="x", help="símbolo do jogador humano")
    ap.add_argument("--auto", type=int, default=0, help="simula N partidas com humano aleatório")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--stats", action="store_true")
    args = ap.parse_args()

    if args.stats:
        print_stats()
        return
    ai = AIClassifier(Path(args.model))
    rng = random.Random(args.seed)
    score = Score()
    print(f"Modelo de IA carregado: {ai.algo} ({ai.approach})")

    if args.auto:
        game_id0 = int(time.time())
        for i in range(args.auto):
            play_game(ai, score, X if i % 2 == 0 else O, "auto", f"auto-{game_id0}-{i}", rng, verbose=False)
        print(f"{args.auto} partidas simuladas (humano substituído por jogador aleatório).")
        print(score.line())
        print(f"Resultados reais (juiz): jogador1 venceu {score.human}, máquina venceu {score.machine}, empates {score.draws}")
        return

    game_id0 = int(time.time())
    n = 0
    print("Casas numeradas assim:\n" + draw_board((B,) * 9))
    while True:
        n += 1
        res = play_game(ai, score, args.play, "human", f"human-{game_id0}-{n}", rng, human_input=ask_move)
        if res is None:
            break
        print(f"\nPlacar: você {score.human} x {score.machine} máquina (empates: {score.draws})")
        print(score.line())
        if input("\nJogar de novo? (s/n): ").strip().lower() not in ("s", "sim", "y"):
            break
    print("\nResumo da sessão:")
    print(score.line())
    print(f"(interações registradas em {LOG.relative_to(ROOT)})")


if __name__ == "__main__":
    main()
