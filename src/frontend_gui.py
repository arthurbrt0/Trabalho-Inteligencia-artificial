"""Front end gráfico (tkinter) do jogo da velha: humano x máquina aleatória.

Uso:  python src/frontend_gui.py [--model models/A1_SVM.joblib]

Mesmas regras do front end de terminal (frontend.py):
  * a cada jogada a IA classifica o tabuleiro (Tem jogo / X venceu / O venceu / Empate);
  * se a IA NÃO detecta um fim de jogo real -> o jogo é encerrado e conta como erro;
  * se a IA detecta um fim de jogo que NÃO existe -> o jogo continua e conta como erro;
  * o painel mostra acertos, erros e acurácia da IA durante as interações.
As interações são gravadas em results/frontend_log.csv (modo "human").
"""
from __future__ import annotations

import argparse
import random
import sys
import time
import tkinter as tk
from pathlib import Path
from tkinter import ttk

sys.path.insert(0, str(Path(__file__).parent))
from frontend import DEFAULT_MODEL, AIClassifier, Score, append_log, evaluate_turn
from ttt.game import B, CLASS_TEXT, O, X, legal_moves, next_player, play

COLOR = {X: "#1f5fbf", O: "#c0392b"}
OK_COLOR, ERR_COLOR = "#1e8449", "#c0392b"


class App:
    def __init__(self, root: tk.Tk, ai: AIClassifier, delay_ms: int = 600):
        self.root, self.ai, self.delay = root, ai, delay_ms
        self.score = Score()
        self.rng = random.Random()
        self.board = (B,) * 9
        self.over = True
        self.busy = False
        self.log_rows: list[dict] = []
        self.session = int(time.time())
        self.n_games = 0
        self.turn = 0
        self.human = X

        root.title("Jogo da Velha - IA verificadora de estado")
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.resizable(False, False)
        main = ttk.Frame(root, padding=10)
        main.grid()

        top = ttk.Frame(main)
        top.grid(row=0, column=0, columnspan=2, sticky="we", pady=(0, 8))
        ttk.Label(top, text="Eu jogo de:").pack(side="left")
        self.sym = tk.StringVar(value="x")
        ttk.Radiobutton(top, text="X (começo)", variable=self.sym, value="x").pack(side="left", padx=4)
        ttk.Radiobutton(top, text="O (máquina começa)", variable=self.sym, value="o").pack(side="left", padx=4)
        ttk.Button(top, text="Nova partida", command=self.new_game).pack(side="left", padx=12)
        ttk.Label(top, text=f"Modelo: {ai.algo} ({ai.approach})", foreground="#555").pack(side="right")

        grid = ttk.Frame(main)
        grid.grid(row=1, column=0, sticky="n")
        self.cells: list[tk.Button] = []
        for i in range(9):
            b = tk.Button(grid, text="", width=4, height=1, font=("Segoe UI", 34, "bold"),
                          relief="groove", bg="#f4f4f4", command=lambda p=i: self.click(p))
            b.grid(row=i // 3, column=i % 3, padx=3, pady=3)
            self.cells.append(b)

        side = ttk.Frame(main, padding=(14, 0, 0, 0))
        side.grid(row=1, column=1, sticky="n")
        ttk.Label(side, text="Placar da IA", font=("Segoe UI", 12, "bold")).pack(anchor="w")
        self.lbl_hits = ttk.Label(side, text="")
        self.lbl_hits.pack(anchor="w")
        self.lbl_acc = ttk.Label(side, text="", font=("Segoe UI", 11, "bold"))
        self.lbl_acc.pack(anchor="w", pady=(0, 8))
        ttk.Label(side, text="Partidas", font=("Segoe UI", 12, "bold")).pack(anchor="w")
        self.lbl_games = ttk.Label(side, text="")
        self.lbl_games.pack(anchor="w", pady=(0, 8))
        ttk.Label(side, text="Histórico das jogadas", font=("Segoe UI", 12, "bold")).pack(anchor="w")
        box = ttk.Frame(side)
        box.pack(anchor="w")
        self.hist = tk.Text(box, width=54, height=11, state="disabled", font=("Consolas", 9), wrap="word")
        sb = ttk.Scrollbar(box, command=self.hist.yview)
        self.hist.configure(yscrollcommand=sb.set)
        self.hist.pack(side="left")
        sb.pack(side="left", fill="y")
        self.hist.tag_configure("ok", foreground=OK_COLOR)
        self.hist.tag_configure("err", foreground=ERR_COLOR)
        self.hist.tag_configure("sep", foreground="#888")

        self.lbl_ai = tk.Label(main, text="", font=("Segoe UI", 13, "bold"), anchor="w", justify="left",
                               wraplength=640)
        self.lbl_ai.grid(row=2, column=0, columnspan=2, sticky="we", pady=(10, 0))
        self.lbl_note = tk.Label(main, text="", font=("Segoe UI", 10), anchor="w", justify="left", wraplength=640)
        self.lbl_note.grid(row=3, column=0, columnspan=2, sticky="we")

        self.refresh()
        self.set_msg("Clique em \"Nova partida\" para começar.", "#333")

    def set_msg(self, ai_text, color, note="", note_color="#333"):
        self.lbl_ai.config(text=ai_text, fg=color)
        self.lbl_note.config(text=note, fg=note_color)

    def log_hist(self, text, tag=None):
        self.hist.config(state="normal")
        self.hist.insert("end", text + "\n", tag)
        self.hist.see("end")
        self.hist.config(state="disabled")

    def refresh(self):
        for i, c in enumerate(self.board):
            sym = {X: "X", O: "O", B: ""}[c]
            can_click = (not self.over and not self.busy and c == B and next_player(self.board) == self.human)
            self.cells[i].config(text=sym, fg=COLOR.get(c, "black"),
                                 state="normal" if can_click else "disabled",
                                 disabledforeground=COLOR.get(c, "#999"))
        s = self.score
        self.lbl_hits.config(text=f"Acertos: {s.hits}    Erros: {s.errors}")
        acc = "-" if not s.total else f"{100 * s.acc:.1f}%"
        self.lbl_acc.config(text=f"Acurácia: {acc}  ({s.total} classificações)")
        self.lbl_games.config(text=f"Você {s.human} x {s.machine} Máquina   (empates: {s.draws})")

    def flush_log(self):
        if self.log_rows:
            append_log(self.log_rows)
            self.log_rows = []

    def new_game(self):
        self.flush_log()
        self.n_games += 1
        self.board = (B,) * 9
        self.over = False
        self.busy = False
        self.turn = 0
        self.human = X if self.sym.get() == "x" else O
        self.game_id = f"gui-{self.session}-{self.n_games}"
        self.log_hist(f"--- Partida {self.n_games}: você é {self.human.upper()} ---", "sep")
        self.set_msg("Jogo iniciado. " + ("Sua vez!" if next_player(self.board) == self.human
                                          else "A máquina começa..."), "#333")
        self.refresh()
        self.after_turn()

    def after_turn(self):
        """Agenda a jogada da máquina se for a vez dela."""
        if not self.over and next_player(self.board) != self.human:
            self.busy = True
            self.refresh()
            self.root.after(self.delay, self.machine_move)
        else:
            self.busy = False
            self.refresh()

    def click(self, pos):
        if self.over or self.busy or self.board[pos] != B or next_player(self.board) != self.human:
            return
        self.move(pos, self.human, "Você")

    def machine_move(self):
        if self.over:
            return
        self.move(self.rng.choice(legal_moves(self.board)), next_player(self.board), "Máquina")

    def move(self, pos, player, who):
        self.board = play(self.board, pos, player)
        self.turn += 1
        pred, truth, ok, game_over = evaluate_turn(
            self.ai, self.board, self.score, self.log_rows,
            dict(mode="human", game_id=self.game_id, turn=self.turn))

        mark = "acertou" if ok else "ERROU"
        self.log_hist(f"{self.turn:>2}. {who} ({player.upper()}) casa {pos + 1}: IA -> "
                      f"{CLASS_TEXT[pred].split(' (')[0]} [{mark}]", "ok" if ok else "err")

        if ok:
            note, note_color = "A IA acertou a classificação.", OK_COLOR
        elif game_over:
            note = (f"ERRO DA IA: o correto era \"{CLASS_TEXT[truth]}\". A IA não detectou o fim da partida; "
                    "o jogo é ENCERRADO.")
            note_color = ERR_COLOR
        else:
            note = (f"ERRO DA IA: ainda há jogo, mas ela indicou \"{CLASS_TEXT[pred]}\". "
                    "Fim de jogo incorreto: o jogo CONTINUA.")
            note_color = ERR_COLOR

        if game_over:
            self.over = True
            if truth == "empate":
                self.score.draws += 1
            elif (truth == "x_venceu") == (self.human == X):
                self.score.human += 1
            else:
                self.score.machine += 1
            self.flush_log()
            self.set_msg(f"IA: {CLASS_TEXT[pred]}  |  FIM DE JOGO: {CLASS_TEXT[truth]}", COLOR.get(
                X if truth == "x_venceu" else O if truth == "o_venceu" else "", "#333"), note, note_color)
            self.busy = False
            self.refresh()
        else:
            self.set_msg(f"IA: {CLASS_TEXT[pred]}", "#333" if ok else ERR_COLOR, note, note_color)
            self.after_turn()

    def close(self):
        self.flush_log()
        self.root.destroy()


def enable_dpi_awareness():
    """Evita janela borrada em telas com escala > 100% no Windows."""
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    args = ap.parse_args()
    ai = AIClassifier(Path(args.model))
    enable_dpi_awareness()
    root = tk.Tk()
    App(root, ai)
    root.mainloop()


if __name__ == "__main__":
    main()
