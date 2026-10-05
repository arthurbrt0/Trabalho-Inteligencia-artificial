"""Estudo extra - curva de aprendizado: quanto o tamanho do dataset (nº de amostras
por classe) afeta cada abordagem de pré-processamento?

Uso:  python src/learning_curve.py     (requer results/best_params.json de train_models.py)

Para cada N (amostras por classe) e 5 sementes, reamostra o dataset (mesmo método de
build_dataset.py), treina com os melhores parâmetros achados para N=200 e mede a
acurácia balanceada em TODOS os estados válidos que ficaram de fora da amostra
(estável, ~5000 estados, classes balanceadas pela métrica).
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

sys.path.insert(0, str(Path(__file__).parent))
from build_dataset import COLS, stratified_sample
from train_models import ALGOS, make_model
from ttt.features import to_matrix
from ttt.game import CLASS_TO_ID, CLASSES, enumerate_reachable, true_state

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
NS = [25, 50, 100, 150, 200, 250, 300]
SEEDS = range(5)


def main():
    best = json.load(open(RES / "best_params.json", encoding="utf-8"))
    reach = enumerate_reachable()
    df = pd.DataFrame([(*b, p, true_state(b)) for b, p in reach.items()], columns=COLS + ["ply", "classe"])
    boards_all = [tuple(r) for r in df[COLS].to_numpy()]
    y_all = df["classe"].map(CLASS_TO_ID).to_numpy()
    rows = []
    for N in NS:
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            parts = [stratified_sample(df[df.classe == c], min(N, int((df.classe == c).sum())), rng) for c in CLASSES]
            ds = pd.concat(parts)
            tr = ds.copy()
            cnt = tr["classe"].value_counts()
            ex = tr[tr.classe == "empate"].sample(n=cnt.max() - cnt["empate"], replace=True, random_state=seed)
            tr = pd.concat([tr, ex])
            used = set(map(tuple, ds[COLS].to_numpy()))
            mask = np.array([b not in used for b in boards_all])
            ytr = tr["classe"].map(CLASS_TO_ID).to_numpy()
            for ap in ("A1", "A2"):
                Xtr = to_matrix([tuple(r) for r in tr[COLS].to_numpy()], ap)
                Xev = to_matrix([b for b, m in zip(boards_all, mask) if m], ap)
                for algo in ALGOS:
                    params = best[f"{ap}|{algo}"]
                    if algo == "MLP":
                        params["hidden_layer_sizes"] = tuple(params["hidden_layer_sizes"])
                    m = make_model(algo, params, ap, seed=seed)
                    m.fit(Xtr, ytr)
                    rows.append(dict(N=N, seed=seed, approach=ap, algo=algo,
                                     bal_acc=balanced_accuracy_score(y_all[mask], m.predict(Xev))))
        print(f"N={N} ok", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(RES / "learning_curve.csv", index=False)
    plot(out)


def plot(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 5, figsize=(19, 3.8), sharey=True)
    for ax, algo in zip(axs, ALGOS):
        for ap, c in (("A1", "#4f81bd"), ("A2", "#f79646")):
            g = out[(out.algo == algo) & (out.approach == ap)].groupby("N")["bal_acc"]
            ax.errorbar(g.mean().index, g.mean().values, yerr=g.std().values, marker="o", capsize=3, color=c,
                        label={"A1": "A1 (tabuleiro)", "A2": "A2 (agregadas)"}[ap])
        ax.set_title(algo)
        ax.set_xlabel("amostras por classe (N)")
        ax.grid(alpha=.3)
    axs[0].set_ylabel("acurácia balanceada (estados não vistos)")
    axs[0].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(RES / "figures" / "learning_curve.png", dpi=140)
    plt.close()


if __name__ == "__main__":
    main()
