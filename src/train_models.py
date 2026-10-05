"""Etapas 2-4 - Pré-processamento, treino, ajuste de parâmetros e avaliação.

Uso:  python src/train_models.py

Protocolo:
  * Mesmos train/val/test.csv para todos os algoritmos e abordagens.
  * Hiperparâmetros escolhidos SOMENTE com o conjunto de validação (grid search).
  * O conjunto de teste é usado uma única vez, com a melhor configuração de cada
    algoritmo/abordagem.
  * Critério de escolha: maior F-measure macro na validação; empates ->
    maior acurácia de validação -> menor gap treino-validação (menos overfitting)
    -> menor custo (tempo de inferência).
"""
from __future__ import annotations

import itertools
import json
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             precision_recall_fscore_support)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, str(Path(__file__).parent))
from ttt.features import APPROACHES, to_matrix
from ttt.game import CLASS_TO_ID, CLASSES, enumerate_reachable, true_state

warnings.filterwarnings("ignore", category=ConvergenceWarning)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
RES = ROOT / "results"
FIG = RES / "figures"
MODELS = ROOT / "models"
SEED = 42
COLS = ["tl", "tm", "tr", "ml", "mm", "mr", "bl", "bm", "br"]

ALGOS = ["kNN", "Árvore de Decisão", "MLP", "Random Forest", "SVM"]
SPEC_APPROACHES = ["A1", "A2"]
ALL_APPROACHES = ["A1", "A2", "A2P"]


def grid(algo: str) -> list[dict]:
    if algo == "kNN":
        return [dict(n_neighbors=k, weights=w, metric=m)
                for k, w, m in itertools.product([1, 3, 5, 7, 9, 11, 15, 21], ["uniform", "distance"],
                                                 ["euclidean", "manhattan"])]
    if algo == "Árvore de Decisão":
        return [dict(criterion=c, max_depth=d, min_samples_leaf=l)
                for c, d, l in itertools.product(["gini", "entropy"], [2, 3, 4, 5, 6, 8, 10, None],
                                                 [1, 2, 5, 10])]
    if algo == "MLP":
        topo = [(8,), (16,), (32,), (64,), (16, 8), (32, 16), (64, 32)]
        return [dict(hidden_layer_sizes=t, activation=a, alpha=al)
                for t, a, al in itertools.product(topo, ["relu", "tanh"], [1e-4, 1e-2])]
    if algo == "Random Forest":
        return [dict(n_estimators=n, max_depth=d, max_features=f)
                for n, d, f in itertools.product([10, 50, 100, 200], [3, 5, 8, None], ["sqrt", None])]
    if algo == "SVM":
        g = [dict(kernel="linear", C=c) for c in [0.1, 1, 10, 100]]
        g += [dict(kernel="rbf", C=c, gamma=ga) for c in [0.1, 1, 10, 100] for ga in ["scale", 0.01, 0.1, 1]]
        g += [dict(kernel="poly", C=c, degree=d, gamma="scale") for c in [0.1, 1, 10, 100] for d in [2, 3]]
        return g
    raise ValueError(algo)


def make_model(algo: str, params: dict, approach: str, seed: int = SEED):
    p = dict(params)
    if algo == "kNN":
        est = KNeighborsClassifier(**p)
    elif algo == "Árvore de Decisão":
        est = DecisionTreeClassifier(random_state=seed, **p)
    elif algo == "MLP":
        est = MLPClassifier(solver="adam", max_iter=2000, random_state=seed, **p)
    elif algo == "Random Forest":
        est = RandomForestClassifier(random_state=seed, n_jobs=1, **p)
    elif algo == "SVM":
        est = SVC(random_state=seed, **p)
    else:
        raise ValueError(algo)
    if approach != "A1" and algo in ("kNN", "MLP", "SVM"):
        return make_pipeline(StandardScaler(), est)
    return est


def metrics(y, pred) -> dict:
    p, r, f, _ = precision_recall_fscore_support(y, pred, average="macro", zero_division=0)
    return dict(acc=accuracy_score(y, pred), prec=p, rec=r, f1=f)


def slug(name: str) -> str:
    import unicodedata
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return s.replace(" ", "_")


def fmt(params: dict) -> str:
    return ", ".join(f"{k}={v}" for k, v in params.items())


def load():
    sets = {}
    for name in ("train", "val", "test"):
        df = pd.read_csv(DATA / f"{name}.csv")
        boards = [tuple(r) for r in df[COLS].to_numpy()]
        y = df["classe"].map(CLASS_TO_ID).to_numpy()
        sets[name] = (boards, y)
    return sets


def main():
    RES.mkdir(exist_ok=True)
    FIG.mkdir(exist_ok=True, parents=True)
    MODELS.mkdir(exist_ok=True)
    sets = load()
    ytr, yva, yte = sets["train"][1], sets["val"][1], sets["test"][1]

    prep_rows = []
    feats: dict[str, dict[str, np.ndarray]] = {}
    for ap in ALL_APPROACHES:
        feats[ap] = {n: to_matrix(sets[n][0], ap) for n in sets}
        t0 = time.perf_counter()
        for b in sets["test"][0]:
            APPROACHES[ap][0](b)
        dt = (time.perf_counter() - t0) / len(sets["test"][0]) * 1e6
        prep_rows.append(dict(approach=ap, n_features=feats[ap]["train"].shape[1], feat_time_us=dt))
    pd.DataFrame(prep_rows).to_csv(RES / "preprocessing_cost.csv", index=False)

    rows = []
    for ap in ALL_APPROACHES:
        Xtr, Xva = feats[ap]["train"], feats[ap]["val"]
        for algo in ALGOS:
            for params in grid(algo):
                m = make_model(algo, params, ap)
                t0 = time.perf_counter()
                m.fit(Xtr, ytr)
                fit_t = time.perf_counter() - t0
                mv = metrics(yva, m.predict(Xva))
                rows.append(dict(approach=ap, algo=algo, params=fmt(params), params_dict=json.dumps(params, default=str),
                                 train_acc=accuracy_score(ytr, m.predict(Xtr)), val_acc=mv["acc"],
                                 val_prec=mv["prec"], val_rec=mv["rec"], val_f1=mv["f1"], fit_time_s=fit_t))
        print(f"[grid] {ap} concluído", flush=True)
    grid_df = pd.DataFrame(rows)
    grid_df["gap"] = grid_df["train_acc"] - grid_df["val_acc"]
    grid_df.drop(columns="params_dict").to_csv(RES / "grid_results.csv", index=False)

    best = {}
    for ap in ALL_APPROACHES:
        for algo in ALGOS:
            sub = grid_df[(grid_df.approach == ap) & (grid_df.algo == algo)].copy()
            sub["f1r"] = sub["val_f1"].round(3)
            sub["accr"] = sub["val_acc"].round(3)
            sub["gapr"] = sub["gap"].abs().round(3)
            sub["ordem"] = range(len(sub))
            sub = sub.sort_values(["f1r", "accr", "gapr", "ordem"], ascending=[False, False, True, True])
            best[(ap, algo)] = json.loads(sub.iloc[0]["params_dict"])
            if algo == "MLP":
                best[(ap, algo)]["hidden_layer_sizes"] = tuple(best[(ap, algo)]["hidden_layer_sizes"])
            if algo == "Árvore de Decisão" and best[(ap, algo)].get("max_depth") in ("None", None):
                best[(ap, algo)]["max_depth"] = None

    json.dump({f"{k[0]}|{k[1]}": {a: (list(b) if isinstance(b, tuple) else b) for a, b in v.items()}
               for k, v in best.items()}, open(RES / "best_params.json", "w", encoding="utf-8"),
              indent=1, ensure_ascii=False)

    reach = enumerate_reachable()
    all_boards = list(reach)
    y_all = np.array([CLASS_TO_ID[true_state(b)] for b in all_boards])
    used = set(sets["train"][0]) | set(sets["val"][0]) | set(sets["test"][0])
    unseen_mask = np.array([b not in used for b in all_boards])

    final_rows, perclass_rows, cms = [], [], {}
    fitted = {}
    rng = np.random.default_rng(0)
    probe = [sets["test"][0][i] for i in rng.choice(len(sets["test"][0]), 100, replace=False)]
    for ap in ALL_APPROACHES:
        Xtr, Xva, Xte = feats[ap]["train"], feats[ap]["val"], feats[ap]["test"]
        X_all = to_matrix(all_boards, ap)
        fn = APPROACHES[ap][0]
        for algo in ALGOS:
            params = best[(ap, algo)]
            m = make_model(algo, params, ap)
            t0 = time.perf_counter()
            m.fit(Xtr, ytr)
            fit_t = time.perf_counter() - t0
            fitted[(ap, algo)] = m
            ptr, pva, pte = m.predict(Xtr), m.predict(Xva), m.predict(Xte)
            mt = metrics(yte, pte)
            mv = metrics(yva, pva)
            t0 = time.perf_counter()
            for b in probe * 3:
                m.predict(np.array([fn(b)], dtype=float))
            inf_us = (time.perf_counter() - t0) / (len(probe) * 3) * 1e6
            p_all = m.predict(X_all)
            exh = accuracy_score(y_all, p_all)
            exh_unseen = accuracy_score(y_all[unseen_mask], p_all[unseen_mask])
            exh_unseen_bal = balanced_accuracy_score(y_all[unseen_mask], p_all[unseen_mask])
            final_rows.append(dict(
                approach=ap, algo=algo, params=fmt(params),
                train_acc=accuracy_score(ytr, ptr), val_acc=mv["acc"], val_f1=mv["f1"],
                test_acc=mt["acc"], test_prec=mt["prec"], test_rec=mt["rec"], test_f1=mt["f1"],
                fit_time_s=fit_t, inference_us=inf_us,
                exhaustive_acc_all=exh, exhaustive_acc_unseen=exh_unseen,
                exhaustive_bal_acc_unseen=exh_unseen_bal,
                n_unseen=int(unseen_mask.sum())))
            p, r, f, s = precision_recall_fscore_support(yte, pte, labels=range(4), zero_division=0)
            for i, c in enumerate(CLASSES):
                perclass_rows.append(dict(approach=ap, algo=algo, classe=c, precision=p[i], recall=r[i],
                                          f1=f[i], support=int(s[i])))
            cms[(ap, algo)] = confusion_matrix(yte, pte, labels=range(4))
            for i, c in enumerate(CLASSES):
                mk = (y_all == i) & unseen_mask
                if mk.any():
                    perclass_rows[-4 + i]["exhaustive_unseen_recall"] = float((p_all[mk] == i).mean())
            joblib.dump(dict(model=m, approach=ap, algo=algo, params=params), MODELS / f"{ap}_{slug(algo)}.joblib")
        print(f"[teste] {ap} concluído", flush=True)

    final = pd.DataFrame(final_rows)
    final.to_csv(RES / "test_results.csv", index=False)
    pd.DataFrame(perclass_rows).to_csv(RES / "per_class_test.csv", index=False)
    json.dump({f"{k[0]}|{k[1]}": v.tolist() for k, v in cms.items()}, open(RES / "confusion_matrices.json", "w"))

    seed_rows = []
    for ap in SPEC_APPROACHES:
        for seed in range(10):
            m = make_model("MLP", best[(ap, "MLP")], ap, seed=seed)
            m.fit(feats[ap]["train"], ytr)
            seed_rows.append(dict(approach=ap, seed=seed, val_acc=accuracy_score(yva, m.predict(feats[ap]["val"])),
                                  test_acc=accuracy_score(yte, m.predict(feats[ap]["test"]))))
    pd.DataFrame(seed_rows).to_csv(RES / "mlp_seeds.csv", index=False)

    TOL = 0.01
    cand = final[final.approach.isin(SPEC_APPROACHES)].copy()
    cand = cand[cand["val_f1"] >= cand["val_f1"].max() - TOL]
    cand = cand.sort_values(["inference_us"])
    top = cand.iloc[0]
    chosen = fitted[(top.approach, top.algo)]
    joblib.dump(dict(model=chosen, approach=top.approach, algo=top.algo, params=best[(top.approach, top.algo)]),
                MODELS / "best_model.joblib")
    json.dump(dict(approach=top.approach, algo=top.algo, params=top.params,
                   val_f1=float(top.val_f1), test_acc=float(top.test_acc), test_f1=float(top.test_f1)),
              open(RES / "best_model.json", "w"), indent=2, ensure_ascii=False)
    print("\nMODELO ESCOLHIDO:", top.approach, top.algo, top.params)

    make_figures(grid_df, final, cms, pd.DataFrame(seed_rows))
    pd.set_option("display.width", 250, "display.max_columns", 30)
    print(final[["approach", "algo", "train_acc", "val_acc", "test_acc", "test_prec", "test_rec", "test_f1",
                 "fit_time_s", "inference_us", "exhaustive_acc_unseen"]].round(4).to_string())


def make_figures(grid_df, final, cms, seeds):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    col = {"A1": "#4f81bd", "A2": "#f79646", "A2P": "#9bbb59"}
    lab = {"A1": "A1 (tabuleiro)", "A2": "A2 (features agregadas)", "A2P": "A2+ (extra)"}

    fig, ax = plt.subplots(figsize=(6.5, 4))
    for ap in ALL_APPROACHES:
        s = grid_df[(grid_df.approach == ap) & (grid_df.algo == "kNN")].copy()
        s["k"] = s["params"].str.extract(r"n_neighbors=(\d+)")[0].astype(int)
        g = s.groupby("k")["val_f1"].max()
        ax.plot(g.index, g.values, marker="o", label=lab[ap], color=col[ap])
    ax.set_xlabel("k (nº de vizinhos)")
    ax.set_ylabel("F-measure macro (validação)")
    ax.set_title("k-NN: efeito de k (melhor peso/métrica por k)")
    ax.legend()
    ax.grid(alpha=.3)
    plt.tight_layout()
    plt.savefig(FIG / "knn_k.png", dpi=150)
    plt.close()

    fig, axs = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, ap in zip(axs, SPEC_APPROACHES):
        s = grid_df[(grid_df.approach == ap) & (grid_df.algo == "Árvore de Decisão")
                    & (grid_df.params.str.contains("criterion=gini")) & (grid_df.params.str.contains("min_samples_leaf=1$"))]
        s = s.copy()
        s["d"] = s["params"].str.extract(r"max_depth=(\w+)")[0]
        order = ["2", "3", "4", "5", "6", "8", "10", "None"]
        s = s.set_index("d").loc[order]
        ax.plot(order, s["train_acc"], marker="o", label="treino", color="#c0504d")
        ax.plot(order, s["val_acc"], marker="s", label="validação", color="#4f81bd")
        ax.set_title(f"Árvore - {lab[ap]}")
        ax.set_xlabel("max_depth")
        ax.grid(alpha=.3)
    axs[0].set_ylabel("acurácia")
    axs[0].legend()
    plt.tight_layout()
    plt.savefig(FIG / "tree_depth.png", dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 4))
    s = grid_df[grid_df.algo == "MLP"].copy()
    s["topo"] = s["params"].str.extract(r"hidden_layer_sizes=(\([^)]*\))")[0]
    order = list(dict.fromkeys(s["topo"]))
    w = 0.27
    for i, ap in enumerate(ALL_APPROACHES):
        g = s[s.approach == ap].groupby("topo")["val_f1"].max().loc[order]
        ax.bar(np.arange(len(order)) + (i - 1) * w, g.values, w, label=lab[ap], color=col[ap])
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=20)
    ax.set_ylabel("F-measure macro (validação)")
    ax.set_title("MLP: topologia da rede (melhor ativação/alpha)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIG / "mlp_topology.png", dpi=150)
    plt.close()

    fig, axs = plt.subplots(2, 2, figsize=(12, 7), sharey=False)
    for ax, (met, title) in zip(axs.ravel(), [("test_acc", "Acurácia"), ("test_prec", "Precision (macro)"),
                                              ("test_rec", "Recall (macro)"), ("test_f1", "F-measure (macro)")]):
        for i, ap in enumerate(ALL_APPROACHES):
            s = final[final.approach == ap].set_index("algo").loc[ALGOS]
            ax.bar(np.arange(len(ALGOS)) + (i - 1) * 0.27, s[met], 0.27, label=lab[ap], color=col[ap])
        ax.set_xticks(range(len(ALGOS)))
        ax.set_xticklabels([a.replace(" de ", "\nde ").replace("Random ", "Random\n") for a in ALGOS], fontsize=8)
        ax.set_title(title + " - teste")
        ax.set_ylim(0.4, 1.02)
        ax.grid(axis="y", alpha=.3)
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=9)
    plt.tight_layout(rect=(0, 0.04, 1, 1))
    plt.savefig(FIG / "compare_test.png", dpi=150)
    plt.close()

    fig, axs = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for ax, ap in zip(axs, ALL_APPROACHES):
        s = final[final.approach == ap].set_index("algo").loc[ALGOS]
        x = np.arange(len(ALGOS))
        for j, (c, l, cc) in enumerate([("train_acc", "treino", "#c0504d"), ("val_acc", "validação", "#4f81bd"),
                                         ("test_acc", "teste", "#9bbb59")]):
            ax.bar(x + (j - 1) * 0.27, s[c], 0.27, label=l, color=cc)
        ax.set_xticks(x)
        ax.set_xticklabels([a.replace(" de ", "\nde ").replace("Random ", "Random\n") for a in ALGOS], fontsize=8)
        ax.set_title(lab[ap])
        ax.set_ylim(0.4, 1.02)
        ax.grid(axis="y", alpha=.3)
    axs[0].set_ylabel("acurácia")
    axs[0].legend()
    plt.suptitle("Overfitting: acurácia em treino / validação / teste")
    plt.tight_layout()
    plt.savefig(FIG / "overfitting.png", dpi=150)
    plt.close()

    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    for i, ap in enumerate(ALL_APPROACHES):
        s = final[final.approach == ap].set_index("algo").loc[ALGOS]
        axs[0].bar(np.arange(len(ALGOS)) + (i - 1) * 0.27, s["fit_time_s"], 0.27, label=lab[ap], color=col[ap])
        axs[1].bar(np.arange(len(ALGOS)) + (i - 1) * 0.27, s["inference_us"], 0.27, label=lab[ap], color=col[ap])
    for ax, t in zip(axs, ["Tempo de treino (s)", "Tempo de inferência por tabuleiro (µs)"]):
        ax.set_yscale("log")
        ax.set_xticks(range(len(ALGOS)))
        ax.set_xticklabels([a.replace(" de ", "\nde ").replace("Random ", "Random\n") for a in ALGOS], fontsize=8)
        ax.set_title(t)
        ax.grid(axis="y", alpha=.3)
    axs[0].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG / "cost.png", dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(9, 4))
    for i, ap in enumerate(ALL_APPROACHES):
        s = final[final.approach == ap].set_index("algo").loc[ALGOS]
        ax.bar(np.arange(len(ALGOS)) + (i - 1) * 0.27, s["exhaustive_bal_acc_unseen"], 0.27, label=lab[ap], color=col[ap])
    ax.set_xticks(range(len(ALGOS)))
    ax.set_xticklabels(ALGOS, fontsize=8)
    ax.set_ylim(0.4, 1.02)
    ax.set_ylabel("acurácia balanceada")
    ax.set_title("Generalização: TODOS os estados válidos fora do dataset (extra)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=.3)
    plt.tight_layout()
    plt.savefig(FIG / "exhaustive_unseen.png", dpi=150)
    plt.close()

    fig, axs = plt.subplots(2, 5, figsize=(18, 7.5))
    short = ["jogo", "X", "O", "emp."]
    for r, ap in enumerate(SPEC_APPROACHES):
        for c, algo in enumerate(ALGOS):
            cm = cms[(ap, algo)]
            ax = axs[r, c]
            ax.imshow(cm, cmap="Blues")
            for i in range(4):
                for j in range(4):
                    ax.text(j, i, cm[i, j], ha="center", va="center",
                            color="white" if cm[i, j] > cm.max() / 2 else "black")
            ax.set_xticks(range(4)); ax.set_yticks(range(4))
            ax.set_xticklabels(short); ax.set_yticklabels(short)
            ax.set_title(f"{algo}\n{ap}", fontsize=9)
            if c == 0:
                ax.set_ylabel("real")
            if r == 1:
                ax.set_xlabel("previsto")
    plt.tight_layout()
    plt.savefig(FIG / "confusion_matrices.png", dpi=130)
    plt.close()

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.boxplot([seeds[seeds.approach == ap]["test_acc"] for ap in SPEC_APPROACHES], tick_labels=[lab[a] for a in SPEC_APPROACHES])
    ax.set_ylabel("acurácia no teste")
    ax.set_title("MLP: variação com 10 sementes aleatórias")
    ax.grid(alpha=.3)
    plt.tight_layout()
    plt.savefig(FIG / "mlp_seeds.png", dpi=150)
    plt.close()


def figures_only():
    """Regenera os gráficos a partir dos CSV/JSON já salvos (sem retreinar)."""
    grid_df = pd.read_csv(RES / "grid_results.csv")
    final = pd.read_csv(RES / "test_results.csv")
    cms = {tuple(k.split("|")): np.array(v) for k, v in json.load(open(RES / "confusion_matrices.json")).items()}
    make_figures(grid_df, final, cms, pd.read_csv(RES / "mlp_seeds.csv"))


if __name__ == "__main__":
    if "--figures-only" in sys.argv:
        figures_only()
    else:
        main()
