"""Etapa 1 - Construção, análise e divisão do dataset.

Uso (a partir da raiz do projeto):  python src/build_dataset.py

Gera:
  data/processed/dataset_balanceado.csv   (dataset final com coluna 'split')
  data/processed/{train,val,test}.csv     (divisão física)
  results/dataset_stats.json              (números usados no relatório)
  docs/dataset_log.md                     (diário de problemas e decisões)
  results/figures/dataset_*.png
"""
from __future__ import annotations

import json
import sys
import urllib.request
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from ttt.game import CLASSES, enumerate_reachable, is_legal, true_state

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "tic-tac-toe.data"
OUT = ROOT / "data" / "processed"
RES = ROOT / "results"
FIG = RES / "figures"
DOCS = ROOT / "docs"

SEED = 42
N_PER_CLASS = 200
FRAC_VAL = FRAC_TEST = 0.20
COLS = ["tl", "tm", "tr", "ml", "mm", "mr", "bl", "bm", "br"]
UCI_URL = "https://archive.ics.uci.edu/static/public/101/tic+tac+toe+endgame.zip"


def download_if_missing():
    if RAW.exists():
        return
    import io
    import zipfile
    RAW.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(UCI_URL, timeout=60) as r:
        z = zipfile.ZipFile(io.BytesIO(r.read()))
    z.extractall(RAW.parent)


def md_table(df: pd.DataFrame) -> str:
    cols = [df.index.name or ""] + [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for idx, row in df.iterrows():
        lines.append("| " + " | ".join([str(idx)] + [str(v) for v in row]) + " |")
    return "\n".join(lines)


def allocate(counts: dict[int, int], total: int, min_each: int = 1) -> dict[int, int]:
    """Alocação proporcional por estrato (ply), método do maior resto, mínimo por estrato."""
    keys = sorted(counts)
    n_all = sum(counts.values())
    total = min(total, n_all)
    raw = {k: total * counts[k] / n_all for k in keys}
    alloc = {k: min(counts[k], max(min_each, int(raw[k]))) for k in keys}
    while sum(alloc.values()) > total:
        k = max((k for k in keys if alloc[k] > min_each), key=lambda k: alloc[k] - raw[k])
        alloc[k] -= 1
    while sum(alloc.values()) < total:
        cand = [k for k in keys if alloc[k] < counts[k]]
        k = max(cand, key=lambda k: raw[k] - alloc[k])
        alloc[k] += 1
    return alloc


def stratified_sample(df: pd.DataFrame, n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Amostra n linhas estratificando por 'ply' (nº de jogadas) para manter a
    distribuição natural de cada classe."""
    counts = df["ply"].value_counts().to_dict()
    alloc = allocate(counts, n)
    parts = []
    for ply, k in alloc.items():
        sub = df[df["ply"] == ply]
        idx = rng.choice(sub.index.to_numpy(), size=k, replace=False)
        parts.append(df.loc[idx])
    return pd.concat(parts)


def main():
    download_if_missing()
    rng = np.random.default_rng(SEED)
    log: list[str] = []
    stats: dict = {}

    def say(s=""):
        log.append(s)
        print(s)

    raw = pd.read_csv(RAW, header=None, names=COLS + ["label_uci"])
    say("# Diário de construção do dataset\n")
    say("## 1. Análise do dataset original (UCI Tic-Tac-Toe Endgame)\n")
    say(f"- Instâncias: **{len(raw)}**; atributos: 9 casas (x/o/b) + 1 rótulo.")
    say(f"- Valores ausentes: **{int(raw.isna().sum().sum())}**; linhas duplicadas: **{int(raw.duplicated().sum())}**.")
    vc = raw["label_uci"].value_counts().to_dict()
    say(f"- Rótulos originais: {vc}  (`positive` = X venceu; `negative` = X **não** venceu).")
    stats["uci_total"] = len(raw)
    stats["uci_labels"] = vc

    boards = [tuple(r) for r in raw[COLS].to_numpy()]
    legal = sum(is_legal(b) for b in boards)
    say(f"- Tabuleiros legais (X começa, jogadas alternadas, 1 só vencedor): **{legal}/{len(raw)}**.")
    true = [true_state(b) for b in boards]
    tc = Counter(true)
    say(f"- Reclassificando pelas regras do jogo: {dict(tc)}.")
    mismatch = sum((t == "x_venceu") != (l == "positive") for t, l in zip(true, raw["label_uci"]))
    say(f"- Divergências entre o rótulo UCI e o juiz de regras (X venceu?): **{mismatch}**.")
    stats["uci_relabel"] = dict(tc)

    say("\n### Problemas encontrados\n")
    say("1. **Só existem tabuleiros terminais** (jogo acabado). Nenhum tem a classe *Tem jogo* "
        "(partida em andamento) -> a classe 1 do enunciado simplesmente não existe no dataset.")
    say("2. **O rótulo é binário** (`positive`/`negative`). A classe `negative` mistura "
        f"*O venceu* ({tc['o_venceu']}) e *Empate* ({tc['empate']}); precisamos de 4 classes.")
    say(f"3. **Classe Empate muito rara**: só **{tc['empate']}** tabuleiros. É o máximo possível: "
        "existem exatamente 16 tabuleiros cheios, legais e sem vencedor no jogo da velha. "
        "Logo é **impossível** chegar a 200 amostras únicas de empate.")
    say("4. Desbalanceamento forte: X venceu é 65% do dataset original.")
    say("5. Os atributos são texto (`x`,`o`,`b`) - precisam ser convertidos para números.")
    say("6. O dataset assume que X sempre começa (confirmado: 100% legais com essa regra), "
        "o que permite deduzir *de quem é a vez* a partir das contagens.")

    say("\n## 2. Adequações realizadas\n")
    reach = enumerate_reachable()
    reach_cls = Counter(true_state(b) for b in reach)
    say(f"- **Enumeração exaustiva** do espaço de estados (busca em largura a partir do tabuleiro vazio): "
        f"**{len(reach)}** tabuleiros alcançáveis -> {dict(reach_cls)}.")
    uci_set = set(boards)
    terminal_reach = {b for b in reach if true_state(b) != "tem_jogo"}
    say(f"- Conferência: os 958 tabuleiros da UCI == conjunto de tabuleiros terminais alcançáveis? "
        f"**{uci_set == terminal_reach}**. (Ou seja, a UCI é completa para jogos terminais.)")
    stats["reachable_total"] = len(reach)
    stats["reachable_by_class"] = dict(reach_cls)
    stats["uci_equals_terminal"] = bool(uci_set == terminal_reach)

    all_df = pd.DataFrame(
        [(*b, ply, true_state(b)) for b, ply in reach.items()],
        columns=COLS + ["ply", "classe"],
    )
    terminal = all_df[all_df.apply(lambda r: tuple(r[COLS]) in uci_set, axis=1)]
    playing = all_df[all_df["classe"] == "tem_jogo"]
    say(f"- Rótulos de 4 classes obtidos aplicando as regras do jogo aos 958 tabuleiros da UCI "
        f"(X venceu={tc['x_venceu']}, O venceu={tc['o_venceu']}, Empate={tc['empate']}).")
    say(f"- Classe *Tem jogo*: **gerada** a partir dos {len(playing)} tabuleiros alcançáveis não terminais.")

    say("\n## 3. Amostragem (conjunto balanceado e representativo)\n")
    say(f"- Meta: {N_PER_CLASS} por classe (sugestão do enunciado), **sem usar todas as instâncias**.")
    say("- Estratégia: amostragem **estratificada por nº de jogadas (ply)** dentro de cada classe, "
        "mantendo a proporção natural (ex.: X só vence nas jogadas 5, 7 e 9). Isso preserva a "
        "diversidade de situações em vez de sortear só os casos mais comuns. Mínimo de 1 por estrato.")
    parts = []
    for cls in CLASSES:
        pool = playing if cls == "tem_jogo" else terminal[terminal["classe"] == cls]
        n = min(N_PER_CLASS, len(pool))
        parts.append(stratified_sample(pool, n, rng))
        say(f"  - {cls}: população {len(pool)} -> amostra **{n}**"
            + ("  (todas as existentes: não há mais que 16 empates possíveis)" if n < N_PER_CLASS else ""))
    ds = pd.concat(parts).reset_index(drop=True)
    assert not ds.duplicated(subset=COLS).any()
    n_uci_used = int((ds["classe"] != "tem_jogo").sum())
    say(f"- Dataset final (estados **únicos**): **{len(ds)}** amostras "
        f"({n_uci_used} das {len(raw)} instâncias da UCI + {len(ds) - n_uci_used} de 'tem_jogo' geradas).")

    say("\n## 4. Divisão treino / validação / teste (física, fixa para todos os algoritmos)\n")
    say(f"- 60% / 20% / 20% **estratificada por classe**, seed={SEED}; os arquivos "
        "`train.csv`, `val.csv`, `test.csv` são os mesmos para todos os experimentos.")
    say("- A divisão é feita sobre estados **únicos** (nenhum tabuleiro aparece em dois conjuntos "
        "-> sem vazamento).")
    split = pd.Series("train", index=ds.index)
    for cls in CLASSES:
        idx = ds.index[ds["classe"] == cls].to_numpy()
        idx = rng.permutation(idx)
        n_t = int(round(len(idx) * FRAC_TEST))
        n_v = int(round(len(idx) * FRAC_VAL))
        split.loc[idx[:n_t]] = "test"
        split.loc[idx[n_t:n_t + n_v]] = "val"
    ds["split"] = split
    ds["oversampled"] = False

    tr = ds[ds["split"] == "train"]
    target = tr["classe"].value_counts().max()
    extra = []
    for cls in CLASSES:
        sub = tr[tr["classe"] == cls]
        miss = target - len(sub)
        if miss > 0:
            pick = sub.sample(n=miss, replace=True, random_state=SEED).copy()
            pick["oversampled"] = True
            extra.append(pick)
    if extra:
        ds_train_bal = pd.concat([tr, *extra]).reset_index(drop=True)
    else:
        ds_train_bal = tr.reset_index(drop=True)
    say(f"- **Balanceamento do treino**: a classe Empate tem só {int((tr['classe']=='empate').sum())} amostras de "
        f"treino contra {target} das demais. Replicamos aleatoriamente (com reposição) as amostras de Empate "
        f"**somente no treino** até {target} (coluna `oversampled`=True). Validação e teste ficam "
        "sem réplicas (distribuição real) -> as métricas não são infladas.")

    val = ds[ds["split"] == "val"].reset_index(drop=True)
    test = ds[ds["split"] == "test"].reset_index(drop=True)
    OUT.mkdir(parents=True, exist_ok=True)
    ds_train_bal.to_csv(OUT / "train.csv", index=False)
    val.to_csv(OUT / "val.csv", index=False)
    test.to_csv(OUT / "test.csv", index=False)
    full = pd.concat([ds_train_bal.assign(split="train"), val, test]).reset_index(drop=True)
    full.to_csv(OUT / "dataset_balanceado.csv", index=False)

    tbl = pd.crosstab(full["split"], full["classe"])[list(CLASSES)].loc[["train", "val", "test"]]
    uniq = pd.crosstab(ds["split"], ds["classe"])[list(CLASSES)].loc[["train", "val", "test"]]
    say("\nAmostras por classe e conjunto (treino já com réplicas de Empate):\n")
    say(md_table(tbl))
    say("\nAmostras **únicas** por classe e conjunto:\n")
    say(md_table(uniq))
    stats["split_table"] = tbl.to_dict()
    stats["split_table_unique"] = uniq.to_dict()
    stats["final_unique"] = len(ds)
    stats["n_per_class_unique"] = ds["classe"].value_counts().to_dict()
    stats["ply_by_class"] = {c: ds[ds["classe"] == c]["ply"].value_counts().sort_index().to_dict()
                             for c in CLASSES}

    RES.mkdir(exist_ok=True)
    DOCS.mkdir(exist_ok=True)
    (RES / "dataset_stats.json").write_text(json.dumps(stats, indent=2, default=int), encoding="utf-8")
    (DOCS / "dataset_log.md").write_text("\n".join(log) + "\n", encoding="utf-8")
    plot(raw, tc, ds, tbl)


def plot(raw, tc, ds, tbl):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    orig = [0, tc["x_venceu"], tc["o_venceu"], tc["empate"]]
    ax[0].bar(CLASSES, orig, color="#c0504d")
    ax[0].set_title("UCI original (reclassificado em 4 classes)")
    for i, v in enumerate(orig):
        ax[0].text(i, v + 5, str(v), ha="center")
    uni = [int((ds["classe"] == c).sum()) for c in CLASSES]
    ax[1].bar(CLASSES, uni, color="#4f81bd")
    ax[1].set_title("Dataset final (amostras únicas)")
    for i, v in enumerate(uni):
        ax[1].text(i, v + 3, str(v), ha="center")
    for a in ax:
        a.set_ylabel("nº de tabuleiros")
        a.tick_params(axis="x", rotation=15)
    plt.tight_layout()
    plt.savefig(FIG / "dataset_classes.png", dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(7, 4))
    plies = sorted(ds["ply"].unique())
    bottom = np.zeros(len(plies))
    colors = ["#4f81bd", "#9bbb59", "#c0504d", "#f79646"]
    for c, col in zip(CLASSES, colors):
        v = np.array([((ds["classe"] == c) & (ds["ply"] == p)).sum() for p in plies])
        ax.bar(plies, v, bottom=bottom, label=c, color=col)
        bottom += v
    ax.set_xlabel("nº de jogadas já feitas (ply)")
    ax.set_ylabel("amostras")
    ax.set_title("Distribuição do dataset final por nº de jogadas")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIG / "dataset_ply.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
