# Diário de construção do dataset

## 1. Análise do dataset original (UCI Tic-Tac-Toe Endgame)

- Instâncias: **958**; atributos: 9 casas (x/o/b) + 1 rótulo.
- Valores ausentes: **0**; linhas duplicadas: **0**.
- Rótulos originais: {'positive': 626, 'negative': 332}  (`positive` = X venceu; `negative` = X **não** venceu).
- Tabuleiros legais (X começa, jogadas alternadas, 1 só vencedor): **958/958**.
- Reclassificando pelas regras do jogo: {'x_venceu': 626, 'o_venceu': 316, 'empate': 16}.
- Divergências entre o rótulo UCI e o juiz de regras (X venceu?): **0**.

### Problemas encontrados

1. **Só existem tabuleiros terminais** (jogo acabado). Nenhum tem a classe *Tem jogo* (partida em andamento) -> a classe 1 do enunciado simplesmente não existe no dataset.
2. **O rótulo é binário** (`positive`/`negative`). A classe `negative` mistura *O venceu* (316) e *Empate* (16); precisamos de 4 classes.
3. **Classe Empate muito rara**: só **16** tabuleiros. É o máximo possível: existem exatamente 16 tabuleiros cheios, legais e sem vencedor no jogo da velha. Logo é **impossível** chegar a 200 amostras únicas de empate.
4. Desbalanceamento forte: X venceu é 65% do dataset original.
5. Os atributos são texto (`x`,`o`,`b`) - precisam ser convertidos para números.
6. O dataset assume que X sempre começa (confirmado: 100% legais com essa regra), o que permite deduzir *de quem é a vez* a partir das contagens.

## 2. Adequações realizadas

- **Enumeração exaustiva** do espaço de estados (busca em largura a partir do tabuleiro vazio): **5478** tabuleiros alcançáveis -> {'tem_jogo': 4520, 'x_venceu': 626, 'o_venceu': 316, 'empate': 16}.
- Conferência: os 958 tabuleiros da UCI == conjunto de tabuleiros terminais alcançáveis? **True**. (Ou seja, a UCI é completa para jogos terminais.)
- Rótulos de 4 classes obtidos aplicando as regras do jogo aos 958 tabuleiros da UCI (X venceu=626, O venceu=316, Empate=16).
- Classe *Tem jogo*: **gerada** a partir dos 4520 tabuleiros alcançáveis não terminais.

## 3. Amostragem (conjunto balanceado e representativo)

- Meta: 200 por classe (sugestão do enunciado), **sem usar todas as instâncias**.
- Estratégia: amostragem **estratificada por nº de jogadas (ply)** dentro de cada classe, mantendo a proporção natural (ex.: X só vence nas jogadas 5, 7 e 9). Isso preserva a diversidade de situações em vez de sortear só os casos mais comuns. Mínimo de 1 por estrato.
  - tem_jogo: população 4520 -> amostra **200**
  - x_venceu: população 626 -> amostra **200**
  - o_venceu: população 316 -> amostra **200**
  - empate: população 16 -> amostra **16**  (todas as existentes: não há mais que 16 empates possíveis)
- Dataset final (estados **únicos**): **616** amostras (416 das 958 instâncias da UCI + 200 de 'tem_jogo' geradas).

## 4. Divisão treino / validação / teste (física, fixa para todos os algoritmos)

- 60% / 20% / 20% **estratificada por classe**, seed=42; os arquivos `train.csv`, `val.csv`, `test.csv` são os mesmos para todos os experimentos.
- A divisão é feita sobre estados **únicos** (nenhum tabuleiro aparece em dois conjuntos -> sem vazamento).
- **Balanceamento do treino**: a classe Empate tem só 10 amostras de treino contra 120 das demais. Replicamos aleatoriamente (com reposição) as amostras de Empate **somente no treino** até 120 (coluna `oversampled`=True). Validação e teste ficam sem réplicas (distribuição real) -> as métricas não são infladas.

Amostras por classe e conjunto (treino já com réplicas de Empate):

| split | tem_jogo | x_venceu | o_venceu | empate |
|---|---|---|---|---|
| train | 120 | 120 | 120 | 120 |
| val | 40 | 40 | 40 | 3 |
| test | 40 | 40 | 40 | 3 |

Amostras **únicas** por classe e conjunto:

| split | tem_jogo | x_venceu | o_venceu | empate |
|---|---|---|---|---|
| train | 120 | 120 | 120 | 10 |
| val | 40 | 40 | 40 | 3 |
| test | 40 | 40 | 40 | 3 |
