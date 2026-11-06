"""As métricas para classe rara (ADR-0009).

Com um golpe em cada mil e quinhentos Pix, a taxa de acerto não diz nada: o
modelo que nunca acusa ninguém acerta 99,93%. A área sob a curva ROC também
engana, porque mede a fila inteira, e o banco só olha o começo dela. O que
importa é o topo:
- **a precisão média** (a área sob a curva de precisão e recall), que pesa o começo da fila;
- **o alerta**: dos 0,5% de Pix que o banco consegue segurar para confirmar, quantos golpes ele pega, quanto
  dinheiro ele salva, e quantos clientes honestos ele segura para cada golpe;
- **cada golpe separado**, porque o modelo que só pega a invasão de conta parece bom no total;
- **a calibração**: quantos golpes o modelo previu, contra quantos houve.
"""

from itertools import pairwise

import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score, roc_auc_score

ALERTA = 0.005
"""A parte dos Pix do mês que o banco consegue segurar para confirmar com o cliente: os de maior pontuação."""

FAIXAS: tuple[float, ...] = (0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0)
"""As faixas da probabilidade prevista para a calibração, de dez em dez vezes: quase todo Pix fica perto de zero."""


def com_o_alerta(pontuacoes: pl.DataFrame, alerta: float = ALERTA) -> pl.DataFrame:
    """As pontuações com a coluna `alertado`: os Pix do topo de cada modelo, em cada mês, até a parte do alerta.

    O empate na fronteira não entra: se os empatados não cabem todos, nenhum
    deles é segurado. Para as regras, que dão a mesma soma a muitos Pix, o
    alerta fica abaixo da parte; o banco não segura metade dos Pix de três
    pontos.
    """
    if not 0 < alerta < 1:
        raise ValueError(f"O alerta é uma parte dos Pix, entre 0 e 1, e não {alerta}.")

    mes = ["modelo", "corte"]
    cabem = (pl.len().over(mes) * alerta).floor()
    # O lugar "max" de um Pix é quantos Pix do mês têm pontuação igual ou maior que a dele.
    lugar = pl.col("pontuacao").rank("max", descending=True).over(mes)

    return pontuacoes.with_columns((lugar <= cabem).alias("alertado"))


def resumo(pontuacoes: pl.DataFrame, alerta: float = ALERTA) -> pl.DataFrame:
    """Uma linha por modelo, com todos os meses de teste juntos."""
    linhas = []

    for (modelo,), dele in com_o_alerta(pontuacoes, alerta).group_by("modelo", maintain_order=True):
        golpe = dele["golpe"].is_not_null()
        golpes = int(golpe.sum())
        pegos = dele.filter(pl.col("alertado") & golpe)
        segurados = dele.filter(pl.col("alertado") & ~golpe).height
        linhas.append(
            {
                "modelo": modelo,
                "precisao_media": _precisao_media(golpe, dele["probabilidade"]),
                "area_roc": float(roc_auc_score(golpe.to_numpy(), dele["probabilidade"].to_numpy())),
                "recall_no_alerta": pegos.height / golpes,
                "recall_em_reais": float(pegos["valor"].sum()) / float(dele.filter(golpe)["valor"].sum()),
                "precisao_no_alerta": pegos.height / max(int(dele["alertado"].sum()), 1),
                "legitimos_por_golpe_pego": segurados / pegos.height if pegos.height else None,
                "golpes": golpes,
                "golpes_previstos": float(dele["probabilidade"].sum()),
            }
        )

    return pl.DataFrame(linhas)


def por_golpe(pontuacoes: pl.DataFrame, alerta: float = ALERTA) -> pl.DataFrame:
    """Uma linha por modelo e golpe: a precisão média contra o Pix legítimo, e o recall no alerta."""
    linhas = []

    for (modelo,), dele in com_o_alerta(pontuacoes, alerta).group_by("modelo", maintain_order=True):
        legitimos = dele.filter(pl.col("golpe").is_null())

        for (golpe,), do_golpe in dele.filter(pl.col("golpe").is_not_null()).group_by("golpe", maintain_order=True):
            contra = pl.concat([legitimos, do_golpe])
            linhas.append(
                {
                    "modelo": modelo,
                    "golpe": golpe,
                    "precisao_media": _precisao_media(contra["golpe"].is_not_null(), contra["probabilidade"]),
                    "recall_no_alerta": int(do_golpe["alertado"].sum()) / do_golpe.height,
                    "golpes": do_golpe.height,
                }
            )

    return pl.DataFrame(linhas).sort("modelo", "golpe", maintain_order=True)


def confiabilidade(pontuacoes: pl.DataFrame, faixas: tuple[float, ...] = FAIXAS) -> pl.DataFrame:
    """Em cada faixa de probabilidade prevista: a média prevista, a parte que foi golpe de verdade, e quantos Pix."""
    rotulos = [f"{de:g} a {ate:g}" for de, ate in pairwise(faixas)]

    return (
        pontuacoes.with_columns(
            pl.col("probabilidade").cut(list(faixas[1:-1]), labels=rotulos, left_closed=True).alias("faixa"),
            pl.col("golpe").is_not_null().alias("foi_golpe"),
        )
        .group_by("modelo", "faixa", maintain_order=True)
        .agg(
            pl.col("probabilidade").mean().alias("prevista"),
            pl.col("foi_golpe").mean().alias("observada"),
            pl.len().alias("pix"),
        )
        .sort("modelo", "faixa", maintain_order=True)
    )


def _precisao_media(golpe: pl.Series, probabilidade: pl.Series) -> float:
    if golpe.all() or not golpe.any():
        raise ValueError("A precisão média pede Pix de golpe e Pix legítimos.")

    return float(average_precision_score(golpe.to_numpy(), np.asarray(probabilidade.to_numpy(), dtype=np.float64)))


def curva_do_alerta(pontuacoes: pl.DataFrame) -> pl.DataFrame:
    """A parte dos golpes pegos para cada parte dos Pix segurados em cada mês: `modelo`, `parte_segurada`, `recall`.

    É a mesma conta do alerta. Um golpe é pego quando a parte segurada no mês
    dele alcança o lugar dele na fila, contados os empatados com ele. Há um
    ponto em cada golpe, do mais fácil ao mais difícil.
    """
    mes = ["modelo", "corte"]
    lugar = pl.col("pontuacao").rank("max", descending=True).over(mes)

    return (
        pontuacoes.with_columns((lugar / pl.len().over(mes)).alias("parte_segurada"))
        .filter(pl.col("golpe").is_not_null())
        .sort("modelo", "parte_segurada", maintain_order=True)
        .select(
            "modelo",
            "parte_segurada",
            ((pl.int_range(pl.len()).over("modelo") + 1) / pl.len().over("modelo")).alias("recall"),
        )
    )
