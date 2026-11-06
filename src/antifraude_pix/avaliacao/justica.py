"""Os erros por idade: o modelo não pode errar mais com os mais velhos (ADR-0011).

O golpe mira os mais velhos: o falso parente e a falsa central escolhem a
vítima pela idade. O modelo pode errar com eles de dois jeitos:
- **deixar passar** mais os golpes contra eles;
- **segurar ou bloquear** mais o Pix honesto deles, por se parecerem com as vítimas.

A medida separa os clientes em faixas de idade e conta, em cada faixa, os dois
erros, e os golpes que o modelo previu contra os que houve.
"""

import polars as pl

from antifraude_pix.decisao.custo import Acao

FAIXAS_DE_IDADE: tuple[tuple[int, int | None], ...] = ((18, 34), (35, 54), (55, 64), (65, None))
"""As faixas de idade, de quantos anos a quantos; a última não tem fim."""

FAIXA = pl.Enum([f"{de} a {ate}" if ate is not None else f"{de} ou mais" for de, ate in FAIXAS_DE_IDADE])
"""A faixa de idade como categoria fechada, da mais nova para a mais velha."""

POR = 10_000
"""Os Pix honestos segurados e bloqueados contam-se a cada dez mil: o bloqueio é raro, e a cada mil seria zero."""


def faixa_de_idade(idade: pl.Expr) -> pl.Expr:
    """A faixa de cada idade. Antes da primeira, fica vazia: os clientes são adultos."""
    faixa = pl.lit(None, dtype=pl.String)

    # A faixa mais velha fica por fora: a primeira condição que vale é a do maior começo que a idade alcança.
    for (de, _), nome in zip(FAIXAS_DE_IDADE, FAIXA.categories.to_list(), strict=True):
        faixa = pl.when(idade >= de).then(pl.lit(nome)).otherwise(faixa)

    return faixa.cast(FAIXA)


def por_idade(decisoes: pl.DataFrame) -> pl.DataFrame:
    """Uma linha por modelo e faixa de idade, de todos os meses de teste juntos.

    A entrada tem, para cada Pix, o `modelo`, a `acao`, a `probabilidade`, o
    `valor`, o `golpe` (vazio no legítimo) e a `idade` do cliente. A saída tem:
    - `golpes` e `golpes_pegos`: os segurados ou bloqueados; e `recall` e `recall_em_reais`;
    - `honestos`, e os `honestos_segurados` e `honestos_bloqueados` a cada dez mil Pix honestos;
    - `golpes_previstos`: a soma da probabilidade, para comparar com os golpes que houve.
    """
    com_a_faixa = decisoes.with_columns(faixa_de_idade(pl.col("idade")).alias("faixa"))

    if com_a_faixa["faixa"].null_count():
        raise ValueError(f"Os clientes são adultos: há idade abaixo de {FAIXAS_DE_IDADE[0][0]} anos, ou sem idade.")

    golpe = pl.col("golpe").is_not_null()
    pego = pl.col("acao") != Acao.APROVAR.value
    honestos = (~golpe).sum()
    # Os modelos na ordem em que vêm, e as faixas da mais nova para a mais velha.
    ordem = pl.Enum(decisoes["modelo"].unique(maintain_order=True).to_list())

    return (
        com_a_faixa.group_by("modelo", "faixa", maintain_order=True)
        .agg(
            golpe.sum().alias("golpes"),
            (golpe & pego).sum().alias("golpes_pegos"),
            ((golpe & pego).sum() / golpe.sum()).alias("recall"),
            (
                pl.when(golpe & pego).then(pl.col("valor")).otherwise(0.0).sum()
                / pl.when(golpe).then(pl.col("valor")).otherwise(0.0).sum()
            ).alias("recall_em_reais"),
            honestos.alias("honestos"),
            ((~golpe & (pl.col("acao") == Acao.SEGURAR.value)).sum() / honestos * POR).alias("honestos_segurados"),
            ((~golpe & (pl.col("acao") == Acao.BLOQUEAR.value)).sum() / honestos * POR).alias("honestos_bloqueados"),
            pl.col("probabilidade").sum().alias("golpes_previstos"),
        )
        .sort(pl.col("modelo").cast(ordem), "faixa")
    )
