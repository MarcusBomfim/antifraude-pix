"""As contas do retrato: como cada golpe se parece com o Pix comum, e onde ele se afasta.

Aqui a verdade do simulador é usada de propósito: o retrato mostra o que o
simulador montou, golpe por golpe, para quem for avaliar se ele parece o
mundo. O modelo não lê nada disto (ADR-0004).
"""

from datetime import timedelta

import polars as pl

from antifraude_pix.simulacao.golpes import GOLPES, NOMES
from antifraude_pix.simulacao.simulador import Simulacao

LEGITIMO = "Pix legítimo"
GRUPOS: tuple[str, ...] = (LEGITIMO, *(NOMES[golpe] for golpe in GOLPES))
"""O Pix legítimo e cada golpe, na ordem em que aparecem nas figuras."""


def _com_grupo(simulacao: Simulacao) -> pl.DataFrame:
    """Cada Pix com o grupo a que pertence: legítimo, ou o nome do golpe."""
    nomes = pl.DataFrame({"golpe": [golpe.value for golpe in GOLPES], "grupo": [NOMES[golpe] for golpe in GOLPES]})
    fraudes = simulacao.verdade.fraudes.with_columns(pl.col("golpe").cast(pl.String)).join(nomes, on="golpe")

    return simulacao.transacoes.join(
        fraudes.select("transacao", "grupo"), left_on="id", right_on="transacao", how="left"
    ).with_columns(pl.col("grupo").fill_null(LEGITIMO))


def por_hora(simulacao: Simulacao) -> pl.DataFrame:
    """A parte dos Pix de cada grupo em cada hora do dia: `grupo`, `hora`, `parte`."""
    horas = pl.DataFrame({"hora": list(range(24))}, schema={"hora": pl.Int8})
    grupos = pl.DataFrame({"grupo": list(GRUPOS)})

    contagem = _com_grupo(simulacao).group_by("grupo", pl.col("momento").dt.hour().cast(pl.Int8).alias("hora")).len()

    return (
        grupos.join(horas, how="cross")
        .join(contagem, on=["grupo", "hora"], how="left")
        .with_columns(pl.col("len").fill_null(0))
        .with_columns((pl.col("len") / pl.col("len").sum().over("grupo")).fill_nan(0).alias("parte"))
        .select("grupo", "hora", "parte")
        .sort(pl.col("grupo").replace_strict(list(GRUPOS), list(range(len(GRUPOS)))), "hora")
    )


def faixas_de_valor(simulacao: Simulacao) -> pl.DataFrame:
    """O valor de cada grupo, em reais: os percentis 10, 25, 50, 75 e 90, e quantos Pix."""
    valor = pl.col("valor") / 100

    return (
        _com_grupo(simulacao)
        .group_by("grupo")
        .agg(
            valor.quantile(0.10).alias("p10"),
            valor.quantile(0.25).alias("p25"),
            valor.median().alias("mediana"),
            valor.quantile(0.75).alias("p75"),
            valor.quantile(0.90).alias("p90"),
            pl.len().alias("pix"),
        )
        .sort(pl.col("grupo").replace_strict(list(GRUPOS), list(range(len(GRUPOS)))))
    )


def demora_da_contestacao(simulacao: Simulacao, ate_dias: int = 30) -> pl.DataFrame:
    """A parte dos ataques de cada golpe já contestada, dia a dia, depois do último Pix.

    As colunas: `golpe`, `dias`, `contestados`.
    """
    ataques = (
        simulacao.verdade.fraudes.join(
            simulacao.transacoes.select(pl.col("id").alias("transacao"), "momento"), on="transacao"
        )
        .join(simulacao.contestacoes, on="transacao", how="left")
        .group_by("ataque")
        .agg(
            pl.col("golpe").first(),
            ((pl.col("contestada_em").min() - pl.col("momento").max()).dt.total_seconds() / 86_400).alias("demora"),
        )
    )
    marcas = pl.DataFrame({"dias": [passo / 4 for passo in range(ate_dias * 4 + 1)]})

    return (
        ataques.join(marcas, how="cross")
        .group_by(pl.col("golpe").cast(pl.String), "dias")
        .agg((pl.col("demora").is_not_null() & (pl.col("demora") <= pl.col("dias"))).mean().alias("contestados"))
        .sort("golpe", "dias")
    )


FAIXAS_DE_PAGADORES: tuple[str, ...] = ("1", "2", "3 a 5", "6 a 10", "mais de 10")


def pagadores_novos_por_recebedor(simulacao: Simulacao, aquecimento: int = 30) -> pl.DataFrame:
    """Quantos clientes pagaram cada recebedor pela primeira vez, na semana mais movimentada dele, por tipo de conta.

    É o rastro da laranja: ela recebe de muitos desconhecidos em poucos dias. A
    pessoa comum recebe de quem já a pagava antes, e o comércio popular recebe
    de muita gente nova — por isso ele aparece no retrato, ao lado. Os
    primeiros dias ficam de fora: no começo da simulação, todo pagador é novo,
    porque a história começa ali.
    """
    laranjas = simulacao.verdade.laranjas.select(
        pl.col("conta").alias("recebedor"), pl.lit(value=True).alias("laranja")
    )
    corte = simulacao.config.inicio + timedelta(days=aquecimento)
    novos = (
        simulacao.transacoes.with_columns(
            (pl.int_range(pl.len()).over("cliente", "recebedor") == 0).alias("primeira_vez")
        )
        .filter(pl.col("primeira_vez") & (pl.col("momento").dt.date() >= corte))
        .group_by("recebedor", "recebedor_pessoa", pl.col("momento").dt.truncate("1w").alias("semana"))
        .agg(pl.col("cliente").n_unique().alias("pagadores"))
        .group_by("recebedor", "recebedor_pessoa")
        .agg(pl.col("pagadores").max())
        .join(laranjas, on="recebedor", how="left")
        .with_columns(
            pl.when(pl.col("laranja").fill_null(value=False))
            .then(pl.lit("conta laranja"))
            .when(pl.col("recebedor_pessoa") == "PJ")
            .then(pl.lit("comércio"))
            .otherwise(pl.lit("pessoa"))
            .alias("tipo"),
            pl.col("pagadores")
            .cut([1, 2, 5, 10], labels=list(FAIXAS_DE_PAGADORES), left_closed=False)
            .cast(pl.String)
            .alias("faixa"),
        )
    )

    return (
        novos.group_by("tipo", "faixa")
        .len()
        .with_columns((pl.col("len") / pl.col("len").sum().over("tipo")).alias("parte"))
        .select("tipo", "faixa", "parte")
        .sort("tipo", pl.col("faixa").replace_strict(list(FAIXAS_DE_PAGADORES), list(range(len(FAIXAS_DE_PAGADORES)))))
    )
