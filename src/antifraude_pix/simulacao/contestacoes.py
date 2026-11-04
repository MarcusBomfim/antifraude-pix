"""A contestação: o rótulo que o banco tem, que chega depois, e que nem sempre chega (ADR-0004).

O banco não sabe na hora que um Pix foi golpe. Sabe quando a vítima reclama,
pelo aplicativo ou pelo MED — horas depois, na invasão de conta, que a vítima
percebe logo; dias depois, na venda falsa, quando o produto não chega. Parte
das vítimas nunca reclama: tem vergonha, não sabe que pode, ou acha que não
adianta. E há quem conteste um Pix legítimo, por arrependimento ou má-fé.

É esse rótulo atrasado e sujo que um modelo de verdade tem para aprender, e é
com ele que o projeto treina. A verdade do simulador fica para medir.
"""

import numpy as np
import polars as pl

from antifraude_pix.simulacao.configuracao import Configuracao
from antifraude_pix.simulacao.golpes import Golpe
from antifraude_pix.simulacao.pix import DIA
from antifraude_pix.simulacao.sorteio import Etapa, sorteio

# Para cada golpe: a chance de a vítima contestar, a demora típica em dias, e o
# quanto ela varia (o desvio do logaritmo).
CONTESTACAO: dict[Golpe, tuple[float, float, float]] = {
    Golpe.INVASAO_DE_CONTA: (0.95, 0.3, 1.2),
    Golpe.FALSA_CENTRAL: (0.90, 0.4, 1.0),
    Golpe.FALSO_PARENTE: (0.85, 1.5, 1.0),
    Golpe.VENDA_FALSA: (0.60, 8.0, 0.7),
}

FALSAS_POR_MILHAO = 50
"""Quantos Pix legítimos, a cada milhão, são contestados mesmo assim: o arrependimento, a má-fé."""

DEMORA_MINIMA = 10 * 60
"""Ninguém contesta antes de dez minutos: é o tempo de perceber, entrar no aplicativo e achar o botão."""


def contestar(config: Configuracao, transacoes: pl.DataFrame, fraudes: pl.DataFrame) -> pl.DataFrame:
    """As contestações, a partir dos Pix (`id`, `instante`) e da verdade de cada golpe (`transacao`, `ataque`, `golpe`).

    Cada vítima que reclama contesta o ataque inteiro, de uma vez, depois do
    último Pix dele. Toda contestação cai dentro do prazo do MED.
    """
    prazo = config.prazo_do_med * DIA
    linhas: list[tuple[int, int]] = []

    por_ataque = (
        fraudes.join(transacoes.select(pl.col("id").alias("transacao"), "instante"), on="transacao")
        .group_by("ataque")
        .agg(
            pl.col("golpe").first(),
            pl.col("transacao"),
            pl.col("instante").min().alias("primeiro"),
            pl.col("instante").max().alias("ultimo"),
        )
        .sort("ataque")
    )

    for ataque, golpe, ids, primeiro, ultimo in por_ataque.iter_rows():
        chance, mediana, desvio = CONTESTACAO[Golpe(golpe)]
        gerador = sorteio(config, Etapa.CONTESTACOES, ataque)

        if gerador.random() >= chance:
            continue

        demora = float(np.exp(gerador.normal(np.log(mediana), desvio))) * DIA
        demora = int(min(max(demora, DEMORA_MINIMA), prazo - (ultimo - primeiro)))
        linhas.extend((transacao, ultimo + demora) for transacao in ids)

    linhas.extend(_contestacoes_falsas(config, transacoes, fraudes))

    return (
        pl.DataFrame(linhas, schema={"transacao": pl.Int64, "instante": pl.Int64}, orient="row")
        .sort("instante", "transacao")
        .select("transacao", "instante")
    )


def _contestacoes_falsas(
    config: Configuracao, transacoes: pl.DataFrame, fraudes: pl.DataFrame
) -> list[tuple[int, int]]:
    gerador = sorteio(config, Etapa.CONTESTACOES)
    legitimas = transacoes.join(fraudes, left_on="id", right_on="transacao", how="anti").select("id", "instante")
    quantas = int(gerador.binomial(legitimas.height, FALSAS_POR_MILHAO / 1_000_000))
    escolhidas = gerador.choice(legitimas.height, size=quantas, replace=False)
    demoras = gerador.integers(DIA, 30 * DIA, size=quantas)

    return [
        (int(legitimas["id"][int(linha)]), int(legitimas["instante"][int(linha)]) + int(demora))
        for linha, demora in zip(escolhidas, demoras, strict=True)
    ]
