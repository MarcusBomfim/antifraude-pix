"""O quanto cada variável, sozinha, separa cada golpe do Pix legítimo.

A medida é a área sob a curva ROC: a chance de um Pix de golpe ficar acima de
um legítimo, pela variável. Meio a meio é não separar nada; um é separar
tudo. O sentido não importa aqui — o valor alto denuncia uns golpes, o baixo,
outros —, e a força vai de 0,5 a 1.

É uma análise da verdade do simulador, de propósito: mostra o que cada golpe
deixa de rastro antes de haver modelo, e o que fica difícil. O modelo não lê
nada disto.
"""

from datetime import timedelta

import polars as pl

from antifraude_pix.simulacao.golpes import GOLPES
from antifraude_pix.simulacao.simulador import Simulacao
from antifraude_pix.variaveis.catalogo import CATALOGO, Tipo

AQUECIMENTO = 30
"""Os primeiros dias ficam de fora: no começo da simulação, todo recebedor é novo, porque a história começa ali."""

ANALISADAS: tuple[str, ...] = tuple(variavel.nome for variavel in CATALOGO if variavel.tipo is not Tipo.CATEGORIA)
"""As variáveis de número e de sim ou não; as de categoria não têm ordem para a curva."""


def forca(valores: pl.Series, golpe: pl.Series) -> float:
    """A área sob a curva ROC da variável para o golpe, no sentido em que ela separa melhor: de 0,5 a 1.

    O vazio entra como o menor valor de todos: o primeiro Pix sem passado, o
    aparelho de sempre sem data de entrada. Os empates dividem a vez.
    """
    numeros = valores.cast(pl.Float64)
    menor = numeros.min()
    piso = (menor if isinstance(menor, float) else 0.0) - 1
    ordem = numeros.fill_null(piso).rank("average")
    positivos = int(golpe.sum())
    negativos = golpe.len() - positivos

    if positivos == 0 or negativos == 0:
        raise ValueError("A força de uma variável pede Pix de golpe e Pix legítimos.")

    area = (float(ordem.filter(golpe).sum()) - positivos * (positivos + 1) / 2) / (positivos * negativos)

    return max(area, 1 - area)


def forcas(simulacao: Simulacao, variaveis: pl.DataFrame) -> pl.DataFrame:
    """A força de cada variável para cada golpe, contra o Pix legítimo: `variavel`, `golpe`, `forca`."""
    inicio = simulacao.config.inicio + timedelta(days=AQUECIMENTO)
    rotulados = (
        variaveis.join(simulacao.transacoes.select("id", "momento"), on="id")
        .filter(pl.col("momento").dt.date() >= inicio)
        .join(
            simulacao.verdade.fraudes.select(pl.col("transacao").alias("id"), pl.col("golpe").cast(pl.String)),
            on="id",
            how="left",
        )
    )
    legitimos = rotulados.filter(pl.col("golpe").is_null())
    linhas: list[tuple[str, str, float]] = []

    for golpe in GOLPES:
        deste = pl.concat([legitimos, rotulados.filter(pl.col("golpe") == golpe.value)])
        alvo = deste["golpe"].is_not_null()

        linhas.extend((nome, golpe.value, forca(deste[nome], alvo)) for nome in ANALISADAS)

    return pl.DataFrame(linhas, schema={"variavel": pl.String, "golpe": pl.String, "forca": pl.Float64}, orient="row")
