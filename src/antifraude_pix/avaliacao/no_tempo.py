"""A avaliação no tempo: em cada corte, o treino com o que o banco sabia, e o teste no mês do corte (ADR-0007).

É a única maneira honesta de medir um modelo de fraude. Sortear o teste entre
os Pix de todos os meses deixa o modelo aprender com o golpe de amanhã, e com a
contestação que ainda não chegou.
"""

from collections.abc import Iterator
from enum import StrEnum

import polars as pl

from antifraude_pix.dados.banco import VisaoDoBanco
from antifraude_pix.modelos.preparo import ENTRADA
from antifraude_pix.modelos.rotulo import Dobra, janela_de_teste, janela_de_treino
from antifraude_pix.modelos.treino import MODELOS, SEMENTE, Calibrado, treinar
from antifraude_pix.simulacao.simulador import Simulacao
from antifraude_pix.variaveis.catalogo import Variavel


class Rotulo(StrEnum):
    """Com que rótulo o modelo aprende."""

    DO_DIA = "do_dia"
    """O que o banco sabia no corte: o contestado até ali. É o único que um banco tem."""
    VERDADE = "verdade"
    """A verdade do simulador, que nenhum banco tem: o quanto o rótulo atrasado custa."""


def treinados(
    simulacao: Simulacao,
    variaveis: pl.DataFrame,
    dobras: tuple[Dobra, ...],
    *,
    modelos: tuple[str, ...] = tuple(MODELOS),
    rotulo: Rotulo = Rotulo.DO_DIA,
    espera_em_dias: int = 0,
    semente: int = SEMENTE,
    entrada: tuple[Variavel, ...] = ENTRADA,
) -> Iterator[tuple[Dobra, pl.DataFrame, dict[str, Calibrado]]]:
    """Em cada dobra, o mês de teste, sem rótulo, e os modelos treinados com o que o banco sabia no corte."""
    if not dobras:
        raise ValueError("A avaliação pede ao menos uma dobra: os dados não têm um mês de teste depois do treino.")

    visao = VisaoDoBanco.da(simulacao)

    for dobra in dobras:
        janela = janela_de_treino(visao, variaveis, dobra.corte, espera_em_dias)

        if rotulo is Rotulo.VERDADE:
            janela = janela.with_columns(pl.col("id").is_in(_fraudes(simulacao)["id"].implode()).alias("rotulo"))

        yield (
            dobra,
            janela_de_teste(visao, variaveis, dobra),
            {nome: treinar(nome, janela, dobra.corte, semente, entrada) for nome in modelos},
        )


def avaliar(
    simulacao: Simulacao,
    variaveis: pl.DataFrame,
    dobras: tuple[Dobra, ...],
    *,
    modelos: tuple[str, ...] = tuple(MODELOS),
    rotulo: Rotulo = Rotulo.DO_DIA,
    espera_em_dias: int = 0,
    semente: int = SEMENTE,
    entrada: tuple[Variavel, ...] = ENTRADA,
) -> pl.DataFrame:
    """A pontuação de cada modelo em cada Pix de teste, com a verdade ao lado.

    As colunas são `id`, `corte`, `modelo`, `pontuacao`, `probabilidade`,
    `valor` (em reais) e `golpe` (vazio no Pix legítimo).
    """
    partes = []

    for dobra, teste, os_modelos in treinados(
        simulacao,
        variaveis,
        dobras,
        modelos=modelos,
        rotulo=rotulo,
        espera_em_dias=espera_em_dias,
        semente=semente,
        entrada=entrada,
    ):
        for nome, modelo in os_modelos.items():
            pontuacao = modelo.pontuar(teste)
            partes.append(
                teste.select(
                    "id",
                    pl.lit(dobra.corte).alias("corte"),
                    pl.lit(nome).alias("modelo"),
                    pl.Series("pontuacao", pontuacao),
                    pl.Series("probabilidade", modelo.probabilidade_da(pontuacao)),
                    "valor",
                )
            )

    return com_a_verdade(simulacao, pl.concat(partes))


def com_a_verdade(simulacao: Simulacao, pix: pl.DataFrame) -> pl.DataFrame:
    """Os Pix com a coluna `golpe` ao lado: o golpe de cada um, pela verdade do simulador, e vazio no legítimo."""
    return pix.join(_fraudes(simulacao), on="id", how="left", maintain_order="left")


def _fraudes(simulacao: Simulacao) -> pl.DataFrame:
    return simulacao.verdade.fraudes.select(pl.col("transacao").alias("id"), pl.col("golpe").cast(pl.String))
