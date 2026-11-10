"""O monitor, semana a semana: o que mudou nos dados, o que o modelo decidiu, e o golpe que passou (ADR-0020).

O campeão entra no banco no dia do corte, com o que o banco sabia até ali. Daí
em diante, cada semana inteira de Pix é posta ao lado do que ele viu:
- a deriva de cada variável e da probabilidade, pelo PSI, contra as quatro
  semanas antes do corte, as últimas que ele viu;
- as saídas dele: a parte dos Pix segurada e a bloqueada;
- a perda nos aprovados: a esperada pelo modelo e a conhecida pelo rótulo, com
  o rótulo que ainda não chegou descontado.

Só entram semanas inteiras, de segunda a domingo: a semana pela metade muda o
dia da semana e o tamanho do Pix, e a deriva acusaria o calendário.
"""

from datetime import date, timedelta
from typing import Protocol

import numpy as np
import numpy.typing as npt
import polars as pl

from antifraude_pix.dados.banco import VisaoDoBanco
from antifraude_pix.decisao.custo import Acao, Custos, decidir
from antifraude_pix.modelos.preparo import ENTRADA
from antifraude_pix.modelos.rotulo import Dobra, janela_de_teste
from antifraude_pix.monitoramento.deriva import por_periodo
from antifraude_pix.monitoramento.rotulo import chegada, perda_por_semana

REFERENCIA = 28
"""Os dias antes do corte que servem de referência para a deriva: as quatro semanas que o modelo viu por último."""


class Modelo(Protocol):
    """O que o monitor pede do modelo: a pontuação e a probabilidade calibrada dela."""

    def pontuar(self, variaveis: pl.DataFrame) -> npt.NDArray[np.float64]: ...

    def probabilidade_da(self, pontuacao: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]: ...


def semanas_inteiras(corte: date, ate: date) -> tuple[date, ...]:
    """As segundas-feiras das semanas inteiras de Pix entre o corte e a véspera de `ate`."""
    primeira = corte + timedelta(days=-corte.weekday() % 7)
    quantas = max((ate - primeira).days // 7, 0)

    return tuple(primeira + timedelta(weeks=semana) for semana in range(quantas))


def semana_a_semana(
    visao: VisaoDoBanco, variaveis: pl.DataFrame, modelo: Modelo, custos: Custos, *, corte: date, ate: date
) -> pl.DataFrame:
    """O monitor do modelo treinado no corte, com o que o banco sabia em `ate`: uma linha por semana inteira.

    As colunas são `semana`, `pix`, `parte_segurada`, `parte_bloqueada`, `psi_da_probabilidade`, a
    `variavel_que_mais_mudou` e o `psi_dela`, e a perda nos aprovados (`aprovados`, `esperada`, `conhecida`
    e `razao`), com a `razao_acumulada` desde a primeira semana.
    """
    segundas = semanas_inteiras(corte, ate)

    if not segundas:
        raise ValueError(f"Entre o corte, {corte}, e {ate}, não há uma semana inteira de Pix para acompanhar.")

    fim = segundas[-1] + timedelta(weeks=1)
    pix = janela_de_teste(visao, variaveis, Dobra(corte - timedelta(days=REFERENCIA), fim))
    pontuacao = modelo.pontuar(pix)
    decididos = decidir(
        pix.with_columns(pl.Series("probabilidade", modelo.probabilidade_da(pontuacao))), custos
    ).with_columns(pl.col("dia").dt.truncate("1w").alias("semana"))
    referencia = decididos.filter(pl.col("dia") < corte)
    atuais = decididos.filter(pl.col("semana").is_in(segundas))
    nomes = [variavel.nome for variavel in ENTRADA]
    deriva = por_periodo(referencia, atuais, [*nomes, "probabilidade"], "semana")
    da_probabilidade = deriva.filter(pl.col("variavel") == "probabilidade").select(
        "semana", pl.col("psi").alias("psi_da_probabilidade")
    )
    das_variaveis = (
        deriva.filter(pl.col("variavel") != "probabilidade")
        .sort("psi", descending=True, maintain_order=True)
        .group_by("semana", maintain_order=True)
        .first()
        .select("semana", pl.col("variavel").alias("variavel_que_mais_mudou"), pl.col("psi").alias("psi_dela"))
    )
    saidas = atuais.group_by("semana").agg(
        pl.len().alias("pix"),
        (pl.col("acao") == Acao.SEGURAR.value).mean().alias("parte_segurada"),
        (pl.col("acao") == Acao.BLOQUEAR.value).mean().alias("parte_bloqueada"),
    )
    com_o_momento = atuais.join(visao.transacoes.select("id", "momento"), on="id")
    perda = perda_por_semana(com_o_momento, visao, ate, chegada(visao, ate))

    return (
        saidas.join(da_probabilidade, on="semana")
        .join(das_variaveis, on="semana")
        .join(perda, on="semana", how="left")
        .sort("semana")
        .with_columns(
            pl.col("aprovados").fill_null(0),
            pl.col("esperada", "conhecida").fill_null(0.0),
        )
        .with_columns((pl.col("conhecida").cum_sum() / pl.col("esperada").cum_sum()).alias("razao_acumulada"))
    )
