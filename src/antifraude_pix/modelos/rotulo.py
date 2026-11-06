"""O rótulo que o banco tem no dia do treino, e as dobras no tempo (ADR-0007).

O banco só sabe que um Pix foi golpe quando a vítima contesta. No dia do
treino, o golpe de ontem ainda parece legítimo, e o golpe que ninguém contesta
parece legítimo para sempre. O treino aprende com esse rótulo, porque é o único
que existe; a avaliação mede contra a verdade do simulador, que só ela lê.
"""

from dataclasses import dataclass
from datetime import date, datetime, timedelta

import polars as pl

from antifraude_pix.dados.banco import VisaoDoBanco
from antifraude_pix.variaveis.catalogo import NOMES

AQUECIMENTO = 30
"""Os primeiros dias da história ficam de fora do treino e do teste.

Ali, todo recebedor é novo, porque a história começa ali, e as variáveis do hábito ainda não têm passado.
"""

VALIDACAO = 1
"""Os meses, depois do primeiro mês de treino, que servem para escolher os parâmetros, e não para o teste."""


@dataclass(frozen=True, slots=True)
class Dobra:
    """Um treino e o seu teste.

    O modelo sabe o que o banco sabia na meia-noite do corte, e é testado do corte até a véspera do fim.
    """

    corte: date
    fim: date

    def __post_init__(self) -> None:
        if self.fim <= self.corte:
            raise ValueError(f"O teste da dobra termina antes de começar: de {self.corte} a {self.fim}.")


def dobras(primeiro_corte: date, ultimo_dia: date) -> tuple[Dobra, ...]:
    """Uma dobra por mês, do primeiro corte em diante: treina com tudo o que veio antes, e testa o mês do corte."""
    if primeiro_corte.day != 1:
        raise ValueError(f"O corte é sempre no primeiro dia do mês, e não em {primeiro_corte}.")

    resultado = []
    corte = primeiro_corte

    while corte <= ultimo_dia:
        seguinte = _mes_seguinte(corte)
        resultado.append(Dobra(corte, min(seguinte, ultimo_dia + timedelta(days=1))))
        corte = seguinte

    return tuple(resultado)


def primeiro_corte(visao: VisaoDoBanco) -> date:
    """O primeiro mês de teste: depois do aquecimento, de um mês inteiro de treino e do mês de validação."""
    inicio = _primeiro_dia(visao) + timedelta(days=AQUECIMENTO + 31)
    corte = inicio.replace(day=1) if inicio.day == 1 else _mes_seguinte(inicio)

    for _ in range(VALIDACAO):
        corte = _mes_seguinte(corte)

    return corte


def janela_de_treino(
    visao: VisaoDoBanco, variaveis: pl.DataFrame, corte: date, espera_em_dias: int = 0
) -> pl.DataFrame:
    """Os Pix do fim do aquecimento até o corte, com o rótulo que o banco tinha no corte.

    As colunas são `id`, `dia`, `rotulo` e as variáveis. O rótulo é verdadeiro
    quando o Pix foi contestado antes do corte. A espera deixa de fora os
    últimos dias, cujos golpes ainda quase não foram contestados; o padrão é
    não esperar (ADR-0007).
    """
    if espera_em_dias < 0:
        raise ValueError("A espera pelo rótulo não pode ser negativa.")

    sabido = visao.ate(corte)
    contestados = sabido.contestacoes.select(pl.col("transacao").alias("id"), pl.lit(value=True).alias("rotulo"))
    inicio = _primeiro_dia(visao) + timedelta(days=AQUECIMENTO)

    return (
        _com_o_dia(sabido, variaveis)
        .filter(pl.col("dia").is_between(inicio, corte - timedelta(days=espera_em_dias), closed="left"))
        .join(contestados, on="id", how="left")
        .select("id", "dia", pl.col("rotulo").fill_null(value=False), *NOMES)
        .sort("id")
    )


def janela_de_teste(visao: VisaoDoBanco, variaveis: pl.DataFrame, dobra: Dobra) -> pl.DataFrame:
    """Os Pix do mês da dobra, sem rótulo nenhum: `id`, `dia` e as variáveis."""
    return (
        _com_o_dia(visao, variaveis)
        .filter(pl.col("dia").is_between(dobra.corte, dobra.fim, closed="left"))
        .select("id", "dia", *NOMES)
        .sort("id")
    )


def _com_o_dia(visao: VisaoDoBanco, variaveis: pl.DataFrame) -> pl.DataFrame:
    dias = visao.transacoes.select("id", pl.col("momento").dt.date().alias("dia"))

    return variaveis.join(dias, on="id")


def _primeiro_dia(visao: VisaoDoBanco) -> date:
    primeiro = visao.transacoes["momento"].min()

    if not isinstance(primeiro, datetime):
        raise ValueError("A visão do banco não tem nenhum Pix.")

    return primeiro.date()


def _mes_seguinte(dia: date) -> date:
    return date(dia.year + dia.month // 12, dia.month % 12 + 1, 1)
