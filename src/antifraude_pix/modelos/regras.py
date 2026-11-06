"""As regras de hoje: o ponto de partida que um modelo tem de bater (ADR-0008).

Antes do modelo, o banco já se defende com regras escritas por analistas:
o aparelho novo, o recebedor já contestado, o Pix muito fora do hábito para
quem o cliente nunca pagou. Cada regra soma pontos, e a soma ordena os Pix: os
de mais pontos são os primeiros a segurar.

Os limites das regras são escritos à mão, como num banco, e não aprendidos.
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import polars as pl


@dataclass(frozen=True, slots=True)
class Regra:
    nome: str
    descricao: str
    pontos: int
    condicao: pl.Expr


REGRAS: tuple[Regra, ...] = (
    Regra(
        "aparelho_novo",
        "O Pix de mais de R$ 200 sai de um aparelho que entrou na conta há menos de um dia.",
        3,
        (pl.col("minutos_desde_o_aparelho_novo") < 24 * 60) & (pl.col("valor") > 200),
    ),
    Regra(
        "recebedor_contestado",
        "O recebedor já teve um Pix contestado por um cliente do banco.",
        3,
        pl.col("contestacoes_do_recebedor") >= 1,
    ),
    Regra(
        "fora_do_habito_para_quem_nunca_pagou",
        "O Pix vai para quem o cliente nunca pagou e vale cinco vezes o Pix típico dele, ou mais.",
        2,
        pl.col("recebedor_novo") & (pl.col("valor_sobre_a_mediana") >= 5),
    ),
    Regra(
        "metade_do_saldo",
        "O Pix de mais de R$ 200 leva metade do saldo, ou mais.",
        2,
        (pl.col("parte_do_saldo") >= 0.5) & (pl.col("valor") > 200),
    ),
    Regra(
        "pressa",
        "É o terceiro Pix do cliente na mesma hora, ou mais.",
        1,
        pl.col("pix_na_ultima_hora") >= 2,
    ),
    Regra(
        "limite_noturno",
        "De noite, o Pix leva quatro quintos do limite noturno, ou mais.",
        1,
        pl.col("parte_do_limite_noturno") >= 0.8,
    ),
)


def acionadas(variaveis: pl.DataFrame) -> pl.DataFrame:
    """Uma coluna por regra, verdadeira quando ela vale para o Pix. A condição sobre o vazio não vale."""
    return variaveis.select(regra.condicao.fill_null(value=False).alias(regra.nome) for regra in REGRAS)


def pontos(variaveis: pl.DataFrame) -> npt.NDArray[np.float64]:
    """A soma dos pontos das regras que valem para cada Pix."""
    dos_que_valem = [pl.when(pl.col(regra.nome)).then(regra.pontos).otherwise(0) for regra in REGRAS]

    return acionadas(variaveis).select(pl.sum_horizontal(dos_que_valem).cast(pl.Float64)).to_series().to_numpy()
