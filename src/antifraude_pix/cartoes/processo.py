"""O mesmo processo do Pix, na base de cartões (ADR-0015).

O que vai igual:
- **a validação no tempo**: o treino no primeiro dia, e o teste no segundo. Sortear o teste entre os dois dias
  deixaria o modelo aprender com a fraude de amanhã;
- **os modelos**: a regressão logística e o LightGBM, com os mesmos parâmetros do Pix, sem ajuste nenhum na base
  de cartões; e, como ponto de partida, o valor sozinho, que faz o papel das regras de hoje;
- **o LightGBM** aprende com todas as fraudes e uma compra legítima em cada dez, com peso dez;
- **a probabilidade calibrada** pelo método de Platt, em um quinto das compras do treino, que o modelo não vê;
- **a medida**: a precisão média, o alerta de 0,5% das compras do dia, a calibração e o custo de cada jeito de
  decidir, com as mesmas funções do Pix.

O que não vai: o rótulo atrasado (a base só tem o final), a explicação em
português (V1 a V28 não têm nome) e os erros por idade (a base não tem idade).
"""

from dataclasses import dataclass, field
from typing import Any

import lightgbm as lgb
import numpy as np
import numpy.typing as npt
import polars as pl
from sklearn.linear_model import LogisticRegression

from antifraude_pix.cartoes.base import COMPONENTES
from antifraude_pix.modelos.treino import (
    CALIBRACAO,
    FIOS,
    LIGHTGBM,
    LOGISTICA,
    SEMENTE,
    UM_LEGITIMO_EM,
    Calibrado,
    calibrar,
)

DIA = 86_400
"""Os segundos de um dia: a base cobre dois, e o tempo conta desde a primeira compra."""

SEGUNDO_DIA = 2
"""O mês de teste do Pix aqui é o segundo dia: é por ele que o alerta conta as compras."""

MODELOS: tuple[str, ...] = ("valor", "logistica", "lightgbm")
"""Os três modelos, do mais simples ao mais forte, como no Pix."""

NOMES = {"valor": "o valor sozinho", "logistica": "a regressão logística", "lightgbm": "o LightGBM"}


def com_as_variaveis(compras: pl.DataFrame) -> pl.DataFrame:
    """As compras com a `hora` do dia, contada desde a primeira compra da base, e o `rotulo`, que é a fraude."""
    return compras.with_columns(
        ((pl.col("segundo") % DIA) / 3_600).alias("hora"),
        pl.col("fraude").alias("rotulo"),
    )


def dividir(compras: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """O treino, com as compras do primeiro dia, e o teste, com as do segundo."""
    primeiro = pl.col("segundo") < DIA

    return compras.filter(primeiro), compras.filter(~primeiro)


ARVORES: tuple[str, ...] = (*COMPONENTES, "valor", "hora")
"""O que as árvores recebem: as componentes, o valor e a hora, como estão."""


@dataclass(frozen=True, slots=True)
class ValorSozinho:
    """O ponto de partida: quanto maior a compra, mais cara de fraude. Não aprende nada."""

    def pontuar(self, compras: pl.DataFrame) -> npt.NDArray[np.float64]:
        return np.log1p(compras["valor"].to_numpy())


@dataclass(frozen=True, slots=True)
class LogisticaDosCartoes:
    """A regressão logística, com as componentes e o valor em logaritmo, padronizados pelas contas do treino.

    A hora fica de fora: ela conta desde a primeira compra da base, e não desde a meia-noite, e numa reta só a
    madrugada não vira nada. As árvores a recebem, e acham sozinhas as horas que importam.
    """

    media: npt.NDArray[np.float64]
    desvio: npt.NDArray[np.float64]
    modelo: Any = field(repr=False)

    def pontuar(self, compras: pl.DataFrame) -> npt.NDArray[np.float64]:
        matriz = (_da_logistica(compras) - self.media) / self.desvio

        return np.asarray(self.modelo.decision_function(matriz), dtype=np.float64)


@dataclass(frozen=True, slots=True)
class ArvoresDosCartoes:
    modelo: lgb.LGBMClassifier = field(repr=False)

    def pontuar(self, compras: pl.DataFrame) -> npt.NDArray[np.float64]:
        matriz = compras.select(ARVORES).to_numpy()

        return np.asarray(self.modelo.predict(matriz, raw_score=True), dtype=np.float64)


def separar(treino: pl.DataFrame, gerador: np.random.Generator) -> tuple[pl.DataFrame, pl.DataFrame]:
    """O ajuste e a calibração: uma compra do treino em cada cinco vai para a calibração, e o modelo não a vê.

    Não há compra madura a esperar, como no Pix: a base só tem o rótulo final.
    """
    na_calibracao = pl.Series(gerador.random(treino.height) < CALIBRACAO)

    return treino.filter(~na_calibracao), treino.filter(na_calibracao)


def treinar(nome: str, treino: pl.DataFrame, semente: int = SEMENTE) -> Calibrado:
    """O modelo treinado com quatro quintos do treino, e calibrado no quinto que ele não viu."""
    if nome not in MODELOS:
        raise ValueError(f"Não conheço o modelo {nome!r}: os que há são {', '.join(MODELOS)}.")

    gerador = np.random.default_rng(semente)
    ajuste, calibracao = separar(treino, gerador)

    if nome == "valor":
        return calibrar(nome, ValorSozinho(), calibracao)

    if nome == "logistica":
        bruta = _da_logistica(ajuste)
        media, desvio = bruta.mean(axis=0), bruta.std(axis=0)
        desvio[desvio == 0] = 1.0
        logistica = LogisticRegression(**LOGISTICA)
        logistica.fit((bruta - media) / desvio, ajuste["rotulo"].to_numpy())

        return calibrar(nome, LogisticaDosCartoes(media, desvio, logistica), calibracao)

    rotulo = ajuste["rotulo"].to_numpy()
    amostra = rotulo | (gerador.random(rotulo.size) < 1 / UM_LEGITIMO_EM)
    arvores = lgb.LGBMClassifier(
        **LIGHTGBM,
        random_state=int(gerador.integers(2**31)),
        n_jobs=FIOS,
        deterministic=True,
        force_col_wise=True,
        verbose=-1,
    )
    arvores.fit(
        ajuste.select(ARVORES).to_numpy()[amostra],
        rotulo[amostra],
        sample_weight=np.where(rotulo, 1.0, float(UM_LEGITIMO_EM))[amostra],
    )

    return calibrar(nome, ArvoresDosCartoes(arvores), calibracao)


def avaliar(compras: pl.DataFrame, modelos: tuple[str, ...] = MODELOS, semente: int = SEMENTE) -> pl.DataFrame:
    """A pontuação de cada modelo em cada compra do segundo dia, no formato da avaliação do Pix.

    As colunas são `id`, `corte` (o dia do teste), `modelo`, `pontuacao`,
    `probabilidade`, `valor` e `golpe` ("fraude", ou vazio na compra
    legítima). Assim, as métricas, o alerta e a decisão pelo custo do Pix
    medem os cartões sem mudar uma linha.
    """
    treino, teste = dividir(com_as_variaveis(compras))
    partes = []

    for nome in modelos:
        modelo = treinar(nome, treino, semente)
        pontuacao = modelo.pontuar(teste)
        partes.append(
            teste.select(
                "id",
                pl.lit(SEGUNDO_DIA).alias("corte"),
                pl.lit(nome).alias("modelo"),
                pl.Series("pontuacao", pontuacao),
                pl.Series("probabilidade", modelo.probabilidade_da(pontuacao)),
                "valor",
                pl.when(pl.col("fraude")).then(pl.lit("fraude")).alias("golpe"),
            )
        )

    return pl.concat(partes)


def _da_logistica(compras: pl.DataFrame) -> npt.NDArray[np.float64]:
    matriz: npt.NDArray[np.float64] = (
        compras.select(*COMPONENTES, pl.col("valor").log1p()).to_numpy().astype(np.float64)
    )

    return matriz
