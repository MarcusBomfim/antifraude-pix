"""O treino dos três modelos e a calibração da probabilidade (ADR-0008).

Cada modelo dá uma pontuação: quanto maior, mais cara de golpe. A pontuação
ordena os Pix, mas não diz a chance de golpe, e é a chance que a Parte 4 pesa
contra o valor e o custo de segurar. A calibração transforma a pontuação em
probabilidade, com uma regressão logística de uma variável só (o método de
Platt), ajustada em Pix que o modelo não viu.

Os Pix da calibração são sorteados entre os maduros, de mais de 28 dias antes
do corte: neles, o rótulo do banco já quase chegou ao que vai ser. O modelo
aprende com todo o resto, inclusive os dias mais recentes, porque esperar o
rótulo amadurecer custa mais que o rótulo atrasado (ADR-0007).
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Protocol

import lightgbm as lgb
import numpy as np
import numpy.typing as npt
import polars as pl
from sklearn.linear_model import LogisticRegression

from antifraude_pix.modelos import regras
from antifraude_pix.modelos.preparo import ENTRADA, PreparoDaLogistica, categoricas, matriz_das_arvores
from antifraude_pix.variaveis.catalogo import Variavel

SEMENTE = 2026

CALIBRACAO = 0.2
"""A parte dos Pix maduros que fica para a calibração."""

MADURO_EM_DIAS = 28
"""A idade a partir da qual o rótulo do banco quase não muda mais: em 28 dias, quase toda contestação já chegou."""

LOGISTICA: dict[str, Any] = {"C": 0.1, "max_iter": 2_000}
"""A regularização escolhida no mês de validação: a mais forte das que deram o melhor; a de 0,01 deu menos."""

LIGHTGBM: dict[str, Any] = {
    "learning_rate": 0.02,
    "n_estimators": 1_000,
    "num_leaves": 15,
    "min_child_samples": 20,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
}
"""As árvores rasas e muitas, escolhidas no mês de validação: com poucas centenas de golpes, a folha de 31 decora."""

UM_LEGITIMO_EM = 10
"""As árvores aprendem com todos os golpes e um legítimo em cada dez, com peso dez: metade do tempo, e a mesma precisão.

O peso devolve aos legítimos a proporção que eles têm, e a pontuação não fica inchada a favor do golpe.
"""

FIOS = 4
"""Os fios do LightGBM, fixos: com o mesmo número de fios, o mesmo treino dá as mesmas árvores."""


class Pontuador(Protocol):
    def pontuar(self, variaveis: pl.DataFrame) -> npt.NDArray[np.float64]: ...


@dataclass(frozen=True, slots=True)
class Regras:
    """As regras de hoje, que não aprendem nada: a pontuação é a soma dos pontos."""

    def pontuar(self, variaveis: pl.DataFrame) -> npt.NDArray[np.float64]:
        return regras.pontos(variaveis)


@dataclass(frozen=True, slots=True)
class Logistica:
    preparo: PreparoDaLogistica
    modelo: Any = field(repr=False)

    def pontuar(self, variaveis: pl.DataFrame) -> npt.NDArray[np.float64]:
        return np.asarray(self.modelo.decision_function(self.preparo.matriz(variaveis)), dtype=np.float64)


@dataclass(frozen=True, slots=True)
class Arvores:
    modelo: lgb.LGBMClassifier = field(repr=False)
    entrada: tuple[Variavel, ...] = ENTRADA

    def pontuar(self, variaveis: pl.DataFrame) -> npt.NDArray[np.float64]:
        matriz = matriz_das_arvores(variaveis, self.entrada)

        return np.asarray(self.modelo.predict(matriz, raw_score=True), dtype=np.float64)


@dataclass(frozen=True, slots=True)
class Calibrado:
    """O modelo e a reta de Platt que leva a pontuação dele à probabilidade."""

    nome: str
    pontuador: Pontuador
    inclinacao: float
    intercepto: float

    def pontuar(self, variaveis: pl.DataFrame) -> npt.NDArray[np.float64]:
        return self.pontuador.pontuar(variaveis)

    def probabilidade(self, variaveis: pl.DataFrame) -> npt.NDArray[np.float64]:
        """A chance de o Pix ser um golpe que o banco vai ficar sabendo."""
        return self.probabilidade_da(self.pontuar(variaveis))

    def probabilidade_da(self, pontuacao: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """A probabilidade de uma pontuação já calculada: pontuar milhões de Pix uma vez só."""
        logito = self.inclinacao * pontuacao + self.intercepto
        chance: npt.NDArray[np.float64] = 1 / (1 + np.exp(-np.clip(logito, -50, 50)))

        return chance


def treinar_regras(treino: pl.DataFrame, gerador: np.random.Generator, entrada: tuple[Variavel, ...]) -> Regras:
    del treino, gerador, entrada
    return Regras()


def treinar_logistica(treino: pl.DataFrame, gerador: np.random.Generator, entrada: tuple[Variavel, ...]) -> Logistica:
    del gerador
    preparo = PreparoDaLogistica.ajustado(treino, entrada)
    modelo = LogisticRegression(**LOGISTICA)
    modelo.fit(preparo.matriz(treino), treino["rotulo"].to_numpy())

    return Logistica(preparo, modelo)


def treinar_arvores(treino: pl.DataFrame, gerador: np.random.Generator, entrada: tuple[Variavel, ...]) -> Arvores:
    rotulo = treino["rotulo"].to_numpy()
    amostra = rotulo | (gerador.random(rotulo.size) < 1 / UM_LEGITIMO_EM)
    peso = np.where(rotulo, 1.0, float(UM_LEGITIMO_EM))[amostra]
    modelo = lgb.LGBMClassifier(
        **LIGHTGBM,
        random_state=int(gerador.integers(2**31)),
        n_jobs=FIOS,
        deterministic=True,
        force_col_wise=True,
        verbose=-1,
    )
    modelo.fit(
        matriz_das_arvores(treino, entrada)[amostra],
        rotulo[amostra],
        sample_weight=peso,
        categorical_feature=list(categoricas(entrada)),
    )

    return Arvores(modelo, entrada)


MODELOS: dict[str, Callable[[pl.DataFrame, np.random.Generator, tuple[Variavel, ...]], Pontuador]] = {
    "regras": treinar_regras,
    "logistica": treinar_logistica,
    "lightgbm": treinar_arvores,
}
"""Os três modelos, do mais simples ao mais forte."""

NOMES = {"regras": "as regras de hoje", "logistica": "a regressão logística", "lightgbm": "o LightGBM"}


def separar(janela: pl.DataFrame, corte: date, gerador: np.random.Generator) -> tuple[pl.DataFrame, pl.DataFrame]:
    """O treino e a calibração: um em cada cinco Pix maduros vai para a calibração, e o resto para o treino."""
    sorteados = pl.Series(gerador.random(janela.height) < CALIBRACAO)
    maduros = janela["dia"] < corte - timedelta(days=MADURO_EM_DIAS)
    na_calibracao = sorteados & maduros

    return janela.filter(~na_calibracao), janela.filter(na_calibracao)


def calibrar(nome: str, pontuador: Pontuador, calibracao: pl.DataFrame) -> Calibrado:
    """A reta de Platt da pontuação para a probabilidade, ajustada nos Pix da calibração."""
    rotulo = calibracao["rotulo"].to_numpy()

    if rotulo.all() or not rotulo.any():
        raise ValueError("A calibração pede golpes e legítimos entre os Pix maduros.")

    # Uma variável só, e quase sem regularização: a reta só muda a escala da pontuação.
    platt = LogisticRegression(C=100.0)
    platt.fit(pontuador.pontuar(calibracao).reshape(-1, 1), rotulo)

    return Calibrado(nome, pontuador, float(platt.coef_[0, 0]), float(platt.intercept_[0]))


def treinar(
    nome: str, janela: pl.DataFrame, corte: date, semente: int = SEMENTE, entrada: tuple[Variavel, ...] = ENTRADA
) -> Calibrado:
    """O modelo treinado na janela, com a probabilidade calibrada nos Pix maduros que ele não viu.

    A entrada são as variáveis que o modelo recebe: todas, menos a idade. Mudá-la é para medir, e não para decidir.
    """
    gerador = np.random.default_rng(semente)
    ajuste, calibracao = separar(janela, corte, gerador)

    return calibrar(nome, MODELOS[nome](ajuste, gerador, entrada), calibracao)
