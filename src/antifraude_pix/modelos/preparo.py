"""As variáveis como matriz de números, do jeito que cada modelo pede (ADR-0008).

A logística pede tudo em número e na mesma escala:
- os números em logaritmo, porque o valor de R$ 5.000 não é cem vezes o de R$ 50 para quem decide;
- o vazio com um indicador ao lado, para o primeiro Pix sem passado não virar o Pix de zero reais;
- a hora, o dia da semana e as categorias em casas, uma para cada valor;
- tudo padronizado pela média e pelo desvio do treino, e só do treino.

As árvores pedem pouco: o número como está, o vazio como vazio, a categoria pelo código.
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import polars as pl

from antifraude_pix.variaveis.catalogo import CATALOGO, Tipo

CASAS = {"hora": 24, "dia_da_semana": 7}
"""Os números que viram casas na logística: a madrugada não é a hora três vezes maior que a uma."""

CATEGORICAS: tuple[int, ...] = tuple(
    posicao for posicao, variavel in enumerate(CATALOGO) if variavel.tipo is Tipo.CATEGORIA
)
"""As posições das categorias na matriz das árvores, para o LightGBM tratá-las como categorias, e não como números."""


@dataclass(frozen=True, slots=True)
class PreparoDaLogistica:
    """As categorias, a média e o desvio de cada coluna, tirados do treino."""

    categorias: dict[str, tuple[str, ...]]
    media: npt.NDArray[np.float64]
    desvio: npt.NDArray[np.float64]

    @staticmethod
    def ajustado(treino: pl.DataFrame) -> "PreparoDaLogistica":
        """O preparo com as contas do treino: as categorias do tipo, a média e o desvio de cada coluna."""
        categorias = {
            variavel.nome: _categorias(treino, variavel.nome)
            for variavel in CATALOGO
            if variavel.tipo is Tipo.CATEGORIA
        }
        bruta = _bruta(treino, categorias)
        desvio = bruta.std(axis=0)
        # A coluna que não varia no treino não ensina nada; fica como está, sem dividir por zero.
        desvio[desvio == 0] = 1.0

        return PreparoDaLogistica(categorias, bruta.mean(axis=0), desvio)

    @property
    def colunas(self) -> tuple[str, ...]:
        """O nome de cada coluna da matriz, na ordem."""
        return tuple(expressao.meta.output_name() for expressao in _expressoes(self.categorias))

    def matriz(self, variaveis: pl.DataFrame) -> npt.NDArray[np.float64]:
        """As variáveis como a logística as recebe, padronizadas com a média e o desvio do treino."""
        padronizada: npt.NDArray[np.float64] = (_bruta(variaveis, self.categorias) - self.media) / self.desvio

        return padronizada


def matriz_das_arvores(variaveis: pl.DataFrame) -> npt.NDArray[np.float64]:
    """As variáveis na ordem do catálogo, como as árvores as recebem.

    O número fica como está, o sim ou não vira 1 ou 0, a categoria vira o código dela, e o vazio vira NaN.
    """
    return variaveis.select(
        (pl.col(variavel.nome).to_physical() if variavel.tipo is Tipo.CATEGORIA else pl.col(variavel.nome)).cast(
            pl.Float64
        )
        for variavel in CATALOGO
    ).to_numpy()


def _categorias(variaveis: pl.DataFrame, nome: str) -> tuple[str, ...]:
    tipo = variaveis.schema[nome]

    if not isinstance(tipo, pl.Enum):
        raise TypeError(f"A variável {nome} devia ser uma categoria fechada, e é {tipo}.")

    return tuple(tipo.categories.to_list())


def _bruta(variaveis: pl.DataFrame, categorias: dict[str, tuple[str, ...]]) -> npt.NDArray[np.float64]:
    bruta: npt.NDArray[np.float64] = variaveis.select(_expressoes(categorias)).to_numpy().astype(np.float64)

    return bruta


def _expressoes(categorias: dict[str, tuple[str, ...]]) -> list[pl.Expr]:
    expressoes: list[pl.Expr] = []

    for variavel in CATALOGO:
        nome, coluna = variavel.nome, pl.col(variavel.nome)

        if nome in CASAS:
            expressoes += [(coluna == casa).cast(pl.Float64).alias(f"{nome}={casa}") for casa in range(CASAS[nome])]
        elif variavel.tipo is Tipo.CATEGORIA:
            expressoes += [
                (coluna == categoria).fill_null(value=False).cast(pl.Float64).alias(f"{nome}={categoria}")
                for categoria in categorias[nome]
            ]
            expressoes.append(coluna.is_null().cast(pl.Float64).alias(f"{nome} vazia"))
        elif variavel.tipo is Tipo.SIM_OU_NAO:
            expressoes.append(coluna.cast(pl.Float64).alias(nome))
        else:
            expressoes.append(coluna.cast(pl.Float64).clip(lower_bound=0).log1p().fill_null(0.0).alias(nome))
            expressoes.append(coluna.is_null().cast(pl.Float64).alias(f"{nome} vazia"))

    return expressoes
