"""As variáveis como matriz de números, do jeito que cada modelo pede (ADR-0008).

A logística pede tudo em número e na mesma escala:
- os números em logaritmo, porque o valor de R$ 5.000 não é cem vezes o de R$ 50 para quem decide;
- o vazio com um indicador ao lado, para o primeiro Pix sem passado não virar o Pix de zero reais;
- a hora, o dia da semana e as categorias em casas, uma para cada valor;
- tudo padronizado pela média e pelo desvio do treino, e só do treino.

As árvores pedem pouco: o número como está, o vazio como vazio, a categoria pelo código.

Nenhum dos dois recebe a idade do cliente (ADR-0011). O golpe mira os mais
velhos, e o modelo que via a idade aprendeu a desconfiar deles: bloqueava o
Pix honesto de quem tem 65 anos ou mais muito mais que o dos outros. A decisão
olha o Pix, o hábito e o recebedor, e não quem a pessoa é.
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import polars as pl

from antifraude_pix.variaveis.catalogo import CATALOGO, Tipo, Variavel

CASAS = {"hora": 24, "dia_da_semana": 7}
"""Os números que viram casas na logística: a madrugada não é a hora três vezes maior que a uma."""

PROTEGIDAS: tuple[str, ...] = ("idade",)
"""As variáveis que o banco tem, mas que os modelos não recebem: a idade (ADR-0011)."""

ENTRADA: tuple[Variavel, ...] = tuple(variavel for variavel in CATALOGO if variavel.nome not in PROTEGIDAS)
"""As variáveis que os modelos recebem, na ordem do catálogo: todas, menos as protegidas."""


def categoricas(entrada: tuple[Variavel, ...] = ENTRADA) -> tuple[int, ...]:
    """As posições das categorias na matriz das árvores, para o LightGBM tratá-las como categorias, e não números."""
    return tuple(posicao for posicao, variavel in enumerate(entrada) if variavel.tipo is Tipo.CATEGORIA)


@dataclass(frozen=True, slots=True)
class PreparoDaLogistica:
    """As variáveis que entram, e as categorias, a média e o desvio de cada coluna, tirados do treino."""

    entrada: tuple[Variavel, ...]
    categorias: dict[str, tuple[str, ...]]
    media: npt.NDArray[np.float64]
    desvio: npt.NDArray[np.float64]

    @staticmethod
    def ajustado(treino: pl.DataFrame, entrada: tuple[Variavel, ...] = ENTRADA) -> "PreparoDaLogistica":
        """O preparo com as contas do treino: as categorias do tipo, a média e o desvio de cada coluna."""
        categorias = {
            variavel.nome: _categorias(treino, variavel.nome) for variavel in entrada if variavel.tipo is Tipo.CATEGORIA
        }
        bruta = _bruta(treino, entrada, categorias)
        desvio = bruta.std(axis=0)
        # A coluna que não varia no treino não ensina nada; fica como está, sem dividir por zero.
        desvio[desvio == 0] = 1.0

        return PreparoDaLogistica(entrada, categorias, bruta.mean(axis=0), desvio)

    @property
    def colunas(self) -> tuple[str, ...]:
        """O nome de cada coluna da matriz, na ordem."""
        return tuple(expressao.meta.output_name() for expressao in _expressoes(self.entrada, self.categorias))

    def matriz(self, variaveis: pl.DataFrame) -> npt.NDArray[np.float64]:
        """As variáveis como a logística as recebe, padronizadas com a média e o desvio do treino."""
        bruta = _bruta(variaveis, self.entrada, self.categorias)
        padronizada: npt.NDArray[np.float64] = (bruta - self.media) / self.desvio

        return padronizada


def matriz_das_arvores(variaveis: pl.DataFrame, entrada: tuple[Variavel, ...] = ENTRADA) -> npt.NDArray[np.float64]:
    """As variáveis da entrada, na ordem do catálogo, como as árvores as recebem.

    O número fica como está, o sim ou não vira 1 ou 0, a categoria vira o código dela, e o vazio vira NaN.
    """
    return variaveis.select(
        (pl.col(variavel.nome).to_physical() if variavel.tipo is Tipo.CATEGORIA else pl.col(variavel.nome)).cast(
            pl.Float64
        )
        for variavel in entrada
    ).to_numpy()


def _categorias(variaveis: pl.DataFrame, nome: str) -> tuple[str, ...]:
    tipo = variaveis.schema[nome]

    if not isinstance(tipo, pl.Enum):
        raise TypeError(f"A variável {nome} devia ser uma categoria fechada, e é {tipo}.")

    return tuple(tipo.categories.to_list())


def _bruta(
    variaveis: pl.DataFrame, entrada: tuple[Variavel, ...], categorias: dict[str, tuple[str, ...]]
) -> npt.NDArray[np.float64]:
    bruta: npt.NDArray[np.float64] = variaveis.select(_expressoes(entrada, categorias)).to_numpy().astype(np.float64)

    return bruta


def _expressoes(entrada: tuple[Variavel, ...], categorias: dict[str, tuple[str, ...]]) -> list[pl.Expr]:
    expressoes: list[pl.Expr] = []

    for variavel in entrada:
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
