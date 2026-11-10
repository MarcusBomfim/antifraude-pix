"""Os dados que mudam: o quanto cada variável, e a probabilidade, se afastam do que o modelo viu (ADR-0020).

A medida é o índice de estabilidade da população (PSI). A distribuição da
semana é comparada com a de referência, o mês que o modelo viu antes do corte:
- o número vai em dez faixas, pelos decis da referência;
- a categoria e o sim ou não vão uma faixa por valor;
- o vazio é uma faixa a mais, porque o vazio que cresce também é mudança.

O índice é `soma((atual - referência) * ln(atual / referência))`, com um piso de
um centésimo de por cento em cada faixa, para a faixa vazia não virar infinito.
Até 0,1, a variável está estável; de 0,1 a 0,25, mudou um pouco; acima de 0,25,
mudou, e alguém tem de olhar.
"""

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
import polars as pl

FAIXAS = 10
"""As faixas do número: os decis da referência."""

PISO = 1e-4
"""A menor parte de uma faixa: sem ela, a faixa que a referência não tem daria um índice infinito."""

ESTAVEL, MUDOU = 0.1, 0.25
"""Os limites de costume do índice: até 0,1, estável; acima de 0,25, mudou."""


def psi(referencia: pl.Series, atual: pl.Series) -> float:
    """O índice de estabilidade da população da série atual contra a de referência."""
    if referencia.is_empty() or atual.is_empty():
        raise ValueError("O índice compara duas distribuições: a referência e a atual precisam de valores.")

    de_referencia, de_agora = _partes(referencia, atual)
    de_referencia, de_agora = np.maximum(de_referencia, PISO), np.maximum(de_agora, PISO)

    return float(np.sum((de_agora - de_referencia) * np.log(de_agora / de_referencia)))


def nivel(indice: float) -> str:
    """O índice em palavras: estável, mudou um pouco, ou mudou."""
    if indice < ESTAVEL:
        return "estável"

    return "mudou um pouco" if indice < MUDOU else "mudou"


def por_periodo(referencia: pl.DataFrame, atual: pl.DataFrame, colunas: Sequence[str], periodo: str) -> pl.DataFrame:
    """O índice de cada coluna em cada período da tabela atual: `periodo`, `variavel` e `psi`, em formato longo."""
    linhas = []

    for (valor,), do_periodo in atual.group_by(periodo, maintain_order=True):
        for coluna in colunas:
            linhas.append({periodo: valor, "variavel": coluna, "psi": psi(referencia[coluna], do_periodo[coluna])})

    return pl.DataFrame(linhas).sort(periodo, maintain_order=True)


def _partes(referencia: pl.Series, atual: pl.Series) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """A parte de cada faixa na referência e na atual, com o vazio na última."""
    if referencia.dtype.is_numeric() and referencia.dtype != pl.Boolean:
        cortes = _cortes(referencia)

        def faixas(serie: pl.Series) -> pl.Series:
            return pl.Series(np.searchsorted(cortes, serie.drop_nulls().to_numpy(), side="right"))

        rotulos: list[object] = list(range(len(cortes) + 1))
    else:
        rotulos = sorted(
            {*referencia.drop_nulls().cast(pl.String).to_list(), *atual.drop_nulls().cast(pl.String).to_list()}
        )

        def faixas(serie: pl.Series) -> pl.Series:
            return serie.drop_nulls().cast(pl.String)

    def partes(serie: pl.Series) -> npt.NDArray[np.float64]:
        contagem = faixas(serie).value_counts()
        por_rotulo = dict(zip(contagem[contagem.columns[0]].to_list(), contagem["count"].to_list(), strict=True))
        cheias = [por_rotulo.get(rotulo, 0) for rotulo in rotulos]

        return np.array([*cheias, serie.null_count()], dtype=np.float64) / len(serie)

    return partes(referencia), partes(atual)


def _cortes(referencia: pl.Series) -> npt.NDArray[np.float64]:
    """Os cortes entre as faixas: os decis da referência, sem repetir, para a variável de poucos valores."""
    valores = referencia.drop_nulls().cast(pl.Float64).to_numpy()

    if valores.size == 0:
        return np.array([], dtype=np.float64)

    decis = np.quantile(valores, np.linspace(0, 1, FAIXAS + 1)[1:-1])

    return np.unique(decis)
