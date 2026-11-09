"""O porquê de cada decisão, em português (ADR-0012).

A LGPD (art. 20) dá ao cliente o direito de pedir a revisão de uma decisão
tomada só por máquina, e de saber os critérios dela. Cada Pix segurado ou
bloqueado sai com as suas razões: as variáveis que mais empurraram a pontuação
para o golpe, cada uma numa frase com o valor que ela teve no Pix.

O quanto cada variável empurrou depende do modelo:
- **o LightGBM**: os valores de Shapley das árvores, que o próprio LightGBM calcula;
- **a regressão logística**: o peso de cada coluna vezes o valor padronizado dela, somado por variável;
- **as regras de hoje**: os pontos de cada regra que valeu.

Nos três, a base mais a soma das contribuições é a pontuação do modelo, e os
testes conferem: a explicação não é uma história contada depois, é a conta
que decidiu.
"""

import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl

from antifraude_pix.decisao.custo import Acao
from antifraude_pix.decisao.frases import frase
from antifraude_pix.modelos.preparo import matriz_das_arvores
from antifraude_pix.modelos.regras import REGRAS, acionadas
from antifraude_pix.modelos.treino import Arvores, Calibrado, Logistica, Regras

QUANTAS = 3
"""As razões de cada decisão: as três que mais pesaram. Mais que isso, quem lê não lê."""

CABECAS = {
    Acao.APROVAR: "Aprovado",
    Acao.SEGURAR: "Segurado para confirmar com o cliente",
    Acao.BLOQUEAR: "Bloqueado",
}


@dataclass(frozen=True, slots=True)
class Razao:
    """Uma razão da decisão: a variável (ou a regra), o quanto ela empurrou a pontuação, e a frase."""

    nome: str
    peso: float
    frase: str


def contribuicoes(modelo: Calibrado, variaveis: pl.DataFrame) -> pl.DataFrame:
    """O quanto cada variável empurrou a pontuação de cada Pix, numa coluna por variável, e a `base`.

    Nas árvores e na logística, há uma coluna para cada variável da entrada do
    modelo; nas regras, uma para cada regra. A base mais a soma das colunas é a pontuação.
    """
    match modelo.pontuador:
        case Arvores() as arvores:
            nomes = [variavel.nome for variavel in arvores.entrada]

            # O LightGBM recusa uma matriz sem linhas: a de um lote em que todo Pix foi aprovado.
            if variaveis.is_empty():
                return pl.DataFrame(schema=dict.fromkeys([*nomes, "base"], pl.Float64))

            matriz = matriz_das_arvores(variaveis, arvores.entrada)
            valores = np.asarray(arvores.modelo.predict(matriz, pred_contrib=True), dtype=np.float64)

            return pl.DataFrame(valores, schema=[*nomes, "base"], orient="row")
        case Logistica() as logistica:
            por_coluna = logistica.preparo.matriz(variaveis) * np.asarray(logistica.modelo.coef_[0], dtype=np.float64)
            # A coluna "valor vazia" e a casa "hora=3" são da variável que vem antes do espaço ou do igual.
            da_variavel = [re.split("[= ]", coluna)[0] for coluna in logistica.preparo.colunas]
            colunas = {
                variavel.nome: por_coluna[:, [i for i, nome in enumerate(da_variavel) if nome == variavel.nome]].sum(
                    axis=1
                )
                for variavel in logistica.preparo.entrada
            }
            base = float(logistica.modelo.intercept_[0])

            return pl.DataFrame(colunas).with_columns(pl.lit(base).alias("base"))
        case Regras():
            return acionadas(variaveis).select(
                *(pl.col(regra.nome).cast(pl.Float64) * regra.pontos for regra in REGRAS), pl.lit(0.0).alias("base")
            )
        case _:
            raise TypeError(f"Não sei explicar o modelo {modelo.nome}.")


def razoes(modelo: Calibrado, variaveis: pl.DataFrame, quantas: int = QUANTAS) -> list[tuple[Razao, ...]]:
    """As razões de cada Pix: as variáveis que mais empurraram a pontuação para o golpe, da que mais pesou.

    Só entra quem empurrou para o golpe. Duas variáveis que dizem a mesma coisa
    ("nunca tinha pagado este recebedor") viram uma razão só.
    """
    pesos = contribuicoes(modelo, variaveis).drop("base")

    return [
        escolher(linha, valores, quantas)
        for linha, valores in zip(pesos.iter_rows(named=True), variaveis.iter_rows(named=True), strict=True)
    ]


def escolher(pesos: dict[str, float], valores: dict[str, Any], quantas: int = QUANTAS) -> tuple[Razao, ...]:
    """As razões de um Pix, pelo quanto cada variável (ou regra) empurrou e pelo valor de cada variável."""
    escolhidas: list[Razao] = []

    for nome, peso in sorted(pesos.items(), key=lambda par: -par[1]):
        if peso <= 0 or len(escolhidas) == quantas:
            break

        dita = _DESCRICOES[nome] if nome in _DESCRICOES else frase(nome, valores[nome])

        if all(razao.frase != dita for razao in escolhidas):
            escolhidas.append(Razao(nome, peso, dita))

    return tuple(escolhidas)


def explicar(modelo: Calibrado, decididos: pl.DataFrame, quantas: int = QUANTAS) -> pl.DataFrame:
    """Os Pix decididos com a coluna `explicacao`: a decisão, a chance de golpe e, quando não aprovado, as razões.

    A entrada tem as variáveis, a `probabilidade` e a `acao` de cada Pix.
    """
    nao_aprovados = decididos.with_row_index("_linha").filter(pl.col("acao") != Acao.APROVAR.value)
    das_razoes = dict(zip(nao_aprovados["_linha"].to_list(), razoes(modelo, nao_aprovados, quantas), strict=True))
    textos = [
        texto(Acao(acao), probabilidade, das_razoes.get(linha, ()))
        for linha, (acao, probabilidade) in enumerate(decididos.select("acao", "probabilidade").iter_rows())
    ]

    return decididos.with_columns(pl.Series("explicacao", textos, dtype=pl.String))


def texto(acao: Acao, probabilidade: float, as_razoes: tuple[Razao, ...]) -> str:
    """A explicação de uma decisão numa frase: "Bloqueado, com 97% de chance de golpe: a; b; e c."."""
    cabeca = f"{CABECAS[acao]}, com {chance(probabilidade)} de chance de golpe"

    if not as_razoes:
        return f"{cabeca}."

    frases = [razao.frase for razao in as_razoes]
    juntas = frases[0] if len(frases) == 1 else "; ".join(frases[:-1]) + f"; e {frases[-1]}"

    return f"{cabeca}: {juntas}."


def chance(probabilidade: float) -> str:
    """A chance de golpe para quem lê: com uma casa até 10%, e sem casas daí em diante."""
    if probabilidade < 0.001:
        return "menos de 0,1%"

    casas = 1 if probabilidade < 0.1 else 0

    return f"{probabilidade:.{casas}%}".replace(".", ",")


def _minuscula(descricao: str) -> str:
    """A descrição da regra dentro da frase: sem a maiúscula do começo, e sem o ponto do fim."""
    return descricao[0].lower() + descricao[1:].removesuffix(".")


_DESCRICOES = {regra.nome: _minuscula(regra.descricao) for regra in REGRAS}
"""A frase de cada regra de hoje, como razão: a descrição dela, dentro da frase."""
