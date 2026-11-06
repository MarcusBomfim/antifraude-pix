"""A decisão pelo custo: aprovar, segurar para confirmar ou bloquear (ADR-0010).

O banco tem três saídas para cada Pix, e cada uma erra de um jeito:
- **aprovar**: o Pix sai na hora. Se era golpe, o dinheiro vai embora;
- **segurar**: o Pix espera alguns minutos, e o banco confirma com o cliente,
  por um canal que o golpista não controla. A confirmação custa ao banco e
  incomoda o cliente honesto, e não para todo golpe: a vítima convencida
  confirma;
- **bloquear**: o Pix não sai. Para o golpe inteiro, e o cliente honesto
  bloqueado reclama, liga, e às vezes vai embora.

Com a probabilidade calibrada `p` e o valor `v`, o custo esperado de cada saída é:
- aprovar: `p * v`;
- segurar: `confirmar + p * v * (1 - confirmacao_para)`;
- bloquear: `(1 - p) * bloquear_um_honesto`.

A decisão é a saída de menor custo esperado; no empate, a mais branda. Por
isso o limiar não é um número só: o Pix de R$ 3.000 é segurado com uma chance
de golpe que deixaria passar o de R$ 50.
"""

from dataclasses import dataclass
from enum import StrEnum

import polars as pl


class Acao(StrEnum):
    """As saídas do banco para um Pix, da mais branda à mais dura."""

    APROVAR = "aprovar"
    SEGURAR = "segurar"
    BLOQUEAR = "bloquear"


ACOES: tuple[Acao, ...] = tuple(Acao)
ACAO = pl.Enum([acao.value for acao in Acao])
"""A coluna da decisão: uma categoria fechada, na ordem da mais branda para a mais dura."""


@dataclass(frozen=True, slots=True)
class Custos:
    """O que custa cada erro, em reais, e o quanto a confirmação para o golpe. São suposições, escritas na ADR-0010."""

    confirmar: float = 10.0
    """Segurar um Pix para confirmar: o atendimento, a espera e o incômodo do cliente honesto."""
    bloquear_um_honesto: float = 150.0
    """Bloquear o Pix de um cliente honesto: a reclamação, a ligação, e o cliente que às vezes vai embora."""
    confirmacao_para: float = 0.7
    """A parte dos golpes que a confirmação para: a vítima da invasão diz que não foi ela; a convencida confirma."""

    def __post_init__(self) -> None:
        if self.confirmar <= 0:
            raise ValueError(f"Confirmar custa alguma coisa: o custo tem de ser positivo, e não {self.confirmar}.")

        if self.bloquear_um_honesto <= self.confirmar:
            raise ValueError(
                "Bloquear um cliente honesto custa mais que confirmar com ele: "
                f"{self.bloquear_um_honesto} não passa de {self.confirmar}."
            )

        if not 0 < self.confirmacao_para <= 1:
            raise ValueError(
                f"A confirmação para uma parte dos golpes, maior que 0 e até 1, e não {self.confirmacao_para}."
            )


def custos_esperados(probabilidade: pl.Expr, valor: pl.Expr, custos: Custos) -> dict[Acao, pl.Expr]:
    """O custo esperado de cada saída, em reais, para a probabilidade e o valor (em reais) de cada Pix."""
    return {
        Acao.APROVAR: probabilidade * valor,
        Acao.SEGURAR: custos.confirmar + probabilidade * valor * (1 - custos.confirmacao_para),
        Acao.BLOQUEAR: (1 - probabilidade) * custos.bloquear_um_honesto,
    }


def acao(probabilidade: pl.Expr, valor: pl.Expr, custos: Custos) -> pl.Expr:
    """A saída de menor custo esperado. No empate, a mais branda: endurecer tem de valer a pena."""
    esperado = custos_esperados(probabilidade, valor, custos)
    escolha = pl.lit(Acao.APROVAR.value)
    menor = esperado[Acao.APROVAR]

    for saida in (Acao.SEGURAR, Acao.BLOQUEAR):
        melhor = esperado[saida] < menor
        escolha = pl.when(melhor).then(pl.lit(saida.value)).otherwise(escolha)
        menor = pl.when(melhor).then(esperado[saida]).otherwise(menor)

    return escolha.cast(ACAO)


def decidir(pontuacoes: pl.DataFrame, custos: Custos | None = None) -> pl.DataFrame:
    """As pontuações com a coluna `acao`, pela `probabilidade` e pelo `valor` (em reais) de cada Pix."""
    return pontuacoes.with_columns(acao(pl.col("probabilidade"), pl.col("valor"), custos or Custos()).alias("acao"))
