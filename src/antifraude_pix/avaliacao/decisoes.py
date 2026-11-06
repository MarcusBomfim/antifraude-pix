"""As decisões contra a verdade: quanto cada jeito de decidir custa de fato (ADR-0010).

Sabendo a verdade, o custo da decisão de cada Pix é:
- o golpe aprovado custa o valor; o segurado, a confirmação e a parte que ela
  não para; o bloqueado, nada;
- o Pix honesto aprovado não custa nada; o segurado custa a confirmação; o
  bloqueado, o bloqueio de um cliente honesto.

A confirmação para uma parte dos golpes, e não se sabe quais: a conta usa o
esperado, e não um sorteio. Assim, a mesma decisão dá sempre o mesmo custo.
"""

from collections.abc import Callable

import polars as pl

from antifraude_pix.avaliacao.metricas import ALERTA, com_o_alerta
from antifraude_pix.decisao.custo import ACAO, Acao, Custos, acao

LIMIAR = 0.5
"""O limiar de costume, o da classificação: o Pix com mais de 50% de chance de golpe é bloqueado."""


def _aprovar_tudo(pontuacoes: pl.DataFrame, custos: Custos) -> pl.Series:
    del custos
    return pl.Series("acao", [Acao.APROVAR.value] * pontuacoes.height, dtype=ACAO)


def _pelo_limiar(pontuacoes: pl.DataFrame, custos: Custos) -> pl.Series:
    del custos
    bloqueia = pontuacoes["probabilidade"] >= LIMIAR

    return pontuacoes.select(
        pl.when(bloqueia).then(pl.lit(Acao.BLOQUEAR.value)).otherwise(pl.lit(Acao.APROVAR.value)).cast(ACAO)
    ).to_series()


def _pelo_alerta(pontuacoes: pl.DataFrame, custos: Custos) -> pl.Series:
    del custos
    return (
        com_o_alerta(pontuacoes)
        .select(
            pl.when(pl.col("alertado"))
            .then(pl.lit(Acao.SEGURAR.value))
            .otherwise(pl.lit(Acao.APROVAR.value))
            .cast(ACAO)
        )
        .to_series()
    )


def _pelo_custo(pontuacoes: pl.DataFrame, custos: Custos) -> pl.Series:
    return pontuacoes.select(acao(pl.col("probabilidade"), pl.col("valor"), custos)).to_series()


POLITICAS: dict[str, Callable[[pl.DataFrame, Custos], pl.Series]] = {
    "aprovar_tudo": _aprovar_tudo,
    "limiar": _pelo_limiar,
    "alerta": _pelo_alerta,
    "custo": _pelo_custo,
}
"""Os jeitos de decidir, do banco sem defesa à decisão pelo custo. Cada um dá a saída de cada Pix."""

NOMES: dict[str, str] = {
    "aprovar_tudo": "aprovar tudo",
    "limiar": f"bloquear de {LIMIAR:.0%} para cima",
    "alerta": f"segurar o alerta de {ALERTA:.1%}".replace(".", ","),
    "custo": "decidir pelo custo",
}


def custo_de_fato(custos: Custos) -> pl.Expr:
    """O custo da decisão (a coluna `acao`) de cada Pix, sabendo a verdade (a coluna `golpe`), em reais."""
    saida, valor = pl.col("acao"), pl.col("valor")

    return (
        pl.when(pl.col("golpe").is_not_null())
        .then(
            pl.when(saida == Acao.APROVAR.value)
            .then(valor)
            .when(saida == Acao.SEGURAR.value)
            .then(custos.confirmar + valor * (1 - custos.confirmacao_para))
            .otherwise(0.0)
        )
        .otherwise(
            pl.when(saida == Acao.APROVAR.value)
            .then(0.0)
            .when(saida == Acao.SEGURAR.value)
            .then(custos.confirmar)
            .otherwise(custos.bloquear_um_honesto)
        )
        .cast(pl.Float64)
    )


def com_a_politica(pontuacoes: pl.DataFrame, politica: str, custos: Custos | None = None) -> pl.DataFrame:
    """As pontuações com a saída da política (`acao`) e o custo de fato dela (`custo`), Pix a Pix."""
    custos = custos or Custos()

    if politica not in POLITICAS:
        raise ValueError(f"Não conheço a política {politica!r}: as que há são {', '.join(POLITICAS)}.")

    return pontuacoes.with_columns(POLITICAS[politica](pontuacoes, custos).alias("acao")).with_columns(
        custo_de_fato(custos).alias("custo")
    )


def balanco(pontuacoes: pl.DataFrame, custos: Custos | None = None) -> pl.DataFrame:
    """Uma linha por modelo e política, com todos os meses de teste juntos.

    - `custo`: o custo de fato, em reais;
    - `perdido`: o dinheiro dos golpes que passaram, inteiro nos aprovados e a parte dos segurados que a
      confirmação não para;
    - `golpes_parados`: os bloqueados, e a parte dos segurados que a confirmação para;
    - `honestos_segurados` e `honestos_bloqueados`;
    - `parte_segurada` e `parte_bloqueada`: entre todos os Pix.
    """
    custos = custos or Custos()
    golpe = pl.col("golpe").is_not_null()
    segurado, bloqueado = pl.col("acao") == Acao.SEGURAR.value, pl.col("acao") == Acao.BLOQUEAR.value
    linhas = []

    for politica in POLITICAS:
        decididas = com_a_politica(pontuacoes, politica, custos)
        linhas.append(
            decididas.group_by("modelo", maintain_order=True).agg(
                pl.lit(politica).alias("politica"),
                pl.col("custo").sum(),
                (
                    pl.when(golpe & (pl.col("acao") == Acao.APROVAR.value))
                    .then(pl.col("valor"))
                    .when(golpe & segurado)
                    .then(pl.col("valor") * (1 - custos.confirmacao_para))
                    .otherwise(0.0)
                    .sum()
                    .alias("perdido")
                ),
                ((golpe & bloqueado).sum() + (golpe & segurado).sum() * custos.confirmacao_para).alias(
                    "golpes_parados"
                ),
                golpe.sum().alias("golpes"),
                (~golpe & segurado).sum().alias("honestos_segurados"),
                (~golpe & bloqueado).sum().alias("honestos_bloqueados"),
                segurado.mean().alias("parte_segurada"),
                bloqueado.mean().alias("parte_bloqueada"),
            )
        )

    return pl.concat(linhas)
