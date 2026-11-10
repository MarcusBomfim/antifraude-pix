"""O retreino: o campeão contra o desafiante, nos mesmos Pix, com o rótulo que o banco tem (ADR-0021).

O desafiante é a mesma receita do campeão, treinada com o que o banco sabe
agora. Ele não ganha por ser novo: ganha se, nos Pix das duas últimas semanas,
que nenhum dos dois viu no treino, medidos com o rótulo que já chegou,
- a precisão média dele passa a do campeão por pelo menos `MARGEM`;
- o custo das decisões dele não passa o do campeão;
- ele não bloqueia mais clientes honestos de 65 anos ou mais do que o acaso
  explica (ADR-0011).

Os dois decidem os mesmos Pix, e o bloqueio dos mais velhos se compara Pix a
Pix: contam os honestos que só o desafiante bloqueia, `b`, e os que só o
campeão bloqueia, `c`. A diferença `b - c` que passa de duas vezes `√(b + c)`
não é sorte: é o desafiante mais duro com eles.

O que perde não some: fica no registro do MLflow, com o apelido de desafiante,
e o veredito na versão dele.
"""

import math
from dataclasses import dataclass

import polars as pl

from antifraude_pix.avaliacao.decisoes import com_a_politica
from antifraude_pix.avaliacao.metricas import precisao_media
from antifraude_pix.decisao.custo import Acao, Custos

MARGEM = 0.01
"""O quanto a precisão média do desafiante tem de passar a do campeão: a diferença menor que isso é ruído."""

JANELA = 14
"""Os dias de Pix em que os dois se comparam, antes do retreino: o desafiante é treinado no começo deles."""

IDOSO = 65
"""A idade a partir da qual o bloqueio do Pix honesto se confere à parte (ADR-0011)."""

ACASO = 2.0
"""Quantos desvios a diferença no bloqueio dos mais velhos pode ter antes de deixar de ser acaso."""


@dataclass(frozen=True, slots=True)
class Medidas:
    """As medidas de um modelo nos Pix da comparação, com o rótulo que o banco tem."""

    precisao_media: float
    custo: float
    golpes: int
    golpes_pegos: int
    idosos_honestos_bloqueados: int


@dataclass(frozen=True, slots=True)
class Veredito:
    """A comparação: as medidas dos dois, o bloqueio dos mais velhos Pix a Pix, e o que o desafiante não passou."""

    campeao: Medidas
    desafiante: Medidas
    so_o_desafiante_bloqueia: int
    so_o_campeao_bloqueia: int
    razoes: tuple[str, ...]
    """O motivo de cada regra que o desafiante não passou; sem nenhum, ele ganha."""

    @property
    def promover(self) -> bool:
        return not self.razoes


def comparar(
    campeao: pl.DataFrame, desafiante: pl.DataFrame, idades: pl.DataFrame, custos: Custos | None = None
) -> Veredito:
    """O veredito entre os dois, das pontuações deles nos mesmos Pix.

    Cada tabela tem `id`, `valor` (em reais), `probabilidade` e `golpe`: o rótulo que o banco tem, vazio no Pix
    que não foi contestado. As `idades` têm `id` e `idade`.
    """
    custos = custos or Custos()

    if not campeao["id"].sort().equals(desafiante["id"].sort()):
        raise ValueError("O campeão e o desafiante têm de ser medidos nos mesmos Pix.")

    do_campeao = _decididos(campeao, idades, custos)
    do_desafiante = _decididos(desafiante, idades, custos)
    medidas_do_campeao, medidas_do_desafiante = _medidas(do_campeao), _medidas(do_desafiante)
    lado_a_lado = do_campeao.select("id", pl.col("idoso_bloqueado").alias("pelo_campeao")).join(
        do_desafiante.select("id", pl.col("idoso_bloqueado").alias("pelo_desafiante")), on="id"
    )
    so_o_desafiante = lado_a_lado.filter(pl.col("pelo_desafiante") & ~pl.col("pelo_campeao")).height
    so_o_campeao = lado_a_lado.filter(pl.col("pelo_campeao") & ~pl.col("pelo_desafiante")).height
    razoes = _razoes(medidas_do_campeao, medidas_do_desafiante, so_o_desafiante, so_o_campeao)

    return Veredito(medidas_do_campeao, medidas_do_desafiante, so_o_desafiante, so_o_campeao, razoes)


def _decididos(pontuacoes: pl.DataFrame, idades: pl.DataFrame, custos: Custos) -> pl.DataFrame:
    """A decisão pelo custo de cada Pix, o custo dela, e se ela bloqueia um cliente honesto de 65 anos ou mais."""
    com_a_idade = pontuacoes.join(idades, on="id", how="left")

    if com_a_idade["idade"].null_count():
        raise ValueError("Falta a idade do cliente de algum Pix da comparação.")

    return com_a_politica(com_a_idade, "custo", custos).with_columns(
        (pl.col("golpe").is_null() & (pl.col("idade") >= IDOSO) & (pl.col("acao") == Acao.BLOQUEAR.value)).alias(
            "idoso_bloqueado"
        )
    )


def _medidas(decididos: pl.DataFrame) -> Medidas:
    golpe = decididos["golpe"].is_not_null()

    return Medidas(
        precisao_media=precisao_media(golpe, decididos["probabilidade"]),
        custo=float(decididos["custo"].sum()),
        golpes=int(golpe.sum()),
        golpes_pegos=int((golpe & (decididos["acao"] != Acao.APROVAR.value)).sum()),
        idosos_honestos_bloqueados=int(decididos["idoso_bloqueado"].sum()),
    )


def _razoes(campeao: Medidas, desafiante: Medidas, so_o_desafiante: int, so_o_campeao: int) -> tuple[str, ...]:
    razoes = []

    if desafiante.precisao_media < campeao.precisao_media + MARGEM:
        razoes.append(
            f"a precisão média dele, {_decimal(desafiante.precisao_media)}, não passa a do campeão, "
            f"{_decimal(campeao.precisao_media)}, por {_decimal(MARGEM, 2)}"
        )

    if desafiante.custo > campeao.custo:
        razoes.append(f"as decisões dele custam mais: {_reais(desafiante.custo)} contra {_reais(campeao.custo)}")

    if so_o_desafiante - so_o_campeao > ACASO * math.sqrt(so_o_desafiante + so_o_campeao):
        razoes.append(
            f"ele bloqueia {so_o_desafiante} Pix honestos de clientes de {IDOSO} anos ou mais que o campeão não "
            f"bloqueia, contra {so_o_campeao} do contrário: mais do que o acaso explica"
        )

    return tuple(razoes)


def _decimal(valor: float, casas: int = 3) -> str:
    return f"{valor:.{casas}f}".replace(".", ",")


def _reais(valor: float) -> str:
    return f"R$ {valor:,.0f}".replace(",", ".")
