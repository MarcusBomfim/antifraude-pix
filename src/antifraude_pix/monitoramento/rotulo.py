"""O rótulo que chega depois: o golpe que passou, e que o banco só conhece quando a vítima reclama (ADR-0020).

A contestação chega horas depois do Pix na invasão de conta, e semanas depois
na venda falsa. A semana passada tem sempre menos golpes conhecidos do que vai
ter: não porque o modelo melhorou, mas porque a vítima ainda não reclamou.

A chegada do rótulo se mede na história do próprio banco, sem a verdade: entre
os Pix contestados que já tiveram todo o prazo do MED, a parte contestada até
k dias depois do Pix. É a curva de chegada, `F(k)`.

Com ela, o monitor compara duas perdas nos Pix que o modelo aprovou:
- a **esperada** pelo próprio modelo: a probabilidade vezes o valor de cada
  Pix, vezes a parte do rótulo que já chegou na idade dele;
- a **conhecida** pelo rótulo: o valor dos aprovados que o banco já sabe contestados.

Com o modelo calibrado e o golpe de sempre, as duas andam juntas. O golpe
novo, que o modelo não reconhece, passa com a probabilidade baixa: a perda
conhecida sobe, e a esperada, não.
"""

from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import numpy.typing as npt
import polars as pl

from antifraude_pix.dados.banco import VisaoDoBanco
from antifraude_pix.dados.esquema import FUSO
from antifraude_pix.decisao.custo import Acao

PRAZO_DO_MED = 80
"""Os dias que a vítima tem para contestar: depois deles, o rótulo de um Pix não muda mais."""

ALARME = 2.0
"""A razão entre a perda conhecida e a esperada que dispara o alarme: o golpe passa o dobro do que o modelo espera."""

CONTESTADO = "contestado"
"""O rótulo que o banco tem: não sabe qual golpe foi, só que a vítima reclamou."""

DIA = 86_400
"""Os segundos de um dia."""


def chegada(visao: VisaoDoBanco, ate: date, prazo: int = PRAZO_DO_MED) -> pl.DataFrame:
    """A curva de chegada do rótulo em `ate`: os `dias` depois do Pix, e a `parte` das contestações que chega até ali.

    Conta só os Pix de mais de `prazo` dias antes de `ate`, que já tiveram todo o tempo de ser contestados.
    """
    limite = _instante(ate - timedelta(days=prazo))
    demoras = (
        visao.ate(ate)
        .contestacoes.join(visao.transacoes.select(pl.col("id").alias("transacao"), "momento"), on="transacao")
        .filter(pl.col("momento") < limite)
        .select(((pl.col("contestada_em") - pl.col("momento")).dt.total_seconds() / DIA).alias("dias"))
    )

    if demoras.is_empty():
        raise ValueError(f"Não há contestação de Pix com mais de {prazo} dias antes de {ate} para medir a chegada.")

    ordenadas = np.sort(demoras["dias"].to_numpy())
    dias = np.arange(prazo + 1)

    return pl.DataFrame(
        {"dias": dias, "parte": np.searchsorted(ordenadas, dias, side="right") / ordenadas.size},
        schema={"dias": pl.Int32, "parte": pl.Float64},
    )


def parte_que_chegou(curva: pl.DataFrame, idades_em_dias: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """A parte do rótulo que já chegou para Pix destas idades: a curva, entre os pontos dela; antes do Pix, nada."""
    idades = np.maximum(np.asarray(idades_em_dias, dtype=np.float64), 0.0)

    return np.asarray(np.interp(idades, curva["dias"].to_numpy(), curva["parte"].to_numpy()), dtype=np.float64)


def conhecidos(visao: VisaoDoBanco, ate: date) -> pl.DataFrame:
    """Os Pix que o banco sabe contestados em `ate`: `id`, e o `golpe` que ele conhece, sempre `contestado`.

    É o rótulo com que o banco mede o modelo, e não a verdade: o golpe que a vítima ainda não contestou fica de
    fora, e o Pix honesto contestado por arrependimento ou má-fé entra.
    """
    return (
        visao.ate(ate)
        .contestacoes.select(pl.col("transacao").alias("id"), pl.lit(CONTESTADO).alias("golpe"))
        .unique("id", keep="first", maintain_order=True)
    )


def perda_por_semana(decididos: pl.DataFrame, visao: VisaoDoBanco, ate: date, curva: pl.DataFrame) -> pl.DataFrame:
    """A perda nos Pix aprovados de cada semana: a esperada pelo modelo e a conhecida pelo rótulo, em `ate`.

    Entram os Pix decididos, com `id`, `momento`, `valor` (em reais), `probabilidade` e `acao`; contam os
    aprovados de antes de `ate`. Sai uma linha por `semana` (a segunda-feira), com os `aprovados`, a perda
    `esperada`, já descontado o rótulo que ainda não chegou, a perda `conhecida`, e a `razao` entre as duas.
    """
    limite = _instante(ate)
    aprovados = decididos.filter((pl.col("acao") == Acao.APROVAR.value) & (pl.col("momento") < limite))
    idades = (limite.timestamp() - aprovados["momento"].dt.epoch("s").to_numpy()) / DIA
    sabidos = conhecidos(visao, ate).select("id", pl.lit(value=True).alias("sabido"))

    return (
        aprovados.with_columns(pl.Series("chegou", parte_que_chegou(curva, idades)))
        .join(sabidos, on="id", how="left")
        .with_columns(
            pl.col("sabido").fill_null(value=False), pl.col("momento").dt.date().dt.truncate("1w").alias("semana")
        )
        .group_by("semana")
        .agg(
            pl.len().alias("aprovados"),
            (pl.col("probabilidade") * pl.col("valor") * pl.col("chegou")).sum().alias("esperada"),
            pl.col("valor").filter(pl.col("sabido")).sum().cast(pl.Float64).alias("conhecida"),
        )
        .sort("semana")
        .with_columns((pl.col("conhecida") / pl.col("esperada")).alias("razao"))
    )


def razao(semanas: pl.DataFrame) -> float:
    """A perda conhecida sobre a esperada, de todas as semanas juntas: de `ALARME` para cima, o golpe passa demais."""
    esperada = float(semanas["esperada"].sum())

    if esperada <= 0:
        raise ValueError(
            "A razão pede alguma perda esperada: não há Pix aprovado, ou o modelo não espera golpe nenhum."
        )

    return float(semanas["conhecida"].sum()) / esperada


def _instante(dia: date) -> datetime:
    """A meia-noite do dia, no horário de Brasília."""
    return datetime.combine(dia, time(), tzinfo=ZoneInfo(FUSO))
