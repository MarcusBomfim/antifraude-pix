"""A visão do banco: só as tabelas que o banco tem, e o que ele sabia em cada instante (ADR-0005).

Tudo o que aprende e tudo o que decide — as variáveis, os modelos, o serviço
que pontua — recebe esta visão, e não a simulação inteira. Assim a verdade do
simulador não tem por onde entrar: não há campo para ela.

E o banco sabe as coisas na hora em que elas acontecem. O corte num instante
devolve o banco daquele momento: os Pix, os aparelhos novos e as contestações
anteriores a ele. É o que o banco tinha para treinar um modelo naquele dia, e
é com ele que se prova que uma variável não olha o futuro.
"""

from dataclasses import dataclass
from datetime import date, datetime, time

import polars as pl

from antifraude_pix.dados.esquema import FUSO, MOMENTO
from antifraude_pix.simulacao.simulador import Simulacao


@dataclass(frozen=True, slots=True)
class VisaoDoBanco:
    """O cadastro, os Pix, os aparelhos novos e as contestações — e nada da verdade."""

    clientes: pl.DataFrame
    transacoes: pl.DataFrame
    aparelhos_novos: pl.DataFrame
    contestacoes: pl.DataFrame

    @staticmethod
    def da(simulacao: Simulacao) -> "VisaoDoBanco":
        """O que o banco vê de uma simulação."""
        return VisaoDoBanco(
            clientes=simulacao.clientes,
            transacoes=simulacao.transacoes,
            aparelhos_novos=simulacao.aparelhos_novos,
            contestacoes=simulacao.contestacoes,
        )

    def ate(self, instante: datetime | date) -> "VisaoDoBanco":
        """O que o banco sabia antes do instante, no horário de Brasília.

        Os Pix e os aparelhos novos de antes dele, e as contestações que já
        tinham chegado — a de um Pix antigo, feita depois do instante, ainda
        não existe. O cadastro fica inteiro: os clientes já estavam no banco.
        """
        corte = pl.lit(_local(instante), dtype=pl.Datetime("us")).dt.replace_time_zone(FUSO).cast(MOMENTO)

        return VisaoDoBanco(
            clientes=self.clientes,
            transacoes=self.transacoes.filter(pl.col("momento") < corte),
            aparelhos_novos=self.aparelhos_novos.filter(pl.col("momento") < corte),
            contestacoes=self.contestacoes.filter(pl.col("contestada_em") < corte),
        )


def _local(instante: datetime | date) -> datetime:
    if isinstance(instante, datetime):
        if instante.tzinfo is not None:
            raise ValueError(
                "O instante do corte é no horário de Brasília, sem fuso: 2025-11-01 00:00, e não 2025-11-01 03:00 UTC."
            )

        return instante

    return datetime.combine(instante, time())
