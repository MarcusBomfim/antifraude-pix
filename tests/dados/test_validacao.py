from collections.abc import Callable
from dataclasses import replace
from datetime import datetime, timedelta

import polars as pl
import pytest

from antifraude_pix.dados.esquema import FUSO
from antifraude_pix.dados.validacao import problemas
from antifraude_pix.simulacao.simulador import Simulacao

type Adulteracao = Callable[[Simulacao], Simulacao]


def _na_linha(tabela: pl.DataFrame, linha: int, coluna: str, valor: object) -> pl.DataFrame:
    """A tabela com um valor trocado numa linha só."""
    return tabela.with_columns(
        pl.when(pl.int_range(pl.len()) == linha)
        .then(pl.lit(valor, dtype=tabela.schema[coluna]))
        .otherwise(pl.col(coluna))
        .alias(coluna)
    )


def _uma_de_dia(simulacao: Simulacao) -> int:
    """Uma linha de um Pix legítimo, de dia, no meio da simulação: onde trocar um valor não mexe no limite noturno."""
    pix = simulacao.transacoes.with_row_index("linha").join(
        simulacao.verdade.fraudes, left_on="id", right_on="transacao", how="anti"
    )

    return int(pix.filter(pl.col("momento").dt.hour() == 12)["linha"][1_000])


def _transacoes(adulterar: Callable[[Simulacao, pl.DataFrame], pl.DataFrame]) -> Adulteracao:
    return lambda simulacao: replace(simulacao, transacoes=adulterar(simulacao, simulacao.transacoes))


def _contestacoes(adulterar: Callable[[Simulacao, pl.DataFrame], pl.DataFrame]) -> Adulteracao:
    return lambda simulacao: replace(simulacao, contestacoes=adulterar(simulacao, simulacao.contestacoes))


def _fraudes(adulterar: Callable[[pl.DataFrame], pl.DataFrame]) -> Adulteracao:
    return lambda simulacao: replace(
        simulacao, verdade=replace(simulacao.verdade, fraudes=adulterar(simulacao.verdade.fraudes))
    )


def _momento_do_pix(simulacao: Simulacao, transacao: int) -> datetime:
    momento = simulacao.transacoes.filter(pl.col("id") == transacao)["momento"].dt.replace_time_zone(None).item()
    assert isinstance(momento, datetime)

    return momento


def _contestada_em(simulacao: Simulacao, contestacoes: pl.DataFrame, deslocamento: timedelta) -> pl.DataFrame:
    """A primeira contestação, movida para o momento do Pix mais o deslocamento."""
    novo = _momento_do_pix(simulacao, int(contestacoes["transacao"][0])) + deslocamento

    return contestacoes.with_columns(
        pl.when(pl.int_range(pl.len()) == 0)
        .then(pl.lit(novo).dt.replace_time_zone(FUSO))
        .otherwise(pl.col("contestada_em"))
        .cast(contestacoes.schema["contestada_em"])
        .alias("contestada_em")
    )


def _cliente_da_noite(simulacao: Simulacao) -> int:
    hora = pl.col("momento").dt.hour()

    return int(simulacao.transacoes.filter((hora >= 21) | (hora < 5))["cliente"][0])


ADULTERACOES: dict[str, tuple[Adulteracao, str]] = {
    "coluna a menos": (
        _transacoes(lambda _, pix: pix.drop("saldo_antes")),
        "A tabela transacoes não segue o contrato",
    ),
    "tipo trocado": (
        _transacoes(lambda _, pix: pix.with_columns(pl.col("valor").cast(pl.Float64))),
        "A tabela transacoes não segue o contrato",
    ),
    "ids fora de sequência": (
        _transacoes(lambda _, pix: pix.with_columns(pl.col("id") * 2)),
        "Os ids dos Pix não vão de 0 ao último, em ordem.",
    ),
    "fora de ordem": (
        _transacoes(
            lambda _, pix: _na_linha(pix, 5_000, "momento", pix["momento"][5_000] - timedelta(days=3)),
        ),
        "Há 1 Pix fora da ordem do tempo.",
    ),
    "fora do período": (
        _transacoes(
            lambda s, pix: _na_linha(
                pix, pix.height - 1, "momento", datetime.combine(s.config.fim, datetime.min.time()).replace(hour=12)
            ).with_columns(pl.col("momento").dt.replace_time_zone(None).dt.replace_time_zone(FUSO))
        ),
        "Há 1 Pix fora do período simulado.",
    ),
    "valor zero": (
        _transacoes(lambda s, pix: _na_linha(pix, _uma_de_dia(s), "valor", 0)),
        "Há 1 Pix de valor zero ou negativo.",
    ),
    "maior que o saldo": (
        _transacoes(lambda s, pix: _na_linha(pix, _uma_de_dia(s), "saldo_antes", 0)),
        "Há 1 Pix maiores que o saldo que o cliente tinha.",
    ),
    "manual com chave": (
        _transacoes(
            lambda s, pix: _na_linha(
                _na_linha(pix, _uma_de_dia(s), "iniciacao", "manual"), _uma_de_dia(s), "chave", "cpf"
            )
        ),
        "Há 1 Pix manuais com chave, ou pela chave sem ela.",
    ),
    "cliente que não existe": (
        _transacoes(lambda s, pix: _na_linha(pix, _uma_de_dia(s), "cliente", 999_999)),
        "Há 1 Pix de clientes que não estão no cadastro.",
    ),
    "aparelho que o cliente não tinha": (
        _transacoes(
            lambda s, pix: _na_linha(pix, _uma_de_dia(s), "aparelho", int(pix["cliente"][_uma_de_dia(s)]) * 16 + 7)
        ),
        "Há 1 Pix de um aparelho que o cliente ainda não tinha.",
    ),
    "limite noturno": (
        lambda s: replace(
            s,
            clientes=s.clientes.with_columns(
                pl.when(pl.col("id") == _cliente_da_noite(s))
                .then(1)
                .otherwise(pl.col("limite_noturno"))
                .alias("limite_noturno")
            ),
        ),
        "noites em que um cliente passou do limite noturno.",
    ),
    "cadastro incompleto": (
        lambda s: replace(s, clientes=s.clientes.slice(1)),
        "Os ids dos clientes não vão de 0 a 1499, em ordem.",
    ),
    "contestação antes do Pix": (
        _contestacoes(lambda s, contestacoes: _contestada_em(s, contestacoes, -timedelta(hours=1))),
        "Há 1 contestações antes do Pix, ou depois do prazo do MED.",
    ),
    "contestação depois do prazo do MED": (
        _contestacoes(lambda s, contestacoes: _contestada_em(s, contestacoes, timedelta(days=81))),
        "Há 1 contestações antes do Pix, ou depois do prazo do MED.",
    ),
    "contestação de Pix que não existe": (
        _contestacoes(lambda _, contestacoes: _na_linha(contestacoes, 0, "transacao", 10**12)),
        "Há 1 contestações de Pix que não existem.",
    ),
    "contestação repetida": (
        _contestacoes(lambda _, contestacoes: pl.concat([contestacoes, contestacoes.head(1)])),
        "Há 2 contestações repetidas do mesmo Pix.",
    ),
    "fraude de Pix que não existe": (
        _fraudes(lambda fraudes: _na_linha(fraudes, 0, "transacao", 10**12)),
        "Há 1 fraudes de Pix que não existem.",
    ),
    "fraude de ataque que não existe": (
        _fraudes(lambda fraudes: _na_linha(fraudes, 0, "ataque", 999_999)),
        "Há 1 fraudes de ataques que não existem.",
    ),
    "golpe trocado": (
        _fraudes(
            lambda fraudes: _na_linha(
                fraudes, 0, "golpe", "venda_falsa" if fraudes["golpe"][0] != "venda_falsa" else "falso_parente"
            )
        ),
        "Há 1 fraudes com um golpe diferente do golpe do ataque.",
    ),
}


@pytest.mark.parametrize("adulteracao", ADULTERACOES.values(), ids=ADULTERACOES.keys())
def test_os_dados_adulterados_sao_recusados_com_o_motivo(
    simulacao: Simulacao, adulteracao: tuple[Adulteracao, str]
) -> None:
    adulterar, motivo = adulteracao

    encontrados = problemas(adulterar(simulacao))

    assert any(motivo in problema for problema in encontrados), encontrados


def test_os_problemas_vem_todos_de_uma_vez(simulacao: Simulacao) -> None:
    adulterada = simulacao

    for nome in ("valor zero", "contestação repetida", "golpe trocado"):
        adulterada = ADULTERACOES[nome][0](adulterada)

    assert len(problemas(adulterada)) >= 3
