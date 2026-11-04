from datetime import date

import polars as pl
from hypothesis import HealthCheck, example, given
from hypothesis import settings as configurar
from hypothesis import strategies as st

from antifraude_pix.dados.esquema import DA_VERDADE, DO_BANCO
from antifraude_pix.dados.validacao import problemas
from antifraude_pix.simulacao.configuracao import Configuracao
from antifraude_pix.simulacao.golpes import Golpe
from antifraude_pix.simulacao.simulador import Simulacao, simular

PEQUENA = Configuracao(clientes=200, dias=30, golpes_por_mil_clientes_ao_mes=80)


def _tabelas(simulacao: Simulacao) -> list[pl.DataFrame]:
    verdade = simulacao.verdade

    return [
        simulacao.clientes,
        simulacao.transacoes,
        simulacao.aparelhos_novos,
        simulacao.contestacoes,
        verdade.fraudes,
        verdade.ataques,
        verdade.laranjas,
    ]


def test_a_mesma_semente_da_as_mesmas_tabelas_e_outra_semente_da_outras() -> None:
    uma, outra = simular(PEQUENA), simular(PEQUENA)
    diferente = simular(Configuracao(clientes=200, dias=30, golpes_por_mil_clientes_ao_mes=80, semente=99))

    assert all(a.equals(b) for a, b in zip(_tabelas(uma), _tabelas(outra), strict=True))
    assert not uma.transacoes.equals(diferente.transacoes)


def test_a_simulacao_media_segue_o_contrato(simulacao: Simulacao) -> None:
    assert problemas(simulacao) == []


def test_o_que_o_banco_ve_nao_diz_o_que_e_golpe(simulacao: Simulacao) -> None:
    colunas_do_banco = {coluna for esquema in DO_BANCO.values() for coluna in esquema}

    assert colunas_do_banco.isdisjoint({"ataque", "golpe", "laranja"})
    assert set(DA_VERDADE) == {"fraudes", "ataques", "laranjas"}
    assert simulacao.transacoes.columns == list(DO_BANCO["transacoes"])


def test_todos_os_golpes_aparecem_e_a_fraude_e_rara(simulacao: Simulacao) -> None:
    fraudes = simulacao.verdade.fraudes

    assert set(fraudes["golpe"].unique().to_list()) == {golpe.value for golpe in Golpe}
    # Com seis vezes mais golpes que o padrão, ainda menos de um Pix em cem.
    assert 0.001 < fraudes.height / simulacao.transacoes.height < 0.01


@configurar(max_examples=12, suppress_health_check=[HealthCheck.too_slow])
@given(
    clientes=st.integers(min_value=5, max_value=60),
    dias=st.integers(min_value=1, max_value=40),
    semente=st.integers(min_value=0, max_value=2**32),
    golpes=st.floats(min_value=0, max_value=300),
    inicio=st.dates(min_value=date(2023, 1, 1), max_value=date(2026, 12, 31)),
)
# Cinco clientes têm só dez pessoas por perto: menos do que os doze contatos que um cliente pode ter.
@example(clientes=5, dias=1, semente=1, golpes=0.0, inicio=date(2023, 1, 1))
def test_qualquer_simulacao_pequena_segue_o_contrato(
    clientes: int, dias: int, semente: int, golpes: float, inicio: date
) -> None:
    config = Configuracao(
        clientes=clientes, dias=dias, semente=semente, golpes_por_mil_clientes_ao_mes=golpes, inicio=inicio
    )

    assert problemas(simular(config)) == []
