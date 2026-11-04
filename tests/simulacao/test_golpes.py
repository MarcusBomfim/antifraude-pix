from datetime import date, timedelta

import pytest

from antifraude_pix.simulacao.configuracao import Configuracao
from antifraude_pix.simulacao.golpes import (
    APARELHOS_DO_GOLPISTA,
    Ataque,
    Golpe,
    passos,
    planejar_ataques,
)
from antifraude_pix.simulacao.pix import Iniciacao, segundos
from antifraude_pix.simulacao.populacao import Populacao, ataques_esperados, gerar_populacao
from tests.apoio import MEDIA

# O último trimestre, com a Black Friday, e golpes a mais para contar.
TRIMESTRE = Configuracao(clientes=3_000, inicio=date(2025, 10, 1), dias=92, golpes_por_mil_clientes_ao_mes=60)


@pytest.fixture(scope="module")
def populacao() -> Populacao:
    return gerar_populacao(MEDIA)


@pytest.fixture(scope="module")
def ataques(populacao: Populacao) -> tuple[Ataque, ...]:
    return planejar_ataques(MEDIA, populacao)


@pytest.fixture(scope="module")
def do_trimestre() -> tuple[Ataque, ...]:
    return planejar_ataques(TRIMESTRE, gerar_populacao(TRIMESTRE))


def test_o_plano_e_sempre_o_mesmo_em_ordem_de_inicio(populacao: Populacao, ataques: tuple[Ataque, ...]) -> None:
    assert planejar_ataques(MEDIA, populacao) == ataques
    assert [ataque.id for ataque in ataques] == list(range(len(ataques)))
    assert [ataque.inicio for ataque in ataques] == sorted(ataque.inicio for ataque in ataques)


def test_sao_tantos_ataques_quantos_a_taxa_pede(ataques: tuple[Ataque, ...]) -> None:
    esperados = ataques_esperados(MEDIA)

    # Quatro desvios de uma Poisson.
    assert abs(len(ataques) - esperados) < 4 * esperados**0.5


def test_todo_golpe_aparece_e_cada_um_na_hora_dele(ataques: tuple[Ataque, ...]) -> None:
    def horas(golpe: Golpe) -> list[int]:
        return [ataque.inicio.hour for ataque in ataques if ataque.golpe is golpe]

    assert {ataque.golpe for ataque in ataques} == set(Golpe)
    assert sum(9 <= hora < 18 for hora in horas(Golpe.FALSA_CENTRAL)) / len(horas(Golpe.FALSA_CENTRAL)) > 0.9
    assert sum(hora < 5 for hora in horas(Golpe.INVASAO_DE_CONTA)) / len(horas(Golpe.INVASAO_DE_CONTA)) > 0.25


def test_o_falso_parente_e_a_falsa_central_miram_os_mais_velhos(
    populacao: Populacao, ataques: tuple[Ataque, ...]
) -> None:
    def idade_media(golpe: Golpe | None) -> float:
        idades = [
            populacao.clientes[ataque.cliente].idade for ataque in ataques if golpe is None or ataque.golpe is golpe
        ]
        return sum(idades) / len(idades)

    todos = sum(cliente.idade for cliente in populacao.clientes) / len(populacao.clientes)

    assert idade_media(Golpe.FALSO_PARENTE) > todos + 8
    assert idade_media(Golpe.FALSA_CENTRAL) > todos + 5
    assert idade_media(Golpe.VENDA_FALSA) < todos


def test_a_falsa_central_liga_em_dia_de_semana(ataques: tuple[Ataque, ...]) -> None:
    da_central = [ataque for ataque in ataques if ataque.golpe is Golpe.FALSA_CENTRAL]

    assert sum(ataque.inicio.weekday() >= 5 for ataque in da_central) / len(da_central) < 0.15


def test_a_venda_falsa_cresce_na_semana_da_black_friday(do_trimestre: tuple[Ataque, ...]) -> None:
    def por_dia(de: date, ate: date) -> float:
        vendas = [a for a in do_trimestre if a.golpe is Golpe.VENDA_FALSA and de <= a.inicio.date() <= ate]
        return len(vendas) / ((ate - de).days + 1)

    assert por_dia(date(2025, 11, 21), date(2025, 11, 30)) > 1.5 * por_dia(date(2025, 10, 1), date(2025, 10, 31))


def test_o_dinheiro_vai_para_laranjas_ativas_no_dia(populacao: Populacao, ataques: tuple[Ataque, ...]) -> None:
    janelas = {laranja.conta: laranja for laranja in populacao.laranjas}
    ativas = [all(janelas[conta].ativa_em(ataque.inicio.date()) for conta in ataque.laranjas) for ataque in ataques]

    assert all(conta in janelas for ataque in ataques for conta in ataque.laranjas)
    assert sum(ativas) / len(ativas) > 0.9


def test_a_invasao_vem_de_poucos_aparelhos_da_quadrilha(ataques: tuple[Ataque, ...]) -> None:
    invasoes = [ataque for ataque in ataques if ataque.golpe is Golpe.INVASAO_DE_CONTA]
    aparelhos = {ataque.aparelho_do_golpista for ataque in invasoes}

    assert all(ataque.aparelho_do_golpista is None for ataque in ataques if ataque.golpe is not Golpe.INVASAO_DE_CONTA)
    assert all(aparelho is not None and aparelho >= APARELHOS_DO_GOLPISTA for aparelho in aparelhos)
    assert len(aparelhos) < len(invasoes) / 2


def _do_golpe(ataques: tuple[Ataque, ...], golpe: Golpe) -> list[Ataque]:
    return [ataque for ataque in ataques if ataque.golpe is golpe]


def test_os_passos_sao_os_mesmos_para_o_mesmo_ataque_e_em_ordem(ataques: tuple[Ataque, ...]) -> None:
    for ataque in ataques:
        dele = passos(MEDIA, ataque)

        assert passos(MEDIA, ataque) == dele
        assert [passo.instante for passo in dele] == sorted(passo.instante for passo in dele)
        assert dele[0].instante >= segundos(ataque.inicio)
        assert {passo.laranja for passo in dele} <= set(ataque.laranjas)


def test_o_falso_parente_pede_valores_redondos_que_diminuem(ataques: tuple[Ataque, ...]) -> None:
    for ataque in _do_golpe(ataques, Golpe.FALSO_PARENTE):
        valores = [passo.valor for passo in passos(MEDIA, ataque)]

        assert 1 <= len(valores) <= 3
        assert all(valor is not None and valor % 1_000 == 0 for valor in valores)
        assert valores == sorted(valores, key=lambda valor: -(valor or 0))


def test_a_falsa_central_e_a_invasao_levam_partes_do_saldo(ataques: tuple[Ataque, ...]) -> None:
    for golpe in (Golpe.FALSA_CENTRAL, Golpe.INVASAO_DE_CONTA):
        for ataque in _do_golpe(ataques, golpe):
            dele = passos(MEDIA, ataque)

            assert 2 <= len(dele) <= 6
            assert all(passo.valor is None and 0.2 <= (passo.fracao_do_saldo or 0) <= 0.7 for passo in dele)


def test_a_invasao_espera_uns_minutos_depois_de_entrar(ataques: tuple[Ataque, ...]) -> None:
    for ataque in _do_golpe(ataques, Golpe.INVASAO_DE_CONTA):
        espera = passos(MEDIA, ataque)[0].instante - segundos(ataque.inicio)

        assert timedelta(minutes=3) <= timedelta(seconds=espera) <= timedelta(minutes=30)


def test_a_venda_falsa_tem_preco_de_anuncio(ataques: tuple[Ataque, ...]) -> None:
    for ataque in _do_golpe(ataques, Golpe.VENDA_FALSA):
        dele = passos(MEDIA, ataque)

        assert 1 <= len(dele) <= 2
        assert all(passo.valor is not None and passo.valor % 100 == 90 for passo in dele)
        assert all(passo.iniciacao in {Iniciacao.CHAVE, Iniciacao.QR_CODE} for passo in dele)
