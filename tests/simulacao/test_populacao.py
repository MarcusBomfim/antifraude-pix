from datetime import timedelta

import pytest

from antifraude_pix.simulacao.configuracao import Configuracao
from antifraude_pix.simulacao.populacao import (
    BANCOS,
    GOLPES_POR_LARANJA,
    SALARIO_MINIMO,
    Populacao,
    TipoDeChave,
    TipoDePessoa,
    ataques_esperados,
    gerar_populacao,
)

PEQUENA = Configuracao(clientes=400, dias=60)


@pytest.fixture(scope="module")
def populacao() -> Populacao:
    return gerar_populacao(PEQUENA)


@pytest.fixture(scope="module")
def grande() -> Populacao:
    # Golpes a mais, para haver laranjas bastantes para contar.
    return gerar_populacao(Configuracao(clientes=5_000, dias=90, golpes_por_mil_clientes_ao_mes=60))


def test_a_mesma_semente_da_a_mesma_populacao_e_outra_semente_da_outra(populacao: Populacao) -> None:
    assert gerar_populacao(PEQUENA) == populacao
    assert gerar_populacao(Configuracao(clientes=400, dias=60, semente=7)) != populacao


def test_o_id_de_cada_conta_e_a_posicao_dela(populacao: Populacao) -> None:
    assert [conta.id for conta in populacao.contas] == list(range(len(populacao.contas)))
    assert [cliente.id for cliente in populacao.clientes] == list(range(PEQUENA.clientes))


def test_o_cliente_e_um_adulto_com_renda_conta_antiga_e_limite_noturno(populacao: Populacao) -> None:
    for cliente in populacao.clientes:
        assert 18 <= cliente.idade <= 90
        assert cliente.renda_mensal >= SALARIO_MINIMO
        assert cliente.conta_aberta_em <= PEQUENA.inicio - timedelta(days=30)
        assert cliente.limite_noturno > 0
        assert cliente.saldo_inicial >= 0
        assert cliente.pix_por_mes > 0


def test_o_cliente_paga_pessoas_e_comercios_que_existem_sem_repetir(populacao: Populacao) -> None:
    pessoas = set(populacao.pessoas)
    comercios = set(populacao.todos_os_comercios)

    for cliente in populacao.clientes:
        assert 3 <= len(cliente.contatos) <= 12
        assert len(set(cliente.contatos)) == len(cliente.contatos)
        assert set(cliente.contatos) <= pessoas
        assert 4 <= len(cliente.comercios) <= 15
        assert len(set(cliente.comercios)) == len(cliente.comercios)
        assert set(cliente.comercios) <= comercios


def test_o_aluguel_vai_para_um_contato_e_cabe_na_renda(populacao: Populacao) -> None:
    com_aluguel = [cliente for cliente in populacao.clientes if cliente.aluguel is not None]

    assert 0.25 < len(com_aluguel) / PEQUENA.clientes < 0.45

    for cliente in com_aluguel:
        assert cliente.aluguel is not None
        assert cliente.aluguel.locador in cliente.contatos
        assert 0 < cliente.aluguel.valor <= 0.36 * cliente.renda_mensal


def test_cada_tipo_de_conta_recebe_pela_chave_que_lhe_cabe(populacao: Populacao) -> None:
    for id_ in populacao.pessoas:
        conta = populacao.contas[id_]
        assert (conta.pessoa, conta.chave != TipoDeChave.CNPJ) == (TipoDePessoa.FISICA, True)

    for id_ in populacao.todos_os_comercios:
        conta = populacao.contas[id_]
        assert (conta.pessoa, conta.chave in {TipoDeChave.CNPJ, TipoDeChave.ALEATORIA}) == (TipoDePessoa.JURIDICA, True)


def test_a_laranja_fica_ativa_de_dois_a_vinte_dias_dentro_da_simulacao(populacao: Populacao) -> None:
    # Uma laranja para cada seis golpes esperados, e pelo menos cinco.
    assert len(populacao.laranjas) == max(5, round(ataques_esperados(PEQUENA) / GOLPES_POR_LARANJA))

    for laranja in populacao.laranjas:
        assert laranja.conta not in populacao.pessoas
        assert laranja.conta not in populacao.todos_os_comercios
        assert PEQUENA.inicio <= laranja.ativa_de < PEQUENA.fim
        assert 2 <= (laranja.ativa_ate - laranja.ativa_de).days <= 20
        assert laranja.ativa_em(laranja.ativa_de)
        assert not laranja.ativa_em(laranja.ativa_ate)


def test_uma_laranja_em_cinco_e_empresa_de_fachada(grande: Populacao) -> None:
    empresas = [laranja for laranja in grande.laranjas if grande.contas[laranja.conta].pessoa is TipoDePessoa.JURIDICA]

    assert len(empresas) == round(len(grande.laranjas) / 5)
    assert all(grande.contas[laranja.conta].chave in {TipoDeChave.CNPJ, TipoDeChave.ALEATORIA} for laranja in empresas)


def test_a_laranja_nasce_mais_nos_bancos_digitais(grande: Populacao) -> None:
    digitais = {banco.codigo for banco, _ in BANCOS if banco.digital}

    def parte_digital(contas: list[int]) -> float:
        return sum(grande.contas[conta].banco in digitais for conta in contas) / len(contas)

    laranjas = [laranja.conta for laranja in grande.laranjas]

    assert len(laranjas) == 150
    assert parte_digital(laranjas) > parte_digital(list(grande.pessoas)) + 0.2


def test_as_idades_e_os_estados_seguem_as_proporcoes_da_populacao(grande: Populacao) -> None:
    idosos = sum(cliente.idade >= 65 for cliente in grande.clientes) / len(grande.clientes)
    paulistas = sum(cliente.uf == "SP" for cliente in grande.clientes) / len(grande.clientes)

    assert 0.15 < idosos < 0.21
    assert 0.19 < paulistas < 0.25


def test_os_jovens_fazem_mais_pix_que_os_idosos(grande: Populacao) -> None:
    def media(de: int, ate: int) -> float:
        frequencias = [cliente.pix_por_mes for cliente in grande.clientes if de <= cliente.idade <= ate]
        return sum(frequencias) / len(frequencias)

    assert media(18, 34) > 2 * media(65, 90)
