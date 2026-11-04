import ast
from datetime import date, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest
from hypothesis import HealthCheck, given
from hypothesis import settings as configurar
from hypothesis import strategies as st

import antifraude_pix.variaveis
from antifraude_pix.dados.banco import VisaoDoBanco
from antifraude_pix.dados.esquema import APARELHOS_NOVOS, CLIENTES, CONTESTACOES, FUSO, TRANSACOES
from antifraude_pix.simulacao.golpes import APARELHOS_DO_GOLPISTA
from antifraude_pix.simulacao.simulador import Simulacao
from antifraude_pix.variaveis.calculo import calcular
from antifraude_pix.variaveis.catalogo import NOMES
from tests.apoio import MEDIA, simulacao_media

GOLPISTA = APARELHOS_DO_GOLPISTA


def _tabela(linhas: list[dict[str, object]], esquema: pl.Schema, *instantes: str) -> pl.DataFrame:
    """Uma tabela do contrato, escrita à mão: os instantes como texto, no horário de Brasília."""
    tabela = pl.DataFrame(
        linhas, schema={nome: pl.String if nome in instantes else tipo for nome, tipo in esquema.items()}
    )

    return tabela.with_columns(
        pl.col(nome).str.to_datetime("%Y-%m-%d %H:%M").dt.replace_time_zone(FUSO).cast(esquema[nome])
        for nome in instantes
    )


# Uma linha da tabela de Pix, com os campos na ordem das colunas: lida como a tabela, e não como uma chamada.
def _pix(  # noqa: PLR0917
    id_: int, momento: str, cliente: int, aparelho: int, recebedor: int, valor: int, saldo: int, **outros: object
) -> dict[str, object]:
    return {
        "id": id_,
        "momento": momento,
        "cliente": cliente,
        "aparelho": aparelho,
        "recebedor": recebedor,
        "recebedor_banco": outros.get("banco", "341"),
        "recebedor_pessoa": outros.get("pessoa", "PF"),
        "chave": outros.get("chave", "cpf"),
        "iniciacao": "chave",
        "valor": valor,
        "saldo_antes": saldo,
    }


@pytest.fixture(scope="module")
def exemplo() -> pl.DataFrame:
    """Dois clientes, seis Pix e uma quadrilha, à mão.

    A dona Maria (cliente 0) paga a vizinha às 10h — no mesmo instante em que o
    seu João (cliente 1) a paga também — e de novo às 10h30. No dia seguinte,
    paga a loja de móveis. Às 20h55, a quadrilha entra na conta dela por um
    aparelho novo e, às 21h, manda R$ 500 para uma laranja. Às 21h01 entra na
    conta do seu João pelo mesmo aparelho; a dona Maria contesta às 21h03; e às
    21h05 sai o Pix do seu João para a mesma laranja.
    """
    clientes = _tabela(
        [
            {
                "id": 0,
                "idade": 70,
                "uf": "SP",
                "renda_mensal": 300_000,
                "conta_aberta_em": None,
                "limite_noturno": 100_000,
            },
            {
                "id": 1,
                "idade": 40,
                "uf": "RJ",
                "renda_mensal": 500_000,
                "conta_aberta_em": None,
                "limite_noturno": 100_000,
            },
        ],
        CLIENTES,
    ).with_columns(pl.date(2020, 1, 1).alias("conta_aberta_em"))
    transacoes = _tabela(
        [
            _pix(0, "2025-07-01 10:00", 0, 0, 100, 5_000, 100_000),
            _pix(1, "2025-07-01 10:00", 1, 16, 100, 3_000, 50_000),
            _pix(2, "2025-07-01 10:30", 0, 0, 100, 7_000, 95_000),
            _pix(3, "2025-07-02 09:00", 0, 0, 200, 100_000, 200_000, pessoa="PJ", chave="cnpj"),
            _pix(4, "2025-07-02 21:00", 0, GOLPISTA, 300, 50_000, 100_000, banco="260", chave="aleatoria"),
            _pix(5, "2025-07-02 21:05", 1, GOLPISTA, 300, 40_000, 47_000, banco="260", chave="aleatoria"),
        ],
        TRANSACOES,
        "momento",
    )
    aparelhos = _tabela(
        [
            {"cliente": 0, "momento": "2025-07-02 20:55", "aparelho": GOLPISTA},
            {"cliente": 1, "momento": "2025-07-02 21:01", "aparelho": GOLPISTA},
        ],
        APARELHOS_NOVOS,
        "momento",
    )
    contestacoes = _tabela([{"transacao": 4, "contestada_em": "2025-07-02 21:03"}], CONTESTACOES, "contestada_em")

    return calcular(VisaoDoBanco(clientes, transacoes, aparelhos, contestacoes))


def _linha(exemplo: pl.DataFrame, id_: int) -> dict[str, object]:
    return exemplo.filter(pl.col("id") == id_).row(0, named=True)


def test_uma_linha_por_pix_na_ordem_com_as_variaveis_do_catalogo(exemplo: pl.DataFrame) -> None:
    assert exemplo["id"].to_list() == list(range(6))
    assert exemplo.columns == ["id", *NOMES]


def test_o_primeiro_pix_nao_tem_passado_e_o_do_mesmo_instante_nao_conta(exemplo: pl.DataFrame) -> None:
    for id_ in (0, 1):
        linha = _linha(exemplo, id_)

        assert (linha["pix_nas_ultimas_24h"], linha["minutos_desde_o_ultimo_pix"], linha["mediana_de_90_dias"]) == (
            0,
            None,
            None,
        )
        assert (linha["recebedor_novo"], linha["vezes_que_pagou_o_recebedor"]) == (True, 0)
        # O Pix do outro cliente, no mesmo segundo, ainda não aconteceu para este.
        assert (linha["pix_recebidos_em_24h"], linha["pagadores_novos_em_24h"]) == (0, 0)


def test_o_habito_e_a_relacao_contam_so_o_que_veio_antes(exemplo: pl.DataFrame) -> None:
    segundo = _linha(exemplo, 2)

    assert (segundo["pix_na_ultima_hora"], segundo["pix_nas_ultimas_24h"], segundo["valor_nas_ultimas_24h"]) == (
        1,
        1,
        50.0,
    )
    assert (segundo["minutos_desde_o_ultimo_pix"], segundo["mediana_de_90_dias"], segundo["valor_sobre_a_mediana"]) == (
        30.0,
        50.0,
        1.4,
    )
    assert (segundo["vezes_que_pagou_o_recebedor"], segundo["recebedor_novo"]) == (1, False)
    assert segundo["dias_desde_o_primeiro_pix_ao_recebedor"] == pytest.approx(30 / 1440)
    # A vizinha recebeu dois Pix antes, de dois clientes que a pagavam pela primeira vez.
    assert (segundo["pix_recebidos_em_24h"], segundo["pagadores_novos_em_24h"]) == (2, 2)

    loja = _linha(exemplo, 3)

    assert (loja["pix_nas_ultimas_24h"], loja["valor_nas_ultimas_24h"], loja["minutos_desde_o_ultimo_pix"]) == (
        2,
        120.0,
        1350.0,
    )
    assert (loja["mediana_de_90_dias"], loja["valor_sobre_o_maior_de_90_dias"]) == (
        60.0,
        pytest.approx(100_000 / 7_000),
    )
    assert (loja["recebedor_novo"], loja["recebedor_pj"], loja["valor_redondo"], loja["chave"]) == (
        True,
        True,
        True,
        "cnpj",
    )


def test_a_quadrilha_aparece_no_aparelho_no_recebedor_e_na_contestacao(exemplo: pl.DataFrame) -> None:
    da_maria = _linha(exemplo, 4)

    assert (da_maria["noturno"], da_maria["parte_do_limite_noturno"], da_maria["parte_do_saldo"]) == (True, 0.5, 0.5)
    assert (da_maria["minutos_desde_o_aparelho_novo"], da_maria["outros_clientes_no_aparelho"]) == (5.0, 0)
    assert (da_maria["recebedor_em_banco_digital"], da_maria["contestacoes_do_recebedor"]) == (True, 0)

    do_joao = _linha(exemplo, 5)

    # O aparelho já tinha entrado na conta da dona Maria; a laranja já tinha um pagador novo e uma contestação.
    assert (do_joao["minutos_desde_o_aparelho_novo"], do_joao["outros_clientes_no_aparelho"]) == (4.0, 1)
    assert (
        do_joao["pagadores_novos_em_24h"],
        do_joao["pix_recebidos_em_24h"],
        do_joao["contestacoes_do_recebedor"],
    ) == (1, 1, 1)


def test_o_aparelho_de_sempre_nao_tem_entrada(exemplo: pl.DataFrame) -> None:
    for id_ in (0, 1, 2, 3):
        linha = _linha(exemplo, id_)
        assert (linha["minutos_desde_o_aparelho_novo"], linha["outros_clientes_no_aparelho"]) == (None, 0)


@pytest.fixture(scope="module")
def bordas() -> pl.DataFrame:
    """As bordas de cada conta, à mão: o mesmo instante, a ponta de cada janela, o domingo e a madrugada.

    O seu Antônio (cliente 0), de conta aberta em 1º de junho, paga o filho
    duas vezes no mesmo segundo, num domingo às 3h, e de novo na segunda às
    10h. Às 10h45, paga a padaria. Na quinta, a dona Carla (cliente 1) troca de
    celular e, no mesmo instante, paga a padaria pelo celular novo, no
    instante em que chega a contestação do Pix do seu Antônio para ela. No dia
    20, a quadrilha entra na conta da dona Carla às 2h e, às 2h10, na do seu
    Antônio, no mesmo instante em que sai o Pix da conta dela. Em setembro, o
    seu Antônio manda ao filho o dobro do maior Pix dos últimos 90 dias.
    """
    clientes = _tabela(
        [
            {
                "id": 0,
                "idade": 60,
                "uf": "BA",
                "renda_mensal": 400_000,
                "conta_aberta_em": None,
                "limite_noturno": 100_000,
            },
            {
                "id": 1,
                "idade": 35,
                "uf": "MG",
                "renda_mensal": 300_000,
                "conta_aberta_em": None,
                "limite_noturno": 100_000,
            },
        ],
        CLIENTES,
    ).with_columns(pl.Series("conta_aberta_em", [date(2025, 6, 1), date(2020, 1, 1)]))
    transacoes = _tabela(
        [
            _pix(0, "2025-07-06 03:00", 0, 0, 100, 5_500, 100_000),
            _pix(1, "2025-07-06 03:00", 0, 0, 100, 2_000, 94_500),
            _pix(2, "2025-07-07 10:00", 0, 0, 100, 10_000, 92_500),
            _pix(3, "2025-07-07 10:45", 0, 0, 101, 3_000, 82_500, pessoa="PJ", chave="cnpj"),
            _pix(4, "2025-07-10 12:00", 1, 17, 101, 4_000, 50_000, pessoa="PJ", chave="cnpj"),
            _pix(5, "2025-07-20 02:10", 1, GOLPISTA, 300, 30_000, 46_000, banco="260"),
            _pix(6, "2025-09-04 10:00", 0, 0, 100, 20_000, 300_000),
        ],
        TRANSACOES,
        "momento",
    )
    aparelhos = _tabela(
        [
            {"cliente": 1, "momento": "2025-07-10 12:00", "aparelho": 17},
            {"cliente": 1, "momento": "2025-07-20 02:00", "aparelho": GOLPISTA},
            {"cliente": 0, "momento": "2025-07-20 02:10", "aparelho": GOLPISTA},
        ],
        APARELHOS_NOVOS,
        "momento",
    )
    contestacoes = _tabela([{"transacao": 3, "contestada_em": "2025-07-10 12:00"}], CONTESTACOES, "contestada_em")

    return calcular(VisaoDoBanco(clientes, transacoes, aparelhos, contestacoes))


def test_dois_pix_para_o_mesmo_recebedor_no_mesmo_segundo_sao_os_dois_a_primeira_vez(bordas: pl.DataFrame) -> None:
    for id_ in (0, 1):
        linha = _linha(bordas, id_)

        assert (linha["vezes_que_pagou_o_recebedor"], linha["recebedor_novo"]) == (0, True)
        assert linha["dias_desde_o_primeiro_pix_ao_recebedor"] is None

    terceiro = _linha(bordas, 2)

    assert terceiro["vezes_que_pagou_o_recebedor"] == 2
    assert terceiro["dias_desde_o_primeiro_pix_ao_recebedor"] == pytest.approx(31 / 24)


def test_o_pix_da_madrugada_de_domingo(bordas: pl.DataFrame) -> None:
    domingo, segunda = _linha(bordas, 0), _linha(bordas, 2)

    assert (domingo["noturno"], domingo["parte_do_limite_noturno"], domingo["dia_da_semana"]) == (
        True,
        pytest.approx(0.055),
        6,
    )
    assert (segunda["noturno"], segunda["parte_do_limite_noturno"], segunda["dia_da_semana"]) == (False, None, 0)
    # Do dia 1º de junho ao dia 6 de julho: 35 dias. E R$ 55 não é redondo; R$ 100 é.
    assert domingo["dias_de_conta"] == 35
    assert (domingo["valor_redondo"], segunda["valor_redondo"]) == (False, True)


def test_cada_janela_tem_o_tamanho_do_nome(bordas: pl.DataFrame) -> None:
    # O Pix das 10h está a 45 minutos do das 10h45: entra na última hora.
    assert _linha(bordas, 3)["pix_na_ultima_hora"] == 1

    # A padaria teve um pagador novo três dias antes: entra nos 7 dias, e não nas 24 horas.
    padaria = _linha(bordas, 4)

    assert (padaria["pagadores_novos_em_7_dias"], padaria["pagadores_novos_em_24h"]) == (1, 0)

    # O maior Pix dos 90 dias foi há 59 dias: R$ 100, metade deste. A mediana é a dos quatro Pix de julho.
    setembro = _linha(bordas, 6)

    assert (setembro["valor_sobre_o_maior_de_90_dias"], setembro["mediana_de_90_dias"]) == (2.0, 42.5)


def test_o_que_acontece_no_mesmo_instante_do_pix(bordas: pl.DataFrame) -> None:
    padaria, da_carla = _linha(bordas, 4), _linha(bordas, 5)

    # A contestação que chega no instante do Pix ainda não tinha chegado.
    assert padaria["contestacoes_do_recebedor"] == 0
    # O celular que entra no instante do Pix é o da mesma sessão: entrou há zero minuto, e em nenhuma outra conta.
    assert (padaria["minutos_desde_o_aparelho_novo"], padaria["outros_clientes_no_aparelho"]) == (0.0, 0)
    # A quadrilha entra na conta do seu Antônio no instante do Pix da dona Carla: para ela, ainda não entrou.
    assert (da_carla["minutos_desde_o_aparelho_novo"], da_carla["outros_clientes_no_aparelho"]) == (10.0, 0)


# O vazamento: as variáveis de um Pix não podem mudar com o que aconteceu depois dele.


@pytest.fixture(scope="module")
def visao() -> VisaoDoBanco:
    return VisaoDoBanco.da(simulacao_media())


@pytest.fixture(scope="module")
def todas(visao: VisaoDoBanco) -> pl.DataFrame:
    return calcular(visao)


def _no_relogio_local(momento: object) -> datetime:
    assert isinstance(momento, datetime)

    return momento.replace(tzinfo=None)


def _confere_o_corte(visao: VisaoDoBanco, todas: pl.DataFrame, corte: datetime) -> None:
    antes = calcular(visao.ate(corte))
    esperados = visao.transacoes.filter(pl.col("momento").dt.replace_time_zone(None) < corte)["id"]

    assert antes["id"].equals(esperados)
    assert antes.equals(todas.filter(pl.col("id").is_in(esperados.implode())))


def _cortes_delicados(visao: VisaoDoBanco) -> list[datetime]:
    """Os cortes em que um vazamento apareceria.

    No instante exato de um Pix, logo depois de uma contestação, e no instante e logo depois de um aparelho novo.
    """
    pix = _no_relogio_local(visao.transacoes["momento"][visao.transacoes.height // 2])
    contestacao = _no_relogio_local(visao.contestacoes["contestada_em"][visao.contestacoes.height // 3])
    aparelho = _no_relogio_local(visao.aparelhos_novos["momento"][visao.aparelhos_novos.height // 2])

    return [pix, contestacao + timedelta(seconds=1), aparelho, aparelho + timedelta(minutes=10)]


def test_as_variaveis_de_antes_do_corte_nao_mudam_com_o_futuro_nos_cortes_delicados(
    visao: VisaoDoBanco, todas: pl.DataFrame
) -> None:
    for corte in _cortes_delicados(visao):
        _confere_o_corte(visao, todas, corte)


@configurar(max_examples=6, suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow])
@given(minutos=st.integers(min_value=60, max_value=MEDIA.dias * 24 * 60))
def test_as_variaveis_de_antes_do_corte_nao_mudam_com_o_futuro_em_qualquer_corte(
    visao: VisaoDoBanco, todas: pl.DataFrame, minutos: int
) -> None:
    _confere_o_corte(visao, todas, datetime.combine(MEDIA.inicio, datetime.min.time()) + timedelta(minutes=minutos))


def test_o_calculo_nao_le_a_verdade() -> None:
    """O pacote das variáveis não importa o simulador nem a gravação, e não toca em nada chamado verdade."""
    pasta = Path(antifraude_pix.variaveis.__file__).parent
    proibidos = {"antifraude_pix.simulacao.simulador", "antifraude_pix.dados.gravacao"}

    for arquivo in pasta.glob("*.py"):
        arvore = ast.parse(arquivo.read_text(encoding="utf-8"))
        importados = {no.module for no in ast.walk(arvore) if isinstance(no, ast.ImportFrom)}
        nomes = {no.attr for no in ast.walk(arvore) if isinstance(no, ast.Attribute)} | {
            no.id for no in ast.walk(arvore) if isinstance(no, ast.Name)
        }

        assert importados.isdisjoint(proibidos), arquivo.name
        assert not any("verdade" in nome.lower() for nome in nomes), arquivo.name


# As variáveis contra a verdade do simulador, que só o teste lê.


@pytest.fixture(scope="module")
def com_a_verdade(todas: pl.DataFrame) -> pl.DataFrame:
    fraudes = simulacao_media().verdade.fraudes.select(
        pl.col("transacao").alias("id"), "ataque", pl.col("golpe").cast(pl.String)
    )

    return todas.join(fraudes, on="id", how="left").with_columns(pl.col("golpe").fill_null("legitimo"))


def test_a_invasao_vem_de_um_aparelho_que_acabou_de_entrar_e_o_legitimo_quase_nunca(
    com_a_verdade: pl.DataFrame,
) -> None:
    def recente(golpe: str) -> float:
        dele = com_a_verdade.filter(pl.col("golpe") == golpe)
        # Até meia hora de espera depois de entrar, e até seis Pix com até oito minutos
        # entre eles: menos de hora e meia.
        return float(dele.select((pl.col("minutos_desde_o_aparelho_novo") <= 90).fill_null(value=False).mean()).item())

    assert recente("invasao_de_conta") == 1.0
    assert recente("legitimo") < 0.01


def test_o_aparelho_em_outras_contas_e_so_da_quadrilha(com_a_verdade: pl.DataFrame) -> None:
    em_outras = com_a_verdade.filter(pl.col("outros_clientes_no_aparelho") > 0)

    assert em_outras.height > 0
    assert set(em_outras["golpe"].unique().to_list()) == {"invasao_de_conta"}


def test_o_primeiro_pix_de_cada_golpe_vai_para_quem_o_cliente_nunca_pagou(
    com_a_verdade: pl.DataFrame, simulacao: Simulacao
) -> None:
    golpes = com_a_verdade.filter(pl.col("golpe") != "legitimo")
    primeiros = golpes.sort("id").group_by("ataque", maintain_order=True).first()
    # Quem cai uma vez cai de novo, às vezes pela mesma laranja: só aí ela não é nova no primeiro Pix do golpe.
    revitimas = simulacao.verdade.ataques.filter(pl.int_range(pl.len()).over("cliente") > 0)["id"]

    # Os Pix seguintes do mesmo ataque vão, às vezes, para a mesma laranja: ela já não é nova para a vítima.
    assert primeiros.filter(~pl.col("ataque").is_in(revitimas.implode()))["recebedor_novo"].all()
    assert com_a_verdade.filter(pl.col("golpe") == "venda_falsa")["termina_em_90_centavos"].all()
    assert golpes.height == simulacao.verdade.fraudes.height
