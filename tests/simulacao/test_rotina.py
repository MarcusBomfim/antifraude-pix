from datetime import date

import polars as pl

from antifraude_pix.simulacao.configuracao import Configuracao
from antifraude_pix.simulacao.golpes import APARELHOS_DO_GOLPISTA, planejar_ataques
from antifraude_pix.simulacao.pix import segundos
from antifraude_pix.simulacao.populacao import gerar_populacao
from antifraude_pix.simulacao.rotina import (
    APARELHOS_POR_CLIENTE,
    Destinos,
    DiasDaSimulacao,
    Registro,
    para_quem,
    simular_cliente,
)
from antifraude_pix.simulacao.simulador import Simulacao
from tests.apoio import MEDIA


def _legitimos(simulacao: Simulacao) -> pl.DataFrame:
    return simulacao.transacoes.join(simulacao.verdade.fraudes, left_on="id", right_on="transacao", how="anti")


def _conhecidos() -> pl.DataFrame:
    """Os contatos de cada cliente da simulação média: `cliente`, `recebedor`."""
    return pl.DataFrame(
        [(cliente.id, contato) for cliente in gerar_populacao(MEDIA).clientes for contato in cliente.contatos],
        schema={"cliente": pl.Int32, "recebedor": pl.Int32},
        orient="row",
    )


def _depois_do_primeiro_mes(legitimos: pl.DataFrame) -> pl.DataFrame:
    return legitimos.filter(pl.col("momento").dt.date() >= date(2025, 8, 1))


def test_o_movimento_do_dia_segue_o_calendario() -> None:
    dias = DiasDaSimulacao.de(Configuracao(inicio=date(2025, 11, 17), dias=21))

    def movimento(dia: date) -> float:
        return float(dias.movimento[dias.dias.index(dia)])

    assert movimento(date(2025, 11, 19)) == 1.0
    # O feriado de 20 de novembro esvazia; o sábado enche; o domingo esvazia; dezembro é dezembro.
    assert movimento(date(2025, 11, 20)) == 0.8
    assert movimento(date(2025, 11, 22)) == 1.25
    assert movimento(date(2025, 11, 23)) == 0.8
    assert movimento(date(2025, 12, 3)) == 1.15
    assert dias.uteis_do_mes[(2025, 11)][:3] == (date(2025, 11, 3), date(2025, 11, 4), date(2025, 11, 5))


def test_a_rotina_de_um_cliente_sai_igual_simulada_sozinha(simulacao: Simulacao) -> None:
    populacao = gerar_populacao(MEDIA)
    ataques = para_quem(planejar_ataques(MEDIA, populacao), MEDIA.clientes)
    vitima = next(cliente for cliente in populacao.clientes if ataques[cliente.id])
    registro = Registro()

    simular_cliente(
        MEDIA,
        vitima,
        destinos=Destinos.de(populacao),
        dias=DiasDaSimulacao.de(MEDIA),
        ataques=ataques[vitima.id],
        registro=registro,
    )

    dele = simulacao.transacoes.filter(pl.col("cliente") == vitima.id)

    assert dele["valor"].to_list() == registro.valor
    assert dele["recebedor"].to_list() == registro.recebedor
    assert [segundos(momento) for momento in dele["momento"].dt.replace_time_zone(None).to_list()] == registro.instante


def test_o_cliente_legitimo_tambem_troca_de_celular(simulacao: Simulacao) -> None:
    trocas = simulacao.aparelhos_novos.filter(pl.col("aparelho") < APARELHOS_DO_GOLPISTA)

    assert (trocas["aparelho"] == trocas["cliente"].cast(pl.Int64) * APARELHOS_POR_CLIENTE + 1).all()
    assert 0.08 < trocas.height / MEDIA.clientes < 0.16


def test_o_cliente_legitimo_tambem_paga_pessoas_que_nunca_pagou_e_de_noite(simulacao: Simulacao) -> None:
    legitimos = _legitimos(simulacao).with_columns(
        pl.col("momento").dt.hour().alias("hora"),
        (pl.int_range(pl.len()).over("cliente", "recebedor") == 0).alias("primeira_vez"),
    )

    # Depois do primeiro mês, quando o hábito já se formou, ainda há Pix para pessoas que o cliente nunca pagou.
    pessoas = _depois_do_primeiro_mes(legitimos).filter(pl.col("recebedor_pessoa") == "PF")

    assert pessoas["primeira_vez"].sum() / pessoas.height > 0.15
    assert legitimos.filter((pl.col("hora") >= 22) | (pl.col("hora") < 5)).height / legitimos.height > 0.03


def test_o_cliente_legitimo_tambem_faz_pix_grande_para_quem_nao_conhece(simulacao: Simulacao) -> None:
    # O carro usado, a reforma: R$ 3.000 ou mais, para quem não é contato nem o locador.
    grandes = (
        _legitimos(simulacao)
        .filter(pl.col("valor") >= 300_000)
        .join(_conhecidos(), on=["cliente", "recebedor"], how="anti")
    )

    assert grandes.height > 100


def test_o_cliente_legitimo_tambem_paga_o_autonomo_que_nunca_pagou_como_no_falso_parente(simulacao: Simulacao) -> None:
    # A diarista, o eletricista: uma pessoa nova, R$ 150 ou mais em valor redondo — o que o falso parente pede.
    depois = _depois_do_primeiro_mes(_legitimos(simulacao)).with_columns(
        (pl.int_range(pl.len()).over("cliente", "recebedor") == 0).alias("primeira_vez")
    )
    redondos_a_quem_nunca_pagou = depois.filter(
        pl.col("primeira_vez")
        & (pl.col("recebedor_pessoa") == "PF")
        & (pl.col("valor") >= 15_000)
        & (pl.col("valor") % 1_000 == 0)
    )
    de_dia = redondos_a_quem_nunca_pagou.filter(pl.col("momento").dt.hour().is_between(9, 17))

    # Sem o autônomo, seriam 2,5% dos Pix; com ele a qualquer hora, 3,9%. E o de dia seria pouco mais da metade.
    assert de_dia.height / depois.height > 0.045
    assert de_dia.height / redondos_a_quem_nunca_pagou.height > 0.65


def test_o_cliente_legitimo_tambem_ajuda_a_familia_com_valor_alto_e_redondo(simulacao: Simulacao) -> None:
    # O aluguel do filho, o remédio da mãe: R$ 500 ou mais, redondo, para um contato.
    depois = _depois_do_primeiro_mes(_legitimos(simulacao))
    ajudas = depois.join(_conhecidos(), on=["cliente", "recebedor"], how="semi").filter(
        (pl.col("valor") >= 50_000) & (pl.col("valor") % 1_000 == 0)
    )

    # O aluguel que o cliente paga ao locador conhecido também entra aqui: sem a ajuda da família, seriam 1,8%.
    assert ajudas.height / depois.height > 0.023


def test_quem_leva_parte_do_saldo_nao_leva_menos_de_50_reais(simulacao: Simulacao) -> None:
    golpes = simulacao.transacoes.join(simulacao.verdade.fraudes, left_on="id", right_on="transacao")
    de_saldo = golpes.filter(pl.col("golpe").is_in(["falsa_central", "invasao_de_conta"]))

    assert de_saldo.height > 0
    assert (de_saldo["valor"] >= 5_000).all()


def test_a_invasao_sai_do_aparelho_da_quadrilha_e_os_outros_golpes_do_da_vitima(simulacao: Simulacao) -> None:
    golpes = simulacao.transacoes.join(simulacao.verdade.fraudes, left_on="id", right_on="transacao")
    da_quadrilha = pl.col("aparelho") >= APARELHOS_DO_GOLPISTA

    assert golpes.filter(pl.col("golpe") == "invasao_de_conta").select(da_quadrilha.all()).item()
    assert not golpes.filter(pl.col("golpe") != "invasao_de_conta").select(da_quadrilha.any()).item()


def test_o_golpe_de_valor_pedido_sai_inteiro(simulacao: Simulacao) -> None:
    golpes = simulacao.transacoes.join(simulacao.verdade.fraudes, left_on="id", right_on="transacao")

    assert (golpes.filter(pl.col("golpe") == "falso_parente")["valor"] % 1_000 == 0).all()
    assert (golpes.filter(pl.col("golpe") == "venda_falsa")["valor"] % 100 == 90).all()
