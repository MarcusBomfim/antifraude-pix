from dataclasses import replace
from datetime import date, datetime, timedelta
from functools import cache
from zoneinfo import ZoneInfo

import polars as pl
import pytest

from antifraude_pix.dados.banco import VisaoDoBanco
from antifraude_pix.dados.esquema import APARELHOS_NOVOS, CLIENTES, CONTESTACOES, TRANSACOES
from antifraude_pix.simulacao.configuracao import Configuracao
from antifraude_pix.simulacao.simulador import simular
from antifraude_pix.tempo_real.estado import Estado, PixNovo
from antifraude_pix.variaveis.calculo import calcular
from tests.apoio import simulacao_media

FUSO = ZoneInfo("America/Sao_Paulo")

LONGA = Configuracao(clientes=200, dias=200, golpes_por_mil_clientes_ao_mes=60)
"""Poucos clientes por muitos dias: cada um passa de 64 Pix fora da janela de 90 dias, e o estado poda a história."""


@cache
def _longa() -> VisaoDoBanco:
    return VisaoDoBanco.da(simular(LONGA))


def _do_lote(visao: VisaoDoBanco) -> pl.DataFrame:
    """As variáveis do cálculo em lote, com as categorias em texto, como o estado as entrega."""
    return calcular(visao).with_columns(pl.col("iniciacao", "chave").cast(pl.String))


@pytest.mark.parametrize("visao", [VisaoDoBanco.da(simulacao_media()), _longa()], ids=["media", "longa"])
def test_o_estado_calcula_as_mesmas_variaveis_que_o_treino_em_todos_os_pix(visao: VisaoDoBanco) -> None:
    em_tempo_real = Estado(visao.clientes).reproduzir(visao, calcular=True)
    em_lote = _do_lote(visao)

    assert em_tempo_real.height == visao.transacoes.height
    # Igual, e não parecido: a mesma conta, o mesmo número, até o último bit de cada variável.
    assert em_tempo_real.equals(em_lote)


def test_comecar_do_meio_da_historia_da_as_mesmas_variaveis() -> None:
    # O serviço sobe com a história até ontem e segue Pix a Pix: o resultado é o mesmo de contar tudo de uma vez.
    visao = VisaoDoBanco.da(simulacao_media())
    corte = datetime(2025, 9, 1, tzinfo=FUSO)
    estado = Estado.da_historia(visao.ate(date(2025, 9, 1)))
    depois = VisaoDoBanco(
        clientes=visao.clientes,
        transacoes=visao.transacoes.filter(pl.col("momento") >= corte),
        aparelhos_novos=visao.aparelhos_novos.filter(pl.col("momento") >= corte),
        contestacoes=visao.contestacoes.filter(pl.col("contestada_em") >= corte),
    )
    em_tempo_real = estado.reproduzir(depois, calcular=True)
    em_lote = _do_lote(visao).join(em_tempo_real.select("id"), on="id", how="semi")

    assert em_tempo_real.height == depois.transacoes.height > 0
    assert em_tempo_real.equals(em_lote)


def _momento(dia: int, hora: int, minuto: int = 0, segundo: int = 0) -> datetime:
    return datetime(2025, 7, dia, hora, minuto, segundo, tzinfo=FUSO)


def _visao_das_bordas() -> VisaoDoBanco:
    """Um cliente, um recebedor e as bordas de cada conta, no mesmo instante ou na ponta da janela."""
    clientes = pl.DataFrame(
        [{"id": 0, "idade": 70, "uf": "SP", "renda_mensal": 300_000, "conta_aberta_em": date(2020, 1, 1),
          "limite_noturno": 100_000},
         {"id": 1, "idade": 30, "uf": "RJ", "renda_mensal": 500_000, "conta_aberta_em": date(2024, 6, 1),
          "limite_noturno": 100_000}],
        schema=CLIENTES,
    )  # fmt: skip
    base = {"recebedor_banco": "260", "recebedor_pessoa": "PF", "chave": "cpf", "iniciacao": "chave"}
    pix = [
        # Dois Pix no mesmo segundo, para o mesmo recebedor: um não conta na conta do outro.
        {"id": 0, "momento": _momento(1, 10), "cliente": 0, "aparelho": 1, "recebedor": 9, "valor": 10_000},
        {"id": 1, "momento": _momento(1, 10), "cliente": 0, "aparelho": 1, "recebedor": 9, "valor": 20_000},
        # Exatamente uma hora e um dia depois: a ponta da janela entra.
        {"id": 2, "momento": _momento(1, 11), "cliente": 0, "aparelho": 1, "recebedor": 9, "valor": 30_000},
        {"id": 3, "momento": _momento(2, 10), "cliente": 0, "aparelho": 2, "recebedor": 9, "valor": 40_000},
        # O aparelho 2 entra na conta do cliente 1 no instante do Pix; a contestação chega no mesmo instante.
        {"id": 4, "momento": _momento(3, 22), "cliente": 1, "aparelho": 2, "recebedor": 9, "valor": 50_000},
        {"id": 5, "momento": _momento(3, 22, 0, 1), "cliente": 1, "aparelho": 2, "recebedor": 9, "valor": 60_000},
    ]
    transacoes = pl.DataFrame([linha | base | {"saldo_antes": 1_000_000} for linha in pix], schema=TRANSACOES)
    aparelhos = pl.DataFrame(
        [
            {"cliente": 0, "momento": _momento(2, 9, 30), "aparelho": 2},
            {"cliente": 1, "momento": _momento(3, 22), "aparelho": 2},
        ],
        schema=APARELHOS_NOVOS,
    )
    contestacoes = pl.DataFrame([{"transacao": 0, "contestada_em": _momento(3, 22)}], schema=CONTESTACOES)

    return VisaoDoBanco(clientes, transacoes, aparelhos, contestacoes)


def test_as_bordas_de_cada_conta_sao_as_do_lote() -> None:
    visao = _visao_das_bordas()
    em_tempo_real = Estado(visao.clientes).reproduzir(visao, calcular=True)
    linhas = em_tempo_real.rows(named=True)

    assert em_tempo_real.equals(_do_lote(visao))
    # O mesmo segundo: nenhum dos dois vê o outro, e os dois são o primeiro Pix para o recebedor.
    assert [linhas[i]["vezes_que_pagou_o_recebedor"] for i in range(3)] == [0, 0, 2]
    # A ponta esquerda da janela entra: os Pix de exatamente uma hora, ou um dia, antes contam.
    assert linhas[2]["pix_na_ultima_hora"] == 2
    assert linhas[3]["pix_nas_ultimas_24h"] == 3
    # Os dois Pix do mesmo segundo foram, cada um, de um pagador novo para o recebedor.
    assert linhas[2]["pagadores_novos_em_24h"] == 2
    # O aparelho que entra no instante do Pix vale para ele; a contestação do mesmo instante, não.
    assert linhas[4]["minutos_desde_o_aparelho_novo"] == 0.0
    assert linhas[4]["outros_clientes_no_aparelho"] == 1
    assert linhas[4]["contestacoes_do_recebedor"] == 0
    assert linhas[5]["contestacoes_do_recebedor"] == 1


def _pix(id_: int, momento: datetime, cliente: int = 0) -> PixNovo:
    return PixNovo(id_, momento, cliente, 1, 9, "260", "PF", "cpf", "chave", 10_000, 100_000)


def test_o_evento_de_antes_do_ultimo_e_recusado() -> None:
    estado = Estado(_visao_das_bordas().clientes)
    estado.registrar(_pix(0, _momento(5, 10)))

    with pytest.raises(ValueError, match="na ordem do tempo"):
        estado.variaveis(_pix(1, _momento(5, 9)))


def test_o_cliente_que_nao_e_do_banco_e_recusado() -> None:
    with pytest.raises(KeyError, match="não é do banco"):
        Estado(_visao_das_bordas().clientes).variaveis(_pix(0, _momento(5, 10), cliente=42))


def test_a_contestacao_de_um_pix_que_o_estado_nao_tem_e_recusada() -> None:
    estado = Estado(_visao_das_bordas().clientes)

    with pytest.raises(KeyError, match="não está no estado"):
        estado.contestacao(7, _momento(5, 10))


def test_o_pix_fica_no_estado_ate_o_fim_do_prazo_do_med() -> None:
    # A vítima tem 80 dias para contestar pelo MED: até lá, o estado sabe para quem o Pix foi.
    estado = Estado(_visao_das_bordas().clientes)
    estado.registrar(_pix(0, _momento(1, 10)))
    estado.registrar(_pix(1, _momento(1, 10) + timedelta(days=80)))
    estado.contestacao(0, _momento(1, 10) + timedelta(days=80))

    assert estado.conhece(0)

    # Passado o prazo, com um dia de folga, ele sai.
    estado.registrar(_pix(2, _momento(1, 10) + timedelta(days=81, seconds=1)))

    assert not estado.conhece(0)
    assert estado.conhece(1)
    assert estado.pix == 3


def test_o_fuso_do_pix_vira_o_horario_de_brasilia() -> None:
    estado = Estado(_visao_das_bordas().clientes)
    em_utc = replace(_pix(0, _momento(5, 10)), momento=_momento(5, 10).astimezone(ZoneInfo("UTC")))

    assert estado.variaveis(em_utc)["hora"] == 10
