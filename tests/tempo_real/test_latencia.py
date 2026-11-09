from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl
import pytest

from antifraude_pix.dados.banco import VisaoDoBanco
from antifraude_pix.decisao.custo import Custos
from antifraude_pix.modelos.preparo import ENTRADA
from antifraude_pix.modelos.rotulo import janela_de_treino
from antifraude_pix.modelos.treino import treinar
from antifraude_pix.rastreio.modelo_do_pix import ModeloDoPix
from antifraude_pix.tempo_real.estado import Estado
from antifraude_pix.tempo_real.latencia import eventos, medir_pela_api, medir_por_dentro, resumir
from antifraude_pix.tempo_real.servico import criar
from antifraude_pix.variaveis.calculo import calcular
from tests.apoio import simulacao_media

CORTE = date(2025, 9, 1)
INICIO = datetime(2025, 9, 1, tzinfo=ZoneInfo("America/Sao_Paulo"))
FIM = INICIO + timedelta(hours=12)


@pytest.fixture(scope="module")
def visao() -> VisaoDoBanco:
    return VisaoDoBanco.da(simulacao_media())


@pytest.fixture(scope="module")
def decisor(visao: VisaoDoBanco) -> ModeloDoPix:
    variaveis = calcular(visao)
    modelo = treinar("lightgbm", janela_de_treino(visao, variaveis, CORTE), CORTE)
    # Custos que seguram e bloqueiam muitos Pix: os três caminhos do serviço passam pela medida.
    custos = Custos(confirmar=0.5, bloquear_um_honesto=5.0)

    return ModeloDoPix(modelo, custos, {variavel.nome: variaveis.schema[variavel.nome] for variavel in ENTRADA})


def test_os_eventos_vem_na_ordem_do_estado(visao: VisaoDoBanco) -> None:
    # Em um dia de setembro, há Pix, aparelhos novos e contestações.
    fim = INICIO + timedelta(days=1)
    os_eventos = eventos(visao, INICIO, fim)
    chaves = [(evento.momento, ("aparelho", "contestacao", "pix").index(evento.tipo)) for evento in os_eventos]
    pix = visao.transacoes.filter(pl.col("momento").is_between(INICIO, fim, closed="left"))

    assert chaves == sorted(chaves)
    assert [evento.dados["id"] for evento in os_eventos if evento.tipo == "pix"] == pix["id"].to_list()
    assert {evento.tipo for evento in os_eventos} == {"pix", "aparelho", "contestacao"}


def test_por_dentro_e_pela_api_o_servico_decide_igual(visao: VisaoDoBanco, decisor: ModeloDoPix) -> None:
    os_eventos = eventos(visao, INICIO, FIM)
    estado = Estado.da_historia(visao.ate(CORTE))
    antes = estado.pix
    por_dentro = medir_por_dentro(estado, decisor, os_eventos)
    pela_api = medir_pela_api(criar(Estado.da_historia(visao.ate(CORTE)), decisor, "teste"), os_eventos)

    # Como na API, só o Pix que saiu entra na história: o bloqueado, não.
    assert estado.pix == antes + int((por_dentro["acao"] != "bloquear").sum())
    assert por_dentro.columns == ["id", "acao", "variaveis", "decisao"]
    assert pela_api.columns == ["id", "acao", "servico", "ida_e_volta"]
    assert por_dentro.select("id", "acao").equals(pela_api.select("id", "acao"))
    assert set(por_dentro["acao"]) == {"aprovar", "segurar", "bloquear"}
    assert (por_dentro["variaveis"] > 0).all()
    assert (por_dentro["decisao"] > 0).all()
    # A ida e volta inclui o que o serviço mede, e mais o HTTP e o JSON.
    assert (pela_api["ida_e_volta"] > pela_api["servico"]).all()


def test_o_resumo_e_a_mediana_os_percentis_e_o_maximo() -> None:
    tempos = pl.Series([1.0, 2.0, 3.0, 4.0, 100.0])
    linha = resumir({"a decisão": tempos}).row(0, named=True)

    assert linha == pytest.approx(
        {
            "medida": "a decisão",
            "pix": 5,
            "p50": 3.0,
            "p95": float(np.percentile(tempos.to_numpy(), 95)),
            "p99": float(np.percentile(tempos.to_numpy(), 99)),
            "maximo": 100.0,
        }
    )
    assert linha["p50"] <= linha["p95"] <= linha["p99"] <= linha["maximo"]
