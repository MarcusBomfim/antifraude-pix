from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from zoneinfo import ZoneInfo

import polars as pl
import pytest
from fastapi.testclient import TestClient

from antifraude_pix.avaliacao.no_tempo import avaliar
from antifraude_pix.dados.banco import VisaoDoBanco
from antifraude_pix.decisao.custo import Custos, decidir
from antifraude_pix.modelos.preparo import ENTRADA
from antifraude_pix.modelos.rotulo import Dobra, janela_de_treino
from antifraude_pix.modelos.treino import Calibrado, treinar
from antifraude_pix.rastreio.modelo_do_pix import ModeloDoPix
from antifraude_pix.rastreio.registro import registrar, treinar_o_final
from antifraude_pix.tempo_real.estado import Estado
from antifraude_pix.tempo_real.servico import PeloMlflow, carregar_o_campeao, criar
from antifraude_pix.variaveis.calculo import calcular
from tests.apoio import simulacao_media

CORTE = date(2025, 9, 1)
FUSO = ZoneInfo("America/Sao_Paulo")


@pytest.fixture(scope="module")
def visao() -> VisaoDoBanco:
    return VisaoDoBanco.da(simulacao_media())


@pytest.fixture(scope="module")
def variaveis(visao: VisaoDoBanco) -> pl.DataFrame:
    return calcular(visao)


@pytest.fixture(scope="module")
def modelo(visao: VisaoDoBanco, variaveis: pl.DataFrame) -> Calibrado:
    return treinar("lightgbm", janela_de_treino(visao, variaveis, CORTE), CORTE)


def _pix(linha: dict[str, Any]) -> dict[str, Any]:
    """O Pix de uma linha das transações, como o banco o manda à API."""
    campos = (
        "id", "cliente", "aparelho", "recebedor", "recebedor_banco", "recebedor_pessoa", "chave", "iniciacao",
        "valor", "saldo_antes",
    )  # fmt: skip

    return {campo: linha[campo] for campo in campos} | {"momento": linha["momento"].isoformat()}


def test_a_api_decide_com_as_variaveis_do_treino(
    visao: VisaoDoBanco, variaveis: pl.DataFrame, modelo: Calibrado
) -> None:
    # Os dois primeiros dias de setembro pela API, na ordem do tempo, com cada evento do banco. O Pix bloqueado não
    # sai, e a história da API deixaria de ser a do treino: aqui, bloquear custa tanto que nenhum Pix é bloqueado. O
    # segurado é confirmado no mesmo instante, e sai como saiu na simulação.
    sem_bloqueio = Custos(bloquear_um_honesto=1e12)
    tipos = {variavel.nome: variaveis.schema[variavel.nome] for variavel in ENTRADA}
    cliente = TestClient(criar(Estado.da_historia(visao.ate(CORTE)), ModeloDoPix(modelo, sem_bloqueio, tipos), "teste"))
    corte = datetime(2025, 9, 1, tzinfo=FUSO)

    fim = corte + timedelta(days=2)

    def depois(tabela: pl.DataFrame, coluna: str) -> list[dict[str, Any]]:
        return tabela.filter(pl.col(coluna).is_between(corte, fim, closed="left")).to_dicts()

    # No mesmo instante, o aparelho e a contestação vêm antes do Pix, como no estado e no lote.
    eventos = sorted(
        [(linha["momento"], 0, 0, "aparelho", linha) for linha in depois(visao.aparelhos_novos, "momento")]
        + [
            (linha["contestada_em"], 1, linha["transacao"], "contestacao", linha)
            for linha in depois(visao.contestacoes, "contestada_em")
        ]
        + [(linha["momento"], 2, linha["id"], "pix", linha) for linha in depois(visao.transacoes, "momento")],
        key=lambda evento: evento[:3],
    )
    esperadas = decidir(
        variaveis.select("id", "valor").with_columns(pl.Series("probabilidade", modelo.probabilidade(variaveis))),
        sem_bloqueio,
    )
    por_id = {linha["id"]: linha for linha in esperadas.iter_rows(named=True)}
    conferidos = segurados = 0

    for momento, _, _, tipo, linha in eventos:
        if tipo == "aparelho":
            dados = {"cliente": linha["cliente"], "aparelho": linha["aparelho"], "momento": momento.isoformat()}
            assert cliente.post("/aparelhos", json=dados).status_code == 204
            continue

        if tipo == "contestacao":
            dados = {"transacao": linha["transacao"], "momento": momento.isoformat()}
            assert cliente.post("/contestacoes", json=dados).status_code == 204
            continue

        resposta = cliente.post("/pix", json=_pix(linha))
        decisao = resposta.json()
        esperada = por_id[linha["id"]]

        assert resposta.status_code == 200
        assert decisao["probabilidade"] == pytest.approx(esperada["probabilidade"], rel=0, abs=1e-12)
        assert decisao["acao"] == esperada["acao"]
        conferidos += 1

        if decisao["acao"] == "segurar":
            confirmacao = {"confirmado": True, "momento": momento.isoformat()}
            assert cliente.post(f"/pix/{linha['id']}/confirmacao", json=confirmacao).json()["saiu"]
            segurados += 1

    assert conferidos == visao.transacoes.filter(pl.col("momento").is_between(corte, fim, closed="left")).height > 1_000
    assert segurados > 0


@dataclass
class _SempreIgual:
    """O decisor de mentira, que toma sempre a mesma decisão."""

    acao: str

    def decidir_um(self, variaveis: dict[str, Any]) -> dict[str, Any]:
        del variaveis
        return {"pontuacao": 0.0, "probabilidade": 0.5, "acao": self.acao, "explicacao": "Pelo decisor de mentira."}


def _api(visao: VisaoDoBanco, acao: str) -> TestClient:
    return TestClient(criar(Estado.da_historia(visao.ate(CORTE)), _SempreIgual(acao), "teste"))


def _primeiro_de_setembro(visao: VisaoDoBanco, **mudancas: Any) -> dict[str, Any]:
    linha = visao.transacoes.filter(pl.col("momento").dt.date() >= CORTE).row(0, named=True)

    return _pix(linha) | mudancas


def test_o_pix_segurado_espera_a_confirmacao_e_sai_quando_o_cliente_confirma(visao: VisaoDoBanco) -> None:
    cliente = _api(visao, "segurar")
    pix = _primeiro_de_setembro(visao)
    antes = cliente.get("/saude").json()["pix_no_estado"]
    depois_de_um_minuto = (datetime.fromisoformat(pix["momento"]) + timedelta(minutes=1)).isoformat()

    assert cliente.post("/pix", json=pix).json()["acao"] == "segurar"
    assert cliente.get("/saude").json() | {"pix_no_estado": antes, "segurados": 1} == cliente.get("/saude").json()
    assert cliente.post(
        f"/pix/{pix['id']}/confirmacao", json={"confirmado": True, "momento": depois_de_um_minuto}
    ).json() == {
        "id": pix["id"],
        "saiu": True,
    }
    assert cliente.get("/saude").json()["pix_no_estado"] == antes + 1
    assert cliente.get("/saude").json()["segurados"] == 0
    # A confirmação vale uma vez só.
    assert (
        cliente.post(
            f"/pix/{pix['id']}/confirmacao", json={"confirmado": True, "momento": depois_de_um_minuto}
        ).status_code
        == 404
    )


def test_o_pix_segurado_entra_na_historia_no_instante_da_confirmacao(visao: VisaoDoBanco) -> None:
    # Confirmado dez minutos depois, ele sai dez minutos depois: o Pix de cinco minutos depois já vem fora de ordem.
    cliente = _api(visao, "segurar")
    pix = _primeiro_de_setembro(visao)
    chegou = datetime.fromisoformat(pix["momento"])
    cliente.post("/pix", json=pix)
    confirmacao = {"confirmado": True, "momento": (chegou + timedelta(minutes=10)).isoformat()}
    cliente.post(f"/pix/{pix['id']}/confirmacao", json=confirmacao)
    depois = pix | {"id": pix["id"] + 10_000_000, "momento": (chegou + timedelta(minutes=5)).isoformat()}

    assert cliente.post("/pix", json=depois).status_code == 409


def test_o_pix_que_o_cliente_nao_confirma_nao_sai(visao: VisaoDoBanco) -> None:
    cliente = _api(visao, "segurar")
    pix = _primeiro_de_setembro(visao)
    antes = cliente.get("/saude").json()["pix_no_estado"]
    cliente.post("/pix", json=pix)

    assert (
        cliente.post(f"/pix/{pix['id']}/confirmacao", json={"confirmado": False, "momento": pix["momento"]}).json()[
            "saiu"
        ]
        is False
    )
    assert cliente.get("/saude").json()["pix_no_estado"] == antes


def test_o_pix_bloqueado_nao_entra_na_historia_e_o_aprovado_entra(visao: VisaoDoBanco) -> None:
    bloqueia, aprova = _api(visao, "bloquear"), _api(visao, "aprovar")
    pix = _primeiro_de_setembro(visao)
    antes = bloqueia.get("/saude").json()["pix_no_estado"]

    assert bloqueia.post("/pix", json=pix).json()["acao"] == "bloquear"
    assert bloqueia.get("/saude").json()["pix_no_estado"] == antes
    assert aprova.post("/pix", json=pix).json()["acao"] == "aprovar"
    assert aprova.get("/saude").json()["pix_no_estado"] == antes + 1
    # O mesmo Pix de novo, depois de decidido, é recusado.
    assert aprova.post("/pix", json=pix).status_code == 409


@pytest.mark.parametrize(
    ("mudancas", "motivo"),
    [
        ({"valor": 2_000_000, "saldo_antes": 1_000}, "O Pix sem saldo não sai"),
        ({"iniciacao": "manual", "chave": "cpf"}, "agência e a conta não tem chave"),
        ({"recebedor_banco": "999"}, "não está na lista"),
        ({"valor": 0}, "greater than 0"),
        ({"momento": "2025-09-01T10:00:00"}, "timezone"),
        ({"chave": "telefone"}, "Input should be"),
    ],
)
def test_o_pix_que_nao_faz_sentido_e_recusado_com_o_motivo(
    visao: VisaoDoBanco, mudancas: dict[str, Any], motivo: str
) -> None:
    resposta = _api(visao, "aprovar").post("/pix", json=_primeiro_de_setembro(visao, **mudancas))

    assert resposta.status_code == 422
    assert motivo in resposta.text


def test_o_cliente_que_nao_e_do_banco_e_o_pix_fora_de_ordem_sao_recusados(visao: VisaoDoBanco) -> None:
    cliente = _api(visao, "aprovar")
    pix = _primeiro_de_setembro(visao)
    ontem = (datetime.fromisoformat(pix["momento"]) - timedelta(days=1)).isoformat()

    assert cliente.post("/pix", json=pix | {"cliente": 99_999}).status_code == 404
    assert cliente.post("/pix", json=pix | {"momento": ontem}).status_code == 409
    assert "ordem do tempo" in cliente.post("/pix", json=pix | {"momento": ontem}).text


def test_os_eventos_do_banco(visao: VisaoDoBanco) -> None:
    cliente = _api(visao, "aprovar")
    pix = _primeiro_de_setembro(visao)

    assert cliente.post("/aparelhos", json={"cliente": 0, "aparelho": 7, "momento": pix["momento"]}).status_code == 204
    assert cliente.post("/contestacoes", json={"transacao": -1, "momento": pix["momento"]}).status_code == 404
    assert cliente.get("/saude").json() == {
        "modelo": "antifraude-pix",
        "versao": "teste",
        "pix_no_estado": cliente.get("/saude").json()["pix_no_estado"],
        "segurados": 0,
    }


def test_o_servico_carrega_o_campeao_do_registro(visao: VisaoDoBanco, variaveis: pl.DataFrame, tmp_path: Path) -> None:
    pontuacoes = avaliar(simulacao_media(), variaveis, (Dobra(CORTE, date(2025, 9, 29)),), modelos=("lightgbm",))
    registrar(simulacao_media(), variaveis, pontuacoes, tmp_path)
    campeao, versao = carregar_o_campeao(tmp_path)
    modelo, _ = treinar_o_final(simulacao_media(), variaveis)
    de_setembro = variaveis.tail(50)
    decididos = [campeao.decidir_um(linha) for linha in de_setembro.with_columns(_em_texto()).to_dicts()]

    assert versao == "1"
    assert [decidido["probabilidade"] for decidido in decididos] == pytest.approx(
        list(modelo.probabilidade(de_setembro)), rel=0, abs=1e-12
    )


def _em_texto() -> pl.Expr:
    return pl.col("iniciacao", "chave").cast(pl.String)


def test_o_caminho_de_um_pix_so_da_o_mesmo_que_o_da_tabela(variaveis: pl.DataFrame, modelo: Calibrado) -> None:
    # Com custos que seguram e bloqueiam muitos Pix, os dois caminhos do decisor passam pela conta.
    tipos = {variavel.nome: variaveis.schema[variavel.nome] for variavel in ENTRADA}
    decisor = ModeloDoPix(modelo, Custos(confirmar=0.5, bloquear_um_honesto=5.0), tipos)
    de_setembro = variaveis.filter(pl.col("id").is_in(variaveis["id"].tail(3_000).implode()))
    pela_tabela = decisor.decidir(de_setembro).to_dicts()
    um_por_um = [decisor.decidir_um(linha) for linha in de_setembro.with_columns(_em_texto()).to_dicts()]

    assert um_por_um == pela_tabela
    assert {decidido["acao"] for decidido in um_por_um} == {"aprovar", "segurar", "bloquear"}


def test_o_campeao_antigo_decide_pelo_predict_do_mlflow_com_a_mesma_resposta(
    variaveis: pl.DataFrame, modelo: Calibrado
) -> None:
    # O modelo registrado antes do caminho de um Pix só só tem o predict, que recebe uma tabela do pandas.
    tipos = {variavel.nome: variaveis.schema[variavel.nome] for variavel in ENTRADA}
    decisor = ModeloDoPix(modelo, Custos(confirmar=0.5, bloquear_um_honesto=5.0), tipos)
    antigo = PeloMlflow(SimpleNamespace(predict=lambda tabela: decisor.predict(None, tabela)))
    linhas = variaveis.tail(200).with_columns(_em_texto()).to_dicts()
    pelo_antigo = [antigo.decidir_um(linha) for linha in linhas]
    pelo_novo = [decisor.decidir_um(linha) for linha in linhas]

    assert pelo_antigo == pelo_novo
    assert {decidido["acao"] for decidido in pelo_novo} >= {"aprovar", "segurar"}
