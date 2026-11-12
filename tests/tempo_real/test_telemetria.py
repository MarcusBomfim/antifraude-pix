from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import polars as pl
import pytest
from fastapi.testclient import TestClient
from opentelemetry import trace
from opentelemetry.metrics import _internal as metricas_globais
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace.sampling import ALWAYS_OFF
from opentelemetry.trace import StatusCode, format_span_id, format_trace_id

from antifraude_pix.dados.banco import VisaoDoBanco
from antifraude_pix.tempo_real.estado import Estado
from antifraude_pix.tempo_real.servico import criar
from antifraude_pix.tempo_real.telemetria import (
    ATRIBUTOS,
    CHANCES,
    ENDERECO,
    MILISSEGUNDOS,
    SERVICO,
    Telemetria,
    configurar,
    faixa_de_valor,
    provedores,
)
from tests.apoio import simulacao_media

CORTE = date(2025, 9, 1)


@pytest.fixture(scope="module")
def visao() -> VisaoDoBanco:
    return VisaoDoBanco.da(simulacao_media())


@pytest.fixture(scope="module")
def estado_de_agosto(visao: VisaoDoBanco) -> Estado:
    return Estado.da_historia(visao.ate(CORTE))


@dataclass
class _EmSequencia:
    """O decisor de mentira, que toma as decisões da lista, uma por Pix, com a chance de golpe dada."""

    acoes: list[str]
    probabilidade: float = 0.3
    vez: int = field(default=0)

    def decidir_um(self, variaveis: dict[str, Any]) -> dict[str, Any]:
        del variaveis
        acao = self.acoes[self.vez % len(self.acoes)]
        self.vez += 1

        return {"pontuacao": 0.0, "probabilidade": self.probabilidade, "acao": acao, "explicacao": "De mentira."}


@dataclass
class _Observada:
    """A API com a telemetria em memória: os traços terminados e as métricas, para conferir."""

    cliente: TestClient
    tracos: InMemorySpanExporter
    leitor: InMemoryMetricReader

    def spans(self) -> tuple[ReadableSpan, ...]:
        return self.tracos.get_finished_spans()

    def pontos(self, nome: str) -> list[Any]:
        dados = self.leitor.get_metrics_data()
        assert dados is not None

        return [
            ponto
            for recurso in dados.resource_metrics
            for escopo in recurso.scope_metrics
            for metrica in escopo.metrics
            if metrica.name == nome
            for ponto in metrica.data.data_points
        ]

    def soma(self, nome: str, **atributos: Any) -> float:
        return float(
            sum(
                ponto.value
                for ponto in self.pontos(nome)
                if all(ponto.attributes.get(k) == v for k, v in atributos.items())
            )
        )


type Api = Callable[[list[str]], _Observada]


@pytest.fixture
def api(visao: VisaoDoBanco) -> Api:
    """A API com a telemetria em memória, e o estado de agosto sem os Pix que outro teste mandou."""

    def montar(acoes: list[str]) -> _Observada:
        tracos, leitor = InMemorySpanExporter(), InMemoryMetricReader()
        rastreador = TracerProvider()
        rastreador.add_span_processor(SimpleSpanProcessor(tracos))
        telemetria = Telemetria.de(rastreador, MeterProvider(metric_readers=[leitor]))
        app = criar(Estado.da_historia(visao.ate(CORTE)), _EmSequencia(acoes), "teste", telemetria=telemetria)

        return _Observada(TestClient(app), tracos, leitor)

    return montar


def _pix_de_setembro(visao: VisaoDoBanco, quantos: int) -> list[dict[str, Any]]:
    """Os primeiros Pix de setembro, na ordem do tempo, como o banco os manda à API."""
    campos = (
        "id", "cliente", "aparelho", "recebedor", "recebedor_banco", "recebedor_pessoa", "chave", "iniciacao",
        "valor", "saldo_antes",
    )  # fmt: skip
    linhas = visao.transacoes.filter(pl.col("momento").dt.date() >= CORTE).head(quantos).to_dicts()

    return [{campo: linha[campo] for campo in campos} | {"momento": linha["momento"].isoformat()} for linha in linhas]


def test_cada_pix_decidido_deixa_um_traco_com_as_duas_etapas(visao: VisaoDoBanco, api: Api) -> None:
    observada = api(["segurar"])
    (pix,) = _pix_de_setembro(visao, 1)
    decisao = observada.cliente.post("/pix", json=pix).json()
    variaveis, modelo, pedido = observada.spans()

    assert (variaveis.name, modelo.name, pedido.name) == ("variaveis", "modelo", "decidir o pix")
    assert pedido.parent is None
    assert variaveis.parent is not None
    assert variaveis.parent.span_id == pedido.context.span_id
    assert modelo.parent is not None
    assert modelo.parent.span_id == pedido.context.span_id
    assert dict(pedido.attributes or {}) == {
        "pix.acao": "segurar",
        "pix.faixa_de_valor": faixa_de_valor(pix["valor"]),
        "pix.iniciacao": pix["iniciacao"],
        "pix.recebedor_pessoa": pix["recebedor_pessoa"],
        "pix.probabilidade": 0.3,
        "modelo.versao": "teste",
    }
    # O banco recebe o id do traço, e guarda junto do Pix do lado dele.
    assert decisao["traco"] == format_trace_id(pedido.context.trace_id)


def test_as_metricas_contam_cada_pix_o_dinheiro_e_quem_espera_o_cliente(visao: VisaoDoBanco, api: Api) -> None:
    observada = api(["aprovar", "segurar", "bloquear"])
    aprovado, segurado, bloqueado = _pix_de_setembro(visao, 3)

    for pix in (aprovado, segurado, bloqueado):
        observada.cliente.post("/pix", json=pix)

    assert observada.soma("antifraude.segurados") == 1

    observada.cliente.post(
        f"/pix/{segurado['id']}/confirmacao", json={"confirmado": False, "momento": segurado["momento"]}
    )

    for acao, pix in (("aprovar", aprovado), ("segurar", segurado), ("bloquear", bloqueado)):
        assert observada.soma("antifraude.pix.decididos", **{"pix.acao": acao}) == 1
        assert observada.soma("antifraude.pix.valor", **{"pix.acao": acao}) == pytest.approx(pix["valor"] / 100)

    assert observada.soma("antifraude.segurados") == 0
    assert observada.soma("antifraude.confirmacoes", **{"pix.confirmado": False}) == 1
    assert observada.soma("antifraude.confirmacoes", **{"pix.confirmado": True}) == 0

    duracoes = {ponto.attributes["etapa"]: ponto for ponto in observada.pontos("antifraude.decisao.duracao")}
    chances = observada.pontos("antifraude.pix.probabilidade")

    assert set(duracoes) == {"variaveis", "modelo", "total"}
    assert all(ponto.count == 3 for ponto in duracoes.values())
    assert duracoes["total"].sum >= duracoes["variaveis"].sum + duracoes["modelo"].sum
    assert tuple(duracoes["total"].explicit_bounds) == MILISSEGUNDOS
    assert sorted(ponto.attributes["pix.acao"] for ponto in chances) == ["aprovar", "bloquear", "segurar"]
    assert all(ponto.count == 1 and tuple(ponto.explicit_bounds) == CHANCES for ponto in chances)


def test_nada_que_identifique_alguem_sai_do_servico(visao: VisaoDoBanco, api: Api) -> None:
    observada = api(["aprovar", "segurar", "bloquear"])
    pixes = _pix_de_setembro(visao, 30)

    for pix in pixes:
        observada.cliente.post("/pix", json=pix)

        if observada.cliente.get("/saude").json()["segurados"]:
            observada.cliente.post(
                f"/pix/{pix['id']}/confirmacao", json={"confirmado": True, "momento": pix["momento"]}
            )

    ultimo = pixes[-1]
    # Os pedidos recusados também: o Pix repetido, o cliente que não é do banco, o aparelho e a contestação.
    observada.cliente.post("/pix", json=ultimo)
    observada.cliente.post("/pix", json=ultimo | {"id": 10**9, "cliente": 99_999})
    observada.cliente.post(
        "/aparelhos", json={"cliente": ultimo["cliente"], "aparelho": 77, "momento": ultimo["momento"]}
    )
    observada.cliente.post("/contestacoes", json={"transacao": pixes[0]["id"], "momento": ultimo["momento"]})

    pessoais = {
        valor
        for pix in pixes
        for campo in ("id", "cliente", "aparelho", "recebedor", "chave", "valor", "saldo_antes")
        if (valor := pix[campo]) is not None
    } | {10**9, 99_999, 77}
    nos_tracos = [(chave, valor) for span in observada.spans() for chave, valor in (span.attributes or {}).items()]
    dados = observada.leitor.get_metrics_data()
    assert dados is not None
    nas_metricas = [
        (chave, valor)
        for recurso in dados.resource_metrics
        for escopo in recurso.scope_metrics
        for metrica in escopo.metrics
        for ponto in metrica.data.data_points
        for chave, valor in (ponto.attributes or {}).items()
    ]

    assert len(nos_tracos) > 100
    assert {chave for chave, _ in nos_tracos + nas_metricas} <= ATRIBUTOS
    assert not [valor for _, valor in nos_tracos + nas_metricas if valor in pessoais and not isinstance(valor, bool)]
    # A exceção não fica no traço: a mensagem dela traz o id do Pix ou do cliente.
    assert all(not span.events for span in observada.spans())
    # A contestação do primeiro Pix, que foi aprovado, chegou e conta.
    assert observada.soma("antifraude.contestacoes") == 1


def test_o_pedido_recusado_fica_no_traco_so_pelo_codigo(visao: VisaoDoBanco, api: Api) -> None:
    observada = api(["aprovar"])
    (pix,) = _pix_de_setembro(visao, 1)
    observada.cliente.post("/pix", json=pix)

    assert observada.cliente.post("/pix", json=pix).status_code == 409

    recusado = observada.spans()[-1]

    assert recusado.name == "decidir o pix"
    assert recusado.status.status_code == StatusCode.ERROR
    assert recusado.status.description is None
    assert dict(recusado.attributes or {}) == {"http.response.status_code": 409}
    assert (
        observada.soma("antifraude.pedidos.recusados", pedido="decidir o pix", **{"http.response.status_code": 409})
        == 1
    )
    assert observada.soma("antifraude.pix.decididos") == 1


def test_o_traco_do_servico_continua_o_do_banco(visao: VisaoDoBanco, api: Api) -> None:
    observada = api(["aprovar"])
    (pix,) = _pix_de_setembro(visao, 1)
    do_banco, pai = "4bf92f3577b34da6a3ce929d0e0e4736", "00f067aa0ba902b7"
    decisao = observada.cliente.post("/pix", json=pix, headers={"traceparent": f"00-{do_banco}-{pai}-01"}).json()
    pedido = observada.spans()[-1]

    assert decisao["traco"] == do_banco
    assert format_trace_id(pedido.context.trace_id) == do_banco
    assert pedido.parent is not None
    assert format_span_id(pedido.parent.span_id) == pai


def test_com_os_provedores_globais_como_no_conteiner_so_a_telemetria_do_servico_sai(
    visao: VisaoDoBanco, estado_de_agosto: Estado, monkeypatch: pytest.MonkeyPatch
) -> None:
    # No contêiner, os provedores são os globais, e o FastAPI também os usaria: o traço dele leva o caminho do
    # pedido, que na confirmação tem o id do Pix. Aqui, os globais são os de memória.
    tracos, leitor = InMemorySpanExporter(), InMemoryMetricReader()
    rastreador = TracerProvider()
    rastreador.add_span_processor(SimpleSpanProcessor(tracos))
    monkeypatch.setattr(trace, "_TRACER_PROVIDER", rastreador)
    monkeypatch.setattr(metricas_globais, "_METER_PROVIDER", MeterProvider(metric_readers=[leitor]))
    (pix,) = _pix_de_setembro(visao, 1)

    with TestClient(criar(Estado.da_historia(visao.ate(CORTE)), _EmSequencia(["segurar"]), "teste")) as cliente:
        cliente.post("/pix", json=pix)
        cliente.post(f"/pix/{pix['id']}/confirmacao", json={"confirmado": True, "momento": pix["momento"]})

    dados = leitor.get_metrics_data()
    assert dados is not None
    metricas = {
        metrica.name
        for recurso in dados.resource_metrics
        for escopo in recurso.scope_metrics
        for metrica in escopo.metrics
    }

    assert [span.name for span in tracos.get_finished_spans()] == [
        "variaveis",
        "modelo",
        "decidir o pix",
        "confirmar o pix",
    ]
    assert metricas
    assert all(nome.startswith("antifraude.") for nome in metricas)


def test_o_pix_fora_da_amostra_conta_nas_metricas_e_nao_tem_traco(
    visao: VisaoDoBanco, estado_de_agosto: Estado
) -> None:
    # Em produção, só uma parte dos traços é guardada; as métricas contam todos os Pix.
    tracos, leitor = InMemorySpanExporter(), InMemoryMetricReader()
    rastreador = TracerProvider(sampler=ALWAYS_OFF)
    rastreador.add_span_processor(SimpleSpanProcessor(tracos))
    telemetria = Telemetria.de(rastreador, MeterProvider(metric_readers=[leitor]))
    observada = _Observada(
        TestClient(criar(estado_de_agosto, _EmSequencia(["bloquear"]), "teste", telemetria=telemetria)), tracos, leitor
    )
    (pix,) = _pix_de_setembro(visao, 1)

    assert observada.cliente.post("/pix", json=pix).json()["traco"] is None
    assert observada.spans() == ()
    assert observada.soma("antifraude.pix.decididos", **{"pix.acao": "bloquear"}) == 1


def test_sem_telemetria_configurada_a_api_responde_sem_traco(
    visao: VisaoDoBanco, estado_de_agosto: Estado, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(ENDERECO, raising=False)
    (pix,) = _pix_de_setembro(visao, 1)
    cliente = TestClient(criar(estado_de_agosto, _EmSequencia(["bloquear"]), "teste"))

    assert provedores("1") is None
    assert configurar("1") is False
    assert cliente.post("/pix", json=pix).json()["traco"] is None


def test_com_o_endereco_do_coletor_os_provedores_exportam_pelo_otlp(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENDERECO, "http://127.0.0.1:4318")
    monkeypatch.delenv("OTEL_SERVICE_NAME", raising=False)
    feitos = provedores("7")

    assert feitos is not None

    rastreador, medidor = feitos

    try:
        assert rastreador.resource.attributes["service.name"] == SERVICO
        assert rastreador.resource.attributes["service.version"] == "7"
    finally:
        rastreador.shutdown()
        medidor.shutdown(timeout_millis=1_000)

    monkeypatch.setenv("OTEL_SERVICE_NAME", "antifraude-pix-homologacao")
    feitos = provedores("7")

    assert feitos is not None

    rastreador, medidor = feitos

    try:
        assert rastreador.resource.attributes["service.name"] == "antifraude-pix-homologacao"
    finally:
        rastreador.shutdown()
        medidor.shutdown(timeout_millis=1_000)


@pytest.mark.parametrize(
    ("centavos", "faixa"),
    [
        (1, "ate-100"),
        (10_000, "ate-100"),
        (10_001, "100-500"),
        (50_000, "100-500"),
        (200_001, "2000-10000"),
        (1_000_000, "2000-10000"),
        (1_000_001, "acima-10000"),
    ],
)
def test_a_faixa_de_valor_diz_o_tamanho_sem_dizer_o_valor(centavos: int, faixa: str) -> None:
    assert faixa_de_valor(centavos) == faixa
