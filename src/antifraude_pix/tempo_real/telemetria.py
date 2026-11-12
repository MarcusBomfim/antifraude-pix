"""O serviço observado: os traços e as métricas de cada Pix, pelo OpenTelemetry, sem dado pessoal (ADR-0022).

Cada pedido à API vira um traço. O do Pix tem as duas etapas dentro dele: as
variáveis, que o estado calcula, e o modelo, que decide. Se o banco manda o
cabeçalho `traceparent`, o traço do serviço continua o dele. E cada Pix
decidido conta nas métricas:
- `antifraude.pix.decididos`: quantos Pix, por ação e faixa de valor;
- `antifraude.pix.valor`: quanto dinheiro, em reais, por ação;
- `antifraude.pix.probabilidade`: a chance de golpe que o modelo deu, por ação;
- `antifraude.decisao.duracao`: quanto cada etapa levou, em milissegundos;
- `antifraude.segurados`: os Pix esperando a confirmação do cliente;
- `antifraude.confirmacoes`: as respostas do cliente ao Pix segurado;
- `antifraude.contestacoes`: as contestações que chegaram;
- `antifraude.pedidos.recusados`: os pedidos que o serviço recusou, por código.

Nada que identifique alguém sai do serviço: nem o cliente, nem o recebedor,
nem o aparelho, nem a chave, nem o id do Pix. O valor exato vai só somado,
nas métricas; no traço vai a faixa. A mensagem de um erro pode trazer um id, e
o traço guarda só o código dele. Os atributos são uma lista fechada, e um
teste confere cada um. Quem precisa achar o traço de um Pix tem o id do traço
na resposta, e o banco o guarda junto do Pix, do lado dele.

O FastAPI tem a telemetria dele, ligada por padrão, e ela fica desligada: o
traço dela guarda o caminho do pedido, que na confirmação tem o id do Pix, e o
registro dela guarda o corpo do pedido recusado.

Sem `OTEL_EXPORTER_OTLP_ENDPOINT`, nada é exportado: o OpenTelemetry fica no
modo que não faz nada, e o serviço roda como antes.
"""

import os
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass

from fastapi import HTTPException
from fastapi.telemetry import TelemetryConfig
from opentelemetry import metrics, trace
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.metrics import Counter, Histogram, MeterProvider, UpDownCounter
from opentelemetry.propagate import extract
from opentelemetry.sdk.metrics import MeterProvider as MedidorDoSdk
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider as RastreadorDoSdk
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import Span, SpanKind, Status, StatusCode, Tracer, TracerProvider

SERVICO = "antifraude-pix"
"""O nome do serviço na telemetria, quando `OTEL_SERVICE_NAME` não diz outro."""

ENDERECO = "OTEL_EXPORTER_OTLP_ENDPOINT"
"""A variável de ambiente com o endereço do coletor: sem ela, nada é exportado."""

INTERVALO = 5_000
"""De quanto em quanto tempo as métricas vão ao coletor, em milissegundos."""

FAIXAS = ((10_000, "ate-100"), (50_000, "100-500"), (200_000, "500-2000"), (1_000_000, "2000-10000"))
"""As faixas de valor, pelo limite de cima em centavos: até R$ 100, de 100 a 500, de 500 a 2 mil, de 2 a 10 mil."""

ACIMA = "acima-10000"

ATRIBUTOS = frozenset(
    {
        "pix.acao",
        "pix.faixa_de_valor",
        "pix.iniciacao",
        "pix.recebedor_pessoa",
        "pix.probabilidade",
        "pix.confirmado",
        "modelo.versao",
        "etapa",
        "pedido",
        "http.response.status_code",
        "error.type",
    }
)
"""Os únicos atributos que a telemetria do serviço usa, nos traços e nas métricas."""

MILISSEGUNDOS = (0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0, 100.0)
"""Os limites do histograma da duração: a decisão leva décimos de milissegundo, e a explicação, alguns."""

CHANCES = (0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0)
"""Os limites do histograma da probabilidade: quase todo Pix fica perto de zero."""


@dataclass(frozen=True)
class Telemetria:
    """O rastreador e os instrumentos do serviço, tirados dos provedores do OpenTelemetry."""

    rastreador: Tracer
    provedor_de_tracos: TracerProvider | None
    provedor_de_metricas: MeterProvider | None
    decididos: Counter
    valor: Counter
    probabilidade: Histogram
    duracao: Histogram
    segurados: UpDownCounter
    confirmacoes: Counter
    contestacoes: Counter
    recusados: Counter

    @classmethod
    def de(cls, rastreador: TracerProvider | None = None, medidor: MeterProvider | None = None) -> "Telemetria":
        """A telemetria pelos provedores dados; sem eles, pelos globais, que não fazem nada até alguém configurá-los."""
        tracer = trace.get_tracer(SERVICO, tracer_provider=rastreador)
        meter = metrics.get_meter(SERVICO, meter_provider=medidor)

        return cls(
            rastreador=tracer,
            provedor_de_tracos=rastreador,
            provedor_de_metricas=medidor,
            decididos=meter.create_counter(
                "antifraude.pix.decididos", unit="{pix}", description="Os Pix decididos, por ação e faixa de valor."
            ),
            valor=meter.create_counter(
                "antifraude.pix.valor", unit="{reais}", description="O dinheiro dos Pix decididos, por ação."
            ),
            probabilidade=meter.create_histogram(
                "antifraude.pix.probabilidade",
                unit="1",
                description="A chance de golpe que o modelo deu ao Pix, por ação.",
                explicit_bucket_boundaries_advisory=list(CHANCES),
            ),
            duracao=meter.create_histogram(
                "antifraude.decisao.duracao",
                unit="ms",
                description="O tempo de cada etapa da decisão, dentro do serviço.",
                explicit_bucket_boundaries_advisory=list(MILISSEGUNDOS),
            ),
            segurados=meter.create_up_down_counter(
                "antifraude.segurados", unit="{pix}", description="Os Pix segurados que esperam a resposta do cliente."
            ),
            confirmacoes=meter.create_counter(
                "antifraude.confirmacoes", unit="{pix}", description="As respostas do cliente ao Pix segurado."
            ),
            contestacoes=meter.create_counter(
                "antifraude.contestacoes", unit="{pix}", description="As contestações que chegaram ao serviço."
            ),
            recusados=meter.create_counter(
                "antifraude.pedidos.recusados",
                unit="{pedido}",
                description="Os pedidos que o serviço recusou, por pedido e código.",
            ),
        )

    def do_fastapi(self) -> TelemetryConfig:
        """A telemetria do próprio FastAPI, desligada, com os mesmos provedores: se alguém a ligar, os testes a veem."""
        configuracao: TelemetryConfig = {"tracing": False, "metrics": False, "logs": False, "operation_spans": False}

        if self.provedor_de_tracos is not None:
            configuracao["tracer_provider"] = self.provedor_de_tracos

        if self.provedor_de_metricas is not None:
            configuracao["meter_provider"] = self.provedor_de_metricas

        return configuracao

    @contextmanager
    def pedido(self, nome: str, cabecalhos: Mapping[str, str] | None = None) -> Iterator[Span]:
        """O traço de um pedido, que continua o do banco se ele mandou o `traceparent`.

        O erro fica no traço só pelo código e pelo tipo: a mensagem dele pode trazer um id.
        """
        contexto = extract(cabecalhos) if cabecalhos is not None else None

        with self.rastreador.start_as_current_span(
            nome, context=contexto, kind=SpanKind.SERVER, record_exception=False, set_status_on_exception=False
        ) as traco:
            try:
                yield traco
            except HTTPException as erro:
                traco.set_attribute("http.response.status_code", erro.status_code)
                traco.set_status(Status(StatusCode.ERROR))
                self.recusados.add(1, {"pedido": nome, "http.response.status_code": erro.status_code})
                raise
            except Exception as erro:
                traco.set_attribute("error.type", type(erro).__name__)
                traco.set_status(Status(StatusCode.ERROR))
                raise

    @contextmanager
    def etapa(self, nome: str) -> Iterator[None]:
        """Uma etapa dentro do traço do pedido."""
        with self.rastreador.start_as_current_span(nome, record_exception=False, set_status_on_exception=False):
            yield

    def decidido(
        self,
        traco: Span,
        *,
        acao: str,
        probabilidade: float,
        valor: int,
        iniciacao: str,
        recebedor_pessoa: str,
        versao: str,
        milissegundos: Mapping[str, float],
    ) -> None:
        """Conta o Pix decidido nas métricas, e põe no traço o que não identifica ninguém."""
        faixa = faixa_de_valor(valor)
        traco.set_attributes(
            {
                "pix.acao": acao,
                "pix.faixa_de_valor": faixa,
                "pix.iniciacao": iniciacao,
                "pix.recebedor_pessoa": recebedor_pessoa,
                "pix.probabilidade": round(probabilidade, 4),
                "modelo.versao": versao,
            }
        )
        self.decididos.add(1, {"pix.acao": acao, "pix.faixa_de_valor": faixa, "modelo.versao": versao})
        self.valor.add(valor / 100, {"pix.acao": acao})
        self.probabilidade.record(probabilidade, {"pix.acao": acao})

        for etapa, tempo in milissegundos.items():
            self.duracao.record(tempo, {"etapa": etapa})

        if acao == "segurar":
            self.segurados.add(1)

    def respondido(self, traco: Span, *, confirmado: bool) -> None:
        """A resposta do cliente ao Pix segurado: ele deixa de esperar."""
        traco.set_attribute("pix.confirmado", confirmado)
        self.confirmacoes.add(1, {"pix.confirmado": confirmado})
        self.segurados.add(-1)


def faixa_de_valor(centavos: int) -> str:
    """A faixa do valor, que diz o tamanho do Pix sem dizer o valor."""
    for limite, faixa in FAIXAS:
        if centavos <= limite:
            return faixa

    return ACIMA


def id_do_traco(traco: Span) -> str | None:
    """O id do traço em hexadecimal, para o banco guardar junto do Pix.

    Nada, se a telemetria está desligada ou se o traço ficou fora da amostra (`OTEL_TRACES_SAMPLER`): o banco não
    guarda o id de um traço que não existe.
    """
    contexto = traco.get_span_context()

    return trace.format_trace_id(contexto.trace_id) if contexto.is_valid and contexto.trace_flags.sampled else None


def provedores(versao: str) -> tuple[RastreadorDoSdk, MedidorDoSdk] | None:
    """Os provedores que exportam ao coletor pelo OTLP, se o endereço dele está no ambiente; senão, nada.

    O endereço, os cabeçalhos e o tempo de espera vêm das variáveis de ambiente de costume do OpenTelemetry
    (`OTEL_EXPORTER_OTLP_*`), e o nome do serviço, de `OTEL_SERVICE_NAME`, se ela existir.
    """
    if not os.environ.get(ENDERECO):
        return None

    atributos: dict[str, str] = {"service.version": versao}

    if not os.environ.get("OTEL_SERVICE_NAME"):
        atributos["service.name"] = SERVICO

    recurso = Resource.create(atributos)
    rastreador = RastreadorDoSdk(resource=recurso)
    rastreador.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    medidor = MedidorDoSdk(
        resource=recurso,
        metric_readers=[PeriodicExportingMetricReader(OTLPMetricExporter(), export_interval_millis=INTERVALO)],
    )

    return rastreador, medidor


def configurar(versao: str) -> bool:
    """Liga a exportação ao coletor para o processo todo, se o endereço dele está no ambiente."""
    feitos = provedores(versao)

    if feitos is None:
        return False

    rastreador, medidor = feitos
    trace.set_tracer_provider(rastreador)
    metrics.set_meter_provider(medidor)

    return True
