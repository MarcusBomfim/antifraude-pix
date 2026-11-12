"""O serviço que decide cada Pix em tempo real: a API em FastAPI (ADR-0017).

O banco chama a API a cada Pix, antes de deixá-lo sair, e a cada evento que
muda o que o banco sabe:
- `POST /pix`: o Pix que chegou. Volta a decisão, a chance de golpe e a explicação;
- `POST /pix/{id}/confirmacao`: o cliente confirmou, ou não, o Pix segurado;
- `POST /aparelhos`: um aparelho entrou na conta de um cliente;
- `POST /contestacoes`: a vítima contestou um Pix;
- `GET /saude`: o modelo que decide, e o tamanho do estado.

O Pix aprovado entra na história na hora. O segurado espera a confirmação, e
entra no instante em que o cliente confirma, que é quando ele sai. O bloqueado
nunca entra: ele não aconteceu. É a mesma história que o treino lê, e por isso
as variáveis do serviço são as do treino (ADR-0016).

O estado não é feito para dois Pix ao mesmo tempo: um cadeado põe os pedidos
em fila. A conta de cada um leva milissegundos (ADR-0018).

Cada pedido deixa um traço e conta nas métricas do OpenTelemetry, sem nada que
identifique alguém (ADR-0022).
"""

import threading
import time
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal, Protocol

import mlflow
import polars as pl
from fastapi import FastAPI, HTTPException, Request, Response
from mlflow.tracking import MlflowClient
from pydantic import AwareDatetime, BaseModel, Field, model_validator

from antifraude_pix.dados.esquema import BANCO
from antifraude_pix.rastreio.experimentos import endereco
from antifraude_pix.rastreio.registro import CAMPEAO, NOME
from antifraude_pix.tempo_real.estado import ESQUEMA, Estado, PixNovo
from antifraude_pix.tempo_real.telemetria import Telemetria, id_do_traco

BANCOS = tuple(BANCO.categories.to_list())


class Decisor(Protocol):
    """O que decide um Pix: as variáveis entram, e saem a pontuação, a probabilidade, a ação e a explicação."""

    def decidir_um(self, variaveis: dict[str, Any]) -> dict[str, Any]: ...


class PixDeEntrada(BaseModel):
    """O Pix como o banco o manda, antes de deixá-lo sair. O dinheiro vem em centavos."""

    id: int
    momento: AwareDatetime
    cliente: int
    aparelho: int
    recebedor: int
    recebedor_banco: str
    recebedor_pessoa: Literal["PF", "PJ"]
    chave: Literal["cpf", "cnpj", "celular", "email", "aleatoria"] | None
    iniciacao: Literal["chave", "qr_code", "manual"]
    valor: int = Field(gt=0, description="Em centavos.")
    saldo_antes: int = Field(gt=0, description="Em centavos.")

    @model_validator(mode="after")
    def _faz_sentido(self) -> "PixDeEntrada":
        if self.recebedor_banco not in BANCOS:
            raise ValueError(f"O banco {self.recebedor_banco} não está na lista: {', '.join(BANCOS)}.")

        if self.valor > self.saldo_antes:
            raise ValueError("O Pix sem saldo não sai: o valor passa do saldo.")

        if (self.iniciacao == "manual") != (self.chave is None):
            raise ValueError("O Pix pela agência e a conta não tem chave, e o Pix pela chave ou pelo QR Code tem.")

        return self


class Decisao(BaseModel):
    id: int
    acao: Literal["aprovar", "segurar", "bloquear"]
    probabilidade: float
    explicacao: str
    modelo: str
    versao: str
    milissegundos: float = Field(description="O tempo da decisão dentro do serviço, sem a rede.")
    traco: str | None = Field(
        default=None, description="O id do traço da decisão, para o banco guardar junto do Pix: nada, sem telemetria."
    )


class Confirmacao(BaseModel):
    confirmado: bool
    momento: AwareDatetime = Field(description="Quando o cliente respondeu: se confirmou, é quando o Pix sai.")


class Confirmado(BaseModel):
    id: int
    saiu: bool


class AparelhoNovo(BaseModel):
    cliente: int
    aparelho: int
    momento: AwareDatetime


class Contestacao(BaseModel):
    transacao: int
    momento: AwareDatetime


class Saude(BaseModel):
    modelo: str
    versao: str
    pix_no_estado: int
    segurados: int


def criar(
    estado: Estado, decisor: Decisor, versao: str, modelo: str = NOME, telemetria: Telemetria | None = None
) -> FastAPI:
    """A API, com o estado do banco e o modelo que decide, e a telemetria: sem ela, a dos provedores globais."""
    observada = telemetria or Telemetria.de()
    app = FastAPI(
        title="Antifraude Pix",
        description="Decide cada Pix antes de ele sair: aprovar, segurar para confirmar ou bloquear, com o porquê.",
        version=versao,
        telemetry=observada.do_fastapi(),
    )
    cadeado = threading.Lock()
    segurados: dict[int, PixNovo] = {}

    @app.post("/pix")
    def decidir_o_pix(entrada: PixDeEntrada, pedido: Request) -> Decisao:
        """Decide o Pix que chegou, com as variáveis do que o banco sabia antes dele."""
        comeco = time.perf_counter()
        pix = PixNovo(**entrada.model_dump())

        with observada.pedido("decidir o pix", pedido.headers) as traco:
            with cadeado:
                if estado.conhece(pix.id) or pix.id in segurados:
                    raise HTTPException(409, f"O Pix {pix.id} já foi decidido.")

                with observada.etapa("variaveis"):
                    variaveis = _ou_erro(lambda: estado.variaveis(pix))

                meio = time.perf_counter()

                with observada.etapa("modelo"):
                    decidido = decisor.decidir_um(variaveis)

                fim = time.perf_counter()

                if decidido["acao"] == "aprovar":
                    estado.registrar(pix)
                elif decidido["acao"] == "segurar":
                    segurados[pix.id] = pix

            milissegundos = (time.perf_counter() - comeco) * 1_000
            observada.decidido(
                traco,
                acao=decidido["acao"],
                probabilidade=decidido["probabilidade"],
                valor=pix.valor,
                iniciacao=pix.iniciacao,
                recebedor_pessoa=pix.recebedor_pessoa,
                versao=versao,
                milissegundos={
                    "variaveis": (meio - comeco) * 1_000,
                    "modelo": (fim - meio) * 1_000,
                    "total": milissegundos,
                },
            )

        return Decisao(
            id=pix.id,
            acao=decidido["acao"],
            probabilidade=decidido["probabilidade"],
            explicacao=decidido["explicacao"],
            modelo=modelo,
            versao=versao,
            milissegundos=milissegundos,
            traco=id_do_traco(traco),
        )

    @app.post("/pix/{id_}/confirmacao")
    def confirmar(id_: int, confirmacao: Confirmacao, pedido: Request) -> Confirmado:
        """A resposta do cliente ao Pix segurado: se ele confirma, o Pix sai e entra na história."""
        with observada.pedido("confirmar o pix", pedido.headers) as traco, cadeado:
            if id_ not in segurados:
                raise HTTPException(404, f"O Pix {id_} não está esperando confirmação.")

            if confirmacao.confirmado:
                _ou_erro(lambda: estado.registrar(replace(segurados[id_], momento=confirmacao.momento)))

            del segurados[id_]
            observada.respondido(traco, confirmado=confirmacao.confirmado)

        return Confirmado(id=id_, saiu=confirmacao.confirmado)

    @app.post("/aparelhos", status_code=204)
    def aparelho_novo(aparelho: AparelhoNovo, pedido: Request) -> Response:
        """Um aparelho entrou na conta do cliente."""
        with observada.pedido("aparelho novo", pedido.headers), cadeado:
            _ou_erro(lambda: estado.aparelho_novo(aparelho.cliente, aparelho.momento, aparelho.aparelho))

        return Response(status_code=204)

    @app.post("/contestacoes", status_code=204)
    def contestar(contestacao: Contestacao, pedido: Request) -> Response:
        """A vítima contestou o Pix: o recebedor dele passa a ter mais uma contestação."""
        with observada.pedido("contestação", pedido.headers), cadeado:
            _ou_erro(lambda: estado.contestacao(contestacao.transacao, contestacao.momento))
            observada.contestacoes.add(1)

        return Response(status_code=204)

    @app.get("/saude")
    def saude() -> Saude:
        """O modelo que decide, e quantos Pix o estado guarda."""
        return Saude(modelo=modelo, versao=versao, pix_no_estado=estado.pix, segurados=len(segurados))

    return app


def carregar_o_campeao(pasta_dos_dados: Path) -> tuple[Decisor, str]:
    """O modelo com o apelido de campeão no registro do MLflow, e a versão dele.

    O MLflow carrega o modelo do código guardado com ele, e a classe vem dessa cópia, e não do pacote instalado:
    o que se confere é que ele decide um Pix. O modelo registrado antes do caminho de um Pix só, como o da Parte
    5, decide pelo `predict` do MLflow, que todo modelo registrado tem: mais lento, e com a mesma resposta.
    """
    mlflow.set_tracking_uri(endereco(pasta_dos_dados))
    versao = str(MlflowClient().get_model_version_by_alias(NOME, CAMPEAO).version)

    return decisor_de(mlflow.pyfunc.load_model(f"models:/{NOME}@{CAMPEAO}")), versao


def decisor_de(carregado: Any) -> Decisor:
    """O decisor de um modelo que o MLflow carregou: o caminho de um Pix só, se ele tem; senão, o `predict`."""
    por_dentro = carregado.unwrap_python_model()

    if callable(getattr(por_dentro, "decidir_um", None)):
        decisor: Decisor = por_dentro

        return decisor

    return PeloMlflow(carregado)


class PeloMlflow:
    """O decisor pelo `predict` do MLflow: um Pix vira uma tabela de uma linha do pandas, conferida pela assinatura."""

    def __init__(self, modelo: Any) -> None:
        self._modelo = modelo

    def decidir_um(self, variaveis: dict[str, Any]) -> dict[str, Any]:
        tabela = pl.DataFrame([variaveis], schema=ESQUEMA).to_pandas()

        return dict(self._modelo.predict(tabela).iloc[0].to_dict())


def _ou_erro[T](conta: Callable[[], T]) -> T:
    """A conta do estado, com o erro dela traduzido para o pedido: 404 se falta algo, 409 se veio fora de ordem."""
    try:
        return conta()
    except KeyError as erro:
        raise HTTPException(404, str(erro.args[0])) from erro
    except ValueError as erro:
        raise HTTPException(409, str(erro)) from erro
