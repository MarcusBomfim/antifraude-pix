"""A latência medida: quanto o serviço leva para decidir cada Pix, por etapa e de ponta a ponta (ADR-0018).

O banco tem alguns milissegundos para deixar cada Pix sair, e a medida é o
que diz se o serviço cabe neles. Ela vem em duas partes, com os Pix e os
eventos de verdade, na ordem do tempo:
- **por dentro**: as variáveis, que o estado calcula, e a decisão, que o
  modelo toma e explica, cronometradas cada uma;
- **pela API**: o serviço de pé num servidor uvicorn de verdade, e cada Pix
  mandado por HTTP, como o banco mandaria. A ida e volta inclui o JSON, a
  validação e a rede da própria máquina.

O resumo de cada medida é a mediana, os percentis 95 e 99 e o máximo: o banco
olha o Pix mais lento, e não o típico.
"""

import http.client
import json
import threading
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

import numpy as np
import polars as pl
import uvicorn
from fastapi import FastAPI

from antifraude_pix.dados.banco import VisaoDoBanco
from antifraude_pix.tempo_real.estado import Estado, PixNovo
from antifraude_pix.tempo_real.servico import Decisor

PERCENTIS = (50, 95, 99)


@dataclass(frozen=True, slots=True)
class Evento:
    """Um evento do banco, na ordem em que aconteceu: um Pix, um aparelho novo ou uma contestação."""

    tipo: Literal["pix", "aparelho", "contestacao"]
    momento: datetime
    dados: dict[str, Any]


def eventos(visao: VisaoDoBanco, de: datetime, ate: datetime) -> list[Evento]:
    """Os eventos da visão de `de` até antes de `ate`, na ordem do estado: no mesmo instante, o Pix vem por último."""
    ordenados: list[tuple[datetime, int, int, Evento]] = []

    for posicao, (tipo, tabela, coluna) in enumerate(
        (
            ("aparelho", visao.aparelhos_novos, "momento"),
            ("contestacao", visao.contestacoes, "contestada_em"),
            ("pix", visao.transacoes, "momento"),
        )
    ):
        for linha in tabela.filter(pl.col(coluna).is_between(de, ate, closed="left")).to_dicts():
            desempate = linha.get("id", linha.get("transacao", 0))
            ordenados.append((linha[coluna], posicao, desempate, Evento(tipo, linha[coluna], linha)))  # type: ignore[arg-type]

    return [evento for *_, evento in sorted(ordenados, key=lambda item: item[:3])]


def medir_por_dentro(estado: Estado, decisor: Decisor, os_eventos: Sequence[Evento]) -> pl.DataFrame:
    """O tempo das variáveis e da decisão de cada Pix, em milissegundos: `id`, `acao`, `variaveis` e `decisao`.

    O estado segue como na API: o Pix aprovado entra na hora, o segurado entra como se o cliente confirmasse no
    mesmo instante, e o bloqueado não entra.
    """
    linhas = []

    for pix in _seguidos(estado, os_eventos):
        comeco = time.perf_counter_ns()
        variaveis = estado.variaveis(pix)
        meio = time.perf_counter_ns()
        decidido = decisor.decidir_um(variaveis)
        fim = time.perf_counter_ns()
        linhas.append(
            {"id": pix.id, "acao": decidido["acao"], "variaveis": (meio - comeco) / 1e6, "decisao": (fim - meio) / 1e6}
        )

        if decidido["acao"] != "bloquear":
            estado.registrar(pix)

    return pl.DataFrame(
        linhas, schema={"id": pl.Int64, "acao": pl.String, "variaveis": pl.Float64, "decisao": pl.Float64}
    )


def medir_pela_api(app: FastAPI, os_eventos: Sequence[Evento]) -> pl.DataFrame:
    """A ida e volta de cada Pix pela API, num servidor de verdade: `id`, `acao`, `servico` e `ida_e_volta`.

    O `servico` é o tempo que o próprio serviço mede, do pedido validado à resposta; a `ida_e_volta`, o que o
    banco espera, com o HTTP e o JSON. O Pix segurado é confirmado no mesmo instante.
    """
    linhas = []

    with _servidor(app) as porta:
        conexao = http.client.HTTPConnection("127.0.0.1", porta, timeout=30)

        for evento in os_eventos:
            if evento.tipo == "aparelho":
                _pedir(
                    conexao,
                    "/aparelhos",
                    {campo: evento.dados[campo] for campo in ("cliente", "aparelho")} | {"momento": evento.momento},
                )
                continue

            if evento.tipo == "contestacao":
                _pedir(conexao, "/contestacoes", {"transacao": evento.dados["transacao"], "momento": evento.momento})
                continue

            comeco = time.perf_counter_ns()
            decisao = _pedir(conexao, "/pix", _do_pix(evento.dados))
            ida_e_volta = (time.perf_counter_ns() - comeco) / 1e6
            linhas.append(
                {
                    "id": decisao["id"],
                    "acao": decisao["acao"],
                    "servico": decisao["milissegundos"],
                    "ida_e_volta": ida_e_volta,
                }
            )

            if decisao["acao"] == "segurar":
                _pedir(conexao, f"/pix/{decisao['id']}/confirmacao", {"confirmado": True, "momento": evento.momento})

        conexao.close()

    return pl.DataFrame(
        linhas, schema={"id": pl.Int64, "acao": pl.String, "servico": pl.Float64, "ida_e_volta": pl.Float64}
    )


def resumir(medidas: dict[str, pl.Series]) -> pl.DataFrame:
    """Uma linha por medida, em milissegundos: quantos Pix, a mediana, os percentis 95 e 99, e o máximo."""
    linhas = []

    for nome, tempos in medidas.items():
        valores = tempos.to_numpy()
        linhas.append(
            {
                "medida": nome,
                "pix": len(valores),
                **{f"p{percentil}": float(np.percentile(valores, percentil)) for percentil in PERCENTIS},
                "maximo": float(valores.max()),
            }
        )

    return pl.DataFrame(linhas)


def _seguidos(estado: Estado, os_eventos: Sequence[Evento]) -> Iterator[PixNovo]:
    """Conta os aparelhos e as contestações no estado, e entrega cada Pix para ser cronometrado."""
    for evento in os_eventos:
        if evento.tipo == "aparelho":
            estado.aparelho_novo(evento.dados["cliente"], evento.momento, evento.dados["aparelho"])
        elif evento.tipo == "contestacao":
            if estado.conhece(evento.dados["transacao"]):
                estado.contestacao(evento.dados["transacao"], evento.momento)
        else:
            yield PixNovo(**{campo: evento.dados[campo] for campo in PixNovo.__dataclass_fields__})


def _do_pix(linha: dict[str, Any]) -> dict[str, Any]:
    return {campo: linha[campo] for campo in PixNovo.__dataclass_fields__}


def _pedir(conexao: http.client.HTTPConnection, caminho: str, corpo: dict[str, Any]) -> dict[str, Any]:
    conexao.request("POST", caminho, json.dumps(corpo, default=_em_json), {"Content-Type": "application/json"})
    resposta = conexao.getresponse()
    conteudo = resposta.read()

    if resposta.status == 404 and caminho == "/contestacoes":
        # A contestação de um Pix que o serviço bloqueou: o Pix não saiu, e não há o que contestar.
        return {}

    if resposta.status >= 300:
        raise RuntimeError(f"O serviço respondeu {resposta.status} a {caminho}: {conteudo.decode()}")

    return json.loads(conteudo) if conteudo else {}


def _em_json(valor: object) -> str:
    if isinstance(valor, datetime):
        return valor.isoformat()

    raise TypeError(f"Não sei mandar {type(valor).__name__} em JSON.")


@contextmanager
def _servidor(app: FastAPI) -> Iterator[int]:
    """O serviço num servidor uvicorn de verdade, numa porta livre da própria máquina, enquanto durar o bloco."""
    servidor = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning"))
    fio = threading.Thread(target=servidor.run, daemon=True)
    fio.start()

    try:
        while not servidor.started:
            time.sleep(0.01)

        yield int(servidor.servers[0].sockets[0].getsockname()[1])
    finally:
        servidor.should_exit = True
        fio.join(timeout=10)
