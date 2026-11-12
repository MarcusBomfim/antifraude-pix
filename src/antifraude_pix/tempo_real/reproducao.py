"""A reprodução: os eventos do banco mandados à API como se acontecessem agora, mais depressa (ADR-0024).

O painel só mostra alguma coisa com Pix chegando. A reprodução faz o papel do
banco e dos clientes: manda à API os eventos de um período, na ordem do tempo
e no ritmo em que aconteceram, multiplicado pela velocidade. Com a velocidade
600, uma hora do banco passa em seis segundos. Sem velocidade, vão o mais
depressa que a API responde.

O Pix segurado espera a resposta do cliente, e quem responde é a reprodução:
- o cliente honesto confirma;
- a vítima de golpe recusa na parte dos golpes que a confirmação para
  (`Custos.confirmacao_para`), sorteada com a semente.

Para saber quem é vítima, a reprodução lê a verdade do simulador. Ela faz o
papel do mundo, e não do banco: a API nunca vê a verdade.
"""

import http.client
import random
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, fields
from datetime import date
from urllib.parse import urlsplit

from antifraude_pix.decisao.custo import Custos
from antifraude_pix.tempo_real.latencia import Evento, do_pix, pedir

ACOES = {"aprovar": "aprovados", "segurar": "segurados", "bloquear": "bloqueados"}


@dataclass
class Placar:
    """O que a reprodução mandou e o que a API respondeu."""

    pix: int = 0
    aprovados: int = 0
    segurados: int = 0
    bloqueados: int = 0
    confirmados: int = 0
    recusados: int = 0
    aparelhos: int = 0
    contestacoes: int = 0

    def somar(self, outro: "Placar") -> None:
        for campo in fields(self):
            setattr(self, campo.name, getattr(self, campo.name) + getattr(outro, campo.name))


def reproduzir(
    os_eventos: Sequence[Evento],
    endereco: str,
    *,
    golpes: frozenset[int],
    velocidade: float | None = None,
    para: float = Custos().confirmacao_para,
    semente: int = 0,
    ao_fim_do_dia: Callable[[date, Placar], None] | None = None,
    relogio: Callable[[], float] = time.monotonic,
    dormir: Callable[[float], None] = time.sleep,
) -> Placar:
    """Manda os eventos à API do endereço, no ritmo deles vezes a velocidade, e devolve o placar.

    `golpes` são os ids dos Pix de golpe, que a verdade conhece. A cada dia que vira, `ao_fim_do_dia` recebe o
    placar do dia que acabou.
    """
    if velocidade is not None and velocidade <= 0:
        raise ValueError("A velocidade multiplica o ritmo do banco: tem de ser maior que zero.")

    url = urlsplit(endereco)

    if url.scheme != "http" or url.hostname is None:
        raise ValueError(f"O endereço da API tem de ser http://máquina:porta, e veio {endereco}.")

    conexao = http.client.HTTPConnection(url.hostname, url.port or 80, timeout=60)
    sorteio = random.Random(semente)
    placar, do_dia = Placar(), Placar()
    dia: date | None = None
    comeco = relogio()

    try:
        for evento in os_eventos:
            if velocidade is not None:
                atraso = (evento.momento - os_eventos[0].momento).total_seconds() / velocidade - (relogio() - comeco)

                if atraso > 0:
                    dormir(atraso)

            if dia is not None and evento.momento.date() != dia:
                _virar(dia, do_dia, placar, ao_fim_do_dia)
                do_dia = Placar()

            dia = evento.momento.date()
            _mandar(conexao, evento, do_dia, golpes=golpes, para=para, sorteio=sorteio)
    finally:
        conexao.close()

    if dia is not None:
        _virar(dia, do_dia, placar, ao_fim_do_dia)

    return placar


def _mandar(
    conexao: http.client.HTTPConnection,
    evento: Evento,
    placar: Placar,
    *,
    golpes: frozenset[int],
    para: float,
    sorteio: random.Random,
) -> None:
    """Manda um evento, e, se o Pix for segurado, a resposta do cliente."""
    if evento.tipo == "aparelho":
        corpo = {"cliente": evento.dados["cliente"], "aparelho": evento.dados["aparelho"], "momento": evento.momento}
        pedir(conexao, "/aparelhos", corpo)
        placar.aparelhos += 1
        return

    if evento.tipo == "contestacao":
        pedir(conexao, "/contestacoes", {"transacao": evento.dados["transacao"], "momento": evento.momento})
        placar.contestacoes += 1
        return

    decisao = pedir(conexao, "/pix", do_pix(evento.dados))
    placar.pix += 1
    setattr(placar, ACOES[decisao["acao"]], getattr(placar, ACOES[decisao["acao"]]) + 1)

    if decisao["acao"] == "segurar":
        # O sorteio só corre para a vítima: a sequência dele não depende de quantos honestos foram segurados.
        recusa = decisao["id"] in golpes and sorteio.random() < para
        pedir(conexao, f"/pix/{decisao['id']}/confirmacao", {"confirmado": not recusa, "momento": evento.momento})

        if recusa:
            placar.recusados += 1
        else:
            placar.confirmados += 1


def _virar(dia: date, do_dia: Placar, placar: Placar, ao_fim_do_dia: Callable[[date, Placar], None] | None) -> None:
    placar.somar(do_dia)

    if ao_fim_do_dia is not None:
        ao_fim_do_dia(dia, do_dia)
