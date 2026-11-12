import json
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import pytest

from antifraude_pix.dados.banco import VisaoDoBanco
from antifraude_pix.tempo_real.estado import Estado
from antifraude_pix.tempo_real.latencia import Evento, eventos, servidor
from antifraude_pix.tempo_real.reproducao import Placar, reproduzir
from antifraude_pix.tempo_real.servico import criar
from tests.apoio import simulacao_media

CORTE = date(2025, 9, 1)
FUSO = ZoneInfo("America/Sao_Paulo")
INICIO = datetime(2025, 9, 1, tzinfo=FUSO)


@pytest.fixture(scope="module")
def visao() -> VisaoDoBanco:
    return VisaoDoBanco.da(simulacao_media())


@pytest.fixture(scope="module")
def dois_dias(visao: VisaoDoBanco) -> list[Evento]:
    return eventos(visao, INICIO, INICIO + timedelta(days=2))


@dataclass
class _SempreIgual:
    acao: str

    def decidir_um(self, variaveis: dict[str, Any]) -> dict[str, Any]:
        del variaveis
        return {"pontuacao": 0.0, "probabilidade": 0.5, "acao": self.acao, "explicacao": "De mentira."}


def _api(visao: VisaoDoBanco, acao: str) -> Any:
    return servidor(criar(Estado.da_historia(visao.ate(CORTE)), _SempreIgual(acao), "teste"))


def _saude(porta: int) -> dict[str, Any]:
    with urllib.request.urlopen(f"http://127.0.0.1:{porta}/saude", timeout=10) as resposta:
        saude: dict[str, Any] = json.loads(resposta.read())

    return saude


def test_a_reproducao_manda_cada_evento_e_o_cliente_responde_ao_segurado(
    visao: VisaoDoBanco, dois_dias: list[Evento]
) -> None:
    # Tudo é segurado, e um Pix a cada dez é de golpe: a vítima sempre recusa, e o honesto sempre confirma.
    pix = [evento.dados["id"] for evento in dois_dias if evento.tipo == "pix"]
    golpes = frozenset(pix[::10])

    with _api(visao, "segurar") as porta:
        antes = _saude(porta)["pix_no_estado"]
        placar = reproduzir(dois_dias, f"http://127.0.0.1:{porta}", golpes=golpes, para=1.0)
        depois = _saude(porta)

    # A API ouviu o cliente: o Pix confirmado saiu e entrou na história, o recusado não, e ninguém ficou esperando.
    assert depois["pix_no_estado"] == antes + len(pix) - len(golpes)
    assert depois["segurados"] == 0
    assert placar == Placar(
        pix=len(pix),
        segurados=len(pix),
        confirmados=len(pix) - len(golpes),
        recusados=len(golpes),
        aparelhos=sum(evento.tipo == "aparelho" for evento in dois_dias),
        contestacoes=sum(evento.tipo == "contestacao" for evento in dois_dias),
    )
    assert placar.pix > 1_000
    assert placar.aparelhos > 0


def test_a_parte_dos_golpes_que_a_confirmacao_para_sai_do_sorteio_com_a_semente(
    visao: VisaoDoBanco, dois_dias: list[Evento]
) -> None:
    golpes = frozenset(evento.dados["id"] for evento in dois_dias if evento.tipo == "pix")
    recusados = []

    for semente in (7, 7, 8):
        with _api(visao, "segurar") as porta:
            placar = reproduzir(dois_dias, f"http://127.0.0.1:{porta}", golpes=golpes, para=0.7, semente=semente)
            recusados.append(placar.recusados)

    assert recusados[0] == recusados[1] != recusados[2]
    assert recusados[0] / placar.pix == pytest.approx(0.7, abs=0.05)


def test_o_ritmo_e_o_do_banco_vezes_a_velocidade(visao: VisaoDoBanco) -> None:
    # O relógio de mentira só anda quando a reprodução dorme: o que ela dorme é o ritmo do banco, sem a API.
    cliente = int(visao.clientes["id"][0])
    aparelhos = [
        Evento("aparelho", INICIO + timedelta(seconds=segundos), {"cliente": cliente, "aparelho": 900 + segundos})
        for segundos in (0, 60, 180, 180)
    ]
    agora = [0.0]
    dormidas: list[float] = []

    def dormir(segundos: float) -> None:
        dormidas.append(segundos)
        agora[0] += segundos

    with _api(visao, "aprovar") as porta:
        placar = reproduzir(
            aparelhos,
            f"http://127.0.0.1:{porta}",
            golpes=frozenset(),
            velocidade=120,
            relogio=lambda: agora[0],
            dormir=dormir,
        )

    # Com 120, um minuto do banco passa em meio segundo. Com 60, o teste não veria a velocidade trocada com a
    # hora: 60 / 3.600 é o mesmo que 1 / 60.
    assert placar.aparelhos == 4
    assert dormidas == pytest.approx([0.5, 1.0])


def test_o_placar_de_cada_dia_soma_o_do_periodo(visao: VisaoDoBanco, dois_dias: list[Evento]) -> None:
    por_dia: dict[date, Placar] = {}

    with _api(visao, "aprovar") as porta:
        placar = reproduzir(
            dois_dias, f"http://127.0.0.1:{porta}", golpes=frozenset(), ao_fim_do_dia=por_dia.__setitem__
        )

    soma = Placar()

    for do_dia in por_dia.values():
        soma.somar(do_dia)

    assert list(por_dia) == [date(2025, 9, 1), date(2025, 9, 2)]
    assert all(do_dia.pix > 0 for do_dia in por_dia.values())
    assert soma == placar
    assert placar.aprovados == placar.pix


@pytest.mark.parametrize(
    ("endereco", "velocidade", "motivo"),
    [
        ("http://127.0.0.1:8000", 0, "maior que zero"),
        ("127.0.0.1:8000", None, "http://máquina:porta"),
        ("https://127.0.0.1:8000", None, "http://máquina:porta"),
    ],
)
def test_o_que_nao_da_para_reproduzir(endereco: str, velocidade: float | None, motivo: str) -> None:
    with pytest.raises(ValueError, match=motivo):
        reproduzir([], endereco, golpes=frozenset(), velocidade=velocidade)
