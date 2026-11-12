"""A implantação conferida contra o código: o painel, o coletor, a imagem e o compose (ADR-0025).

Os arquivos de configuração não rodam nos testes, e por isso envelhecem calados:
o painel consulta uma métrica que mudou de nome e mostra um gráfico vazio; o
coletor deixa passar um atributo novo. Estes testes leem os arquivos e os
comparam com o que o código grava.
"""

import json
import re
from pathlib import Path

import pytest
from opentelemetry.sdk.metrics import Counter, Histogram, MeterProvider, UpDownCounter

from antifraude_pix import cli
from antifraude_pix.tempo_real.telemetria import ATRIBUTOS, Telemetria

RAIZ = Path(__file__).parent.parent
IMPLANTACAO = RAIZ / "implantacao"
UNIDADES = {"ms": "_milliseconds", "1": ""}
"""O sufixo que o coletor põe no nome da métrica pela unidade; a unidade entre chaves não vira sufixo."""


def _no_prometheus() -> dict[str, set[str]]:
    """Os nomes que cada instrumento da telemetria ganha no Prometheus, pelas regras do coletor."""
    telemetria = Telemetria.de(medidor=MeterProvider())
    nomes: dict[str, set[str]] = {}

    for instrumento in vars(telemetria).values():
        if not isinstance(instrumento, Counter | UpDownCounter | Histogram):
            continue

        base = instrumento.name.replace(".", "_") + UNIDADES.get(instrumento.unit or "", "")

        if isinstance(instrumento, Counter):
            nomes[instrumento.name] = {f"{base}_total"}
        elif isinstance(instrumento, UpDownCounter):
            nomes[instrumento.name] = {base}
        else:
            nomes[instrumento.name] = {f"{base}_bucket", f"{base}_sum", f"{base}_count"}

    return nomes


def _consultas() -> list[str]:
    painel = json.loads((IMPLANTACAO / "grafana" / "painel.json").read_text(encoding="utf-8"))

    return [alvo["expr"] for quadro in painel["panels"] for alvo in quadro["targets"]]


def test_o_painel_consulta_todas_as_metricas_do_servico_e_so_elas() -> None:
    nomes = _no_prometheus()
    conhecidos = set().union(*nomes.values())
    usados = {nome for consulta in _consultas() for nome in re.findall(r"\bantifraude_[a-z_]+", consulta)}

    assert len(nomes) == 8
    assert usados <= conhecidos
    assert all(usados & do_instrumento for do_instrumento in nomes.values())


def test_o_painel_filtra_e_agrupa_so_pelos_atributos_da_telemetria() -> None:
    rotulos = {
        rotulo
        for consulta in _consultas()
        for rotulo in re.findall(r"([a-z_]+)\s*(?:=|!=)\s*\"", consulta)
        + [nome.strip() for grupo in re.findall(r"by \(([^)]*)\)", consulta) for nome in grupo.split(",")]
    }

    assert rotulos
    assert rotulos <= {atributo.replace(".", "_") for atributo in ATRIBUTOS} | {"le"}


def test_o_coletor_apaga_dos_tracos_o_que_nao_esta_na_lista_da_telemetria() -> None:
    coletor = (IMPLANTACAO / "coletor.yaml").read_text(encoding="utf-8")
    (lista,) = re.findall(r"keep_keys\(span\.attributes, \[([^\]]*)\]\)", coletor)

    assert set(re.findall(r'"([^"]+)"', lista)) == ATRIBUTOS
    assert "processors: [memory_limiter, transform/so_a_lista, batch]" in coletor


def _comando_do_dockerfile() -> list[str]:
    (cmd,) = re.findall(r"^CMD (\[.*\])$", (RAIZ / "Dockerfile").read_text(encoding="utf-8"), re.MULTILINE)
    comando: list[str] = json.loads(cmd)

    return comando


def _comando_do_compose() -> list[str]:
    (linha,) = re.findall(
        r"^\s+command: (\[\"servir\".*\])$", (RAIZ / "compose.yaml").read_text(encoding="utf-8"), re.M
    )
    # O compose troca ${VARIAVEL:-padrão} pelo padrão, se a variável não existe.
    comando: list[str] = json.loads(re.sub(r"\$\{[A-Z]+:-([^}]*)\}", r"\1", linha))

    return comando


@pytest.mark.parametrize("comando", [_comando_do_dockerfile(), _comando_do_compose()], ids=["Dockerfile", "compose"])
def test_a_imagem_serve_o_pacote_para_fora_do_conteiner(comando: list[str]) -> None:
    pedido = cli._leitor().parse_args(comando)

    assert pedido.comando == "servir"
    assert pedido.pasta == Path("/dados")
    assert pedido.pacote == Path("/dados/campeao")
    assert pedido.host == "0.0.0.0"
    assert pedido.porta == 8000


def test_cada_imagem_e_fixada_pelo_digest() -> None:
    imagens = re.findall(r"^FROM (\S+)", (RAIZ / "Dockerfile").read_text(encoding="utf-8"), re.MULTILINE)
    imagens += re.findall(r"^\s+image: (\S+)", (RAIZ / "compose.yaml").read_text(encoding="utf-8"), re.MULTILINE)
    de_fora = [imagem for imagem in imagens if not imagem.startswith("antifraude-pix")]

    assert len(de_fora) == 7
    assert all(re.fullmatch(r"[\w./-]+:[\w.-]+@sha256:[0-9a-f]{64}", imagem) for imagem in de_fora)
