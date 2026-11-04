"""As figuras do README: o retrato de cada golpe, desenhado com o matplotlib.

As cores seguem o golpe em todas as figuras — o falso parente é sempre azul —,
e o Pix legítimo é sempre o cinza neutro. Cada série tem o nome escrito ao
lado dela ou no título do quadro: a cor ajuda, mas nunca está sozinha.
"""

from pathlib import Path
from typing import Any

import matplotlib as mpl
import polars as pl
from matplotlib.figure import Figure
from matplotlib.layout_engine import ConstrainedLayoutEngine
from matplotlib.ticker import FuncFormatter, MultipleLocator

from antifraude_pix.analise import retrato
from antifraude_pix.simulacao.golpes import GOLPES, NOMES, Golpe
from antifraude_pix.simulacao.simulador import Simulacao

SUPERFICIE = "#fcfcfb"
TINTA = "#0b0b0b"
TINTA_SECUNDARIA = "#52514e"
MUDO = "#898781"
GRADE = "#e1e0d9"
EIXO = "#c3c2b7"

CORES: dict[str, str] = {
    retrato.LEGITIMO: MUDO,
    NOMES[Golpe.FALSO_PARENTE]: "#2a78d6",
    NOMES[Golpe.FALSA_CENTRAL]: "#eb6834",
    NOMES[Golpe.INVASAO_DE_CONTA]: "#1baf7a",
    NOMES[Golpe.VENDA_FALSA]: "#eda100",
    "pessoa": MUDO,
    "comércio": "#6250d6",
    "conta laranja": "#e34948",
}

_ESTILO: dict[Any, Any] = {
    "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"],
    "font.size": 10,
    "axes.edgecolor": EIXO,
    "axes.labelcolor": TINTA_SECUNDARIA,
    "axes.titlecolor": TINTA,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.facecolor": SUPERFICIE,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": GRADE,
    "grid.linewidth": 0.6,
    "xtick.color": MUDO,
    "ytick.color": MUDO,
    "xtick.labelcolor": TINTA_SECUNDARIA,
    "ytick.labelcolor": TINTA_SECUNDARIA,
    "figure.facecolor": SUPERFICIE,
    "savefig.facecolor": SUPERFICIE,
}


def _porcentagem(valor: float, _: object = None) -> str:
    return f"{valor:.0%}".replace(".", ",")


def _reais(valor: float, _: object = None) -> str:
    return "R$ " + f"{valor:,.0f}".replace(",", ".")


def _espaco_para_o_titulo(topo: float) -> ConstrainedLayoutEngine:
    """O arranjo automático dos quadros, deixando o alto da figura para o título e o subtítulo."""
    return ConstrainedLayoutEngine(rect=(0, 0, 1, topo))


def _titulo(figura: Figure, titulo: str, subtitulo: str) -> None:
    figura.text(0.012, 0.975, titulo, fontsize=14, fontweight="bold", color=TINTA, va="top")
    figura.text(0.012, 0.915, subtitulo, fontsize=10, color=TINTA_SECUNDARIA, va="top")


def horas(simulacao: Simulacao) -> Figure:
    """A hora de cada golpe, num quadro para cada um, contra a hora do Pix legítimo."""
    tabela = retrato.por_hora(simulacao)
    legitimo = tabela.filter(pl.col("grupo") == retrato.LEGITIMO)

    with mpl.rc_context(_ESTILO):
        figura = Figure(figsize=(10, 6.2), layout=_espaco_para_o_titulo(0.86))
        quadros = figura.subplots(2, 2, sharex=True, sharey=True)

        for quadro, golpe in zip(quadros.ravel(), GOLPES, strict=True):
            nome = NOMES[golpe]
            dele = tabela.filter(pl.col("grupo") == nome)
            quadro.bar(dele["hora"], dele["parte"], width=0.72, color=CORES[nome], label=nome)
            quadro.plot(legitimo["hora"], legitimo["parte"], color=MUDO, linewidth=2, label=retrato.LEGITIMO)
            quadro.set_title(nome[0].upper() + nome[1:])
            quadro.set_xticks([0, 6, 12, 18, 23], ["0h", "6h", "12h", "18h", "23h"])
            quadro.yaxis.set_major_locator(MultipleLocator(0.05))
            quadro.yaxis.set_major_formatter(FuncFormatter(_porcentagem))
            quadro.grid(axis="x", visible=False)

        quadros[0][0].legend(
            handles=[quadros[0][0].lines[0]], labels=[retrato.LEGITIMO], frameon=False, loc="upper left", fontsize=9
        )
        _titulo(
            figura,
            "A hora do golpe",
            "A parte dos Pix de cada golpe em cada hora do dia; a linha cinza é a hora do Pix legítimo.",
        )

    return figura


def valores(simulacao: Simulacao) -> Figure:
    """O valor de cada grupo: a faixa do meio, a mediana escrita, e os extremos de cada lado."""
    tabela = retrato.faixas_de_valor(simulacao)

    with mpl.rc_context(_ESTILO):
        figura = Figure(figsize=(10, 4.2), layout=_espaco_para_o_titulo(0.8))
        quadro = figura.subplots()
        nomes = tabela["grupo"].to_list()

        for posicao, linha in enumerate(tabela.iter_rows(named=True)):
            cor = CORES[linha["grupo"]]
            y = len(nomes) - 1 - posicao
            quadro.plot([linha["p10"], linha["p90"]], [y, y], color=cor, linewidth=1.5, solid_capstyle="round")
            quadro.plot([linha["p25"], linha["p75"]], [y, y], color=cor, linewidth=7, solid_capstyle="round")
            quadro.plot(
                [linha["mediana"]], [y], "o", color=cor, markersize=9, markeredgecolor=SUPERFICIE, markeredgewidth=2
            )
            quadro.text(linha["p90"] * 1.25, y, _reais(linha["mediana"]), va="center", color=TINTA, fontsize=10)

        quadro.set_xscale("log")
        quadro.set_xlim(5, 40_000)
        quadro.set_yticks(range(len(nomes)), [nome[0].upper() + nome[1:] for nome in reversed(nomes)])
        quadro.xaxis.set_major_formatter(FuncFormatter(_reais))
        quadro.grid(axis="y", visible=False)
        quadro.tick_params(axis="y", length=0)
        quadro.spines["left"].set_visible(False)
        _titulo(
            figura,
            "O tamanho do Pix: o golpe pede mais",
            "A faixa grossa vai do quartil de baixo ao de cima; a fina, do décimo ao nonagésimo percentil. "
            "Ao lado, a mediana.",
        )

    return figura


def contestacao(simulacao: Simulacao) -> Figure:
    """A parte dos ataques já contestada, dia a dia depois do último Pix.

    O que fica abaixo de 100% nunca é contestado.
    """
    tabela = retrato.demora_da_contestacao(simulacao)

    with mpl.rc_context(_ESTILO):
        figura = Figure(figsize=(10, 4.8), layout=_espaco_para_o_titulo(0.84))
        quadro = figura.subplots()

        finais = {}

        for golpe in GOLPES:
            nome = NOMES[golpe]
            dele = tabela.filter(pl.col("golpe") == golpe.value)
            quadro.plot(dele["dias"], dele["contestados"], color=CORES[nome], linewidth=2)
            finais[nome] = float(dele["contestados"][-1])

        # O nome na ponta de cada linha; os que terminam perto uns dos outros se afastam, para não se cobrir.
        for nome, altura in _afastados(finais, distancia=0.05).items():
            quadro.text(30.6, altura, f"{nome}: {_porcentagem(finais[nome])}", va="center", color=TINTA, fontsize=10)

        quadro.set_xlim(0, 30)
        quadro.set_ylim(0, 1.02)
        quadro.set_xticks([0, 7, 14, 21, 30], ["0", "7", "14", "21", "30 dias"])
        quadro.yaxis.set_major_formatter(FuncFormatter(_porcentagem))
        _titulo(
            figura,
            "Quando o banco fica sabendo",
            "A parte dos ataques de cada golpe já contestada, dias depois do último Pix. "
            "O que falta para 100% nunca é contestado.",
        )

    return figura


def recebedores(simulacao: Simulacao) -> Figure:
    """Quantos pagadores novos cada recebedor tem na semana mais movimentada, por tipo de conta."""
    tabela = retrato.pagadores_novos_por_recebedor(simulacao)
    tipos = ("pessoa", "comércio", "conta laranja")
    largura = 0.26

    with mpl.rc_context(_ESTILO):
        figura = Figure(figsize=(10, 4.6), layout=_espaco_para_o_titulo(0.82))
        quadro = figura.subplots()

        for indice, tipo in enumerate(tipos):
            partes = {faixa: parte for _, faixa, parte in tabela.filter(pl.col("tipo") == tipo).iter_rows()}
            posicoes = [faixa + (indice - 1) * (largura + 0.02) for faixa in range(len(retrato.FAIXAS_DE_PAGADORES))]
            alturas = [partes.get(faixa, 0.0) for faixa in retrato.FAIXAS_DE_PAGADORES]
            quadro.bar(posicoes, alturas, width=largura, color=CORES[tipo], label=tipo[0].upper() + tipo[1:])

        quadro.set_xticks(range(len(retrato.FAIXAS_DE_PAGADORES)), list(retrato.FAIXAS_DE_PAGADORES))
        quadro.set_xlabel("Clientes que pagaram a conta pela primeira vez, na semana mais movimentada dela")
        quadro.yaxis.set_major_formatter(FuncFormatter(_porcentagem))
        quadro.grid(axis="x", visible=False)
        quadro.legend(frameon=False, loc="upper right")
        _titulo(
            figura,
            "A laranja recebe de desconhecidos",
            "A parte das contas de cada tipo em cada faixa. "
            "A pessoa comum recebe de quem já a pagava; o comércio, de muita gente.",
        )

    return figura


def _afastados(alturas: dict[str, float], distancia: float) -> dict[str, float]:
    """As alturas dos rótulos, de baixo para cima, cada uma ao menos a distância dada da anterior."""
    resultado: dict[str, float] = {}
    anterior = float("-inf")

    for nome, altura in sorted(alturas.items(), key=lambda item: item[1]):
        anterior = max(altura, anterior + distancia)
        resultado[nome] = anterior

    return resultado


FIGURAS = {"horas": horas, "valores": valores, "contestacao": contestacao, "recebedores": recebedores}


def gerar(simulacao: Simulacao, pasta: Path) -> list[Path]:
    """Desenha as figuras e as grava em PNG na pasta."""
    pasta.mkdir(parents=True, exist_ok=True)
    gravadas = []

    for nome, desenhar in FIGURAS.items():
        destino = pasta / f"{nome}.png"
        desenhar(simulacao).savefig(destino, dpi=150)
        gravadas.append(destino)

    return gravadas
