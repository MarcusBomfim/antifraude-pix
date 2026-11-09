"""As figuras do README: o retrato de cada golpe, desenhado com o matplotlib.

As cores seguem o golpe em todas as figuras — o falso parente é sempre azul —,
e o Pix legítimo é sempre o cinza neutro. Cada série tem o nome escrito ao
lado dela ou no título do quadro: a cor ajuda, mas nunca está sozinha. Na
tabela da força das variáveis, o azul vai do claro ao escuro, e o número está
escrito em cada casa.

Nas figuras dos modelos, as regras de hoje ficam no cinza, como referência; a
regressão logística é violeta, e o LightGBM, verde.

Nas figuras da decisão, as saídas têm as cores de estado, sempre com o nome
escrito: segurar é o amarelo de alerta, e bloquear, o vermelho. O modelo com a
idade, que saiu, fica no cinza, ao lado do verde do LightGBM sem ela.
"""

from collections.abc import Callable
from datetime import datetime, timedelta
from functools import cached_property
from pathlib import Path
from typing import Any, Literal

import matplotlib as mpl
import numpy as np
import polars as pl
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.figure import Figure
from matplotlib.layout_engine import ConstrainedLayoutEngine
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter, MultipleLocator

from antifraude_pix.analise import retrato, separacao
from antifraude_pix.avaliacao import metricas
from antifraude_pix.avaliacao.decisoes import NOMES as NOMES_DAS_POLITICAS
from antifraude_pix.avaliacao.decisoes import balanco, com_a_politica
from antifraude_pix.avaliacao.justica import por_idade
from antifraude_pix.avaliacao.no_tempo import Rotulo, avaliar
from antifraude_pix.cartoes.base import ler as ler_os_cartoes
from antifraude_pix.cartoes.processo import MODELOS as MODELOS_DOS_CARTOES
from antifraude_pix.cartoes.processo import NOMES as NOMES_NOS_CARTOES
from antifraude_pix.cartoes.processo import avaliar as avaliar_os_cartoes
from antifraude_pix.dados.banco import VisaoDoBanco
from antifraude_pix.decisao.custo import Custos, decidir
from antifraude_pix.modelos.rotulo import Dobra, dobras, primeiro_corte
from antifraude_pix.modelos.treino import MODELOS
from antifraude_pix.modelos.treino import NOMES as NOMES_DOS_MODELOS
from antifraude_pix.simulacao.golpes import GOLPES, NOMES, Golpe
from antifraude_pix.simulacao.simulador import Simulacao
from antifraude_pix.variaveis.calculo import calcular
from antifraude_pix.variaveis.catalogo import CATALOGO

SEGURAR = "#fab219"
BLOQUEAR = "#d03b3b"
SEM_A_IDADE = "#008300"

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

CORES_DOS_MODELOS: dict[str, str] = {"regras": MUDO, "valor": MUDO, "logistica": "#6250d6", "lightgbm": "#008300"}
"""O ponto de partida no cinza: as regras de hoje no Pix, e o valor sozinho nos cartões."""
MODELOS_EM_ORDEM: tuple[str, ...] = tuple(MODELOS)

AZUIS = (
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
    "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
)  # fmt: skip
"""A escala do azul, do claro ao escuro: a força que vai de 0,5, não separar nada, a 1, separar tudo."""

_COM_ACENTO = {
    "pix": "Pix",
    "pj": "é empresa",
    "ultima": "última",
    "ultimas": "últimas",
    "ultimo": "último",
    "contestacoes": "contestações",
}
"""Os nomes das variáveis são identificadores, sem acento; na figura, cada um se escreve como se lê."""

_BRANCO_A_PARTIR_DE = 7 / 12
"""Do sétimo azul em diante, o número da casa é branco: no claro, a tinta escura se lê melhor."""

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


# O alto de toda figura, em polegadas: o título, o subtítulo, e onde começam os quadros.
_TITULO = 0.12
_SUBTITULO = 0.44
_QUADROS = 0.78


def _figura(altura: float) -> Figure:
    """Uma figura da largura do README, com o arranjo automático dos quadros abaixo do título e do subtítulo."""
    return Figure(figsize=(10, altura), layout=ConstrainedLayoutEngine(rect=(0, 0, 1, 1 - _QUADROS / altura)))


def _titulo(figura: Figure, titulo: str, subtitulo: str) -> None:
    altura = figura.get_figheight()
    figura.text(0.012, 1 - _TITULO / altura, titulo, fontsize=14, fontweight="bold", color=TINTA, va="top")
    figura.text(0.012, 1 - _SUBTITULO / altura, subtitulo, fontsize=10, color=TINTA_SECUNDARIA, va="top")


def horas(simulacao: Simulacao) -> Figure:
    """A hora de cada golpe, num quadro para cada um, contra a hora do Pix legítimo."""
    tabela = retrato.por_hora(simulacao)
    legitimo = tabela.filter(pl.col("grupo") == retrato.LEGITIMO)

    with mpl.rc_context(_ESTILO):
        figura = _figura(6.2)
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
        figura = _figura(4.2)
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
        figura = _figura(4.8)
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
        figura = _figura(4.6)
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


def forca_das_variaveis(simulacao: Simulacao, variaveis: pl.DataFrame) -> Figure:
    """A força de cada variável, sozinha, contra cada golpe: uma tabela de calor, com os grupos do catálogo."""
    tabela = separacao.forcas(simulacao, variaveis)
    forca = {(variavel, golpe): valor for variavel, golpe, valor in tabela.iter_rows()}
    escala = LinearSegmentedColormap.from_list("azuis", AZUIS)

    # Uma linha para o nome de cada grupo e uma para cada variável dele.
    linhas: list[tuple[str, bool]] = []

    for variavel in CATALOGO:
        if variavel.nome not in separacao.ANALISADAS:
            continue

        titulo = variavel.grupo.value[0].upper() + variavel.grupo.value[1:]

        if (titulo, True) not in linhas:
            linhas.append((titulo, True))

        linhas.append((variavel.nome, False))

    with mpl.rc_context(_ESTILO):
        figura = _figura(10.5)
        quadro = figura.subplots()

        for y, (nome, e_grupo) in enumerate(linhas):
            if e_grupo:
                continue

            for x, golpe in enumerate(GOLPES):
                valor = forca[nome, golpe.value]
                tom = (valor - 0.5) / 0.5
                quadro.add_patch(
                    Rectangle((x - 0.5, y - 0.5), 1, 1, facecolor=escala(tom), edgecolor=SUPERFICIE, linewidth=2)
                )
                quadro.text(
                    x,
                    y,
                    f"{valor:.2f}".replace(".", ","),
                    ha="center",
                    va="center",
                    fontsize=9,
                    color=SUPERFICIE if tom >= _BRANCO_A_PARTIR_DE else TINTA,
                )

        quadro.set_xlim(-0.5, len(GOLPES) - 0.5)
        quadro.set_ylim(len(linhas) - 0.5, -0.5)
        quadro.xaxis.tick_top()
        quadro.set_xticks(range(len(GOLPES)), [NOMES[golpe][0].upper() + NOMES[golpe][1:] for golpe in GOLPES])
        quadro.set_yticks(range(len(linhas)), ["" if e_grupo else _rotulo(nome) for nome, e_grupo in linhas])

        # O nome do grupo vai sobre as casas, na linha vazia que abre o grupo.
        for y, (nome, e_grupo) in enumerate(linhas):
            if e_grupo:
                quadro.text(-0.48, y + 0.15, nome, ha="left", va="center", fontsize=10, fontweight="bold", color=TINTA)

        quadro.tick_params(length=0)
        quadro.tick_params(axis="y", labelcolor=TINTA_SECUNDARIA)
        quadro.tick_params(axis="x", labelcolor=TINTA)
        quadro.grid(visible=False)

        for lado in quadro.spines.values():
            lado.set_visible(False)

        _titulo(
            figura,
            "O que cada variável, sozinha, separa",
            "A área sob a curva ROC de cada variável, contra o Pix legítimo: 0,5 é não separar nada; 1, separar tudo.",
        )

    return figura


def golpes_pegos(
    pontuacoes: pl.DataFrame,
    *,
    modelos: tuple[str, ...] = MODELOS_EM_ORDEM,
    nomes: dict[str, str] = NOMES_DOS_MODELOS,
    eixo: str = "A parte dos Pix segurados para confirmar, dos de mais cara de golpe para baixo",
    titulo: str = "Os golpes que cada modelo pega",
    subtitulo: str = "A parte dos golpes dos meses de teste que cada modelo pega, para cada parte dos Pix que o banco "
    "segura.",
    legenda: Literal["lower right", "center right"] = "lower right",
) -> Figure:
    """A parte dos golpes que cada modelo pega, para cada parte dos Pix que o banco segura, com o alerta marcado.

    A mesma figura serve à base de cartões, com os modelos, os nomes e os textos dela.
    """
    curva = metricas.curva_do_alerta(pontuacoes)
    alerta = metricas.ALERTA
    no_alerta = {
        linha["modelo"]: linha["recall_no_alerta"] for linha in metricas.resumo(pontuacoes).iter_rows(named=True)
    }

    with mpl.rc_context(_ESTILO):
        figura = _figura(4.8)
        quadro = figura.subplots()

        for modelo in modelos:
            dele = curva.filter(pl.col("modelo") == modelo)
            # A curva começa na origem: sem segurar nenhum Pix, nenhum golpe.
            x = [1e-5, *dele["parte_segurada"].to_list()]
            y = [0.0, *dele["recall"].to_list()]
            quadro.plot(x, y, color=CORES_DOS_MODELOS[modelo], linewidth=2, drawstyle="steps-post")
            pegos = no_alerta[modelo]
            quadro.plot(
                [alerta],
                [pegos],
                "o",
                color=CORES_DOS_MODELOS[modelo],
                markersize=8,
                markeredgecolor=SUPERFICIE,
                markeredgewidth=2,
                label=f"{nomes[modelo]}: {_porcentagem(pegos)}",
            )

        quadro.axvline(alerta, color=EIXO, linewidth=1, zorder=0)
        quadro.text(alerta * 0.92, 0.04, "o alerta de 0,5%", ha="right", color=TINTA_SECUNDARIA, fontsize=9)
        quadro.set_xscale("log")
        quadro.set_xlim(1e-4, 0.1)
        quadro.set_ylim(0, 1.02)
        quadro.set_xticks([1e-4, 1e-3, 1e-2, 0.1], ["0,01%", "0,1%", "1%", "10%"])
        quadro.set_xlabel(eixo)
        quadro.yaxis.set_major_formatter(FuncFormatter(_porcentagem))
        # A legenda traz o número do alerta, no canto que as curvas deixam vazio.
        ordem = sorted(range(len(modelos)), key=lambda indice: -no_alerta[modelos[indice]])
        marcas, rotulos = quadro.get_legend_handles_labels()
        quadro.legend(
            [marcas[indice] for indice in ordem],
            [rotulos[indice] for indice in ordem],
            title="No alerta de 0,5%:",
            title_fontsize=9,
            alignment="left",
            frameon=False,
            loc=legenda,
            fontsize=10,
        )
        _titulo(figura, titulo, subtitulo)

    return figura


def fraudes_nos_cartoes(pontuacoes: pl.DataFrame) -> Figure:
    """A parte das fraudes que cada modelo pega na base de cartões, com o mesmo alerta do Pix (ADR-0015)."""
    return golpes_pegos(
        pontuacoes,
        modelos=MODELOS_DOS_CARTOES,
        nomes=NOMES_NOS_CARTOES,
        eixo="A parte das compras seguradas para confirmar, das de mais cara de fraude para baixo",
        titulo="As fraudes que cada modelo pega, nos cartões",
        subtitulo="Na base pública da ULB, com o treino no primeiro dia, o teste no segundo, e os mesmos parâmetros "
        "do Pix.",
        # Embaixo, à direita, passa a curva do valor sozinho; no meio, à direita, não passa nada.
        legenda="center right",
    )


def golpes_por_modelo(pontuacoes: pl.DataFrame) -> Figure:
    """A parte de cada golpe que cada modelo pega no alerta: um grupo de barras por golpe."""
    tabela = metricas.por_golpe(pontuacoes)
    largura = 0.26

    with mpl.rc_context(_ESTILO):
        figura = _figura(5.0)
        quadro = figura.subplots()

        for indice, modelo in enumerate(MODELOS):
            dele = {golpe: recall for _, golpe, _, recall, _ in tabela.filter(pl.col("modelo") == modelo).iter_rows()}
            posicoes = [posicao + (indice - 1) * (largura + 0.03) for posicao in range(len(GOLPES))]
            alturas = [dele[golpe.value] for golpe in GOLPES]
            quadro.barh(
                posicoes, alturas, height=largura, color=CORES_DOS_MODELOS[modelo], label=NOMES_DOS_MODELOS[modelo]
            )

            for posicao, altura in zip(posicoes, alturas, strict=True):
                quadro.text(altura + 0.01, posicao, _porcentagem(altura), va="center", fontsize=9, color=TINTA)

        quadro.set_yticks(range(len(GOLPES)), [NOMES[golpe][0].upper() + NOMES[golpe][1:] for golpe in GOLPES])
        quadro.invert_yaxis()
        quadro.set_xlim(0, 1.1)
        quadro.xaxis.set_major_formatter(FuncFormatter(_porcentagem))
        quadro.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        quadro.grid(axis="y", visible=False)
        quadro.tick_params(axis="y", length=0)
        quadro.spines["left"].set_visible(False)
        quadro.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1), ncols=3, fontsize=9)
        _titulo(
            figura,
            "O que cada modelo pega de cada golpe",
            "A parte dos golpes de cada tipo entre os 0,5% de Pix de cada mês com mais cara de golpe.",
        )

    return figura


def calibracao(pontuacoes: pl.DataFrame) -> Figure:
    """A probabilidade prevista contra a parte que foi golpe, por faixa, para os dois modelos que aprendem."""
    faixas = (0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0)
    tabela = metricas.confiabilidade(pontuacoes, faixas).filter(pl.col("observada") > 0)

    with mpl.rc_context(_ESTILO):
        figura = _figura(5.4)
        quadro = figura.subplots()
        quadro.plot([1e-5, 1], [1e-5, 1], color=EIXO, linewidth=1, linestyle="--", zorder=0)
        quadro.text(
            0.25, 0.012, "a linha em que o previsto\né o que acontece", color=TINTA_SECUNDARIA, fontsize=9, ha="right"
        )

        for modelo in ("logistica", "lightgbm"):
            dele = tabela.filter(pl.col("modelo") == modelo)
            quadro.plot(
                dele["prevista"],
                dele["observada"],
                "o-",
                color=CORES_DOS_MODELOS[modelo],
                linewidth=2,
                markersize=8,
                markeredgecolor=SUPERFICIE,
                markeredgewidth=2,
                label=NOMES_DOS_MODELOS[modelo],
            )

        quadro.set_xscale("log")
        quadro.set_yscale("log")
        quadro.set_xlim(1e-5, 1)
        quadro.set_ylim(1e-5, 1)
        marcas, nomes = [1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0], ["0,001%", "0,01%", "0,1%", "1%", "10%", "100%"]
        quadro.set_xticks(marcas, nomes)
        quadro.set_yticks(marcas, nomes)
        quadro.set_xlabel("A probabilidade prevista, na média da faixa")
        quadro.set_ylabel("A parte que foi golpe")
        quadro.legend(frameon=False, loc="upper left", fontsize=9)
        _titulo(
            figura,
            "A probabilidade, conferida com a verdade",
            "Em cada faixa de probabilidade prevista, a parte dos Pix que foi golpe. No meio, onde o banco decide, "
            "as duas batem; nas pontas, o modelo é tímido.",
        )

    return figura


def rotulo_atrasado(do_dia: pl.DataFrame, com_a_verdade: pl.DataFrame) -> Figure:
    """A precisão média do LightGBM em cada golpe, treinado com o rótulo do dia e com a verdade."""
    lightgbm = pl.col("modelo") == "lightgbm"
    do_dia_por_golpe = {
        linha["golpe"]: linha["precisao_media"]
        for linha in metricas.por_golpe(do_dia.filter(lightgbm)).iter_rows(named=True)
    }
    verdade_por_golpe = {
        linha["golpe"]: linha["precisao_media"]
        for linha in metricas.por_golpe(com_a_verdade.filter(lightgbm)).iter_rows(named=True)
    }
    cor = CORES_DOS_MODELOS["lightgbm"]

    with mpl.rc_context(_ESTILO):
        figura = _figura(4.2)
        quadro = figura.subplots()

        for posicao, golpe in enumerate(GOLPES):
            dia, verdade = do_dia_por_golpe[golpe.value], verdade_por_golpe[golpe.value]
            quadro.plot([dia, verdade], [posicao, posicao], color=EIXO, linewidth=2, zorder=1)
            quadro.plot(
                [verdade], [posicao], "o", color=SUPERFICIE, markeredgecolor=cor, markersize=9, markeredgewidth=2
            )
            quadro.plot([dia], [posicao], "o", color=cor, markersize=9, markeredgecolor=SUPERFICIE, markeredgewidth=2)
            esquerda, direita = sorted((dia, verdade))
            quadro.text(esquerda - 0.008, posicao, _decimal(esquerda), ha="right", va="center", fontsize=9)
            quadro.text(direita + 0.008, posicao, _decimal(direita), ha="left", va="center", fontsize=9)

        quadro.plot([], [], "o", color=cor, markersize=9, label="com o rótulo do dia, o que o banco tem")
        quadro.plot(
            [],
            [],
            "o",
            color=SUPERFICIE,
            markeredgecolor=cor,
            markersize=9,
            markeredgewidth=2,
            label="com a verdade, que nenhum banco tem",
        )
        quadro.set_yticks(range(len(GOLPES)), [NOMES[golpe][0].upper() + NOMES[golpe][1:] for golpe in GOLPES])
        quadro.invert_yaxis()
        quadro.set_xlim(0.7, 1.03)
        quadro.set_xticks([0.7, 0.8, 0.9, 1.0])
        quadro.xaxis.set_major_formatter(FuncFormatter(lambda valor, _: f"{valor:.1f}".replace(".", ",")))
        quadro.set_ylim(len(GOLPES) - 0.4, -1.1)
        quadro.grid(axis="y", visible=False)
        quadro.tick_params(axis="y", length=0)
        quadro.spines["left"].set_visible(False)
        quadro.set_xlabel("A precisão média do LightGBM no golpe, contra o Pix legítimo")
        quadro.legend(frameon=False, loc="upper left", fontsize=9, ncols=2)
        _titulo(
            figura,
            "O preço do rótulo atrasado",
            "O golpe que demora a ser contestado é o que o banco aprende menos: a venda falsa.",
        )

    return figura


def mapa_da_decisao(pontuacoes: pl.DataFrame, custos: Custos | None = None) -> Figure:
    """As três saídas no plano do valor e da chance de golpe, com os Pix de teste do LightGBM por cima."""
    custos = custos or Custos()
    valores, chances = np.logspace(0, np.log10(30_000), 500), np.logspace(-6, 0, 600)
    em_valor, em_chance = np.meshgrid(valores, chances)
    grade = decidir(pl.DataFrame({"probabilidade": em_chance.ravel(), "valor": em_valor.ravel()}), custos)
    saidas = grade["acao"].to_physical().to_numpy().reshape(em_chance.shape)
    lightgbm = pontuacoes.filter(pl.col("modelo") == "lightgbm")
    golpes = lightgbm.filter(pl.col("golpe").is_not_null())
    honestos = lightgbm.filter(pl.col("golpe").is_null())
    amostra = honestos.sample(n=min(5_000, honestos.height), seed=2026)

    def no_quadro(pix: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        # O Pix de mais de R$ 30.000 fica na borda direita do quadro, e não fora dele.
        return np.clip(pix["valor"].to_numpy(), 1.0, 30_000.0), pix["probabilidade"].to_numpy()

    with mpl.rc_context(_ESTILO):
        figura = _figura(6.2)
        quadro = figura.subplots()
        tons = ListedColormap([_misturar(GRADE, 0.5), _misturar(SEGURAR, 0.35), _misturar(BLOQUEAR, 0.3)])
        quadro.pcolormesh(em_valor, em_chance, saidas, cmap=tons, vmin=-0.5, vmax=2.5, shading="auto", zorder=0)
        quadro.scatter(
            *no_quadro(amostra), s=3, color=MUDO, alpha=0.6, linewidths=0, label="Pix honestos, 5 mil sorteados"
        )
        quadro.scatter(
            *no_quadro(golpes),
            s=14,
            color=TINTA,
            edgecolors=SUPERFICIE,
            linewidths=0.5,
            label=f"os {_numero(golpes.height)} golpes",
        )

        regioes = (
            (5.0, 1e-3, "aprovar", _misturar(GRADE, 0.5)),
            (6_000.0, 2e-2, "segurar para confirmar", _misturar(SEGURAR, 0.35)),
            (7_000.0, 0.25, "bloquear", _misturar(BLOQUEAR, 0.3)),
        )

        for valor, chance, nome, fundo in regioes:
            quadro.text(
                valor,
                chance,
                nome,
                fontsize=12,
                fontweight="bold",
                color=TINTA,
                ha="center",
                va="center",
                bbox={"boxstyle": "round,pad=0.25", "facecolor": fundo, "edgecolor": "none"},
                zorder=3,
            )

        quadro.set_xscale("log")
        quadro.set_yscale("log")
        quadro.set_xlim(1, 30_000)
        quadro.set_ylim(1e-6, 1)
        quadro.set_xticks([1, 10, 100, 1_000, 10_000], ["R$ 1", "R$ 10", "R$ 100", "R$ 1.000", "R$ 10.000"])
        marcas = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0]
        quadro.set_yticks(marcas, ["0,0001%", "0,001%", "0,01%", "0,1%", "1%", "10%", "100%"])
        quadro.grid(visible=False)
        quadro.set_xlabel("O valor do Pix")
        quadro.set_ylabel("A chance de golpe, calibrada")
        quadro.legend(frameon=False, loc="upper left", fontsize=9, markerscale=1.5)
        _titulo(
            figura,
            "O mapa da decisão",
            "A saída de menor custo para cada valor e cada chance de golpe. Os pontos são os Pix de outubro a "
            "dezembro, pelo LightGBM.",
        )

    return figura


def custo_das_politicas(pontuacoes: pl.DataFrame, custos: Custos | None = None) -> Figure:
    """O custo de fato de cada jeito de decidir, com o LightGBM, do mais caro ao mais barato."""
    linhas = sorted(
        balanco(pontuacoes.filter(pl.col("modelo") == "lightgbm"), custos).iter_rows(named=True),
        key=lambda linha: -linha["custo"],
    )

    with mpl.rc_context(_ESTILO):
        figura = _figura(4.4)
        quadro = figura.subplots()
        maior = linhas[0]["custo"]

        for posicao, linha in enumerate(linhas):
            escolhida = linha["politica"] == "custo"
            quadro.barh(posicao, linha["custo"], height=0.6, color=SEM_A_IDADE if escolhida else MUDO)
            partes = [f"{_mil(linha['perdido'])} em golpes que passaram"]

            if linha["honestos_segurados"]:
                partes.append(f"{_numero(linha['honestos_segurados'])} honestos segurados")

            if linha["honestos_bloqueados"]:
                partes.append(f"{_numero(linha['honestos_bloqueados'])} honestos bloqueados")

            detalhe = "; ".join(partes)
            quadro.text(
                linha["custo"] + maior * 0.01,
                posicao - 0.1,
                _mil(linha["custo"]),
                va="center",
                fontsize=11,
                fontweight="bold",
                color=TINTA,
            )
            quadro.text(
                linha["custo"] + maior * 0.01, posicao + 0.2, detalhe, va="center", fontsize=8, color=TINTA_SECUNDARIA
            )

        quadro.set_yticks(range(len(linhas)), [NOMES_DAS_POLITICAS[linha["politica"]].capitalize() for linha in linhas])
        quadro.invert_yaxis()
        quadro.set_xlim(0, maior * 1.45)
        quadro.set_xticks([0, 500_000, 1_000_000, 1_500_000], ["0", "R$ 500 mil", "R$ 1 milhão", "R$ 1,5 milhão"])
        quadro.grid(axis="y", visible=False)
        quadro.tick_params(axis="y", length=0)
        quadro.spines["left"].set_visible(False)
        _titulo(
            figura,
            "O custo de cada jeito de decidir",
            "O custo de fato dos três meses de teste, com o LightGBM: os golpes que passaram, as confirmações e "
            "os bloqueios de clientes honestos.",
        )

    return figura


def erros_por_idade(com_a_idade: pl.DataFrame, sem_a_idade: pl.DataFrame, idades: pl.DataFrame) -> Figure:
    """Os dois erros em cada faixa de idade, pelo custo, com o LightGBM que via a idade e com o que não vê."""

    def tabela(pontuacoes: pl.DataFrame) -> pl.DataFrame:
        decididas = com_a_politica(pontuacoes.filter(pl.col("modelo") == "lightgbm"), "custo")
        return por_idade(decididas.join(idades, on="id"))

    tabelas = {"com a idade, como era": tabela(com_a_idade), "sem a idade, como ficou": tabela(sem_a_idade)}
    cores = dict(zip(tabelas, (MUDO, SEM_A_IDADE), strict=True))
    paineis: tuple[tuple[str, str, int | None], ...] = (
        ("recall", "Os golpes pegos", None),
        ("honestos_segurados", "Os honestos segurados,\na cada 10 mil Pix", 1),
        ("honestos_bloqueados", "Os honestos bloqueados,\na cada 10 mil Pix", 2),
    )
    faixas = next(iter(tabelas.values()))["faixa"].cast(pl.String).to_list()
    largura = 0.36

    with mpl.rc_context(_ESTILO):
        figura = _figura(4.6)
        quadros = figura.subplots(1, 3, sharey=True)

        for quadro, (coluna, titulo, casas) in zip(quadros, paineis, strict=True):
            maior = max(max(dela[coluna].to_list()) for dela in tabelas.values())

            def escrever(valor: float, casas: int | None = casas) -> str:
                # A parte dos golpes vai em porcentagem; os honestos, em número com as casas do painel.
                return _porcentagem(valor) if casas is None else _decimal_curto(valor, casas)

            for indice, (nome, dela) in enumerate(tabelas.items()):
                posicoes = [posicao + (indice - 0.5) * (largura + 0.04) for posicao in range(len(faixas))]
                alturas = dela[coluna].to_list()
                quadro.barh(posicoes, alturas, height=largura, color=cores[nome], label=nome)

                for posicao, altura in zip(posicoes, alturas, strict=True):
                    quadro.text(altura + maior * 0.03, posicao, escrever(altura), va="center", fontsize=8, color=TINTA)

            quadro.set_title(titulo, fontsize=10, loc="left")
            quadro.set_xlim(0, maior * 1.35)
            quadro.set_xticks([])
            quadro.grid(visible=False)
            quadro.tick_params(axis="y", length=0)
            quadro.spines["left"].set_visible(False)
            quadro.spines["bottom"].set_visible(False)

        quadros[0].set_yticks(
            range(len(faixas)),
            [faixa.replace(" ou mais", "") + " anos" + (" ou mais" if "ou mais" in faixa else "") for faixa in faixas],
        )
        quadros[0].invert_yaxis()
        marcas, nomes = quadros[0].get_legend_handles_labels()
        figura.legend(marcas, nomes, frameon=False, loc="outside lower center", ncols=2, fontsize=9)
        _titulo(
            figura,
            "Os erros com cada idade",
            "Pelo custo, com o LightGBM. Com a idade, o modelo bloqueava o honesto de 65 anos ou mais 19 vezes mais "
            "que o de 18 a 34.",
        )

    return figura


def _misturar(cor: str, parte: float) -> str:
    """A cor misturada à superfície: a parte dela, e o resto da superfície. É o tom de fundo de uma região."""
    de = [int(cor[i : i + 2], 16) for i in (1, 3, 5)]
    para = [int(SUPERFICIE[i : i + 2], 16) for i in (1, 3, 5)]

    return "#" + "".join(f"{round(a * parte + b * (1 - parte)):02x}" for a, b in zip(de, para, strict=True))


def _mil(valor: float) -> str:
    """O dinheiro em mil e em milhão: R$ 101 mil, R$ 1,76 milhão."""
    if valor >= 1_000_000:
        return f"R$ {valor / 1_000_000:.2f} milhão".replace(".", ",")

    return f"R$ {valor / 1_000:.0f} mil"


def _numero(valor: int) -> str:
    return f"{valor:,}".replace(",", ".")


def _decimal_curto(valor: float, casas: int) -> str:
    return f"{valor:.{casas}f}".replace(".", ",")


def _decimal(valor: float) -> str:
    """Três casas: a invasão de conta fica entre 0,995 e 0,996, e duas casas a mostrariam como 0,99 e 1,00."""
    return f"{valor:.3f}".replace(".", ",")


def _rotulo(nome: str) -> str:
    """O nome da variável como se lê: `pix_na_ultima_hora` vira "Pix na última hora"."""
    return " ".join(_COM_ACENTO.get(palavra, palavra) for palavra in nome.split("_"))


def _afastados(alturas: dict[str, float], distancia: float) -> dict[str, float]:
    """As alturas dos rótulos, de baixo para cima, cada uma ao menos a distância dada da anterior."""
    resultado: dict[str, float] = {}
    anterior = float("-inf")

    for nome, altura in sorted(alturas.items(), key=lambda item: item[1]):
        anterior = max(altura, anterior + distancia)
        resultado[nome] = anterior

    return resultado


class Insumos:
    """O que as figuras pedem, calculado uma vez só, e só quando alguma figura pede: as variáveis e as avaliações."""

    def __init__(
        self, simulacao: Simulacao, as_dobras: tuple[Dobra, ...] | None = None, cartoes: Path | None = None
    ) -> None:
        self.simulacao = simulacao
        self._dobras = as_dobras
        self.arquivo_dos_cartoes = cartoes

    @cached_property
    def nos_cartoes(self) -> pl.DataFrame:
        """As pontuações da base de cartões, do arquivo dado: a figura dos cartões só é desenhada com ele."""
        if self.arquivo_dos_cartoes is None:
            raise ValueError("A figura dos cartões pede o arquivo da base de cartões.")

        return avaliar_os_cartoes(ler_os_cartoes(self.arquivo_dos_cartoes))

    @cached_property
    def variaveis(self) -> pl.DataFrame:
        return calcular(VisaoDoBanco.da(self.simulacao))

    @cached_property
    def dobras(self) -> tuple[Dobra, ...]:
        """As dobras pedidas, ou as do padrão: do primeiro mês de teste ao fim dos dados."""
        if self._dobras is not None:
            return self._dobras

        visao = VisaoDoBanco.da(self.simulacao)
        ultimo = visao.transacoes["momento"].max()
        fim = ultimo.date() if isinstance(ultimo, datetime) else self.simulacao.config.fim - timedelta(days=1)

        return dobras(primeiro_corte(visao), fim)

    @cached_property
    def pontuacoes(self) -> pl.DataFrame:
        return avaliar(self.simulacao, self.variaveis, self.dobras)

    @cached_property
    def com_a_verdade(self) -> pl.DataFrame:
        return avaliar(self.simulacao, self.variaveis, self.dobras, modelos=("lightgbm",), rotulo=Rotulo.VERDADE)

    @cached_property
    def com_a_idade(self) -> pl.DataFrame:
        """O LightGBM como era na Parte 3, recebendo a idade: só para medir o que mudou sem ela (ADR-0011)."""
        return avaliar(self.simulacao, self.variaveis, self.dobras, modelos=("lightgbm",), entrada=CATALOGO)

    @cached_property
    def idades(self) -> pl.DataFrame:
        return self.variaveis.select("id", "idade")


FIGURAS: dict[str, Callable[[Insumos], Figure]] = {
    "horas": lambda insumos: horas(insumos.simulacao),
    "valores": lambda insumos: valores(insumos.simulacao),
    "contestacao": lambda insumos: contestacao(insumos.simulacao),
    "recebedores": lambda insumos: recebedores(insumos.simulacao),
    "separacao": lambda insumos: forca_das_variaveis(insumos.simulacao, insumos.variaveis),
    "golpes_pegos": lambda insumos: golpes_pegos(insumos.pontuacoes),
    "golpes_por_modelo": lambda insumos: golpes_por_modelo(insumos.pontuacoes),
    "calibracao": lambda insumos: calibracao(insumos.pontuacoes),
    "rotulo_atrasado": lambda insumos: rotulo_atrasado(insumos.pontuacoes, insumos.com_a_verdade),
    "mapa_da_decisao": lambda insumos: mapa_da_decisao(insumos.pontuacoes),
    "custo_das_politicas": lambda insumos: custo_das_politicas(insumos.pontuacoes),
    "erros_por_idade": lambda insumos: erros_por_idade(insumos.com_a_idade, insumos.pontuacoes, insumos.idades),
}


FIGURAS_DOS_CARTOES: dict[str, Callable[[Insumos], Figure]] = {
    "fraudes_nos_cartoes": lambda insumos: fraudes_nos_cartoes(insumos.nos_cartoes),
}
"""As figuras da base de cartões: só com o arquivo dela, que fica fora do repositório."""


def gerar(
    simulacao: Simulacao, pasta: Path, as_dobras: tuple[Dobra, ...] | None = None, cartoes: Path | None = None
) -> list[Path]:
    """Desenha as figuras e as grava em PNG na pasta.

    As dobras da avaliação são as do padrão, se não vierem. A figura dos cartões só sai com o arquivo da base.
    """
    pasta.mkdir(parents=True, exist_ok=True)
    insumos = Insumos(simulacao, as_dobras, cartoes)
    gravadas = []
    todas = FIGURAS | (FIGURAS_DOS_CARTOES if cartoes is not None else {})

    for nome, desenhar in todas.items():
        destino = pasta / f"{nome}.png"
        desenhar(insumos).savefig(destino, dpi=150)
        gravadas.append(destino)

    return gravadas
