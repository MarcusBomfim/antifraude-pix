"""A linha de comando do projeto.

- `simular` gera os dados;
- `validar` os confere;
- `variaveis` calcula as variáveis de cada Pix;
- `avaliar` treina os modelos mês a mês com o rótulo do dia, e os mede contra a verdade.
"""

import argparse
import sys
import time
from collections.abc import Sequence
from datetime import date, datetime, timedelta
from pathlib import Path

import polars as pl

from antifraude_pix.avaliacao.metricas import ALERTA, resumo
from antifraude_pix.avaliacao.no_tempo import avaliar
from antifraude_pix.dados.banco import VisaoDoBanco
from antifraude_pix.dados.gravacao import gravar, ler, ler_o_banco
from antifraude_pix.dados.validacao import problemas
from antifraude_pix.modelos.rotulo import Dobra, dobras, primeiro_corte
from antifraude_pix.modelos.treino import NOMES as NOMES_DOS_MODELOS
from antifraude_pix.simulacao.configuracao import Configuracao
from antifraude_pix.simulacao.golpes import NOMES, Golpe
from antifraude_pix.simulacao.simulador import Simulacao, simular
from antifraude_pix.variaveis.calculo import calcular
from antifraude_pix.variaveis.catalogo import CATALOGO


def principal(argumentos: Sequence[str] | None = None) -> int:
    """O comando `antifraude-pix`.

    Devolve 0 quando dá certo, 1 quando os dados têm problema, e 2 com o pedido errado.
    """
    pedido = _leitor().parse_args(argumentos)

    if pedido.comando == "simular":
        return _simular(pedido)

    if pedido.comando == "variaveis":
        return _variaveis(pedido.pasta, pedido.saida)

    if pedido.comando == "avaliar":
        return _avaliar(pedido.pasta, pedido.saida)

    return _validar(pedido.pasta)


def _leitor() -> argparse.ArgumentParser:
    padrao = Configuracao()
    leitor = argparse.ArgumentParser(prog="antifraude-pix", description="Fraude em Pix: o simulador e os modelos.")
    comandos = leitor.add_subparsers(dest="comando", required=True)

    simular_ = comandos.add_parser("simular", help="gera os dados de uma simulação e os grava em Parquet")
    simular_.add_argument(
        "--clientes", type=int, default=padrao.clientes, help="quantos clientes (padrão: %(default)s)"
    )
    simular_.add_argument("--dias", type=int, default=padrao.dias, help="quantos dias (padrão: %(default)s)")
    simular_.add_argument(
        "--inicio",
        type=date.fromisoformat,
        default=padrao.inicio,
        help="o primeiro dia, AAAA-MM-DD (padrão: %(default)s)",
    )
    simular_.add_argument(
        "--semente", type=int, default=padrao.semente, help="a semente do sorteio (padrão: %(default)s)"
    )
    simular_.add_argument(
        "--golpes-por-mil",
        type=float,
        default=padrao.golpes_por_mil_clientes_ao_mes,
        help="ataques por mês a cada mil clientes (padrão: %(default)s)",
    )
    simular_.add_argument("--saida", type=Path, default=Path("dados"), help="a pasta dos dados (padrão: %(default)s)")

    validar = comandos.add_parser("validar", help="confere os dados gravados numa pasta")
    validar.add_argument("pasta", type=Path, nargs="?", default=Path("dados"), help="a pasta dos dados (padrão: dados)")

    variaveis = comandos.add_parser(
        "variaveis", help="calcula as variáveis de cada Pix, só com o que o banco sabia antes dele"
    )
    variaveis.add_argument(
        "pasta", type=Path, nargs="?", default=Path("dados"), help="a pasta dos dados (padrão: dados)"
    )
    variaveis.add_argument(
        "--saida", type=Path, help="o arquivo das variáveis (padrão: variaveis.parquet, na pasta dos dados)"
    )

    avaliar_ = comandos.add_parser(
        "avaliar", help="treina os modelos mês a mês com o rótulo do dia, e os mede contra a verdade do simulador"
    )
    avaliar_.add_argument(
        "pasta", type=Path, nargs="?", default=Path("dados"), help="a pasta dos dados (padrão: dados)"
    )
    avaliar_.add_argument(
        "--saida", type=Path, help="o arquivo das pontuações (padrão: avaliacao.parquet, na pasta dos dados)"
    )

    return leitor


def _simular(pedido: argparse.Namespace) -> int:
    try:
        config = Configuracao(
            clientes=pedido.clientes,
            dias=pedido.dias,
            inicio=pedido.inicio,
            semente=pedido.semente,
            golpes_por_mil_clientes_ao_mes=pedido.golpes_por_mil,
        )
    except ValueError as erro:
        print(f"antifraude-pix simular: {erro}", file=sys.stderr)
        return 2

    comeco = time.perf_counter()
    simulacao = simular(config)
    encontrados = problemas(simulacao)

    if encontrados:
        _mostrar_problemas(encontrados)
        return 1

    gravar(simulacao, pedido.saida)
    print(_resumo(simulacao))
    print(f"Gravado em {pedido.saida}, em {_numero(round(time.perf_counter() - comeco))} s.")

    return 0


def _validar(pasta: Path) -> int:
    try:
        simulacao = ler(pasta)
    except (OSError, ValueError, KeyError) as erro:
        print(f"antifraude-pix validar: não consegui ler {pasta}: {erro}", file=sys.stderr)
        return 1

    encontrados = problemas(simulacao)

    if encontrados:
        _mostrar_problemas(encontrados)
        return 1

    print(_resumo(simulacao))
    print("Os dados seguem o contrato.")

    return 0


def _variaveis(pasta: Path, saida: Path | None) -> int:
    """Lê só o que o banco vê, calcula as variáveis e as grava em Parquet, ao lado dos dados."""
    try:
        visao = ler_o_banco(pasta)
    except (OSError, ValueError, KeyError) as erro:
        print(f"antifraude-pix variaveis: não consegui ler {pasta}: {erro}", file=sys.stderr)
        return 1

    comeco = time.perf_counter()
    destino = saida or pasta / "variaveis.parquet"
    destino.parent.mkdir(parents=True, exist_ok=True)
    calcular(visao).write_parquet(destino, compression="zstd")

    print(
        f"{_numero(len(CATALOGO))} variáveis de {_numero(visao.transacoes.height)} Pix, "
        f"gravadas em {destino}, em {_numero(round(time.perf_counter() - comeco))} s."
    )

    return 0


def _avaliar(pasta: Path, saida: Path | None) -> int:
    """Treina em cada corte com o que o banco sabia, pontua o mês do corte, e mede contra a verdade (ADR-0009)."""
    try:
        simulacao = ler(pasta)
    except (OSError, ValueError, KeyError) as erro:
        print(f"antifraude-pix avaliar: não consegui ler {pasta}: {erro}", file=sys.stderr)
        return 1

    comeco = time.perf_counter()
    visao = VisaoDoBanco.da(simulacao)
    ultimo = visao.transacoes["momento"].max()
    as_dobras = dobras(primeiro_corte(visao), ultimo.date()) if isinstance(ultimo, datetime) else ()

    if not as_dobras:
        print(
            f"antifraude-pix avaliar: os dados de {pasta} não chegam a um mês de teste. Antes dele, são precisos "
            "o aquecimento, um mês de treino e um mês de validação.",
            file=sys.stderr,
        )
        return 1

    pontuacoes = avaliar(simulacao, calcular(visao), as_dobras)
    destino = saida or pasta / "avaliacao.parquet"
    destino.parent.mkdir(parents=True, exist_ok=True)
    pontuacoes.write_parquet(destino, compression="zstd")

    print(_quadro(pontuacoes, as_dobras))
    print(f"Pontuações gravadas em {destino}, em {_numero(round(time.perf_counter() - comeco))} s.")

    return 0


def _quadro(pontuacoes: pl.DataFrame, as_dobras: tuple[Dobra, ...]) -> str:
    """O resumo da avaliação em texto: uma linha por modelo."""
    um_modelo = pontuacoes.filter(pl.col("modelo") == pontuacoes["modelo"][0])
    golpes = int(um_modelo["golpe"].is_not_null().sum())
    ultimo_dia = as_dobras[-1].fim - timedelta(days=1)
    meses = "mês" if len(as_dobras) == 1 else "meses"
    cabecalho = "".join(
        f"{titulo:>{largura}}"
        for titulo, largura in (
            ("", 24),
            ("precisão média", 16),
            ("golpes pegos", 14),
            ("dinheiro salvo", 16),
            ("honestos por golpe", 20),
            ("golpes previstos", 18),
        )
    )
    linhas = [
        f"Teste de {as_dobras[0].corte:%d/%m/%Y} a {ultimo_dia:%d/%m/%Y}, em {len(as_dobras)} {meses}: "
        f"{_numero(um_modelo.height)} Pix, {_numero(golpes)} de golpe.",
        "O treino usa só o rótulo do dia; a medida, a verdade do simulador.",
        f"Segurando, em cada mês, os {_porcentagem(ALERTA, casas=1)} de Pix com mais cara de golpe:",
        "",
        cabecalho,
    ]

    for linha in resumo(pontuacoes).iter_rows(named=True):
        honestos = linha["legitimos_por_golpe_pego"]
        colunas = (
            (NOMES_DOS_MODELOS[linha["modelo"]], 24, "<"),
            (_decimal(linha["precisao_media"]), 16, ">"),
            (_porcentagem(linha["recall_no_alerta"], casas=0), 14, ">"),
            (_porcentagem(linha["recall_em_reais"], casas=0), 16, ">"),
            ("—" if honestos is None else _decimal(honestos, casas=1), 20, ">"),
            (_numero(round(linha["golpes_previstos"])), 18, ">"),
        )
        linhas.append("".join(f"{texto:{lado}{largura}}" for texto, largura, lado in colunas))

    return "\n".join(linhas)


def _mostrar_problemas(encontrados: list[str]) -> None:
    print("Os dados têm problemas:", file=sys.stderr)

    for problema in encontrados:
        print(f"- {problema}", file=sys.stderr)


def _resumo(simulacao: Simulacao) -> str:
    config = simulacao.config
    pix = simulacao.transacoes.height
    fraudes = simulacao.verdade.fraudes
    contestadas = simulacao.contestacoes.join(fraudes, on="transacao", how="semi").height
    por_golpe = ", ".join(
        f"{NOMES[Golpe(golpe)]} {_numero(quantos)}"
        for golpe, quantos in fraudes.group_by("golpe").len().sort("golpe").iter_rows()
    )

    return (
        f"{_numero(config.clientes)} clientes, {_numero(config.dias)} dias "
        f"a partir de {config.inicio:%d/%m/%Y}, semente {config.semente}.\n"
        f"{_numero(pix)} Pix; {_numero(fraudes.height)} de golpe ({_porcentagem(fraudes.height / max(pix, 1))}), "
        f"em {_numero(simulacao.verdade.ataques.height)} ataques: {por_golpe}.\n"
        f"{_numero(contestadas)} Pix de golpe contestados ({_porcentagem(contestadas / max(fraudes.height, 1))}), "
        f"e {_numero(simulacao.contestacoes.height - contestadas)} Pix legítimos contestados mesmo assim."
    )


def _numero(valor: int) -> str:
    return f"{valor:,}".replace(",", ".")


def _porcentagem(fracao: float, casas: int = 2) -> str:
    return f"{fracao:.{casas}%}".replace(".", ",")


def _decimal(valor: float, casas: int = 2) -> str:
    return f"{valor:.{casas}f}".replace(".", ",")


if __name__ == "__main__":
    sys.exit(principal())
