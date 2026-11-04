"""A linha de comando: `antifraude-pix simular` gera os dados, e `antifraude-pix validar` os confere."""

import argparse
import sys
import time
from collections.abc import Sequence
from datetime import date
from pathlib import Path

from antifraude_pix.dados.gravacao import gravar, ler
from antifraude_pix.dados.validacao import problemas
from antifraude_pix.simulacao.configuracao import Configuracao
from antifraude_pix.simulacao.golpes import NOMES, Golpe
from antifraude_pix.simulacao.simulador import Simulacao, simular


def principal(argumentos: Sequence[str] | None = None) -> int:
    """O comando `antifraude-pix`.

    Devolve 0 quando dá certo, 1 quando os dados têm problema, e 2 com o pedido errado.
    """
    pedido = _leitor().parse_args(argumentos)

    if pedido.comando == "simular":
        return _simular(pedido)

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


def _porcentagem(fracao: float) -> str:
    return f"{fracao:.2%}".replace(".", ",")


if __name__ == "__main__":
    sys.exit(principal())
