"""As figuras do README a partir dos dados gravados: `uv run python -m antifraude_pix.analise`."""

import argparse
import sys
from pathlib import Path

from antifraude_pix.analise.figuras import gerar
from antifraude_pix.dados.gravacao import ler


def principal() -> int:
    leitor = argparse.ArgumentParser(
        prog="python -m antifraude_pix.analise", description="Desenha o retrato dos dados."
    )
    leitor.add_argument("--dados", type=Path, default=Path("dados"), help="a pasta dos dados (padrão: %(default)s)")
    leitor.add_argument("--saida", type=Path, default=Path("docs/imagens"), help="onde gravar (padrão: %(default)s)")
    pedido = leitor.parse_args()

    for figura in gerar(ler(pedido.dados), pedido.saida):
        print(figura)

    return 0


if __name__ == "__main__":
    sys.exit(principal())
