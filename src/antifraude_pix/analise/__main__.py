"""As figuras do README a partir dos dados gravados: `uv run python -m antifraude_pix.analise`.

Com a simulação padrão, leva uns quatro minutos: as figuras dos modelos treinam
os três modelos em cada mês de teste, e o LightGBM mais duas vezes: com a
verdade, e com a idade que saiu do modelo. A figura dos cartões só sai se a base
de cartões já foi baixada, com `antifraude-pix cartoes`; a do monitor, se a
simulação com a virada já foi gravada, com `antifraude-pix simular --virada`.
"""

import argparse
import sys
from pathlib import Path

from antifraude_pix.analise.figuras import gerar
from antifraude_pix.cartoes.base import ARQUIVO
from antifraude_pix.dados.gravacao import ler


def principal() -> int:
    leitor = argparse.ArgumentParser(
        prog="python -m antifraude_pix.analise",
        description="Desenha o retrato dos dados, a força das variáveis, a avaliação dos modelos e a decisão.",
    )
    leitor.add_argument("--dados", type=Path, default=Path("dados"), help="a pasta dos dados (padrão: %(default)s)")
    leitor.add_argument("--saida", type=Path, default=Path("docs/imagens"), help="onde gravar (padrão: %(default)s)")
    leitor.add_argument(
        "--virada",
        type=Path,
        default=Path("dados-virada"),
        help="a pasta da simulação com a virada, para a figura do monitor (padrão: %(default)s)",
    )
    pedido = leitor.parse_args()

    cartoes = pedido.dados / "cartoes" / ARQUIVO

    if not cartoes.exists():
        print(f"A figura dos cartões fica de fora: falta {cartoes}, que antifraude-pix cartoes baixa.")

    if not (pedido.virada / "manifesto.json").exists():
        print(f"A figura do monitor fica de fora: falta {pedido.virada}, que antifraude-pix simular --virada grava.")

    virada = ler(pedido.virada) if (pedido.virada / "manifesto.json").exists() else None

    for figura in gerar(ler(pedido.dados), pedido.saida, cartoes=cartoes if cartoes.exists() else None, virada=virada):
        print(figura)

    return 0


if __name__ == "__main__":
    sys.exit(principal())
