"""O que os testes da simulação, dos dados e da base de cartões têm em comum."""

from functools import cache
from pathlib import Path

import numpy as np

from antifraude_pix.cartoes.base import ARQUIVO, COMPONENTES
from antifraude_pix.simulacao.configuracao import Configuracao
from antifraude_pix.simulacao.simulador import Simulacao, simular

MEDIA = Configuracao(clientes=1_500, dias=90, golpes_por_mil_clientes_ao_mes=60)
"""Uma simulação de tamanho médio, com seis vezes mais golpes que o padrão: o bastante de cada golpe para contar."""


@cache
def simulacao_media() -> Simulacao:
    """A simulação média, feita uma vez só por execução dos testes, para todas as pastas que a usam."""
    return simular(MEDIA)


COMPRAS = 4_000
"""As compras da base de cartões inventada: duas mil por dia."""


def cartoes_inventados(
    pasta: Path, linhas: int = COMPRAS, colunas: tuple[str, ...] = ("Time", *COMPONENTES, "Amount")
) -> Path:
    """Uma base de cartões pequena e inventada, no formato do OpenML: a fraude empurra V1 e V3 para longe."""
    sorteio = np.random.default_rng(7)
    fraude = sorteio.random(linhas) < 0.03
    tempo = np.sort(sorteio.uniform(0, 2 * 86_400, linhas))
    componentes = sorteio.normal(0, 1, (linhas, len(COMPONENTES)))
    componentes[fraude, 0] += 3.0
    componentes[fraude, 2] -= 2.5
    valor = np.round(np.exp(sorteio.normal(3.5, 1.2, linhas)), 2)
    cabecalho = ["@relation creditcard", *(f"@attribute {coluna} numeric" for coluna in colunas)]
    cabecalho += ["@attribute Class {'0','1'}", "@data"]
    corpo = [
        ",".join([f"{tempo[i]:.0f}", *(f"{x:.6f}" for x in componentes[i]), f"{valor[i]:.2f}", f"'{int(fraude[i])}'"])
        for i in range(linhas)
    ]
    pasta.mkdir(parents=True, exist_ok=True)
    arquivo = pasta / ARQUIVO
    arquivo.write_text("\n".join(cabecalho + corpo) + "\n", encoding="utf-8")

    return arquivo
