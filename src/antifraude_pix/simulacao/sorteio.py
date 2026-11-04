"""De onde sai cada número sorteado da simulação (ADR-0002).

Cada etapa — a população, os ataques, a rotina de cada cliente — tem o seu
fluxo de números, tirado da semente e do nome da etapa; a rotina, um fluxo
por cliente. Assim, mudar uma etapa não muda o que as outras sorteiam, e a
rotina de um cliente não depende da ordem em que os outros são simulados.
"""

from enum import IntEnum

import numpy as np

from antifraude_pix.simulacao.configuracao import Configuracao


class Etapa(IntEnum):
    POPULACAO = 1
    ATAQUES = 2
    ROTINA = 3
    CONTESTACOES = 4


def sorteio(config: Configuracao, etapa: Etapa, *chaves: int) -> np.random.Generator:
    """O gerador da etapa — e, com as chaves, o de um cliente ou de um ataque dentro dela."""
    return np.random.default_rng([config.semente, int(etapa), *chaves])
