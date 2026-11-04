"""O que os testes da simulação e dos dados têm em comum."""

from functools import cache

from antifraude_pix.simulacao.configuracao import Configuracao
from antifraude_pix.simulacao.simulador import Simulacao, simular

MEDIA = Configuracao(clientes=1_500, dias=90, golpes_por_mil_clientes_ao_mes=60)
"""Uma simulação de tamanho médio, com seis vezes mais golpes que o padrão: o bastante de cada golpe para contar."""


@cache
def simulacao_media() -> Simulacao:
    """A simulação média, feita uma vez só por execução dos testes, para todas as pastas que a usam."""
    return simular(MEDIA)
