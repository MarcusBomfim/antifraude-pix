import pytest

from antifraude_pix.simulacao.simulador import Simulacao
from tests.apoio import simulacao_media


@pytest.fixture
def simulacao() -> Simulacao:
    return simulacao_media()
