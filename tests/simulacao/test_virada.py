import itertools
import json
import re
from dataclasses import replace
from datetime import date, datetime, time, timedelta
from pathlib import Path

import pytest

from antifraude_pix.dados.gravacao import MANIFESTO, gravar, ler
from antifraude_pix.dados.validacao import problemas
from antifraude_pix.simulacao.configuracao import Configuracao
from antifraude_pix.simulacao.golpes import Ataque, Golpe, passos, planejar_ataques
from antifraude_pix.simulacao.pix import DIA, HORA, Iniciacao, segundos
from antifraude_pix.simulacao.populacao import Populacao, gerar_populacao
from antifraude_pix.simulacao.simulador import simular
from tests.apoio import MEDIA

VIRADA = date(2025, 8, 15)
COM_A_VIRADA = replace(MEDIA, virada=VIRADA)


@pytest.fixture(scope="module")
def populacao() -> Populacao:
    return gerar_populacao(MEDIA)


@pytest.fixture(scope="module")
def ataques(populacao: Populacao) -> tuple[Ataque, ...]:
    return planejar_ataques(MEDIA, populacao)


def _depois_da_virada(ataque: Ataque) -> bool:
    return ataque.golpe is Golpe.FALSA_CENTRAL and ataque.inicio.date() >= VIRADA


def test_a_virada_nao_muda_o_plano_dos_ataques(populacao: Populacao, ataques: tuple[Ataque, ...]) -> None:
    assert planejar_ataques(COM_A_VIRADA, populacao) == ataques


def test_so_a_falsa_central_da_virada_em_diante_muda_de_tatica(ataques: tuple[Ataque, ...]) -> None:
    mudaram = [ataque for ataque in ataques if passos(COM_A_VIRADA, ataque) != passos(MEDIA, ataque)]

    assert mudaram
    assert mudaram == [ataque for ataque in ataques if _depois_da_virada(ataque)]


def test_a_falsa_central_que_comeca_no_dia_da_virada_ja_e_a_nova(ataques: tuple[Ataque, ...]) -> None:
    antiga = next(ataque for ataque in ataques if ataque.golpe is Golpe.FALSA_CENTRAL and not _depois_da_virada(ataque))
    no_dia = replace(antiga, inicio=datetime.combine(VIRADA, time(10)))
    na_vespera = replace(antiga, inicio=datetime.combine(VIRADA - timedelta(days=1), time(23, 59)))

    assert passos(COM_A_VIRADA, no_dia) != passos(MEDIA, no_dia)
    assert passos(COM_A_VIRADA, na_vespera) == passos(MEDIA, na_vespera)


def test_a_falsa_central_parcelada_pede_um_pix_por_dia_a_mesma_conta(ataques: tuple[Ataque, ...]) -> None:
    fim = _fim(COM_A_VIRADA)
    quantos = []

    for ataque in filter(_depois_da_virada, ataques):
        dele = passos(COM_A_VIRADA, ataque)
        intervalos = [depois.instante - antes.instante for antes, depois in itertools.pairwise(dele)]
        quantos.append(len(dele))

        assert dele[0].instante == segundos(ataque.inicio)
        assert all(DIA - 4 * HORA <= intervalo <= DIA + 4 * HORA for intervalo in intervalos)
        assert {passo.laranja for passo in dele} == {ataque.laranjas[0]}
        assert all(passo.iniciacao is Iniciacao.CHAVE and passo.valor is None for passo in dele)
        assert all(0.2 <= (passo.fracao_do_saldo or 0) <= 0.35 for passo in dele)
        assert all(passo.instante < fim for passo in dele)

    # De três a seis Pix; menos só quando o fim da simulação corta o ataque no meio.
    assert max(quantos) <= 6
    assert sum(quantidade >= 3 for quantidade in quantos) / len(quantos) > 0.9


def test_a_parcelada_para_no_fim_da_simulacao(ataques: tuple[Ataque, ...]) -> None:
    ultimo = max(filter(_depois_da_virada, ataques), key=lambda ataque: ataque.inicio)
    no_fim = replace(COM_A_VIRADA, dias=(ultimo.inicio.date() - MEDIA.inicio).days + 2)
    dele = passos(no_fim, ultimo)

    assert 1 <= len(dele) <= 2
    assert all(passo.instante < _fim(no_fim) for passo in dele)


@pytest.mark.parametrize("virada", [MEDIA.inicio - timedelta(days=1), MEDIA.fim])
def test_a_virada_fora_da_simulacao_e_recusada(virada: date) -> None:
    with pytest.raises(ValueError, match=re.escape("A virada tem de cair dentro da simulação.")):
        replace(MEDIA, virada=virada)


def test_a_simulacao_com_a_virada_passa_na_validacao_e_volta_do_disco(tmp_path: Path) -> None:
    config = Configuracao(clientes=300, dias=75, golpes_por_mil_clientes_ao_mes=150, virada=date(2025, 8, 1))
    simulacao = simular(config)
    gravar(simulacao, tmp_path)
    lida = ler(tmp_path)

    assert problemas(simulacao) == []
    assert lida.config == config
    assert json.loads((tmp_path / MANIFESTO).read_text(encoding="utf-8"))["configuracao"]["virada"] == "2025-08-01"


def test_sem_a_virada_o_manifesto_fica_como_era(tmp_path: Path) -> None:
    gravar(simular(Configuracao(clientes=30, dias=10)), tmp_path)

    assert "virada" not in json.loads((tmp_path / MANIFESTO).read_text(encoding="utf-8"))["configuracao"]
    assert ler(tmp_path).config.virada is None


def _fim(config: Configuracao) -> int:
    """O instante da meia-noite em que a simulação acaba."""
    return segundos(datetime.combine(config.fim, time()))
