from datetime import date

import pytest

from antifraude_pix.simulacao.configuracao import Configuracao


def test_o_padrao_cobre_o_segundo_semestre_de_2025_com_o_limite_noturno_do_banco_central() -> None:
    config = Configuracao()

    assert (config.inicio, config.fim) == (date(2025, 7, 1), date(2026, 1, 1))
    assert config.limite_noturno == 100_000
    assert config.prazo_do_med == 80


def test_o_periodo_noturno_vai_das_20h_as_6h() -> None:
    config = Configuracao()

    assert [hora for hora in range(24) if config.noturno(hora)] == [0, 1, 2, 3, 4, 5, 20, 21, 22, 23]


def test_a_configuracao_invalida_diz_todos_os_problemas_de_uma_vez() -> None:
    with pytest.raises(ValueError, match="clientes") as erro:
        Configuracao(clientes=0, dias=0, semente=-1)

    assert str(erro.value) == (
        "O número de clientes tem de ser positivo. "
        "A semente não pode ser negativa. "
        "O número de dias tem de ser positivo."
    )
