from datetime import date, timedelta

import pytest
from hypothesis import given
from hypothesis import strategies as st

from antifraude_pix.simulacao.calendario import dia_util, feriados, pascoa, quinto_dia_util


@pytest.mark.parametrize(
    ("ano", "domingo"),
    [
        (2008, date(2008, 3, 23)),
        (2019, date(2019, 4, 21)),
        (2024, date(2024, 3, 31)),
        (2025, date(2025, 4, 20)),
        (2026, date(2026, 4, 5)),
        (2038, date(2038, 4, 25)),
        (2285, date(2285, 3, 22)),
    ],
)
def test_a_pascoa_cai_no_domingo_dos_calendarios_conhecidos(ano: int, domingo: date) -> None:
    assert pascoa(ano) == domingo


@given(st.integers(min_value=1583, max_value=4099))
def test_a_pascoa_e_sempre_um_domingo_entre_22_de_marco_e_25_de_abril(ano: int) -> None:
    domingo = pascoa(ano)

    assert domingo.weekday() == 6
    assert date(ano, 3, 22) <= domingo <= date(ano, 4, 25)


def test_os_feriados_de_2025_sao_os_do_expediente_bancario() -> None:
    assert sorted(feriados(2025)) == [
        date(2025, 1, 1),
        date(2025, 3, 3),
        date(2025, 3, 4),
        date(2025, 4, 18),
        date(2025, 4, 21),
        date(2025, 5, 1),
        date(2025, 6, 19),
        date(2025, 9, 7),
        date(2025, 10, 12),
        date(2025, 11, 2),
        date(2025, 11, 15),
        date(2025, 11, 20),
        date(2025, 12, 25),
    ]


def test_o_fim_de_semana_e_o_feriado_nao_sao_dia_util() -> None:
    assert dia_util(date(2025, 11, 19))
    assert not dia_util(date(2025, 11, 20))
    assert not dia_util(date(2025, 11, 22))
    assert not dia_util(date(2025, 11, 23))


@pytest.mark.parametrize(
    ("ano", "mes", "dia"),
    [
        # O 1º de janeiro tira um dia da conta.
        (2025, 1, 8),
        # O Carnaval tira dois.
        (2025, 3, 11),
        # O mês que começa no sábado.
        (2025, 11, 7),
        (2025, 12, 5),
    ],
)
def test_o_salario_cai_ate_o_quinto_dia_util(ano: int, mes: int, dia: int) -> None:
    assert quinto_dia_util(ano, mes) == date(ano, mes, dia)


@given(st.integers(min_value=2000, max_value=2100), st.integers(min_value=1, max_value=12))
def test_o_quinto_dia_util_e_util_e_tem_quatro_dias_uteis_antes_no_mes(ano: int, mes: int) -> None:
    quinto = quinto_dia_util(ano, mes)
    antes = [date(ano, mes, 1) + timedelta(days=dias) for dias in range(quinto.day - 1)]

    assert quinto.month == mes
    assert dia_util(quinto)
    assert sum(dia_util(dia) for dia in antes) == 4
