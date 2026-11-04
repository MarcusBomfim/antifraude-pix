"""O calendário bancário: feriados nacionais, dias úteis e o dia do salário.

O dinheiro do brasileiro anda no ritmo do calendário: o salário cai até o
quinto dia útil (CLT, art. 459), o comércio enche no sábado e na Black
Friday, o feriado esvazia a segunda-feira. O simulador usa este calendário
para que o Pix de mentira tenha o mesmo pulso do Pix de verdade.
"""

from datetime import date, timedelta
from functools import cache

# Os feriados nacionais de data fixa, como (mês, dia). O Dia da Consciência
# Negra é feriado nacional desde a Lei 14.759/2023.
_FIXOS: tuple[tuple[int, int], ...] = (
    (1, 1),  # Confraternização Universal
    (4, 21),  # Tiradentes
    (5, 1),  # Dia do Trabalho
    (9, 7),  # Independência
    (10, 12),  # Nossa Senhora Aparecida
    (11, 2),  # Finados
    (11, 15),  # Proclamação da República
    (11, 20),  # Dia Nacional de Zumbi e da Consciência Negra
    (12, 25),  # Natal
)


def pascoa(ano: int) -> date:
    """O domingo de Páscoa do ano, pelo algoritmo de Meeus/Jones/Butcher para o calendário gregoriano."""
    a = ano % 19
    b, c = divmod(ano, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    m = (32 + 2 * e + 2 * i - h - k) % 7
    n = (a + 11 * h + 22 * m) // 451
    mes, dia = divmod(h + m - 7 * n + 114, 31)

    return date(ano, mes, dia + 1)


@cache
def feriados(ano: int) -> frozenset[date]:
    """Os dias sem expediente bancário no ano: os feriados nacionais e os móveis da Páscoa.

    A segunda e a terça de Carnaval e o Corpus Christi são ponto facultativo
    para o governo, mas os bancos não abrem — e é o expediente bancário que
    decide o quinto dia útil do salário.
    """
    domingo = pascoa(ano)
    moveis = {
        domingo - timedelta(days=48),  # segunda de Carnaval
        domingo - timedelta(days=47),  # terça de Carnaval
        domingo - timedelta(days=2),  # Sexta-feira da Paixão
        domingo + timedelta(days=60),  # Corpus Christi
    }

    return frozenset({date(ano, mes, dia) for mes, dia in _FIXOS} | moveis)


def dia_util(dia: date) -> bool:
    """O banco abre: de segunda a sexta, fora dos feriados."""
    return dia.weekday() < 5 and dia not in feriados(dia.year)


def quinto_dia_util(ano: int, mes: int) -> date:
    """O último dia em que o salário do mês pode cair."""
    dia = date(ano, mes, 1)
    contados = 0

    while True:
        if dia_util(dia):
            contados += 1

            if contados == 5:
                return dia

        dia += timedelta(days=1)
