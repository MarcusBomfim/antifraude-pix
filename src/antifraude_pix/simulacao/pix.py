"""O que todo Pix tem: como foi iniciado, e o relógio em que a simulação o marca.

Dentro do simulador, o instante é um inteiro: os segundos desde 1970 no
horário de Brasília, sem fuso. Milhões de Pix ordenam-se depressa assim, e o
Brasil não tem horário de verão desde 2019 — a hora do relógio é a hora do
comércio, do salário e do golpe. O fuso só entra quando os dados saem.
"""

from datetime import date, datetime, timedelta
from enum import StrEnum

from antifraude_pix.simulacao.configuracao import Configuracao

# O relógio local da simulação não tem fuso, de propósito.
_ZERO = datetime(1970, 1, 1)  # noqa: DTZ001
HORA = 3600
DIA = 24 * HORA


class Iniciacao(StrEnum):
    CHAVE = "chave"
    QR_CODE = "qr_code"
    MANUAL = "manual"
    """Pela agência e pela conta, digitadas — sem chave."""


def segundos(momento: datetime) -> int:
    """O instante da simulação de uma data e hora locais."""
    return int((momento - _ZERO).total_seconds())


def momento(instante: int) -> datetime:
    return _ZERO + timedelta(seconds=instante)


def inicio_do_dia(dia: date) -> int:
    return (dia - _ZERO.date()).days * DIA


def hora(instante: int) -> int:
    return (instante // HORA) % 24


def noite(config: Configuracao, instante: int) -> int | None:
    """De que noite é o instante, para o limite noturno, ou nada, de dia.

    As 22h de segunda e as 3h de terça são a mesma noite: a que começou na segunda às 20h.
    """
    if not config.noturno(hora(instante)):
        return None

    return (instante - config.inicio_da_noite * HORA) // DIA
