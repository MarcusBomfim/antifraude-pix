"""Os golpes: quais acontecem, quando, contra quem, e para que contas laranja (ADR-0003).

Quatro golpes, do jeito que acontecem no Brasil:

- **falso parente**: alguém diz no WhatsApp que é o filho, com número novo, e
  pede um Pix para uma emergência. A vítima paga do próprio celular, de dia,
  e às vezes paga de novo, "só mais um pouco";
- **falsa central**: alguém liga dizendo ser do banco, fala de uma compra
  suspeita e convence a vítima a passar o dinheiro para uma "conta segura",
  em vários Pix seguidos, até esvaziar a conta;
- **invasão de conta**: o golpista entra na conta por outro aparelho — com a
  senha roubada ou o chip clonado — e manda o dinheiro para várias contas,
  muitas vezes de madrugada, dentro do limite noturno;
- **venda falsa**: o anúncio de um produto que não existe; a vítima paga um
  desconhecido, e o produto nunca chega.

Aqui sai só o plano de cada ataque. Quanto cada Pix leva depende do saldo da
vítima na hora, e isso se decide na rotina dela.
"""

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from enum import StrEnum

import numpy as np
import numpy.typing as npt

from antifraude_pix.simulacao.configuracao import Configuracao
from antifraude_pix.simulacao.pix import DIA, HORA, Iniciacao, segundos
from antifraude_pix.simulacao.populacao import Cliente, Populacao, ataques_esperados
from antifraude_pix.simulacao.sorteio import Etapa, sorteio


class Golpe(StrEnum):
    FALSO_PARENTE = "falso_parente"
    FALSA_CENTRAL = "falsa_central"
    INVASAO_DE_CONTA = "invasao_de_conta"
    VENDA_FALSA = "venda_falsa"


GOLPES: tuple[Golpe, ...] = tuple(Golpe)

NOMES: dict[Golpe, str] = {
    Golpe.FALSO_PARENTE: "falso parente",
    Golpe.FALSA_CENTRAL: "falsa central",
    Golpe.INVASAO_DE_CONTA: "invasão de conta",
    Golpe.VENDA_FALSA: "venda falsa",
}
"""O nome de cada golpe, para quem lê."""

# A parte de cada golpe nos ataques de um mês comum.
MISTURA: dict[Golpe, float] = {
    Golpe.FALSO_PARENTE: 0.28,
    Golpe.FALSA_CENTRAL: 0.22,
    Golpe.INVASAO_DE_CONTA: 0.20,
    Golpe.VENDA_FALSA: 0.30,
}

APARELHOS_DO_GOLPISTA = 1_000_000_000
"""Onde começam os ids dos aparelhos das quadrilhas, longe dos aparelhos dos clientes."""


@dataclass(frozen=True, slots=True)
class Ataque:
    id: int
    golpe: Golpe
    cliente: int
    inicio: datetime
    laranjas: tuple[int, ...]
    """As contas para onde o dinheiro vai, na ordem em que o golpista as usa."""
    aparelho_do_golpista: int | None
    """Na invasão de conta, o aparelho de onde a quadrilha entra.

    É o mesmo, às vezes, em contas de vítimas diferentes: a quadrilha usa poucos aparelhos.
    """


def planejar_ataques(config: Configuracao, populacao: Populacao) -> tuple[Ataque, ...]:
    """Os ataques da simulação, em ordem de início."""
    gerador = sorteio(config, Etapa.ATAQUES)
    dias = [config.inicio + timedelta(days=dia) for dia in range(config.dias)]

    # O peso de cada golpe em cada dia: a mistura do mês comum, mexida pela época.
    pesos = np.array([[MISTURA[golpe] * _epoca(golpe, dia) for golpe in GOLPES] for dia in dias])
    quantos = int(gerador.poisson(ataques_esperados(config)))
    sorteados = gerador.choice(pesos.size, size=quantos, p=(pesos / pesos.sum()).ravel())

    invasoes = sum(1 for celula in sorteados if GOLPES[int(celula) % len(GOLPES)] is Golpe.INVASAO_DE_CONTA)
    quadrilhas = max(3, invasoes // 4)
    alvos = {golpe: _alvos(populacao.clientes, golpe) for golpe in GOLPES}
    planos = []

    for celula in sorted(int(celula) for celula in sorteados):
        dia, golpe = dias[celula // len(GOLPES)], GOLPES[celula % len(GOLPES)]
        cliente = populacao.clientes[int(gerador.choice(config.clientes, p=alvos[golpe]))]
        inicio = datetime.combine(
            dia, time(_hora(gerador, golpe), int(gerador.integers(60)), int(gerador.integers(60)))
        )
        laranjas = _laranjas(gerador, populacao, dia, golpe)
        aparelho = (
            APARELHOS_DO_GOLPISTA + int(gerador.integers(quadrilhas)) if golpe is Golpe.INVASAO_DE_CONTA else None
        )
        planos.append((inicio, golpe, cliente.id, laranjas, aparelho))

    planos.sort(key=lambda plano: (plano[0], plano[2]))

    return tuple(
        Ataque(id_, golpe, cliente, inicio, laranjas, aparelho)
        for id_, (inicio, golpe, cliente, laranjas, aparelho) in enumerate(planos)
    )


def _epoca(golpe: Golpe, dia: date) -> float:
    if golpe is Golpe.VENDA_FALSA:
        # A semana da Black Friday e as compras de Natal.
        if dia.month == 11 and dia.day >= 21:
            return 2.0
        if dia.month == 12 and dia.day <= 24:
            return 1.5

    if golpe is Golpe.FALSA_CENTRAL:
        # A "central do banco" liga em horário comercial, de segunda a sexta.
        return 0.3 if dia.weekday() >= 5 else 1.2

    return 1.0


def _alvos(clientes: tuple[Cliente, ...], golpe: Golpe) -> npt.NDArray[np.float64]:
    """O peso de cada cliente como vítima do golpe: o falso parente e a falsa central miram os mais velhos."""
    pesos = np.array([_peso_da_idade(golpe, cliente.idade) for cliente in clientes])
    normalizados: npt.NDArray[np.float64] = pesos / pesos.sum()

    return normalizados


def _peso_da_idade(golpe: Golpe, idade: int) -> float:
    match golpe:
        case Golpe.FALSO_PARENTE:
            return 4.0 if idade >= 65 else 2.5 if idade >= 55 else 1.0 if idade >= 35 else 0.5
        case Golpe.FALSA_CENTRAL:
            return 3.0 if idade >= 65 else 2.0 if idade >= 55 else 1.0
        case Golpe.INVASAO_DE_CONTA:
            return 1.2 if idade < 35 else 1.0
        case Golpe.VENDA_FALSA:
            return 1.5 if idade < 35 else 1.2 if idade < 55 else 0.7


# As horas em que cada golpe começa, com o peso de cada uma.
_HORAS: dict[Golpe, tuple[float, ...]] = {
    Golpe.FALSO_PARENTE: tuple(1.0 if 9 <= hora < 20 else 0.05 for hora in range(24)),
    Golpe.FALSA_CENTRAL: tuple(1.0 if 9 <= hora < 18 else 0.03 for hora in range(24)),
    # A madrugada, quando a vítima dorme e não vê o aviso do aplicativo.
    Golpe.INVASAO_DE_CONTA: tuple(1.6 if hora < 5 else 0.6 for hora in range(24)),
    Golpe.VENDA_FALSA: tuple(1.0 if 8 <= hora < 23 else 0.1 for hora in range(24)),
}


def _hora(gerador: np.random.Generator, golpe: Golpe) -> int:
    pesos = np.array(_HORAS[golpe])

    return int(gerador.choice(24, p=pesos / pesos.sum()))


def _laranjas(gerador: np.random.Generator, populacao: Populacao, dia: date, golpe: Golpe) -> tuple[int, ...]:
    quantas = {
        Golpe.FALSO_PARENTE: 1,
        Golpe.VENDA_FALSA: 1,
        Golpe.FALSA_CENTRAL: int(gerador.integers(1, 3)),
        Golpe.INVASAO_DE_CONTA: int(gerador.integers(2, 5)),
    }[golpe]

    ativas = [laranja for laranja in populacao.laranjas if laranja.ativa_em(dia)]

    if len(ativas) < quantas:
        # Sem laranjas bastantes naquele dia, entram as de janela mais próxima: o golpe não espera.
        distancia = sorted(
            populacao.laranjas,
            key=lambda laranja: min(abs((laranja.ativa_de - dia).days), abs((laranja.ativa_ate - dia).days)),
        )
        ativas = distancia[: max(quantas, len(ativas))]

    escolhidas = gerador.choice(len(ativas), size=quantas, replace=False)

    return tuple(ativas[int(indice)].conta for indice in escolhidas)


@dataclass(frozen=True, slots=True)
class Passo:
    """Um Pix do golpe, como o golpista o pede: para qual laranja, quando, e quanto."""

    instante: int
    laranja: int
    iniciacao: Iniciacao
    valor: int | None
    """O valor pedido, em centavos — ou nada, quando o golpista leva uma parte do saldo."""
    fracao_do_saldo: float | None
    """A parte do saldo que o golpista leva, quando ele vê o saldo: na falsa central e na invasão de conta."""


def passos(config: Configuracao, ataque: Ataque) -> tuple[Passo, ...]:
    """Os Pix do ataque, em ordem. Sempre os mesmos para o mesmo ataque, venha ele antes ou depois de outros."""
    gerador = sorteio(config, Etapa.ATAQUES, ataque.id)
    inicio = segundos(ataque.inicio)

    match ataque.golpe:
        case Golpe.FALSO_PARENTE:
            return _falso_parente(gerador, ataque, inicio)
        case Golpe.FALSA_CENTRAL if config.virada is not None and ataque.inicio.date() >= config.virada:
            return _falsa_central_parcelada(gerador, ataque, inicio, fim=segundos(datetime.combine(config.fim, time())))
        case Golpe.FALSA_CENTRAL:
            return _falsa_central(gerador, ataque, inicio)
        case Golpe.INVASAO_DE_CONTA:
            return _invasao_de_conta(gerador, ataque, inicio)
        case Golpe.VENDA_FALSA:
            return _venda_falsa(gerador, ataque, inicio)


def _log_uniforme(gerador: np.random.Generator, de: int, ate: int) -> float:
    return float(np.exp(gerador.uniform(np.log(de), np.log(ate))))


def _redondo(centavos: float) -> int:
    """O valor que o golpista pede: em dezenas de reais, como quem pede ajuda."""
    return max(1_000, round(centavos / 1_000) * 1_000)


def _falso_parente(gerador: np.random.Generator, ataque: Ataque, inicio: int) -> tuple[Passo, ...]:
    # Um, dois ou três Pix: depois do primeiro, "deu um problema, manda mais um pouco".
    quantos = int(gerador.choice([1, 2, 3], p=[0.6, 0.3, 0.1]))
    # De R$ 150 — "pra pagar uma conta que venceu hoje" — a R$ 3.000.
    valor = _redondo(_log_uniforme(gerador, 15_000, 300_000))
    instante, resultado = inicio, []

    for _ in range(quantos):
        iniciacao = Iniciacao.CHAVE if gerador.random() < 0.9 else Iniciacao.MANUAL
        resultado.append(Passo(instante, ataque.laranjas[0], iniciacao, valor, None))
        instante += int(gerador.integers(5, 61)) * 60
        valor = _redondo(valor * float(gerador.uniform(0.5, 1.0)))

    return tuple(resultado)


def _falsa_central(gerador: np.random.Generator, ataque: Ataque, inicio: int) -> tuple[Passo, ...]:
    # De dois a seis Pix em minutos, cada um levando uma parte do que sobrou, até
    # esvaziar a conta. A "central" às vezes dita a agência e a conta, em vez de
    # uma chave.
    quantos = int(gerador.integers(2, 7))
    instante, resultado = inicio, []

    for vez in range(quantos):
        iniciacao = Iniciacao.MANUAL if gerador.random() < 0.4 else Iniciacao.CHAVE
        laranja = ataque.laranjas[vez % len(ataque.laranjas)]
        resultado.append(Passo(instante, laranja, iniciacao, None, float(gerador.uniform(0.3, 0.7))))
        instante += int(gerador.integers(2, 16)) * 60

    return tuple(resultado)


def _falsa_central_parcelada(
    gerador: np.random.Generator, ataque: Ataque, inicio: int, *, fim: int
) -> tuple[Passo, ...]:
    """A falsa central depois da virada: um Pix por dia, de três a seis dias, para a mesma "conta segura" (ADR-0019).

    A "central" diz que a conta foi invadida e que o dinheiro tem de sair aos
    poucos, "para o sistema não bloquear". Cada Pix leva de um quinto a um
    terço do saldo, pela chave, perto da hora do primeiro. Sem a pressa dos
    Pix em minutos, e com o recebedor que, do segundo Pix em diante, o cliente
    já pagou, o golpe se parece com um pagamento combinado.
    """
    quantos = int(gerador.integers(3, 7))
    resultado = []

    for dia in range(quantos):
        instante = inicio + dia * DIA + (int(gerador.integers(-2 * HORA, 2 * HORA)) if dia else 0)

        if instante >= fim:
            break

        resultado.append(Passo(instante, ataque.laranjas[0], Iniciacao.CHAVE, None, float(gerador.uniform(0.2, 0.35))))

    return tuple(resultado)


def _invasao_de_conta(gerador: np.random.Generator, ataque: Ataque, inicio: int) -> tuple[Passo, ...]:
    # O golpista entra pelo aparelho dele, espera uns minutos e espalha o dinheiro por várias laranjas, depressa.
    quantos = int(gerador.integers(2, 7))
    instante, resultado = inicio + int(gerador.integers(3, 31)) * 60, []

    for vez in range(quantos):
        laranja = ataque.laranjas[vez % len(ataque.laranjas)]
        resultado.append(Passo(instante, laranja, Iniciacao.CHAVE, None, float(gerador.uniform(0.2, 0.6))))
        instante += int(gerador.integers(1, 9)) * 60

    return tuple(resultado)


def _venda_falsa(gerador: np.random.Generator, ataque: Ataque, inicio: int) -> tuple[Passo, ...]:
    # O preço do anúncio, que termina em noventa centavos; às vezes, depois, o "frete".
    preco = max(1_990, round(_log_uniforme(gerador, 12_000, 250_000) / 100) * 100 - 10)
    iniciacao = Iniciacao.CHAVE if gerador.random() < 0.7 else Iniciacao.QR_CODE
    resultado = [Passo(inicio, ataque.laranjas[0], iniciacao, preco, None)]

    if gerador.random() < 0.15:
        frete = round(preco * float(gerador.uniform(0.1, 0.3)) / 100) * 100 - 10
        resultado.append(
            Passo(inicio + int(gerador.integers(10, 181)) * 60, ataque.laranjas[0], iniciacao, frete, None)
        )

    return tuple(resultado)
