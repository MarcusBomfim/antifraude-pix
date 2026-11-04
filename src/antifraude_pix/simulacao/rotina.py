"""A rotina de cada cliente: o salário, o aluguel, as compras, os Pix para quem ele conhece — e os golpes (ADR-0003).

O cliente legítimo também faz o que parece golpe: troca de celular, paga um
desconhecido na feira, compra um carro usado e manda o valor de uma vez, faz
Pix de madrugada. Sem isso, o modelo aprenderia que todo recebedor novo é
golpe, e bloquearia metade do país.

A rotina monta primeiro a agenda do cliente — o que ele pretende fazer, e o
que o golpista vai pedir — e depois a executa em ordem, como o banco: o Pix
sem saldo não sai, e o noturno não passa do limite.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, timedelta
from enum import IntEnum

import numpy as np
import numpy.typing as npt

from antifraude_pix.simulacao.calendario import dia_util
from antifraude_pix.simulacao.configuracao import Configuracao
from antifraude_pix.simulacao.golpes import Ataque, Passo, passos
from antifraude_pix.simulacao.pix import DIA, HORA, Iniciacao, inicio_do_dia, noite, segundos
from antifraude_pix.simulacao.populacao import Cliente, Populacao
from antifraude_pix.simulacao.sorteio import Etapa, sorteio

APARELHOS_POR_CLIENTE = 16
"""Cada cliente tem dezesseis ids de aparelho, a partir de dezesseis vezes o id dele.

O primeiro é o aparelho com que ele chega; o segundo, o que ele compra depois.
"""

MINIMO_DO_GOLPE = 5_000
"""Abaixo de R$ 50 de saldo, o golpista não insiste."""


class Categoria(IntEnum):
    """Para quem vai um Pix comum."""

    COMERCIO_HABITUAL = 0
    CONTATO = 1
    COMERCIO_NOVO = 2
    PESSOA_NOVA = 3
    GRANDE = 4
    """O Pix grande e raro: o carro usado, a reforma, o conserto — quase sempre para alguém novo."""
    SERVICO = 5
    """O autônomo: a diarista, o eletricista, o frete.

    Uma pessoa nova, de dia, em valor redondo: como no falso parente.
    """
    AJUDA_DA_FAMILIA = 6
    """Os pais ajudando o filho, o filho ajudando a mãe: um contato, em valor redondo, às vezes alto."""


# A chance de cada categoria num Pix comum.
CHANCES: dict[Categoria, float] = {
    Categoria.COMERCIO_HABITUAL: 0.50,
    Categoria.CONTATO: 0.23,
    Categoria.COMERCIO_NOVO: 0.09,
    Categoria.PESSOA_NOVA: 0.10,
    Categoria.GRANDE: 0.006,
    Categoria.SERVICO: 0.05,
    Categoria.AJUDA_DA_FAMILIA: 0.024,
}

# O valor típico de cada categoria, em centavos, e o quanto ele varia (o desvio do logaritmo).
VALORES: dict[Categoria, tuple[int, float]] = {
    Categoria.COMERCIO_HABITUAL: (3_800, 0.9),
    Categoria.CONTATO: (9_000, 1.0),
    Categoria.COMERCIO_NOVO: (6_000, 1.0),
    Categoria.PESSOA_NOVA: (8_000, 1.1),
}

# As horas de cada categoria: o comércio no almoço e no fim do dia; as pessoas a qualquer hora, um pouco de madrugada.
_COMERCIO = (
    0.2, 0.1, 0.05, 0.05, 0.05, 0.1, 0.3, 0.8, 1.2, 1.4, 1.6, 2.0,
    2.6, 2.2, 1.6, 1.4, 1.5, 1.8, 2.3, 2.6, 2.4, 1.8, 1.0, 0.5,
)  # fmt: skip
_PESSOAS = (
    0.6, 0.3, 0.2, 0.1, 0.1, 0.2, 0.5, 0.9, 1.2, 1.4, 1.5, 1.6,
    1.7, 1.6, 1.5, 1.5, 1.6, 1.8, 2.0, 2.2, 2.2, 2.0, 1.6, 1.0,
)  # fmt: skip
_HORARIO_COMERCIAL = tuple(1.0 if 9 <= hora < 18 else 0.05 for hora in range(24))
HORAS: dict[Categoria, tuple[float, ...]] = {
    Categoria.COMERCIO_HABITUAL: _COMERCIO,
    Categoria.COMERCIO_NOVO: _COMERCIO,
    Categoria.CONTATO: _PESSOAS,
    Categoria.PESSOA_NOVA: _PESSOAS,
    Categoria.GRANDE: _HORARIO_COMERCIAL,
    Categoria.SERVICO: _HORARIO_COMERCIAL,
    Categoria.AJUDA_DA_FAMILIA: _PESSOAS,
}

# Como cada categoria inicia o Pix: o comércio pelo QR Code; as pessoas pela
# chave, às vezes digitando a agência e a conta.
INICIACOES: dict[Categoria, tuple[tuple[Iniciacao, float], ...]] = {
    Categoria.COMERCIO_HABITUAL: ((Iniciacao.QR_CODE, 0.7), (Iniciacao.CHAVE, 0.3)),
    Categoria.COMERCIO_NOVO: ((Iniciacao.QR_CODE, 0.6), (Iniciacao.CHAVE, 0.4)),
    Categoria.CONTATO: ((Iniciacao.CHAVE, 0.9), (Iniciacao.MANUAL, 0.1)),
    Categoria.PESSOA_NOVA: ((Iniciacao.CHAVE, 0.85), (Iniciacao.QR_CODE, 0.1), (Iniciacao.MANUAL, 0.05)),
    Categoria.GRANDE: ((Iniciacao.CHAVE, 0.7), (Iniciacao.MANUAL, 0.3)),
    Categoria.SERVICO: ((Iniciacao.CHAVE, 0.9), (Iniciacao.MANUAL, 0.1)),
    Categoria.AJUDA_DA_FAMILIA: ((Iniciacao.CHAVE, 0.9), (Iniciacao.MANUAL, 0.1)),
}


@dataclass(slots=True)
class Registro:
    """O que as rotinas produzem, em colunas: milhões de Pix cabem na memória assim, e viram tabela sem cópia."""

    instante: list[int] = field(default_factory=list)
    cliente: list[int] = field(default_factory=list)
    aparelho: list[int] = field(default_factory=list)
    recebedor: list[int] = field(default_factory=list)
    iniciacao: list[str] = field(default_factory=list)
    valor: list[int] = field(default_factory=list)
    saldo_antes: list[int] = field(default_factory=list)
    ataque: list[int] = field(default_factory=list)
    """O ataque a que o Pix pertence, ou -1. É verdade do simulador: o banco não vê esta coluna."""
    trocas: list[tuple[int, int, int]] = field(default_factory=list)
    """Os aparelhos novos, como o banco os vê: (cliente, instante, aparelho) — do cliente e da quadrilha."""


@dataclass(frozen=True, slots=True)
class DiasDaSimulacao:
    """O que vale para todos os clientes em cada dia: o movimento do comércio e os dias úteis de cada mês."""

    dias: tuple[date, ...]
    movimento: npt.NDArray[np.float64]
    uteis_do_mes: dict[tuple[int, int], tuple[date, ...]]

    @staticmethod
    def de(config: Configuracao) -> "DiasDaSimulacao":
        dias = tuple(config.inicio + timedelta(days=dia) for dia in range(config.dias))
        meses = sorted({(dia.year, dia.month) for dia in dias})
        uteis = {
            (ano, mes): tuple(
                dia
                for dia in (date(ano, mes, 1) + timedelta(days=d) for d in range(31))
                if dia.month == mes and dia_util(dia)
            )
            for ano, mes in meses
        }

        return DiasDaSimulacao(dias, np.array([_movimento(dia) for dia in dias]), uteis)


@dataclass(frozen=True, slots=True)
class Destinos:
    """As contas novas que um Pix comum pode encontrar: qualquer pessoa, qualquer comércio.

    Montadas uma vez para todos os clientes, e não uma vez por cliente.
    """

    pessoas: npt.NDArray[np.int64]
    comercios: npt.NDArray[np.int64]

    @staticmethod
    def de(populacao: Populacao) -> "Destinos":
        return Destinos(
            np.array(populacao.pessoas, dtype=np.int64), np.array(populacao.todos_os_comercios, dtype=np.int64)
        )


def _movimento(dia: date) -> float:
    # O sábado enche o comércio, o domingo e o feriado esvaziam, e dezembro é dezembro.
    fator = {5: 1.25, 6: 0.8}.get(dia.weekday(), 1.0)

    if dia.weekday() < 5 and not dia_util(dia):
        fator = 0.8

    return fator * (1.15 if dia.month == 12 else 1.0)


# O que pode acontecer na conta do cliente.


@dataclass(frozen=True, slots=True)
class _Salario:
    renda: int


@dataclass(frozen=True, slots=True)
class _Troca:
    aparelho: int


@dataclass(frozen=True, slots=True)
class _Pedido:
    recebedor: int
    iniciacao: Iniciacao
    valor: int


@dataclass(frozen=True, slots=True)
class _DoGolpe:
    ataque: Ataque
    passo: Passo


type _Acontecimento = _Salario | _Troca | _Pedido | _DoGolpe
type _Agendar = Callable[[int, _Acontecimento], None]

# A ordem do que cai no mesmo segundo: o salário entra antes de o dinheiro sair, e o golpista entra antes de mandar.
_ORDEM: dict[type, int] = {_Salario: 0, _Troca: 1, _DoGolpe: 2, _Pedido: 3}


def simular_cliente(
    config: Configuracao,
    cliente: Cliente,
    *,
    destinos: Destinos,
    dias: DiasDaSimulacao,
    ataques: tuple[Ataque, ...],
    registro: Registro,
) -> None:
    """Simula os dias do cliente, com os ataques que ele sofre, e anota o que aconteceu."""
    gerador = sorteio(config, Etapa.ROTINA, cliente.id)
    agenda: list[tuple[int, int, int, _Acontecimento]] = []

    def agendar(instante: int, acontecimento: _Acontecimento) -> None:
        agenda.append((instante, _ORDEM[type(acontecimento)], len(agenda), acontecimento))

    dias_do_salario = _salario(gerador, config, dias, cliente, agendar)
    _aluguel(gerador, dias, cliente, agendar)
    _troca_de_aparelho(gerador, config, cliente, agendar)
    _pix_comuns(
        gerador, config, cliente, destinos=destinos, dias=dias, dias_do_salario=dias_do_salario, agendar=agendar
    )

    for ataque in ataques:
        if ataque.aparelho_do_golpista is not None:
            agendar(segundos(ataque.inicio), _Troca(ataque.aparelho_do_golpista))

        for passo in passos(config, ataque):
            agendar(passo.instante, _DoGolpe(ataque, passo))

    agenda.sort(key=lambda item: item[:3])
    _executar(
        gerador,
        config,
        cliente,
        acontecimentos=[acontecimento for *_, acontecimento in agenda],
        instantes=[item[0] for item in agenda],
        registro=registro,
    )


def _salario(
    gerador: np.random.Generator, config: Configuracao, dias: DiasDaSimulacao, cliente: Cliente, agendar: _Agendar
) -> list[int]:
    # Cada empregador paga num dos cinco primeiros dias úteis, sempre no mesmo; o crédito cai de manhã cedo.
    qual = int(gerador.integers(0, 5))
    indices = []

    for uteis in dias.uteis_do_mes.values():
        dia = uteis[qual]

        if config.inicio <= dia < config.fim:
            agendar(inicio_do_dia(dia) + 7 * HORA, _Salario(cliente.renda_mensal))
            indices.append((dia - config.inicio).days)

    return indices


def _aluguel(gerador: np.random.Generator, dias: DiasDaSimulacao, cliente: Cliente, agendar: _Agendar) -> None:
    if cliente.aluguel is None:
        return

    vencimento = int(gerador.integers(5, 11))

    for ano, mes in dias.uteis_do_mes:
        dia = date(ano, mes, vencimento)

        if dias.dias[0] <= dia <= dias.dias[-1]:
            instante = inicio_do_dia(dia) + int(gerador.integers(9 * HORA, 21 * HORA))
            agendar(instante, _Pedido(cliente.aluguel.locador, Iniciacao.CHAVE, cliente.aluguel.valor))


def _troca_de_aparelho(gerador: np.random.Generator, config: Configuracao, cliente: Cliente, agendar: _Agendar) -> None:
    # Um em cada oito clientes troca de celular no semestre: o banco vê um aparelho novo, e não é golpe.
    if gerador.random() < 0.12:
        dia = config.inicio + timedelta(days=int(gerador.integers(config.dias)))
        instante = inicio_do_dia(dia) + int(gerador.integers(9 * HORA, 22 * HORA))
        agendar(instante, _Troca(cliente.id * APARELHOS_POR_CLIENTE + 1))


def _pix_comuns(
    gerador: np.random.Generator,
    config: Configuracao,
    cliente: Cliente,
    *,
    destinos: Destinos,
    dias: DiasDaSimulacao,
    dias_do_salario: list[int],
    agendar: _Agendar,
) -> None:
    # O dinheiro sai mais nos três dias depois do salário.
    animo = np.ones(config.dias)

    for indice in dias_do_salario:
        animo[indice : indice + 3] *= 1.4

    quantos = gerador.poisson(cliente.pix_por_mes / 30 * dias.movimento * animo)
    total = int(quantos.sum())
    categorias = gerador.choice(len(Categoria), size=total, p=_pesos(tuple(CHANCES.values())))

    # O comércio e o contato de sempre vêm do círculo do cliente, os mais
    # frequentes primeiro; o novo, de qualquer lugar.
    recebedores = np.select(
        [
            categorias == Categoria.COMERCIO_HABITUAL,
            (categorias == Categoria.CONTATO) | (categorias == Categoria.AJUDA_DA_FAMILIA),
            categorias == Categoria.COMERCIO_NOVO,
        ],
        [
            _por_posicao(gerador, cliente.comercios, total, 0.9),
            _por_posicao(gerador, cliente.contatos, total, 0.8),
            gerador.choice(destinos.comercios, size=total),
        ],
        default=gerador.choice(destinos.pessoas, size=total),
    )

    horas = np.zeros(total, dtype=np.int64)
    iniciacoes = np.zeros(total, dtype=np.int64)
    valores = np.zeros(total, dtype=np.int64)

    # Quem ganha mais gasta mais em cada Pix, mas não na mesma proporção.
    escala = (cliente.renda_mensal / 280_000) ** 0.4

    for categoria in Categoria:
        onde = np.flatnonzero(categorias == categoria)
        horas[onde] = gerador.choice(24, size=onde.size, p=_pesos(HORAS[categoria]))
        iniciacoes[onde] = _iniciacoes(gerador, categoria, onde.size)
        valores[onde] = _valores(gerador, categoria, escala, onde.size)

    dia_de_cada = np.repeat(np.arange(config.dias), quantos)
    instantes = inicio_do_dia(config.inicio) + dia_de_cada * DIA + horas * HORA + gerador.integers(0, HORA, size=total)
    todas = tuple(Iniciacao)

    for instante, recebedor, iniciacao, valor in zip(
        instantes.tolist(), recebedores.tolist(), iniciacoes.tolist(), valores.tolist(), strict=True
    ):
        agendar(instante, _Pedido(recebedor, todas[iniciacao], valor))


def _pesos(pesos: tuple[float, ...]) -> npt.NDArray[np.float64]:
    vetor = np.array(pesos)
    normalizado: npt.NDArray[np.float64] = vetor / vetor.sum()

    return normalizado


def _por_posicao(
    gerador: np.random.Generator, opcoes: tuple[int, ...], quantos: int, decaimento: float
) -> npt.NDArray[np.int64]:
    pesos = 1 / np.arange(1, len(opcoes) + 1) ** decaimento

    return gerador.choice(np.array(opcoes), size=quantos, p=pesos / pesos.sum())


def _iniciacoes(gerador: np.random.Generator, categoria: Categoria, quantas: int) -> npt.NDArray[np.int64]:
    opcoes = INICIACOES[categoria]
    posicoes = {iniciacao: posicao for posicao, iniciacao in enumerate(Iniciacao)}
    codigos = np.array([posicoes[iniciacao] for iniciacao, _ in opcoes])

    return codigos[gerador.choice(len(opcoes), size=quantas, p=_pesos(tuple(peso for _, peso in opcoes)))]


def _valores(gerador: np.random.Generator, categoria: Categoria, escala: float, quantos: int) -> npt.NDArray[np.int64]:
    if categoria is Categoria.GRANDE:
        # De R$ 1.500 a R$ 15.000, em valor redondo: quem compra um carro usado não manda centavos.
        grandes = np.exp(gerador.uniform(np.log(150_000), np.log(1_500_000), size=quantos)) * escala
        return np.maximum(10_000, np.round(grandes, -4)).astype(np.int64)

    if categoria is Categoria.SERVICO:
        # Por volta de R$ 250, de R$ 100 a R$ 1.000 quase sempre, e três em quatro redondos: o preço combinado.
        servicos = np.exp(gerador.normal(np.log(25_000), 0.6, size=quantos)) * escala**0.5
        redondos = gerador.random(quantos) < 0.75
        valores_de_servico: npt.NDArray[np.int64] = np.where(
            redondos, np.maximum(1_000, np.round(servicos, -3)), np.round(servicos)
        ).astype(np.int64)
        return valores_de_servico

    if categoria is Categoria.AJUDA_DA_FAMILIA:
        # De R$ 200 a R$ 2.000, quase sempre redondo: a ajuda com o aluguel, com o remédio.
        ajudas = np.exp(gerador.uniform(np.log(20_000), np.log(200_000), size=quantos)) * escala**0.5
        redondos = gerador.random(quantos) < 0.85
        valores_de_ajuda: npt.NDArray[np.int64] = np.where(
            redondos, np.maximum(1_000, np.round(ajudas, -3)), np.round(ajudas)
        ).astype(np.int64)
        return valores_de_ajuda

    tipico, desvio = VALORES[categoria]
    valores = np.maximum(100, np.round(np.exp(gerador.normal(np.log(tipico), desvio, size=quantos)) * escala))

    # Entre pessoas, quase metade dos valores é redonda: R$ 50, R$ 120.
    if categoria in {Categoria.CONTATO, Categoria.PESSOA_NOVA}:
        redondos = gerador.random(quantos) < 0.45
        valores[redondos] = np.maximum(1_000, np.round(valores[redondos], -3))

    return valores.astype(np.int64)


def _executar(
    gerador: np.random.Generator,
    config: Configuracao,
    cliente: Cliente,
    *,
    acontecimentos: list[_Acontecimento],
    instantes: list[int],
    registro: Registro,
) -> None:
    """O banco processa o que o cliente — ou o golpista — pediu, em ordem.

    Sem saldo, o Pix não sai; de noite, só até o limite.
    """
    saldo = cliente.saldo_inicial
    aparelho = cliente.id * APARELHOS_POR_CLIENTE
    gasto_na_noite: dict[int, int] = {}

    for instante, acontecimento in zip(instantes, acontecimentos, strict=True):
        qual_noite = noite(config, instante)
        restante = None if qual_noite is None else cliente.limite_noturno - gasto_na_noite.get(qual_noite, 0)
        valor: int | None

        match acontecimento:
            case _Salario(renda):
                saldo += renda
                continue
            case _Troca(novo):
                registro.trocas.append((cliente.id, instante, novo))

                # O aparelho do golpista não vira o do cliente: o cliente continua no dele.
                if novo < (cliente.id + 1) * APARELHOS_POR_CLIENTE:
                    aparelho = novo
                continue
            case _Pedido(recebedor, iniciacao, pedido):
                # Sem saldo para tudo, o cliente manda menos — ou não manda.
                valor = pedido if pedido <= saldo else round(saldo * float(gerador.uniform(0.3, 0.9)))

                if valor is not None and (valor < 100 or (restante is not None and valor > restante)):
                    valor = None

                usado, ataque = aparelho, -1
            case _DoGolpe(ataque_do_passo, passo):
                valor = _valor_do_golpe(passo, saldo, restante)
                recebedor, iniciacao = passo.laranja, passo.iniciacao
                usado = (
                    ataque_do_passo.aparelho_do_golpista
                    if ataque_do_passo.aparelho_do_golpista is not None
                    else aparelho
                )
                ataque = ataque_do_passo.id

        if valor is None:
            continue

        registro.instante.append(instante)
        registro.cliente.append(cliente.id)
        registro.aparelho.append(usado)
        registro.recebedor.append(recebedor)
        registro.iniciacao.append(iniciacao.value)
        registro.valor.append(valor)
        registro.saldo_antes.append(saldo)
        registro.ataque.append(ataque)
        saldo -= valor

        if qual_noite is not None:
            gasto_na_noite[qual_noite] = gasto_na_noite.get(qual_noite, 0) + valor


def _valor_do_golpe(passo: Passo, saldo: int, restante_da_noite: int | None) -> int | None:
    teto = saldo if restante_da_noite is None else min(saldo, restante_da_noite)

    if passo.valor is not None:
        # O valor pedido sai inteiro, ou não sai: a vítima não manda metade da "emergência".
        return passo.valor if passo.valor <= teto else None

    assert passo.fracao_do_saldo is not None
    valor = int(min(teto, round(saldo * passo.fracao_do_saldo, -2)))

    return valor if valor >= MINIMO_DO_GOLPE else None


def para_quem(ataques: tuple[Ataque, ...], clientes: int) -> list[tuple[Ataque, ...]]:
    """Os ataques de cada cliente, na ordem dos clientes."""
    por_cliente: list[list[Ataque]] = [[] for _ in range(clientes)]

    for ataque in ataques:
        por_cliente[ataque.cliente].append(ataque)

    return [tuple(dele) for dele in por_cliente]
