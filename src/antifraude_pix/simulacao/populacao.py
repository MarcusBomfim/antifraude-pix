"""A população: os clientes do banco e as contas para onde o dinheiro deles vai.

O simulador olha o Pix do lado de quem paga: os clientes são os de um banco
só, que decide se deixa cada Pix sair. Do outro lado estão as pessoas que eles
conhecem, os comércios onde compram e as contas laranja, que recebem o
dinheiro dos golpes e o repassam. O banco não sabe quem é laranja; o
simulador sabe, e guarda essa verdade à parte (ADR-0002).
"""

from dataclasses import dataclass
from datetime import date, timedelta
from enum import StrEnum

import numpy as np
import numpy.typing as npt

from antifraude_pix.simulacao.configuracao import Configuracao
from antifraude_pix.simulacao.sorteio import Etapa, sorteio


class TipoDePessoa(StrEnum):
    FISICA = "PF"
    JURIDICA = "PJ"


class TipoDeChave(StrEnum):
    """A chave Pix por que a conta recebe."""

    CPF = "cpf"
    CNPJ = "cnpj"
    CELULAR = "celular"
    EMAIL = "email"
    ALEATORIA = "aleatoria"


@dataclass(frozen=True, slots=True)
class Banco:
    codigo: str
    nome: str
    digital: bool
    """Banco que abre conta pelo aplicativo, em minutos — onde a conta laranja nasce com mais facilidade."""


# Os bancos das contas que recebem, com a parte de cada um. É uma suposição
# simples, de ordem de grandeza: os grandes, e os digitais, que somam quase
# metade das contas.
BANCOS: tuple[tuple[Banco, float], ...] = (
    (Banco("260", "Nubank", digital=True), 0.18),
    (Banco("341", "Itaú", digital=False), 0.13),
    (Banco("104", "Caixa", digital=False), 0.13),
    (Banco("001", "Banco do Brasil", digital=False), 0.12),
    (Banco("237", "Bradesco", digital=False), 0.11),
    (Banco("033", "Santander", digital=False), 0.08),
    (Banco("077", "Inter", digital=True), 0.06),
    (Banco("323", "Mercado Pago", digital=True), 0.06),
    (Banco("380", "PicPay", digital=True), 0.05),
    (Banco("336", "C6 Bank", digital=True), 0.04),
    (Banco("290", "PagBank", digital=True), 0.04),
)

# A população adulta por estado, em proporção aproximada à do Censo 2022.
UFS: tuple[tuple[str, float], ...] = (
    ("SP", 21.9), ("MG", 10.0), ("RJ", 7.9), ("BA", 6.9), ("PR", 5.6), ("RS", 5.3), ("PE", 4.5),
    ("CE", 4.4), ("PA", 4.0), ("SC", 3.7), ("GO", 3.5), ("MA", 3.3), ("AM", 1.9), ("ES", 1.9),
    ("PB", 1.9), ("MT", 1.8), ("RN", 1.6), ("PI", 1.6), ("AL", 1.5), ("DF", 1.4), ("MS", 1.4),
    ("SE", 1.1), ("RO", 0.8), ("TO", 0.7), ("AC", 0.4), ("AP", 0.4), ("RR", 0.3),
)  # fmt: skip

# As faixas de idade dos clientes adultos, com o peso de cada uma.
FAIXAS_DE_IDADE: tuple[tuple[int, int, float], ...] = (
    (18, 24, 0.12),
    (25, 34, 0.19),
    (35, 44, 0.19),
    (45, 54, 0.17),
    (55, 64, 0.15),
    (65, 74, 0.11),
    (75, 90, 0.07),
)

GOLPES_POR_LARANJA = 6
"""Quantos golpes, em média, cada conta laranja recebe antes de sumir.

É recebendo de muitas vítimas, em poucos dias, que ela se denuncia.
"""

LARANJAS_DE_FACHADA = 0.2
"""A parte das contas laranja que é de empresa: o MEI aberto para receber golpe."""

SALARIO_MINIMO = 151_800
"""O salário mínimo de 2025, em centavos: o piso da renda dos clientes."""


@dataclass(frozen=True, slots=True)
class Conta:
    """Uma conta que recebe Pix: de uma pessoa, de um comércio ou de um laranja."""

    id: int
    pessoa: TipoDePessoa
    banco: str
    chave: TipoDeChave


@dataclass(frozen=True, slots=True)
class Laranja:
    """Uma conta laranja e os dias em que ela recebe golpes, antes de ser bloqueada ou largada."""

    conta: int
    ativa_de: date
    ativa_ate: date
    """O primeiro dia em que ela já não recebe."""

    def ativa_em(self, dia: date) -> bool:
        return self.ativa_de <= dia < self.ativa_ate


@dataclass(frozen=True, slots=True)
class Aluguel:
    locador: int
    valor: int


@dataclass(frozen=True, slots=True)
class Cliente:
    """Um cliente do banco, com o que o cadastro sabe dele e com os hábitos que só o simulador conhece."""

    id: int
    idade: int
    uf: str
    renda_mensal: int
    """Em centavos."""
    conta_aberta_em: date
    limite_noturno: int
    """Em centavos. A maioria fica no padrão; alguns aumentaram antes de a simulação começar, alguns diminuíram."""
    saldo_inicial: int
    pix_por_mes: float
    """Quantos Pix ele costuma fazer por mês, sem contar o aluguel."""
    contatos: tuple[int, ...]
    """As pessoas para quem ele costuma mandar dinheiro: família, amigos, quem presta serviço."""
    comercios: tuple[int, ...]
    """Os comércios onde ele costuma comprar, do preferido para o menos frequente."""
    aluguel: Aluguel | None


@dataclass(frozen=True, slots=True)
class Populacao:
    clientes: tuple[Cliente, ...]
    contas: tuple[Conta, ...]
    """Todas as contas que recebem; o id de cada uma é a posição dela aqui."""
    pessoas: tuple[int, ...]
    """As contas de pessoa física que não são laranja: de onde sai o recebedor novo de um Pix comum."""
    todos_os_comercios: tuple[int, ...]
    laranjas: tuple[Laranja, ...]


def ataques_esperados(config: Configuracao) -> float:
    """Quantos golpes a simulação deve ter, em média, pela taxa da configuração."""
    return config.golpes_por_mil_clientes_ao_mes * config.clientes / 1000 * config.dias / 30


def gerar_populacao(config: Configuracao) -> Populacao:
    """Sorteia a população inteira, sempre a mesma para a mesma configuração."""
    gerador = sorteio(config, Etapa.POPULACAO)
    contas: list[Conta] = []

    pessoas = _contas(gerador, contas, TipoDePessoa.FISICA, 2 * config.clientes, _CHAVES_DE_PESSOA, laranja=False)
    comercios = _contas(
        gerador, contas, TipoDePessoa.JURIDICA, max(20, config.clientes // 8), _CHAVES_DE_COMERCIO, laranja=False
    )
    # A laranja é quase sempre uma pessoa; uma em cada cinco é uma empresa de fachada, um MEI aberto para isso.
    quantas_laranjas = max(5, round(ataques_esperados(config) / GOLPES_POR_LARANJA))
    de_fachada = round(quantas_laranjas * LARANJAS_DE_FACHADA)
    contas_laranja = _contas(
        gerador, contas, TipoDePessoa.FISICA, quantas_laranjas - de_fachada, _CHAVES_DE_LARANJA, laranja=True
    ) + _contas(gerador, contas, TipoDePessoa.JURIDICA, de_fachada, _CHAVES_DE_COMERCIO, laranja=True)

    laranjas = tuple(_laranja(gerador, config, conta) for conta in contas_laranja)
    popularidade = _popularidade(len(comercios))
    clientes = tuple(
        _cliente(gerador, config, id_, pessoas=pessoas, comercios=comercios, popularidade=popularidade)
        for id_ in range(config.clientes)
    )

    return Populacao(clientes, tuple(contas), tuple(pessoas), tuple(comercios), laranjas)


# Por que chave cada tipo de conta recebe. O laranja prefere a chave
# aleatória, que não diz de quem é, e o CPF de quem emprestou a conta.
_CHAVES_DE_PESSOA = (
    (TipoDeChave.CELULAR, 0.45),
    (TipoDeChave.CPF, 0.30),
    (TipoDeChave.EMAIL, 0.10),
    (TipoDeChave.ALEATORIA, 0.15),
)
_CHAVES_DE_COMERCIO = ((TipoDeChave.CNPJ, 0.55), (TipoDeChave.ALEATORIA, 0.45))
_CHAVES_DE_LARANJA = (
    (TipoDeChave.ALEATORIA, 0.45),
    (TipoDeChave.CPF, 0.30),
    (TipoDeChave.CELULAR, 0.20),
    (TipoDeChave.EMAIL, 0.05),
)


def _escolher[T](gerador: np.random.Generator, opcoes: tuple[tuple[T, float], ...]) -> T:
    pesos = np.array([peso for _, peso in opcoes])

    return opcoes[int(gerador.choice(len(opcoes), p=pesos / pesos.sum()))][0]


def _contas(
    gerador: np.random.Generator,
    contas: list[Conta],
    pessoa: TipoDePessoa,
    quantas: int,
    chaves: tuple[tuple[TipoDeChave, float], ...],
    *,
    laranja: bool,
) -> list[int]:
    # O laranja nasce três vezes mais nos bancos digitais, onde a conta abre pelo aplicativo em minutos.
    bancos = tuple((banco, parte * (3 if laranja and banco.digital else 1)) for banco, parte in BANCOS)
    ids = []

    for _ in range(quantas):
        conta = Conta(len(contas), pessoa, _escolher(gerador, bancos).codigo, _escolher(gerador, chaves))
        contas.append(conta)
        ids.append(conta.id)

    return ids


def _laranja(gerador: np.random.Generator, config: Configuracao, conta: int) -> Laranja:
    # Ativa por alguns dias, de dois a vinte: até a vítima reclamar e o banco dela bloquear, ou até o golpista largar.
    inicio = config.inicio + timedelta(days=int(gerador.integers(0, config.dias)))

    return Laranja(conta, inicio, inicio + timedelta(days=int(gerador.integers(2, 21))))


def _popularidade(comercios: int) -> npt.NDArray[np.float64]:
    # Uns poucos comércios recebem de muita gente — o mercado do bairro, o
    # aplicativo de entrega —, e a maioria, de poucos.
    pesos = 1 / np.arange(1, comercios + 1) ** 1.1

    return pesos / pesos.sum()


def _idade(gerador: np.random.Generator) -> int:
    pesos = np.array([peso for _, _, peso in FAIXAS_DE_IDADE])
    de, ate, _ = FAIXAS_DE_IDADE[int(gerador.choice(len(FAIXAS_DE_IDADE), p=pesos / pesos.sum()))]

    return int(gerador.integers(de, ate + 1))


def _frequencia(gerador: np.random.Generator, idade: int) -> float:
    # Por volta de vinte Pix por mês, mais entre os jovens e bem menos entre os mais velhos.
    fator = 1.3 if idade < 35 else 1.0 if idade < 55 else 0.7 if idade < 65 else 0.45

    return float(np.exp(gerador.normal(np.log(20), 0.6)) * fator)


def _limite_noturno(gerador: np.random.Generator, config: Configuracao) -> int:
    sorteado = gerador.random()

    if sorteado < 0.06:
        return int(gerador.integers(3, 11)) * 100_000

    if sorteado < 0.08:
        return int(gerador.choice([30_000, 50_000]))

    return config.limite_noturno


def _cliente(
    gerador: np.random.Generator,
    config: Configuracao,
    id_: int,
    *,
    pessoas: list[int],
    comercios: list[int],
    popularidade: npt.NDArray[np.float64],
) -> Cliente:
    idade = _idade(gerador)
    renda = max(SALARIO_MINIMO, round(float(np.exp(gerador.normal(np.log(280_000), 0.8))), -2))
    renda = min(renda, 8_000_000)
    # De três a doze contatos e de quatro a quinze comércios — ou todos os que houver, numa população pequena.
    quantos_contatos = min(int(gerador.integers(3, 13)), len(pessoas))
    quantos_comercios = min(int(gerador.integers(4, 16)), len(comercios))
    contatos = tuple(int(conta) for conta in gerador.choice(pessoas, size=quantos_contatos, replace=False))
    preferidos = gerador.choice(comercios, size=quantos_comercios, replace=False, p=popularidade)

    aluguel = None

    if gerador.random() < 0.35:
        # O aluguel vai para um contato, todo mês, entre um quinto e um terço da renda, em dezenas de reais.
        valor = round(renda * float(gerador.uniform(0.2, 0.35)), -3)
        aluguel = Aluguel(contatos[0], int(valor))

    return Cliente(
        id=id_,
        idade=idade,
        uf=_escolher(gerador, UFS),
        renda_mensal=int(renda),
        conta_aberta_em=config.inicio - timedelta(days=int(gerador.integers(30, 15 * 365))),
        limite_noturno=_limite_noturno(gerador, config),
        saldo_inicial=int(round(renda * float(gerador.uniform(0.3, 2.5)), -2)),
        pix_por_mes=_frequencia(gerador, idade),
        contatos=contatos,
        comercios=tuple(int(conta) for conta in preferidos),
        aluguel=aluguel,
    )
