"""O catálogo das variáveis: o nome de cada uma, o grupo, e o que ela quer dizer, em português (ADR-0006).

A frase de cada variável é para quem lê: o analista de fraude, o cliente que
teve o Pix segurado. Na Parte 4, é dela que sai a explicação de cada decisão.
"""

from dataclasses import dataclass
from enum import StrEnum


class Grupo(StrEnum):
    PIX = "o Pix"
    CLIENTE = "o cliente"
    HABITO = "o hábito do cliente"
    RELACAO = "o cliente e o recebedor"
    RECEBEDOR = "o recebedor"
    APARELHO = "o aparelho"


class Tipo(StrEnum):
    NUMERO = "número"
    SIM_OU_NAO = "sim ou não"
    CATEGORIA = "categoria"


@dataclass(frozen=True, slots=True)
class Variavel:
    nome: str
    grupo: Grupo
    tipo: Tipo
    descricao: str


def _v(nome: str, grupo: Grupo, tipo: Tipo, descricao: str) -> Variavel:
    return Variavel(nome, grupo, tipo, descricao)


N, S, C = Tipo.NUMERO, Tipo.SIM_OU_NAO, Tipo.CATEGORIA

CATALOGO: tuple[Variavel, ...] = (
    _v("valor", Grupo.PIX, N, "O valor do Pix, em reais."),
    _v("hora", Grupo.PIX, N, "A hora do Pix, no horário de Brasília."),
    _v("noturno", Grupo.PIX, S, "O Pix é do período noturno, das 20h às 6h, em que vale o limite noturno."),
    _v("dia_da_semana", Grupo.PIX, N, "O dia da semana do Pix, de segunda (0) a domingo (6)."),
    _v(
        "iniciacao", Grupo.PIX, C, "Como o Pix foi iniciado: pela chave, pelo QR Code ou digitando a agência e a conta."
    ),
    _v(
        "chave",
        Grupo.PIX,
        C,
        "O tipo de chave por que o recebedor recebe; nenhum, quando o Pix é pela agência e a conta.",
    ),
    _v("recebedor_pj", Grupo.PIX, S, "O recebedor é uma empresa."),
    _v(
        "recebedor_em_banco_digital",
        Grupo.PIX,
        S,
        "A conta do recebedor é de um banco digital, em que a conta abre pelo aplicativo.",
    ),
    _v("valor_redondo", Grupo.PIX, S, "O valor é redondo, em dezenas de reais, como quem pede ajuda."),
    _v("termina_em_90_centavos", Grupo.PIX, S, "O valor termina em 90 centavos, como o preço de um anúncio."),
    _v("parte_do_saldo", Grupo.PIX, N, "A parte do saldo que o Pix leva."),
    _v("parte_do_limite_noturno", Grupo.PIX, N, "De noite, a parte do limite noturno que o Pix leva; de dia, nada."),
    _v("idade", Grupo.CLIENTE, N, "A idade do cliente."),
    _v("renda_mensal", Grupo.CLIENTE, N, "A renda mensal do cliente, em reais."),
    _v("dias_de_conta", Grupo.CLIENTE, N, "Há quantos dias o cliente tem conta no banco."),
    _v("valor_sobre_a_renda", Grupo.CLIENTE, N, "O valor do Pix como parte da renda mensal do cliente."),
    _v("pix_na_ultima_hora", Grupo.HABITO, N, "Quantos Pix o cliente fez na hora anterior."),
    _v("pix_nas_ultimas_24h", Grupo.HABITO, N, "Quantos Pix o cliente fez nas 24 horas anteriores."),
    _v("valor_nas_ultimas_24h", Grupo.HABITO, N, "Quanto o cliente mandou por Pix nas 24 horas anteriores, em reais."),
    _v(
        "minutos_desde_o_ultimo_pix",
        Grupo.HABITO,
        N,
        "Há quantos minutos o cliente fez o Pix anterior; nada, se é o primeiro.",
    ),
    _v("mediana_de_90_dias", Grupo.HABITO, N, "O valor típico dos Pix do cliente nos 90 dias anteriores, em reais."),
    _v("valor_sobre_a_mediana", Grupo.HABITO, N, "Quantas vezes o valor do Pix é o valor típico do cliente."),
    _v(
        "valor_sobre_o_maior_de_90_dias",
        Grupo.HABITO,
        N,
        "O valor do Pix como parte do maior Pix do cliente nos 90 dias anteriores.",
    ),
    _v("pix_noturnos_em_90_dias", Grupo.HABITO, N, "Quantos Pix noturnos o cliente fez nos 90 dias anteriores."),
    _v("vezes_que_pagou_o_recebedor", Grupo.RELACAO, N, "Quantas vezes o cliente já tinha pagado este recebedor."),
    _v("recebedor_novo", Grupo.RELACAO, S, "O cliente nunca tinha pagado este recebedor."),
    _v(
        "dias_desde_o_primeiro_pix_ao_recebedor",
        Grupo.RELACAO,
        N,
        "Há quantos dias o cliente pagou este recebedor pela primeira vez.",
    ),
    _v(
        "pix_recebidos_em_24h",
        Grupo.RECEBEDOR,
        N,
        "Quantos Pix de clientes do banco o recebedor recebeu nas 24 horas anteriores.",
    ),
    _v(
        "pagadores_novos_em_24h",
        Grupo.RECEBEDOR,
        N,
        "Quantos clientes do banco pagaram o recebedor pela primeira vez nas 24 horas anteriores.",
    ),
    _v(
        "pagadores_novos_em_7_dias",
        Grupo.RECEBEDOR,
        N,
        "Quantos clientes do banco pagaram o recebedor pela primeira vez nos 7 dias anteriores.",
    ),
    _v(
        "contestacoes_do_recebedor",
        Grupo.RECEBEDOR,
        N,
        "Quantos Pix para este recebedor já tinham sido contestados como golpe.",
    ),
    _v(
        "minutos_desde_o_aparelho_novo",
        Grupo.APARELHO,
        N,
        "Há quantos minutos o aparelho do Pix entrou na conta; nada, se é o aparelho de sempre.",
    ),
    _v(
        "outros_clientes_no_aparelho",
        Grupo.APARELHO,
        N,
        "Em quantas contas de outros clientes este aparelho já tinha entrado.",
    ),
)

NOMES: tuple[str, ...] = tuple(variavel.nome for variavel in CATALOGO)
