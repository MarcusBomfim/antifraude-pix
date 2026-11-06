"""A frase de cada variável com o valor dela, em português: o fato do Pix que pesou na decisão (ADR-0012).

O catálogo diz o que cada variável quer dizer ("há quantos minutos o aparelho
entrou na conta"); a frase diz o que ela valeu neste Pix ("o aparelho entrou
na conta há 12 minutos"). É para quem lê a decisão: o analista que a revisa, e
o cliente que pede a revisão.

Só há frase para as variáveis que o modelo recebe. A idade não entra no
modelo, e por isso nunca vira razão de uma decisão.
"""

import math
from collections.abc import Callable
from typing import Any

from antifraude_pix.modelos.preparo import ENTRADA

NOMES_DA_ENTRADA = frozenset(variavel.nome for variavel in ENTRADA)

DIAS_DA_SEMANA = (
    "numa segunda-feira",
    "numa terça-feira",
    "numa quarta-feira",
    "numa quinta-feira",
    "numa sexta-feira",
    "num sábado",
    "num domingo",
)

INICIACOES = {
    "chave": "o Pix foi feito pela chave",
    "qr_code": "o Pix foi feito pelo QR Code",
    "manual": "o Pix foi feito digitando a agência e a conta",
}

CHAVES = {
    "cpf": "o recebedor recebe pelo CPF",
    "cnpj": "o recebedor recebe pelo CNPJ",
    "celular": "o recebedor recebe pelo número de celular",
    "email": "o recebedor recebe pelo e-mail",
    "aleatoria": "o recebedor recebe por uma chave aleatória, que não diz de quem é",
}


def reais(valor: float) -> str:
    """O dinheiro como se escreve no Brasil: R$ 1.250,00."""
    return "R$ " + f"{valor:,.2f}".replace(",", "_").replace(".", ",").replace("_", ".")


def porcentagem(fracao: float) -> str:
    """A parte em porcentagem, sem casas: 64%. Abaixo de 1%, com uma casa: 0,4%."""
    if 0 < fracao < 0.01:
        return f"{fracao:.1%}".replace(".", ",")

    return f"{fracao:.0%}"


def vezes(quantas: float) -> str:
    """Quantas vezes, com uma casa abaixo de dez: "2,5 vezes", "12 vezes"."""
    texto = f"{quantas:.1f}".replace(".", ",").removesuffix(",0") if quantas < 10 else f"{quantas:.0f}"

    return f"{texto} {'vez' if texto == '1' else 'vezes'}"


def contagem(quantos: int, singular: str, plural: str) -> str:
    """O número com a palavra no singular ou no plural: "1 Pix noturno", "3 Pix noturnos"."""
    return f"{quantos} {singular if quantos == 1 else plural}"


def tempo(minutos: float) -> str:
    """Há quanto tempo, na unidade que se fala: minutos, horas ou dias."""
    if minutos < 1:
        return "menos de um minuto"

    if minutos < 60:
        return contagem(math.floor(minutos), "minuto", "minutos")

    if minutos < 48 * 60:
        return contagem(math.floor(minutos / 60), "hora", "horas")

    return contagem(math.floor(minutos / (24 * 60)), "dia", "dias")


def _sim_ou_nao(sim: str, nao: str) -> Callable[[Any], str]:
    return lambda valor: sim if valor else nao


def _hora(hora: int) -> str:
    return f"o Pix foi feito entre {hora}h e {(hora + 1) % 24}h"


def _chave(chave: str | None) -> str:
    return "o Pix não usou chave" if chave is None else CHAVES[chave]


def _limite_noturno(parte: float | None) -> str:
    return (
        "o Pix é de dia, fora do limite noturno"
        if parte is None
        else f"o Pix leva {porcentagem(parte)} do limite noturno"
    )


def _pix_antes(periodo: str) -> Callable[[Any], str]:
    def frase(quantos: int) -> str:
        if quantos == 0:
            return f"o cliente não fez nenhum Pix {periodo}"

        return f"o cliente fez {contagem(quantos, 'Pix', 'Pix')} {periodo}"

    return frase


def _valor_nas_ultimas_24h(valor: float) -> str:
    if valor == 0:
        return "o cliente não mandou nada por Pix nas 24 horas anteriores"

    return f"o cliente mandou {reais(valor)} por Pix nas 24 horas anteriores"


def _ultimo_pix(minutos: float | None) -> str:
    return "é o primeiro Pix do cliente" if minutos is None else f"o Pix anterior do cliente foi há {tempo(minutos)}"


def _mediana(mediana: float | None) -> str:
    if mediana is None:
        return "o cliente não fez Pix nos 90 dias anteriores"

    return f"o Pix típico do cliente é de {reais(mediana)}"


def _sobre_a_mediana(quantas: float | None) -> str:
    if quantas is None:
        return "o cliente não tem Pix típico: não fez Pix nos 90 dias anteriores"

    if quantas < 1:
        return f"o valor é {porcentagem(quantas)} do Pix típico do cliente"

    return f"o valor é {vezes(quantas)} o Pix típico do cliente"


def _sobre_o_maior(quantas: float | None) -> str:
    if quantas is None:
        return "o cliente não fez Pix nos 90 dias anteriores"

    if quantas > 1:
        return f"o valor passa do maior Pix do cliente em 90 dias: é {vezes(quantas)} ele"

    return f"o valor é {porcentagem(quantas)} do maior Pix do cliente em 90 dias"


def _noturnos(quantos: int) -> str:
    if quantos == 0:
        return "o cliente não fez nenhum Pix noturno nos 90 dias anteriores"

    return f"o cliente fez {contagem(quantos, 'Pix noturno', 'Pix noturnos')} nos 90 dias anteriores"


def _vezes_que_pagou(quantas: int) -> str:
    if quantas == 0:
        return "o cliente nunca tinha pagado este recebedor"

    return f"o cliente já tinha pagado este recebedor {contagem(quantas, 'vez', 'vezes')}"


def _primeiro_pix_ao_recebedor(dias: float | None) -> str:
    if dias is None:
        return "é o primeiro Pix do cliente para este recebedor"

    return f"o primeiro Pix do cliente para este recebedor foi há {tempo(dias * 24 * 60)}"


def _recebidos(quantos: int) -> str:
    if quantos == 0:
        return "o recebedor não recebeu nenhum Pix de clientes do banco nas 24 horas anteriores"

    return f"o recebedor recebeu {contagem(quantos, 'Pix', 'Pix')} de clientes do banco nas 24 horas anteriores"


def _pagadores_novos(periodo: str) -> Callable[[Any], str]:
    def frase(quantos: int) -> str:
        if quantos == 0:
            return f"nenhum cliente do banco pagou o recebedor pela primeira vez {periodo}"

        if quantos == 1:
            return f"1 cliente do banco pagou o recebedor pela primeira vez {periodo}"

        return f"{quantos} clientes do banco pagaram o recebedor pela primeira vez {periodo}"

    return frase


def _contestacoes(quantas: int) -> str:
    if quantas == 0:
        return "nenhum Pix para este recebedor tinha sido contestado"

    if quantas == 1:
        return "1 Pix para este recebedor já tinha sido contestado como golpe"

    return f"{quantas} Pix para este recebedor já tinham sido contestados como golpe"


def _aparelho_novo(minutos: float | None) -> str:
    return "o Pix saiu do aparelho de sempre" if minutos is None else f"o aparelho entrou na conta há {tempo(minutos)}"


def _outros_clientes(quantos: int) -> str:
    if quantos == 0:
        return "o aparelho não tinha entrado na conta de nenhum outro cliente"

    return f"o aparelho já tinha entrado na conta de {contagem(quantos, 'outro cliente', 'outros clientes')}"


FRASES: dict[str, Callable[[Any], str]] = {
    "valor": lambda valor: f"o Pix é de {reais(valor)}",
    "hora": _hora,
    "noturno": _sim_ou_nao("o Pix é do período noturno", "o Pix é do período diurno"),
    "dia_da_semana": lambda dia: f"o Pix foi feito {DIAS_DA_SEMANA[dia]}",
    "iniciacao": lambda iniciacao: INICIACOES[iniciacao],
    "chave": _chave,
    "recebedor_pj": _sim_ou_nao("o recebedor é uma empresa", "o recebedor é uma pessoa"),
    "recebedor_em_banco_digital": _sim_ou_nao(
        "a conta do recebedor é de um banco digital", "a conta do recebedor é de um banco tradicional"
    ),
    "valor_redondo": _sim_ou_nao("o valor é redondo, em dezenas de reais", "o valor não é redondo"),
    "termina_em_90_centavos": _sim_ou_nao(
        "o valor termina em 90 centavos, como o preço de um anúncio", "o valor não termina em 90 centavos"
    ),
    "parte_do_saldo": lambda parte: f"o Pix leva {porcentagem(parte)} do saldo",
    "parte_do_limite_noturno": _limite_noturno,
    "renda_mensal": lambda renda: f"a renda do cliente é de {reais(renda)} por mês",
    "dias_de_conta": lambda dias: f"o cliente tem conta há {contagem(dias, 'dia', 'dias')}",
    "valor_sobre_a_renda": lambda parte: f"o Pix é {porcentagem(parte)} da renda mensal do cliente",
    "pix_na_ultima_hora": _pix_antes("na hora anterior"),
    "pix_nas_ultimas_24h": _pix_antes("nas 24 horas anteriores"),
    "valor_nas_ultimas_24h": _valor_nas_ultimas_24h,
    "minutos_desde_o_ultimo_pix": _ultimo_pix,
    "mediana_de_90_dias": _mediana,
    "valor_sobre_a_mediana": _sobre_a_mediana,
    "valor_sobre_o_maior_de_90_dias": _sobre_o_maior,
    "pix_noturnos_em_90_dias": _noturnos,
    "vezes_que_pagou_o_recebedor": _vezes_que_pagou,
    "recebedor_novo": _sim_ou_nao(
        "o cliente nunca tinha pagado este recebedor", "o cliente já tinha pagado este recebedor"
    ),
    "dias_desde_o_primeiro_pix_ao_recebedor": _primeiro_pix_ao_recebedor,
    "pix_recebidos_em_24h": _recebidos,
    "pagadores_novos_em_24h": _pagadores_novos("nas 24 horas anteriores"),
    "pagadores_novos_em_7_dias": _pagadores_novos("nos 7 dias anteriores"),
    "contestacoes_do_recebedor": _contestacoes,
    "minutos_desde_o_aparelho_novo": _aparelho_novo,
    "outros_clientes_no_aparelho": _outros_clientes,
}
"""A frase de cada variável que o modelo recebe, na ordem do catálogo, com o valor que ela tem no Pix."""


def frase(variavel: str, valor: Any) -> str:
    """A frase da variável com o valor que ela tem no Pix."""
    if variavel not in FRASES or variavel not in NOMES_DA_ENTRADA:
        raise ValueError(f"A variável {variavel!r} não entra no modelo, e não vira razão de decisão.")

    return FRASES[variavel](valor)
