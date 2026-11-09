"""O estado do banco em tempo real: as variáveis de um Pix que acabou de chegar, só com o que veio antes (ADR-0016).

No treino, as 33 variáveis saem de uma vez, em lote, com as janelas do Polars
sobre a história inteira (`variaveis/calculo.py`). No banco, o Pix chega um
de cada vez, e a variável tem de sair em microssegundos, sem reler a
história. O estado guarda o que as variáveis pedem, e só isso:
- de cada cliente, os Pix dos últimos 90 dias: a hora, o valor e se foi noturno;
- de cada cliente e recebedor, quantas vezes e desde quando;
- de cada recebedor, os Pix recebidos nos últimos 7 dias, e as contestações;
- de cada aparelho, quando entrou em cada conta;
- de cada Pix dos últimos 81 dias, o recebedor: a contestação chega até 80 dias depois, pelo MED.

As duas contas têm de dar o mesmo número, Pix a Pix, ou o modelo decide com
variáveis que não são as do treino. As regras são as do lote:
- toda janela é fechada à esquerda e aberta à direita, `[t - período, t)`:
  o Pix do mesmo segundo não entra na conta do outro;
- o aparelho que entra no instante do Pix já vale para ele;
- a contestação feita no instante do Pix, não.

O teste de paridade confere isso em todos os Pix de uma simulação.
"""

from bisect import bisect_left, bisect_right
from collections import deque
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any
from zoneinfo import ZoneInfo

import polars as pl

from antifraude_pix.dados.banco import VisaoDoBanco
from antifraude_pix.dados.esquema import FUSO
from antifraude_pix.variaveis.calculo import DIGITAIS, FIM_DA_NOITE, INICIO_DA_NOITE

MINUTO, HORA, DIA = 60, 3_600, 86_400

# O Polars divide por uma constante multiplicando pelo inverso dela, e a conta
# difere da divisão no último bit: 7.521 / 100 dá 75,21, e 7.521 * 0,01 dá
# 75,21000000000001. O estado faz a conta do mesmo jeito, e a paridade é exata.
POR_CENTAVO = 1 / 100
POR_MINUTO = 1 / MINUTO
POR_DIA = 1 / DIA
JANELA_DO_CLIENTE = 90 * DIA
"""A janela mais longa do hábito do cliente: a mediana, o maior Pix e os noturnos dos 90 dias anteriores."""

JANELA_DO_RECEBEDOR = 7 * DIA
"""A janela mais longa do recebedor: os pagadores novos dos 7 dias anteriores."""

JANELA_DA_CONTESTACAO = 81 * DIA
"""O recebedor de cada Pix fica guardado um dia além do prazo do MED, de 80 dias: a contestação chega até lá."""

_FUSO = ZoneInfo(FUSO)
_ZERO = datetime(1970, 1, 1)  # noqa: DTZ001 - o relógio do lote é o horário de Brasília, sem fuso


@dataclass(frozen=True, slots=True)
class PixNovo:
    """Um Pix que acabou de chegar ao banco, antes da decisão: o que o banco sabe dele naquele instante."""

    id: int
    momento: datetime
    """Com o fuso: o horário de Brasília, ou qualquer outro, que vira o de Brasília."""
    cliente: int
    aparelho: int
    recebedor: int
    recebedor_banco: str
    recebedor_pessoa: str
    chave: str | None
    iniciacao: str
    valor: int
    """Em centavos."""
    saldo_antes: int
    """Em centavos."""


@dataclass(frozen=True, slots=True)
class _Cadastro:
    idade: int
    renda_mensal: int
    conta_aberta_em: date
    limite_noturno: int


@dataclass(slots=True)
class _Cliente:
    """Os Pix do cliente nos últimos 90 dias, e os dois últimos instantes distintos, de qualquer tempo."""

    segundos: list[int] = field(default_factory=list)
    valores: list[int] = field(default_factory=list)
    noturnos: list[bool] = field(default_factory=list)
    ultimo: int | None = None
    penultimo: int | None = None


@dataclass(slots=True)
class _Par:
    """O cliente e o recebedor: quantos Pix, o primeiro, o último instante e quantos Pix nele."""

    vezes: int = 0
    primeiro: int = 0
    ultimo: int = 0
    no_ultimo: int = 0


@dataclass(slots=True)
class _Recebedor:
    """Os Pix que o recebedor recebeu nos últimos 7 dias, se cada um foi de um pagador novo, e as contestações."""

    segundos: list[int] = field(default_factory=list)
    novos: list[bool] = field(default_factory=list)
    contestacoes: list[int] = field(default_factory=list)


class Estado:
    """O que o banco sabe agora, para calcular as variáveis do próximo Pix.

    Os eventos chegam na ordem do tempo: os Pix, os aparelhos novos e as contestações.
    """

    def __init__(self, clientes: pl.DataFrame) -> None:
        self._cadastro = {
            linha["id"]: _Cadastro(
                linha["idade"], linha["renda_mensal"], linha["conta_aberta_em"], linha["limite_noturno"]
            )
            for linha in clientes.iter_rows(named=True)
        }
        self._clientes: dict[int, _Cliente] = {}
        self._pares: dict[tuple[int, int], _Par] = {}
        self._recebedores: dict[int, _Recebedor] = {}
        self._registros: dict[tuple[int, int], list[int]] = {}
        self._primeiras_entradas: dict[tuple[int, int], int] = {}
        self._entradas_do_aparelho: dict[int, list[int]] = {}
        self._recebedor_do_pix: dict[int, int] = {}
        self._pix_recentes: deque[tuple[int, int]] = deque()
        self.agora: int | None = None
        """O instante do último evento, em segundos no horário de Brasília: o próximo não pode vir antes."""
        self.pix = 0

    @staticmethod
    def da_historia(visao: VisaoDoBanco) -> "Estado":
        """O estado depois de toda a história da visão, contada na ordem do tempo, como o banco a viveu."""
        estado = Estado(visao.clientes)
        estado.reproduzir(visao)

        return estado

    def reproduzir(self, visao: VisaoDoBanco, *, calcular: bool = False) -> pl.DataFrame:
        """Conta os eventos da visão na ordem do tempo. Com `calcular`, devolve as variáveis de cada Pix.

        No mesmo instante, o aparelho novo e a contestação vêm antes do Pix, e os Pix vêm na ordem dos ids. As
        variáveis saem como no cálculo em lote: o `id` e as variáveis, na ordem dos ids, com as categorias em texto.
        """
        eventos: list[tuple[int, int, int, dict[str, Any]]] = []

        for linha in visao.aparelhos_novos.iter_rows(named=True):
            eventos.append((_segundos(linha["momento"]), 0, 0, linha))

        for linha in visao.contestacoes.iter_rows(named=True):
            eventos.append((_segundos(linha["contestada_em"]), 1, linha["transacao"], linha))

        for linha in visao.transacoes.iter_rows(named=True):
            eventos.append((_segundos(linha["momento"]), 2, linha["id"], linha))

        eventos.sort(key=lambda evento: evento[:3])
        calculadas = []

        for _, tipo, _, linha in eventos:
            if tipo == 0:
                self.aparelho_novo(linha["cliente"], linha["momento"], linha["aparelho"])
            elif tipo == 1:
                self.contestacao(linha["transacao"], linha["contestada_em"])
            else:
                pix = PixNovo(
                    **{campo: linha[campo] for campo in PixNovo.__dataclass_fields__ if campo != "chave"},
                    chave=linha["chave"],
                )

                if calcular:
                    calculadas.append({"id": pix.id, **self.variaveis(pix)})

                self.registrar(pix)

        esquema = pl.Schema({"id": pl.Int64(), **ESQUEMA})

        return pl.DataFrame(calculadas, schema=esquema).sort("id") if calcular else pl.DataFrame(schema=esquema)

    def variaveis(self, pix: PixNovo) -> dict[str, Any]:
        """As 33 variáveis do Pix, só com o que o banco sabia antes dele, com os nomes e na ordem do catálogo."""
        if pix.cliente not in self._cadastro:
            raise KeyError(f"O cliente {pix.cliente} não é do banco.")

        local = pix.momento.astimezone(_FUSO).replace(tzinfo=None)
        t = _segundos_locais(local)
        self._em_ordem(t)
        cadastro = self._cadastro[pix.cliente]
        noturno = local.hour >= INICIO_DA_NOITE or local.hour < FIM_DA_NOITE
        do_cliente = self._clientes.get(pix.cliente, _Cliente())
        par = self._pares.get((pix.cliente, pix.recebedor))
        vezes = 0 if par is None else par.vezes - (par.no_ultimo if par.ultimo == t else 0)
        primeiro = par.primeiro if par is not None and par.primeiro < t else None
        recebedor = self._recebedores.get(pix.recebedor, _Recebedor())
        ultimo = _ultimo_antes(do_cliente, t)
        noventa = _janela(do_cliente.segundos, t, JANELA_DO_CLIENTE)
        valores_de_90_dias = do_cliente.valores[noventa]
        um_dia = _janela(do_cliente.segundos, t, DIA)
        mediana = _mediana(valores_de_90_dias)
        maior = max(valores_de_90_dias) if valores_de_90_dias else None
        registrado = _ultimo_ate(self._registros.get((pix.cliente, pix.aparelho), []), t)
        entradas = self._entradas_do_aparelho.get(pix.aparelho, [])
        minha = self._primeiras_entradas.get((pix.aparelho, pix.cliente))

        return {
            "valor": pix.valor * POR_CENTAVO,
            "hora": local.hour,
            "noturno": noturno,
            "dia_da_semana": local.weekday(),
            "iniciacao": pix.iniciacao,
            "chave": pix.chave,
            "recebedor_pj": pix.recebedor_pessoa == "PJ",
            "recebedor_em_banco_digital": pix.recebedor_banco in DIGITAIS,
            "valor_redondo": pix.valor % 1_000 == 0,
            "termina_em_90_centavos": pix.valor % 100 == 90,
            "parte_do_saldo": pix.valor / pix.saldo_antes,
            "parte_do_limite_noturno": pix.valor / cadastro.limite_noturno if noturno else None,
            "idade": cadastro.idade,
            "renda_mensal": cadastro.renda_mensal * POR_CENTAVO,
            "dias_de_conta": (local.date() - cadastro.conta_aberta_em).days,
            "valor_sobre_a_renda": pix.valor / cadastro.renda_mensal,
            "pix_na_ultima_hora": _contar(do_cliente.segundos, t, HORA),
            "pix_nas_ultimas_24h": _contar(do_cliente.segundos, t, DIA),
            "valor_nas_ultimas_24h": sum(do_cliente.valores[um_dia]) * POR_CENTAVO,
            "minutos_desde_o_ultimo_pix": None if ultimo is None else (t - ultimo) * POR_MINUTO,
            "mediana_de_90_dias": None if mediana is None else mediana * POR_CENTAVO,
            "valor_sobre_a_mediana": None if mediana is None else pix.valor / mediana,
            "valor_sobre_o_maior_de_90_dias": None if maior is None else pix.valor / maior,
            "pix_noturnos_em_90_dias": sum(do_cliente.noturnos[noventa]),
            "vezes_que_pagou_o_recebedor": vezes,
            "recebedor_novo": vezes == 0,
            "dias_desde_o_primeiro_pix_ao_recebedor": None if primeiro is None else (t - primeiro) * POR_DIA,
            "pix_recebidos_em_24h": _contar(recebedor.segundos, t, DIA),
            "pagadores_novos_em_24h": sum(recebedor.novos[_janela(recebedor.segundos, t, DIA)]),
            "pagadores_novos_em_7_dias": sum(recebedor.novos[_janela(recebedor.segundos, t, JANELA_DO_RECEBEDOR)]),
            "contestacoes_do_recebedor": bisect_left(recebedor.contestacoes, t),
            "minutos_desde_o_aparelho_novo": None if registrado is None else (t - registrado) * POR_MINUTO,
            "outros_clientes_no_aparelho": bisect_left(entradas, t) - (1 if minha is not None and minha < t else 0),
        }

    def registrar(self, pix: PixNovo) -> None:
        """O Pix que saiu entra na história: depois dele, as variáveis do próximo já o contam."""
        t = _segundos(pix.momento)
        self._em_ordem(t)
        local = pix.momento.astimezone(_FUSO)
        noturno = local.hour >= INICIO_DA_NOITE or local.hour < FIM_DA_NOITE
        do_cliente = self._clientes.setdefault(pix.cliente, _Cliente())
        _podar(do_cliente.segundos, t - JANELA_DO_CLIENTE, do_cliente.valores, do_cliente.noturnos)
        do_cliente.segundos.append(t)
        do_cliente.valores.append(pix.valor)
        do_cliente.noturnos.append(noturno)

        if do_cliente.ultimo is None or t > do_cliente.ultimo:
            do_cliente.penultimo, do_cliente.ultimo = do_cliente.ultimo, t

        par = self._pares.get((pix.cliente, pix.recebedor))
        primeira_vez = par is None or (par.vezes == par.no_ultimo and par.ultimo == t)

        if par is None:
            par = self._pares[(pix.cliente, pix.recebedor)] = _Par(primeiro=t, ultimo=t)

        par.vezes += 1
        par.no_ultimo = par.no_ultimo + 1 if par.ultimo == t else 1
        par.ultimo = t
        recebedor = self._recebedores.setdefault(pix.recebedor, _Recebedor())
        _podar(recebedor.segundos, t - JANELA_DO_RECEBEDOR, recebedor.novos)
        recebedor.segundos.append(t)
        recebedor.novos.append(primeira_vez)
        self._recebedor_do_pix[pix.id] = pix.recebedor
        self._pix_recentes.append((t, pix.id))

        while self._pix_recentes and self._pix_recentes[0][0] < t - JANELA_DA_CONTESTACAO:
            del self._recebedor_do_pix[self._pix_recentes.popleft()[1]]

        self.pix += 1

    def aparelho_novo(self, cliente: int, momento: datetime, aparelho: int) -> None:
        """Um aparelho entrou na conta do cliente: vale para o Pix do mesmo instante em diante."""
        t = _segundos(momento)
        self._em_ordem(t)
        self._registros.setdefault((cliente, aparelho), []).append(t)

        if (aparelho, cliente) not in self._primeiras_entradas:
            self._primeiras_entradas[(aparelho, cliente)] = t
            self._entradas_do_aparelho.setdefault(aparelho, []).append(t)

    def contestacao(self, transacao: int, momento: datetime) -> None:
        """A vítima contestou o Pix: o recebedor dele passa a ter mais uma contestação, depois deste instante."""
        t = _segundos(momento)

        if transacao not in self._recebedor_do_pix:
            raise KeyError(f"O Pix {transacao} não está no estado: é de antes do prazo do MED, ou nunca saiu.")

        self._em_ordem(t)
        self._recebedores.setdefault(self._recebedor_do_pix[transacao], _Recebedor()).contestacoes.append(t)

    def conhece(self, transacao: int) -> bool:
        """O Pix ainda está no estado, dentro do prazo em que pode ser contestado."""
        return transacao in self._recebedor_do_pix

    def _em_ordem(self, t: int) -> None:
        if self.agora is not None and t < self.agora:
            raise ValueError("Os eventos chegam na ordem do tempo: este é de antes do último que o banco viu.")

        self.agora = t


def _segundos(momento: datetime) -> int:
    """O instante em segundos no horário de Brasília, sem fuso: o relógio do cálculo em lote."""
    return _segundos_locais(momento.astimezone(_FUSO).replace(tzinfo=None))


def _segundos_locais(local: datetime) -> int:
    return int((local - _ZERO).total_seconds())


def _janela(segundos: list[int], t: int, periodo: int) -> slice:
    """As posições dos instantes em `[t - período, t)`: a janela fechada à esquerda e aberta à direita."""
    return slice(bisect_left(segundos, t - periodo), bisect_left(segundos, t))


def _contar(segundos: list[int], t: int, periodo: int) -> int:
    """Quantos instantes há em `[t - período, t)`."""
    return bisect_left(segundos, t) - bisect_left(segundos, t - periodo)


def _ultimo_antes(cliente: _Cliente, t: int) -> int | None:
    """O último instante de Pix antes de `t`: o do mesmo segundo não conta."""
    if cliente.ultimo is None:
        return None

    return cliente.ultimo if cliente.ultimo < t else cliente.penultimo


def _ultimo_ate(instantes: list[int], t: int) -> int | None:
    """O último instante até `t`, inclusive."""
    posicao = bisect_right(instantes, t)

    return instantes[posicao - 1] if posicao else None


def _mediana(valores: list[int]) -> float | None:
    if not valores:
        return None

    ordenados = sorted(valores)
    meio = len(ordenados) // 2

    if len(ordenados) % 2:
        return float(ordenados[meio])

    return (ordenados[meio - 1] + ordenados[meio]) / 2


def _podar(segundos: list[int], antes_de: int, *outras: list[Any]) -> None:
    """Tira da frente o que saiu da janela mais longa, de vez em quando: apagar um por um custaria caro."""
    corte = bisect_left(segundos, antes_de)

    if corte > 64:
        del segundos[:corte]

        for outra in outras:
            del outra[:corte]


ESQUEMA = pl.Schema(
    {
        "valor": pl.Float64(),
        "hora": pl.Int8(),
        "noturno": pl.Boolean(),
        "dia_da_semana": pl.Int8(),
        "iniciacao": pl.String(),
        "chave": pl.String(),
        "recebedor_pj": pl.Boolean(),
        "recebedor_em_banco_digital": pl.Boolean(),
        "valor_redondo": pl.Boolean(),
        "termina_em_90_centavos": pl.Boolean(),
        "parte_do_saldo": pl.Float64(),
        "parte_do_limite_noturno": pl.Float64(),
        "idade": pl.Int16(),
        "renda_mensal": pl.Float64(),
        "dias_de_conta": pl.Int32(),
        "valor_sobre_a_renda": pl.Float64(),
        "pix_na_ultima_hora": pl.Int32(),
        "pix_nas_ultimas_24h": pl.Int32(),
        "valor_nas_ultimas_24h": pl.Float64(),
        "minutos_desde_o_ultimo_pix": pl.Float64(),
        "mediana_de_90_dias": pl.Float64(),
        "valor_sobre_a_mediana": pl.Float64(),
        "valor_sobre_o_maior_de_90_dias": pl.Float64(),
        "pix_noturnos_em_90_dias": pl.Int32(),
        "vezes_que_pagou_o_recebedor": pl.Int32(),
        "recebedor_novo": pl.Boolean(),
        "dias_desde_o_primeiro_pix_ao_recebedor": pl.Float64(),
        "pix_recebidos_em_24h": pl.Int32(),
        "pagadores_novos_em_24h": pl.Int32(),
        "pagadores_novos_em_7_dias": pl.Int32(),
        "contestacoes_do_recebedor": pl.Int32(),
        "minutos_desde_o_aparelho_novo": pl.Float64(),
        "outros_clientes_no_aparelho": pl.Int32(),
    }
)
"""As variáveis como o estado as calcula: as do lote, com as categorias em texto, que o modelo converte."""
