"""O cálculo das variáveis de cada Pix, só com o que o banco sabia antes dele (ADR-0006).

Toda janela no tempo é fechada à esquerda e aberta à direita, [t - período, t):
o Pix do instante t não entra nas contas dele mesmo, nem nas de outro Pix do
mesmo instante. O que vem de outras tabelas entra pelo que já tinha
acontecido: a contestação que já tinha chegado, o aparelho que já tinha
entrado na conta.

A entrada é a visão do banco, e não a simulação: a verdade do simulador não
tem por onde entrar aqui.
"""

import polars as pl

from antifraude_pix.dados.banco import VisaoDoBanco
from antifraude_pix.simulacao.configuracao import Configuracao
from antifraude_pix.simulacao.populacao import BANCOS
from antifraude_pix.variaveis.catalogo import NOMES

_REGRA = Configuracao()
INICIO_DA_NOITE = _REGRA.inicio_da_noite
FIM_DA_NOITE = _REGRA.fim_da_noite
"""O período noturno do Pix é regra do Banco Central, e não do simulador: das 20h às 6h."""

DIGITAIS: tuple[str, ...] = tuple(banco.codigo for banco, _ in BANCOS if banco.digital)

SEMPRE = "36500d"
"""Uma janela de cem anos: a que vê todo o passado."""


def calcular(visao: VisaoDoBanco) -> pl.DataFrame:
    """As variáveis de cada Pix da visão, uma linha por Pix, na ordem dos ids."""
    pix = _com_o_cadastro(visao)
    pix = _do_passado(pix)
    pix = _contestacoes_anteriores(pix, visao)
    pix = _aparelhos_anteriores(pix, visao)

    return _variaveis(pix).select("id", *NOMES)


def _com_o_cadastro(visao: VisaoDoBanco) -> pl.DataFrame:
    cadastro = visao.clientes.select(
        pl.col("id").alias("cliente"), "idade", "renda_mensal", "conta_aberta_em", "limite_noturno"
    )
    local = pl.col("momento").dt.replace_time_zone(None)
    hora = local.dt.hour()

    return (
        visao.transacoes.join(cadastro, on="cliente", how="left")
        # As janelas no tempo andam na ordem dos Pix: a ordem dos ids.
        .sort("id")
        .with_columns(
            pl.lit(1, dtype=pl.Int32).alias("_um"),
            local.dt.epoch("s").alias("_segundos"),
            (pl.col("cliente").cast(pl.Int64) * (1 << 32) + pl.col("recebedor")).alias("_par"),
            ((hora >= INICIO_DA_NOITE) | (hora < FIM_DA_NOITE)).alias("_noturno"),
        )
    )


def _antes(expressao: pl.Expr, periodo: str, *grupo: str) -> pl.Expr:
    """A soma da expressão nos Pix do grupo no período anterior, sem o instante do Pix."""
    return expressao.rolling_sum_by("momento", periodo, closed="left").over(list(grupo))


def _do_passado(pix: pl.DataFrame) -> pl.DataFrame:
    """O hábito do cliente, a relação com o recebedor, e o movimento do recebedor entre todos os clientes."""
    um, valor, segundos, momento = pl.col("_um"), pl.col("valor"), pl.col("_segundos"), pl.col("momento")

    # O cliente e o recebedor formam centenas de milhares de duplas pequenas, e
    # uma janela no tempo para cada uma sai cara. As duas contas da dupla têm uma
    # forma exata e barata, que o teste do corte confere como as outras:
    # - as vezes de antes são o lugar do Pix na ordem do tempo, e o Pix do mesmo
    #   instante divide o lugar, sem contar o outro;
    # - o primeiro Pix é o mais antigo da dupla, quando é anterior a este.
    vezes = (momento.rank("min").over("_par") - 1).cast(pl.Int32)
    primeiro = pl.when(momento > momento.min().over("_par")).then(segundos.min().over("_par"))

    return (
        pix.with_columns(
            _antes(um, "1h", "cliente").alias("_na_ultima_hora"),
            _antes(um, "1d", "cliente").alias("_nas_ultimas_24h"),
            _antes(valor, "1d", "cliente").alias("_valor_nas_ultimas_24h"),
            _antes(pl.col("_noturno").cast(pl.Int32), "90d", "cliente").alias("_noturnos_em_90_dias"),
            segundos.rolling_max_by("momento", SEMPRE, closed="left").over("cliente").alias("_ultimo_pix"),
            valor.rolling_median_by("momento", "90d", closed="left").over("cliente").alias("_mediana"),
            valor.rolling_max_by("momento", "90d", closed="left").over("cliente").alias("_maior"),
            vezes.alias("_vezes"),
            primeiro.alias("_primeiro"),
            _antes(um, "1d", "recebedor").alias("_recebidos_em_24h"),
        )
        # A primeira vez de cada cliente com cada recebedor: é o que conta como pagador novo.
        .with_columns((pl.col("_vezes") == 0).cast(pl.Int32).alias("_primeira_vez"))
        .with_columns(
            _antes(pl.col("_primeira_vez"), "1d", "recebedor").alias("_novos_em_24h"),
            _antes(pl.col("_primeira_vez"), "7d", "recebedor").alias("_novos_em_7_dias"),
        )
    )


def _contestacoes_anteriores(pix: pl.DataFrame, visao: VisaoDoBanco) -> pl.DataFrame:
    """Quantos Pix para o mesmo recebedor já tinham sido contestados antes do instante do Pix."""
    por_instante = (
        visao.contestacoes.join(visao.transacoes.select(pl.col("id").alias("transacao"), "recebedor"), on="transacao")
        .group_by("recebedor", "contestada_em")
        .len()
        .sort("contestada_em", "recebedor")
        .with_columns(pl.col("len").cum_sum().over("recebedor").cast(pl.Int32).alias("_contestacoes"))
        .select("recebedor", "contestada_em", "_contestacoes")
    )

    return pix.join_asof(
        por_instante,
        left_on="momento",
        right_on="contestada_em",
        by="recebedor",
        strategy="backward",
        allow_exact_matches=False,
        check_sortedness=False,
    )


def _aparelhos_anteriores(pix: pl.DataFrame, visao: VisaoDoBanco) -> pl.DataFrame:
    """Quando o aparelho entrou na conta, e em quantas contas de outros clientes ele já tinha entrado."""
    registros = visao.aparelhos_novos.sort("momento")
    # A primeira entrada de cada aparelho em cada conta: o mesmo aparelho em contas diferentes é sinal de quadrilha.
    primeiras = registros.group_by("aparelho", "cliente").agg(pl.col("momento").min()).sort("momento", "aparelho")
    contas_do_aparelho = (
        primeiras.group_by("aparelho", "momento")
        .len()
        .sort("momento", "aparelho")
        .with_columns(pl.col("len").cum_sum().over("aparelho").cast(pl.Int32).alias("_contas_ate"))
        .select("aparelho", pl.col("momento").alias("_desde"), "_contas_ate")
    )

    return (
        # A entrada do próprio aparelho é da mesma sessão do Pix: vale até o instante dele.
        pix.join_asof(
            registros.select("cliente", "aparelho", pl.col("momento").alias("_registrado_em")),
            left_on="momento",
            right_on="_registrado_em",
            by=["cliente", "aparelho"],
            allow_exact_matches=True,
            strategy="backward",
            check_sortedness=False,
        )
        .join_asof(
            contas_do_aparelho,
            left_on="momento",
            right_on="_desde",
            by="aparelho",
            allow_exact_matches=False,
            strategy="backward",
            check_sortedness=False,
        )
        .join_asof(
            primeiras.select("aparelho", "cliente", pl.col("momento").alias("_minha_entrada")),
            left_on="momento",
            right_on="_minha_entrada",
            by=["aparelho", "cliente"],
            allow_exact_matches=False,
            strategy="backward",
            check_sortedness=False,
        )
    )


def _variaveis(pix: pl.DataFrame) -> pl.DataFrame:
    local = pl.col("momento").dt.replace_time_zone(None)
    valor = pl.col("valor")
    noturno = pl.col("_noturno")

    return pix.select(
        "id",
        (valor / 100).alias("valor"),
        local.dt.hour().cast(pl.Int8).alias("hora"),
        noturno.alias("noturno"),
        (local.dt.weekday() - 1).cast(pl.Int8).alias("dia_da_semana"),
        "iniciacao",
        "chave",
        (pl.col("recebedor_pessoa") == "PJ").alias("recebedor_pj"),
        pl.col("recebedor_banco").cast(pl.String).is_in(DIGITAIS).alias("recebedor_em_banco_digital"),
        (valor % 1_000 == 0).alias("valor_redondo"),
        (valor % 100 == 90).alias("termina_em_90_centavos"),
        (valor / pl.col("saldo_antes")).alias("parte_do_saldo"),
        pl.when(noturno).then(valor / pl.col("limite_noturno")).alias("parte_do_limite_noturno"),
        "idade",
        (pl.col("renda_mensal") / 100).alias("renda_mensal"),
        (local.dt.date() - pl.col("conta_aberta_em")).dt.total_days().cast(pl.Int32).alias("dias_de_conta"),
        (valor / pl.col("renda_mensal")).alias("valor_sobre_a_renda"),
        pl.col("_na_ultima_hora").alias("pix_na_ultima_hora"),
        pl.col("_nas_ultimas_24h").alias("pix_nas_ultimas_24h"),
        (pl.col("_valor_nas_ultimas_24h") / 100).alias("valor_nas_ultimas_24h"),
        ((pl.col("_segundos") - pl.col("_ultimo_pix")) / 60).alias("minutos_desde_o_ultimo_pix"),
        (pl.col("_mediana") / 100).alias("mediana_de_90_dias"),
        (valor / pl.col("_mediana")).alias("valor_sobre_a_mediana"),
        (valor / pl.col("_maior")).alias("valor_sobre_o_maior_de_90_dias"),
        pl.col("_noturnos_em_90_dias").alias("pix_noturnos_em_90_dias"),
        pl.col("_vezes").alias("vezes_que_pagou_o_recebedor"),
        (pl.col("_vezes") == 0).alias("recebedor_novo"),
        ((pl.col("_segundos") - pl.col("_primeiro")) / 86_400).alias("dias_desde_o_primeiro_pix_ao_recebedor"),
        pl.col("_recebidos_em_24h").alias("pix_recebidos_em_24h"),
        pl.col("_novos_em_24h").alias("pagadores_novos_em_24h"),
        pl.col("_novos_em_7_dias").alias("pagadores_novos_em_7_dias"),
        pl.col("_contestacoes").fill_null(0).alias("contestacoes_do_recebedor"),
        ((pl.col("momento") - pl.col("_registrado_em")).dt.total_seconds() / 60).alias("minutos_desde_o_aparelho_novo"),
        (pl.col("_contas_ate").fill_null(0) - pl.col("_minha_entrada").is_not_null().cast(pl.Int32)).alias(
            "outros_clientes_no_aparelho"
        ),
    )
