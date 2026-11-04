"""O que tem de valer nos dados, conferido de fora: o simulador não se confere sozinho (ADR-0004).

A validação lê só as tabelas e a configuração, e não o código do simulador. É
a mesma que roda depois de gravar e antes de treinar: dados que não passam
por ela não chegam ao modelo. Os problemas vêm todos de uma vez, cada um em
uma frase, com quantas linhas o têm.
"""

from datetime import datetime, time

import polars as pl

from antifraude_pix.dados.esquema import DA_VERDADE, DO_BANCO
from antifraude_pix.simulacao.pix import DIA, HORA
from antifraude_pix.simulacao.rotina import APARELHOS_POR_CLIENTE
from antifraude_pix.simulacao.simulador import Simulacao


def problemas(simulacao: Simulacao) -> list[str]:
    """Os problemas dos dados; nenhum, quando eles estão certos."""
    encontrados = _esquemas(simulacao)

    # Com uma coluna fora do contrato, as outras conferências não têm o que conferir.
    if encontrados:
        return encontrados

    return [
        *_clientes(simulacao),
        *_transacoes(simulacao),
        *_limite_noturno(simulacao),
        *_aparelhos(simulacao),
        *_contestacoes(simulacao),
        *_verdade(simulacao),
    ]


def _esquemas(simulacao: Simulacao) -> list[str]:
    tabelas = {
        "clientes": simulacao.clientes,
        "transacoes": simulacao.transacoes,
        "aparelhos_novos": simulacao.aparelhos_novos,
        "contestacoes": simulacao.contestacoes,
        "fraudes": simulacao.verdade.fraudes,
        "ataques": simulacao.verdade.ataques,
        "laranjas": simulacao.verdade.laranjas,
    }

    return [
        f"A tabela {nome} não segue o contrato: esperava {dict(esperado)}, e veio {dict(tabelas[nome].schema)}."
        for nome, esperado in {**DO_BANCO, **DA_VERDADE}.items()
        if tabelas[nome].schema != esperado
    ]


def _contagem(quantas: int, frase: str) -> list[str]:
    return [frase.format(n=quantas)] if quantas else []


def _sequencia(tabela: pl.DataFrame, coluna: str) -> bool:
    return tabela[coluna].to_list() == list(range(tabela.height))


def _local(coluna: str) -> pl.Expr:
    """O instante no relógio de Brasília, sem o fuso: para comparar com as datas da configuração."""
    return pl.col(coluna).dt.replace_time_zone(None)


def _clientes(simulacao: Simulacao) -> list[str]:
    if simulacao.clientes.height != simulacao.config.clientes or not _sequencia(simulacao.clientes, "id"):
        return [f"Os ids dos clientes não vão de 0 a {simulacao.config.clientes - 1}, em ordem."]

    return []


def _transacoes(simulacao: Simulacao) -> list[str]:
    pix = simulacao.transacoes
    config = simulacao.config
    inicio, fim = datetime.combine(config.inicio, time()), datetime.combine(config.fim, time())

    resultado = [] if _sequencia(pix, "id") else ["Os ids dos Pix não vão de 0 ao último, em ordem."]

    contas = pix.select(
        (pl.col("momento").diff() < pl.duration(seconds=0)).sum().alias("fora_de_ordem"),
        ((_local("momento") < inicio) | (_local("momento") >= fim)).sum().alias("fora_do_periodo"),
        (pl.col("valor") <= 0).sum().alias("sem_valor"),
        (pl.col("valor") > pl.col("saldo_antes")).sum().alias("sem_saldo"),
        ((pl.col("iniciacao") == "manual") != pl.col("chave").is_null()).sum().alias("chave_errada"),
        (~pl.col("cliente").is_in(simulacao.clientes["id"].implode())).sum().alias("sem_cliente"),
    ).row(0, named=True)

    return [
        *resultado,
        *_contagem(contas["fora_de_ordem"], "Há {n} Pix fora da ordem do tempo."),
        *_contagem(contas["fora_do_periodo"], "Há {n} Pix fora do período simulado."),
        *_contagem(contas["sem_valor"], "Há {n} Pix de valor zero ou negativo."),
        *_contagem(contas["sem_saldo"], "Há {n} Pix maiores que o saldo que o cliente tinha."),
        *_contagem(contas["chave_errada"], "Há {n} Pix manuais com chave, ou pela chave sem ela."),
        *_contagem(contas["sem_cliente"], "Há {n} Pix de clientes que não estão no cadastro."),
    ]


def _limite_noturno(simulacao: Simulacao) -> list[str]:
    config = simulacao.config
    hora = _local("momento").dt.hour()
    noturno = (hora >= config.inicio_da_noite) | (hora < config.fim_da_noite)
    segundos = _local("momento").dt.epoch("s")

    acima = (
        simulacao.transacoes.filter(noturno)
        .group_by("cliente", ((segundos - config.inicio_da_noite * HORA) // DIA).alias("noite"))
        .agg(pl.col("valor").sum())
        .join(simulacao.clientes.select(pl.col("id").alias("cliente"), "limite_noturno"), on="cliente")
        .filter(pl.col("valor") > pl.col("limite_noturno"))
        .height
    )

    return _contagem(acima, "Há {n} noites em que um cliente passou do limite noturno.")


def _aparelhos(simulacao: Simulacao) -> list[str]:
    # Todo Pix sai do primeiro aparelho do cliente, ou de um que apareceu antes dele.
    desde = simulacao.aparelhos_novos.group_by("cliente", "aparelho").agg(pl.col("momento").min().alias("desde"))
    sem_origem = (
        simulacao.transacoes.join(desde, on=["cliente", "aparelho"], how="left")
        .filter(
            (pl.col("aparelho") != pl.col("cliente").cast(pl.Int64) * APARELHOS_POR_CLIENTE)
            & (pl.col("desde").is_null() | (pl.col("desde") > pl.col("momento")))
        )
        .height
    )

    return _contagem(sem_origem, "Há {n} Pix de um aparelho que o cliente ainda não tinha.")


def _contestacoes(simulacao: Simulacao) -> list[str]:
    prazo = pl.duration(days=simulacao.config.prazo_do_med)
    contestados = simulacao.contestacoes.join(
        simulacao.transacoes.select(pl.col("id").alias("transacao"), "momento"), on="transacao", how="left"
    )
    contas = contestados.select(
        pl.col("momento").is_null().sum().alias("sem_pix"),
        (
            pl.col("momento").is_not_null()
            & ((pl.col("contestada_em") <= pl.col("momento")) | (pl.col("contestada_em") > pl.col("momento") + prazo))
        )
        .sum()
        .alias("fora_do_prazo"),
        pl.col("transacao").is_duplicated().sum().alias("repetidas"),
    ).row(0, named=True)

    return [
        *_contagem(contas["sem_pix"], "Há {n} contestações de Pix que não existem."),
        *_contagem(contas["fora_do_prazo"], "Há {n} contestações antes do Pix, ou depois do prazo do MED."),
        *_contagem(contas["repetidas"], "Há {n} contestações repetidas do mesmo Pix."),
    ]


def _verdade(simulacao: Simulacao) -> list[str]:
    verdade = simulacao.verdade
    ataques = verdade.ataques.select(pl.col("id").alias("ataque"), pl.col("golpe").alias("golpe_do_ataque"))
    fraudes = verdade.fraudes.join(ataques, on="ataque", how="left")
    contas = fraudes.select(
        (~pl.col("transacao").is_in(simulacao.transacoes["id"].implode())).sum().alias("sem_pix"),
        pl.col("transacao").is_duplicated().sum().alias("repetidas"),
        pl.col("golpe_do_ataque").is_null().sum().alias("sem_ataque"),
        (pl.col("golpe") != pl.col("golpe_do_ataque")).sum().alias("golpe_trocado"),
    ).row(0, named=True)

    return [
        *_contagem(contas["sem_pix"], "Há {n} fraudes de Pix que não existem."),
        *_contagem(contas["repetidas"], "Há {n} fraudes repetidas do mesmo Pix."),
        *_contagem(contas["sem_ataque"], "Há {n} fraudes de ataques que não existem."),
        *_contagem(contas["golpe_trocado"], "Há {n} fraudes com um golpe diferente do golpe do ataque."),
    ]
