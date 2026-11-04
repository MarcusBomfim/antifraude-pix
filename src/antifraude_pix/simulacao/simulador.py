"""O simulador inteiro: da semente às tabelas que o banco vê, e à verdade que só o simulador sabe (ADR-0002)."""

from dataclasses import dataclass

import numpy as np
import polars as pl

from antifraude_pix.dados.esquema import BANCO, CHAVE, FUSO, GOLPE, INICIACAO, MOMENTO, PESSOA, UF
from antifraude_pix.simulacao.configuracao import Configuracao
from antifraude_pix.simulacao.contestacoes import contestar
from antifraude_pix.simulacao.golpes import Ataque, planejar_ataques
from antifraude_pix.simulacao.pix import Iniciacao
from antifraude_pix.simulacao.populacao import Populacao, gerar_populacao
from antifraude_pix.simulacao.rotina import Destinos, DiasDaSimulacao, Registro, para_quem, simular_cliente


@dataclass(frozen=True, slots=True)
class Verdade:
    """O que só o simulador sabe. O modelo nunca lê estas tabelas: elas servem para medir."""

    fraudes: pl.DataFrame
    """Cada Pix de golpe: a transação, o ataque e o golpe."""
    ataques: pl.DataFrame
    laranjas: pl.DataFrame
    """As contas laranja e os dias em que estiveram ativas."""


@dataclass(frozen=True, slots=True)
class Simulacao:
    """O que o banco vê — o cadastro, os Pix, os aparelhos novos e as contestações — e a verdade, à parte."""

    config: Configuracao
    clientes: pl.DataFrame
    transacoes: pl.DataFrame
    aparelhos_novos: pl.DataFrame
    contestacoes: pl.DataFrame
    verdade: Verdade


def simular(config: Configuracao) -> Simulacao:
    """A simulação inteira, sempre a mesma para a mesma configuração."""
    populacao = gerar_populacao(config)
    ataques = planejar_ataques(config, populacao)
    dias = DiasDaSimulacao.de(config)
    destinos = Destinos.de(populacao)
    registro = Registro()

    for cliente, deles in zip(populacao.clientes, para_quem(ataques, config.clientes), strict=True):
        simular_cliente(config, cliente, destinos=destinos, dias=dias, ataques=deles, registro=registro)

    pix = _pix(populacao, registro)
    fraudes = (
        pix.filter(pl.col("ataque") >= 0)
        .select(pl.col("id").alias("transacao"), "ataque")
        .join(_ataques(ataques).select(pl.col("id").alias("ataque"), "golpe"), on="ataque")
        .sort("transacao")
    )
    contestacoes = contestar(config, pix, fraudes)

    return Simulacao(
        config=config,
        clientes=_clientes(populacao),
        transacoes=pix.select(
            "id",
            _momento("instante"),
            "cliente",
            "aparelho",
            "recebedor",
            "recebedor_banco",
            "recebedor_pessoa",
            "chave",
            "iniciacao",
            "valor",
            "saldo_antes",
        ),
        aparelhos_novos=pl.DataFrame(
            registro.trocas, schema={"cliente": pl.Int32, "instante": pl.Int64, "aparelho": pl.Int64}, orient="row"
        )
        .sort("instante", "cliente", maintain_order=True)
        .select("cliente", _momento("instante"), "aparelho"),
        contestacoes=contestacoes.select("transacao", _momento("instante", "contestada_em")),
        verdade=Verdade(fraudes=fraudes, ataques=_ataques(ataques), laranjas=_laranjas(populacao)),
    )


def _momento(coluna: str, nome: str = "momento") -> pl.Expr:
    """O instante da simulação, com o fuso de Brasília: o relógio local vira data e hora de verdade."""
    return pl.from_epoch(coluna, time_unit="s").dt.replace_time_zone(FUSO).cast(MOMENTO).alias(nome)


def _pix(populacao: Populacao, registro: Registro) -> pl.DataFrame:
    recebedores = np.array(registro.recebedor, dtype=np.int64)
    bancos = np.array([conta.banco for conta in populacao.contas])
    pessoas = np.array([conta.pessoa.value for conta in populacao.contas])
    chaves = np.array([conta.chave.value for conta in populacao.contas])

    return (
        pl.DataFrame(
            {
                "instante": pl.Series(registro.instante, dtype=pl.Int64),
                "cliente": pl.Series(registro.cliente, dtype=pl.Int32),
                "aparelho": pl.Series(registro.aparelho, dtype=pl.Int64),
                "recebedor": pl.Series(recebedores, dtype=pl.Int32),
                "recebedor_banco": pl.Series(bancos[recebedores], dtype=pl.String).cast(BANCO),
                "recebedor_pessoa": pl.Series(pessoas[recebedores], dtype=pl.String).cast(PESSOA),
                "chave": pl.Series(chaves[recebedores], dtype=pl.String).cast(CHAVE),
                "iniciacao": pl.Series(registro.iniciacao, dtype=pl.String).cast(INICIACAO),
                "valor": pl.Series(registro.valor, dtype=pl.Int64),
                "saldo_antes": pl.Series(registro.saldo_antes, dtype=pl.Int64),
                "ataque": pl.Series(registro.ataque, dtype=pl.Int32),
            }
        )
        # Pela agência e pela conta não há chave.
        .with_columns(
            pl.when(pl.col("iniciacao") == Iniciacao.MANUAL.value).then(None).otherwise("chave").alias("chave")
        )
        # Em ordem de tempo; no mesmo segundo, pelo cliente, e no mesmo cliente, na ordem em que o banco processou.
        .sort("instante", "cliente", maintain_order=True)
        .with_row_index("id")
        .with_columns(pl.col("id").cast(pl.Int64))
    )


def _clientes(populacao: Populacao) -> pl.DataFrame:
    """O cadastro, como o banco o conhece: sem os hábitos, que só o simulador sabe."""
    clientes = populacao.clientes

    return pl.DataFrame(
        {
            "id": pl.Series([cliente.id for cliente in clientes], dtype=pl.Int32),
            "idade": pl.Series([cliente.idade for cliente in clientes], dtype=pl.Int16),
            "uf": pl.Series([cliente.uf for cliente in clientes], dtype=pl.String).cast(UF),
            "renda_mensal": pl.Series([cliente.renda_mensal for cliente in clientes], dtype=pl.Int64),
            "conta_aberta_em": pl.Series([cliente.conta_aberta_em for cliente in clientes], dtype=pl.Date),
            "limite_noturno": pl.Series([cliente.limite_noturno for cliente in clientes], dtype=pl.Int64),
        }
    )


def _ataques(ataques: tuple[Ataque, ...]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "id": pl.Series([ataque.id for ataque in ataques], dtype=pl.Int32),
            "golpe": pl.Series([ataque.golpe.value for ataque in ataques], dtype=pl.String).cast(GOLPE),
            "cliente": pl.Series([ataque.cliente for ataque in ataques], dtype=pl.Int32),
            "inicio": pl.Series([ataque.inicio for ataque in ataques], dtype=pl.Datetime("us")),
            "aparelho_do_golpista": pl.Series([ataque.aparelho_do_golpista for ataque in ataques], dtype=pl.Int64),
        }
    ).with_columns(pl.col("inicio").dt.replace_time_zone(FUSO))


def _laranjas(populacao: Populacao) -> pl.DataFrame:
    laranjas = populacao.laranjas

    return pl.DataFrame(
        {
            "conta": pl.Series([laranja.conta for laranja in laranjas], dtype=pl.Int32),
            "ativa_de": pl.Series([laranja.ativa_de for laranja in laranjas], dtype=pl.Date),
            "ativa_ate": pl.Series([laranja.ativa_ate for laranja in laranjas], dtype=pl.Date),
        }
    )
