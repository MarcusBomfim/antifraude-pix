"""O contrato dos dados: as colunas e os tipos de cada tabela (ADR-0004).

As categorias são tipos fechados (`pl.Enum`): um banco que não está na lista,
uma chave de tipo inventado, um golpe com outro nome não entram. O instante é
data e hora com o fuso de Brasília. O dinheiro é inteiro, em centavos — conta
de dinheiro com vírgula flutuante perde centavo.
"""

import polars as pl

from antifraude_pix.simulacao.golpes import Golpe
from antifraude_pix.simulacao.pix import Iniciacao
from antifraude_pix.simulacao.populacao import BANCOS, UFS, TipoDeChave, TipoDePessoa

FUSO = "America/Sao_Paulo"

UF = pl.Enum([uf for uf, _ in UFS])
BANCO = pl.Enum([banco.codigo for banco, _ in BANCOS])
PESSOA = pl.Enum([pessoa.value for pessoa in TipoDePessoa])
CHAVE = pl.Enum([chave.value for chave in TipoDeChave])
INICIACAO = pl.Enum([iniciacao.value for iniciacao in Iniciacao])
GOLPE = pl.Enum([golpe.value for golpe in Golpe])
MOMENTO = pl.Datetime("us", FUSO)

# O que o banco vê.
CLIENTES = pl.Schema(
    {
        "id": pl.Int32(),
        "idade": pl.Int16(),
        "uf": UF,
        "renda_mensal": pl.Int64(),
        "conta_aberta_em": pl.Date(),
        "limite_noturno": pl.Int64(),
    }
)
TRANSACOES = pl.Schema(
    {
        "id": pl.Int64(),
        "momento": MOMENTO,
        "cliente": pl.Int32(),
        "aparelho": pl.Int64(),
        "recebedor": pl.Int32(),
        "recebedor_banco": BANCO,
        "recebedor_pessoa": PESSOA,
        "chave": CHAVE,
        "iniciacao": INICIACAO,
        "valor": pl.Int64(),
        "saldo_antes": pl.Int64(),
    }
)
APARELHOS_NOVOS = pl.Schema({"cliente": pl.Int32(), "momento": MOMENTO, "aparelho": pl.Int64()})
CONTESTACOES = pl.Schema({"transacao": pl.Int64(), "contestada_em": MOMENTO})

# O que só o simulador sabe.
FRAUDES = pl.Schema({"transacao": pl.Int64(), "ataque": pl.Int32(), "golpe": GOLPE})
ATAQUES = pl.Schema(
    {
        "id": pl.Int32(),
        "golpe": GOLPE,
        "cliente": pl.Int32(),
        "inicio": MOMENTO,
        "aparelho_do_golpista": pl.Int64(),
    }
)
LARANJAS = pl.Schema({"conta": pl.Int32(), "ativa_de": pl.Date(), "ativa_ate": pl.Date()})

DO_BANCO: dict[str, pl.Schema] = {
    "clientes": CLIENTES,
    "transacoes": TRANSACOES,
    "aparelhos_novos": APARELHOS_NOVOS,
    "contestacoes": CONTESTACOES,
}
DA_VERDADE: dict[str, pl.Schema] = {"fraudes": FRAUDES, "ataques": ATAQUES, "laranjas": LARANJAS}
