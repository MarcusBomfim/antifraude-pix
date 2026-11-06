from dataclasses import replace
from datetime import date

import numpy as np
import polars as pl
import pytest

from antifraude_pix.avaliacao.metricas import resumo
from antifraude_pix.avaliacao.no_tempo import Rotulo, avaliar
from antifraude_pix.dados.banco import VisaoDoBanco
from antifraude_pix.modelos.rotulo import Dobra
from antifraude_pix.simulacao.simulador import Simulacao
from antifraude_pix.variaveis.calculo import calcular
from tests.apoio import simulacao_media

SETEMBRO = (Dobra(date(2025, 9, 1), date(2025, 9, 29)),)


@pytest.fixture(scope="module")
def variaveis() -> pl.DataFrame:
    return calcular(VisaoDoBanco.da(simulacao_media()))


@pytest.fixture(scope="module")
def pontuacoes(variaveis: pl.DataFrame) -> pl.DataFrame:
    return avaliar(simulacao_media(), variaveis, SETEMBRO)


def test_cada_pix_de_teste_tem_uma_pontuacao_de_cada_modelo_e_a_verdade_ao_lado(
    pontuacoes: pl.DataFrame, simulacao: Simulacao
) -> None:
    de_setembro = simulacao.transacoes.filter(pl.col("momento").dt.date() >= date(2025, 9, 1))
    golpes = simulacao.verdade.fraudes.select(pl.col("transacao").alias("id"), pl.col("golpe").cast(pl.String))

    assert pontuacoes.columns == ["id", "corte", "modelo", "pontuacao", "probabilidade", "valor", "golpe"]
    assert pontuacoes.group_by("modelo", maintain_order=True).len()["modelo"].to_list() == [
        "regras",
        "logistica",
        "lightgbm",
    ]

    for (_,), do_modelo in pontuacoes.group_by("modelo"):
        assert do_modelo["id"].sort().equals(de_setembro["id"].sort())
        assert (
            do_modelo.filter(pl.col("golpe").is_not_null())["id"]
            .sort()
            .equals(golpes.join(de_setembro.select("id"), on="id", how="semi")["id"].sort())
        )

    assert pontuacoes["probabilidade"].is_between(0, 1).all()


def test_a_pontuacao_que_sai_da_avaliacao_e_a_do_modelo(pontuacoes: pl.DataFrame) -> None:
    precisao = {linha["modelo"]: linha["precisao_media"] for linha in resumo(pontuacoes).iter_rows(named=True)}

    assert precisao["lightgbm"] > 0.8
    assert precisao["logistica"] > 0.8
    assert precisao["regras"] < 0.7


def test_o_que_vem_depois_de_um_pix_de_teste_nao_muda_a_pontuacao_dele(
    pontuacoes: pl.DataFrame, simulacao: Simulacao
) -> None:
    # A simulação inteira cortada em 15 de setembro: sem os Pix, os aparelhos e as contestações dali em diante.
    def antes(tabela: pl.DataFrame, coluna: str) -> pl.DataFrame:
        return tabela.filter(pl.col(coluna).dt.date() < date(2025, 9, 15))

    cortada = replace(
        simulacao,
        transacoes=antes(simulacao.transacoes, "momento"),
        aparelhos_novos=antes(simulacao.aparelhos_novos, "momento"),
        contestacoes=antes(simulacao.contestacoes, "contestada_em"),
    )
    da_cortada = avaliar(cortada, calcular(VisaoDoBanco.da(cortada)), SETEMBRO)

    da_inteira = pontuacoes.join(da_cortada.select("id").unique(), on="id", how="semi").sort("modelo", "id")
    da_cortada = da_cortada.sort("modelo", "id")

    assert da_cortada.height > 0
    assert da_cortada.drop("pontuacao", "probabilidade").equals(da_inteira.drop("pontuacao", "probabilidade"))
    # A conta de matriz da logística agrupa as somas pelo tamanho da matriz, e o último algarismo pode mudar.
    for coluna in ("pontuacao", "probabilidade"):
        assert np.allclose(da_cortada[coluna].to_numpy(), da_inteira[coluna].to_numpy(), rtol=0, atol=1e-12)


def test_com_a_verdade_o_modelo_aprende_outra_coisa(variaveis: pl.DataFrame, pontuacoes: pl.DataFrame) -> None:
    com_a_verdade = avaliar(simulacao_media(), variaveis, SETEMBRO, modelos=("lightgbm",), rotulo=Rotulo.VERDADE)
    do_dia = pontuacoes.filter(pl.col("modelo") == "lightgbm")

    assert com_a_verdade["id"].equals(do_dia["id"])
    assert not np.allclose(com_a_verdade["probabilidade"].to_numpy(), do_dia["probabilidade"].to_numpy())


def test_a_espera_pelo_rotulo_muda_o_que_o_modelo_aprende(variaveis: pl.DataFrame, pontuacoes: pl.DataFrame) -> None:
    esperando = avaliar(simulacao_media(), variaveis, SETEMBRO, modelos=("lightgbm",), espera_em_dias=14)
    sem_esperar = pontuacoes.filter(pl.col("modelo") == "lightgbm")

    assert esperando["id"].equals(sem_esperar["id"])
    assert not np.allclose(esperando["probabilidade"].to_numpy(), sem_esperar["probabilidade"].to_numpy())


def test_sem_dobra_nao_ha_avaliacao(variaveis: pl.DataFrame) -> None:
    with pytest.raises(ValueError, match="ao menos uma dobra"):
        avaliar(simulacao_media(), variaveis, ())
