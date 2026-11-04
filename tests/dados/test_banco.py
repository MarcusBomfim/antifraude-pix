import shutil
from dataclasses import fields
from datetime import UTC, date, datetime
from pathlib import Path

import polars as pl
import pytest

from antifraude_pix.dados.banco import VisaoDoBanco
from antifraude_pix.dados.esquema import DO_BANCO, FUSO
from antifraude_pix.dados.gravacao import gravar, ler_o_banco
from antifraude_pix.simulacao.simulador import Simulacao

CORTE = datetime(2025, 8, 15, 12, 0)  # noqa: DTZ001 — o corte é no relógio de Brasília, sem fuso.


def _corte_com_fuso() -> pl.Expr:
    return pl.lit(CORTE, dtype=pl.Datetime("us")).dt.replace_time_zone(FUSO)


def test_a_visao_do_banco_tem_as_tabelas_do_banco_e_nenhuma_outra(simulacao: Simulacao) -> None:
    visao = VisaoDoBanco.da(simulacao)

    assert [campo.name for campo in fields(VisaoDoBanco)] == list(DO_BANCO)
    assert visao.transacoes.equals(simulacao.transacoes)
    assert not hasattr(visao, "verdade")


def test_o_corte_tem_so_o_que_aconteceu_antes_do_instante(simulacao: Simulacao) -> None:
    antes = VisaoDoBanco.da(simulacao).ate(CORTE)

    assert 0 < antes.transacoes.height < simulacao.transacoes.height
    assert antes.transacoes.select((pl.col("momento") < _corte_com_fuso()).all()).item()
    assert antes.aparelhos_novos.select((pl.col("momento") < _corte_com_fuso()).all()).item()
    assert antes.transacoes.equals(simulacao.transacoes.filter(pl.col("momento") < _corte_com_fuso()))
    assert antes.clientes.equals(simulacao.clientes)


def test_a_contestacao_que_chegou_depois_do_corte_ainda_nao_existe_nele(simulacao: Simulacao) -> None:
    antes = VisaoDoBanco.da(simulacao).ate(CORTE)
    pix_de_antes = antes.transacoes.select(pl.col("id").alias("transacao"))
    ainda_por_chegar = simulacao.contestacoes.join(pix_de_antes, on="transacao").filter(
        pl.col("contestada_em") >= _corte_com_fuso()
    )

    # Há Pix de antes do corte contestados depois dele: no corte, o banco ainda os tem por legítimos.
    assert ainda_por_chegar.height > 0
    assert antes.contestacoes.join(ainda_por_chegar, on="transacao", how="semi").height == 0
    assert antes.contestacoes.select((pl.col("contestada_em") < _corte_com_fuso()).all()).item()


def test_o_corte_num_dia_e_a_meia_noite_dele(simulacao: Simulacao) -> None:
    visao = VisaoDoBanco.da(simulacao)

    assert visao.ate(date(2025, 8, 15)).transacoes.equals(visao.ate(datetime(2025, 8, 15)).transacoes)  # noqa: DTZ001


def test_o_corte_com_fuso_e_recusado(simulacao: Simulacao) -> None:
    with pytest.raises(ValueError, match="no horário de Brasília, sem fuso"):
        VisaoDoBanco.da(simulacao).ate(datetime(2025, 8, 15, 15, tzinfo=UTC))


def test_ler_o_banco_nao_abre_a_pasta_da_verdade(simulacao: Simulacao, tmp_path: Path) -> None:
    gravar(simulacao, tmp_path)
    shutil.rmtree(tmp_path / "verdade")

    lida = ler_o_banco(tmp_path)

    assert lida.transacoes.equals(simulacao.transacoes)
    assert lida.contestacoes.equals(simulacao.contestacoes)


def test_ler_o_banco_recusa_o_arquivo_do_banco_mexido(simulacao: Simulacao, tmp_path: Path) -> None:
    gravar(simulacao, tmp_path)

    with (tmp_path / "banco" / "contestacoes.parquet").open("ab") as arquivo:
        arquivo.write(b"\0")

    with pytest.raises(ValueError, match=r"O arquivo banco/contestacoes.parquet mudou depois de gravado"):
        ler_o_banco(tmp_path)
