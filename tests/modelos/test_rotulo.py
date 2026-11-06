import ast
from datetime import date, timedelta
from pathlib import Path

import polars as pl
import pytest

import antifraude_pix.modelos
from antifraude_pix.dados.banco import VisaoDoBanco
from antifraude_pix.modelos.rotulo import Dobra, dobras, janela_de_teste, janela_de_treino, primeiro_corte
from antifraude_pix.simulacao.simulador import Simulacao
from antifraude_pix.variaveis.calculo import calcular
from antifraude_pix.variaveis.catalogo import NOMES
from tests.apoio import simulacao_media

CORTE = date(2025, 9, 1)
"""Na simulação média, de 1º de julho a 28 de setembro: um mês de treino depois do aquecimento, e setembro de teste."""


@pytest.fixture(scope="module")
def visao() -> VisaoDoBanco:
    return VisaoDoBanco.da(simulacao_media())


@pytest.fixture(scope="module")
def variaveis(visao: VisaoDoBanco) -> pl.DataFrame:
    return calcular(visao)


def test_uma_dobra_por_mes_e_a_ultima_vai_ate_o_fim_dos_dados() -> None:
    assert dobras(date(2025, 10, 1), date(2025, 12, 31)) == (
        Dobra(date(2025, 10, 1), date(2025, 11, 1)),
        Dobra(date(2025, 11, 1), date(2025, 12, 1)),
        Dobra(date(2025, 12, 1), date(2026, 1, 1)),
    )
    assert dobras(date(2025, 9, 1), date(2025, 9, 28)) == (Dobra(date(2025, 9, 1), date(2025, 9, 29)),)
    assert dobras(date(2026, 1, 1), date(2025, 12, 31)) == ()
    # O dado que termina no primeiro dia do mês ainda tem um dia de teste naquele mês.
    assert dobras(date(2025, 11, 1), date(2025, 12, 1))[-1] == Dobra(date(2025, 12, 1), date(2025, 12, 2))


def test_o_corte_e_no_primeiro_dia_do_mes_e_o_teste_nao_termina_antes_de_comecar() -> None:
    with pytest.raises(ValueError, match="primeiro dia do mês"):
        dobras(date(2025, 10, 2), date(2025, 12, 31))

    with pytest.raises(ValueError, match="termina antes de começar"):
        Dobra(date(2025, 10, 1), date(2025, 10, 1))


def test_o_primeiro_teste_vem_depois_do_aquecimento_de_um_mes_de_treino_e_da_validacao(visao: VisaoDoBanco) -> None:
    # De 1º de julho: o aquecimento vai até 30 de julho, o treino ocupa agosto, setembro é a validação.
    assert primeiro_corte(visao) == date(2025, 10, 1)

    # De 15 de julho: o aquecimento vai até 13 de agosto, e o primeiro mês inteiro de treino termina em setembro.
    do_dia_15 = VisaoDoBanco(
        visao.clientes,
        visao.transacoes.filter(pl.col("momento").dt.date() >= date(2025, 7, 15)),
        visao.aparelhos_novos,
        visao.contestacoes,
    )

    assert primeiro_corte(do_dia_15) == date(2025, 11, 1)


def test_o_treino_vai_do_fim_do_aquecimento_ate_a_vespera_do_corte(
    visao: VisaoDoBanco, variaveis: pl.DataFrame
) -> None:
    janela = janela_de_treino(visao, variaveis, CORTE)

    assert janela.columns == ["id", "dia", "rotulo", *NOMES]
    assert janela["dia"].min() == date(2025, 7, 31)
    assert janela["dia"].max() == date(2025, 8, 31)
    assert janela["id"].is_sorted()


def test_o_rotulo_e_o_que_o_banco_sabia_no_corte(visao: VisaoDoBanco, variaveis: pl.DataFrame) -> None:
    janela = janela_de_treino(visao, variaveis, CORTE)
    contestadas = visao.contestacoes.select(
        pl.col("transacao").alias("id"), (pl.col("contestada_em").dt.date() < CORTE).alias("antes_do_corte")
    )
    com_a_contestacao = janela.join(contestadas, on="id", how="left")

    # O contestado antes do corte é golpe; o contestado depois ainda não é; o nunca contestado, nunca.
    assert com_a_contestacao.filter(pl.col("antes_do_corte").fill_null(value=False))["rotulo"].all()
    depois = com_a_contestacao.filter(~pl.col("antes_do_corte").fill_null(value=True))
    assert depois.height > 0
    assert not depois["rotulo"].any()
    assert not com_a_contestacao.filter(pl.col("antes_do_corte").is_null())["rotulo"].any()


def test_o_rotulo_do_dia_nao_sabe_de_todos_os_golpes_e_acusa_alguns_legitimos(
    visao: VisaoDoBanco, variaveis: pl.DataFrame, simulacao: Simulacao
) -> None:
    janela = janela_de_treino(visao, variaveis, CORTE)
    golpes = simulacao.verdade.fraudes.select(pl.col("transacao").alias("id"), pl.lit(value=True).alias("golpe"))
    comparado = janela.join(golpes, on="id", how="left").with_columns(pl.col("golpe").fill_null(value=False))

    quantos_golpes = comparado.filter(pl.col("golpe")).height
    golpes_sem_rotulo = comparado.filter(pl.col("golpe") & ~pl.col("rotulo")).height
    legitimos_com_rotulo = comparado.filter(~pl.col("golpe") & pl.col("rotulo")).height

    assert golpes_sem_rotulo > 0.1 * quantos_golpes
    assert legitimos_com_rotulo > 0


def test_o_treino_e_o_mesmo_com_o_banco_inteiro_e_com_o_banco_do_dia_do_corte(
    visao: VisaoDoBanco, variaveis: pl.DataFrame
) -> None:
    do_dia = visao.ate(CORTE)

    assert janela_de_treino(visao, variaveis, CORTE).equals(janela_de_treino(do_dia, calcular(do_dia), CORTE))


def test_a_espera_deixa_de_fora_os_ultimos_dias(visao: VisaoDoBanco, variaveis: pl.DataFrame) -> None:
    janela = janela_de_treino(visao, variaveis, CORTE, espera_em_dias=14)

    assert janela["dia"].max() == CORTE - timedelta(days=15)

    with pytest.raises(ValueError, match="não pode ser negativa"):
        janela_de_treino(visao, variaveis, CORTE, espera_em_dias=-1)


def test_o_teste_vai_do_corte_ate_a_vespera_do_fim_sem_rotulo(visao: VisaoDoBanco, variaveis: pl.DataFrame) -> None:
    teste = janela_de_teste(visao, variaveis, Dobra(CORTE, date(2025, 9, 15)))

    assert teste.columns == ["id", "dia", *NOMES]
    assert (teste["dia"].min(), teste["dia"].max()) == (CORTE, date(2025, 9, 14))


def test_os_modelos_nao_leem_a_verdade() -> None:
    """O pacote dos modelos não importa o simulador, a gravação, as análises nem a avaliação, e não toca na verdade."""
    pasta = Path(antifraude_pix.modelos.__file__).parent
    proibidos = (
        "antifraude_pix.simulacao",
        "antifraude_pix.dados.gravacao",
        "antifraude_pix.analise",
        "antifraude_pix.avaliacao",
    )

    for arquivo in pasta.glob("*.py"):
        arvore = ast.parse(arquivo.read_text(encoding="utf-8"))
        importados = {no.module or "" for no in ast.walk(arvore) if isinstance(no, ast.ImportFrom)}
        importados |= {nome.name for no in ast.walk(arvore) if isinstance(no, ast.Import) for nome in no.names}
        nomes = {no.attr for no in ast.walk(arvore) if isinstance(no, ast.Attribute)} | {
            no.id for no in ast.walk(arvore) if isinstance(no, ast.Name)
        }

        assert not any(modulo.startswith(proibidos) for modulo in importados), arquivo.name
        assert not any("verdade" in nome.lower() for nome in nomes), arquivo.name
