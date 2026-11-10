from datetime import date, timedelta

import numpy as np
import numpy.typing as npt
import polars as pl
import pytest

from antifraude_pix.dados.banco import VisaoDoBanco
from antifraude_pix.decisao.custo import Custos
from antifraude_pix.modelos.preparo import ENTRADA
from antifraude_pix.modelos.rotulo import Dobra, janela_de_teste
from antifraude_pix.monitoramento.deriva import psi
from antifraude_pix.monitoramento.semanas import REFERENCIA, semana_a_semana, semanas_inteiras
from antifraude_pix.variaveis.calculo import calcular
from tests.apoio import simulacao_media

CORTE, ATE = date(2025, 9, 1), date(2025, 9, 29)  # uma segunda-feira e o fim da simulação média


class PeloValor:
    """Um modelo de mentira, sem treino: quanto maior o Pix, maior a chance de golpe."""

    def pontuar(self, variaveis: pl.DataFrame) -> npt.NDArray[np.float64]:
        return np.log1p(variaveis["valor"].to_numpy().astype(np.float64))

    def probabilidade_da(self, pontuacao: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return 1 / (1 + np.exp(-(pontuacao - 9)))


@pytest.fixture(scope="module")
def visao() -> VisaoDoBanco:
    return VisaoDoBanco.da(simulacao_media())


@pytest.fixture(scope="module")
def variaveis(visao: VisaoDoBanco) -> pl.DataFrame:
    return calcular(visao)


@pytest.fixture(scope="module")
def semanas(visao: VisaoDoBanco, variaveis: pl.DataFrame) -> pl.DataFrame:
    return semana_a_semana(visao, variaveis, PeloValor(), Custos(), corte=CORTE, ate=ATE)


@pytest.mark.parametrize(
    ("corte", "ate", "segundas"),
    [
        # O corte num sábado começa na segunda seguinte; a semana que não termina antes de `ate` fica de fora.
        (date(2025, 11, 1), date(2025, 12, 1), tuple(date(2025, 11, dia) for dia in (3, 10, 17, 24))),
        (date(2025, 11, 3), date(2025, 11, 20), (date(2025, 11, 3), date(2025, 11, 10))),
        (date(2025, 11, 4), date(2025, 11, 17), (date(2025, 11, 10),)),
        (date(2025, 11, 4), date(2025, 11, 16), ()),
    ],
)
def test_as_semanas_inteiras_vao_de_segunda_a_domingo(corte: date, ate: date, segundas: tuple[date, ...]) -> None:
    assert semanas_inteiras(corte, ate) == segundas


def test_o_monitor_da_uma_linha_por_semana_inteira_com_todos_os_pix_dela(
    visao: VisaoDoBanco, semanas: pl.DataFrame
) -> None:
    pix_por_semana = (
        visao.transacoes.group_by(pl.col("momento").dt.date().dt.truncate("1w").alias("semana"))
        .len()
        .filter(pl.col("semana").is_in(semanas_inteiras(CORTE, ATE)))
        .sort("semana")
    )

    assert semanas["semana"].to_list() == list(semanas_inteiras(CORTE, ATE))
    assert semanas["pix"].to_list() == pix_por_semana["len"].to_list()
    assert semanas.columns == [
        "semana",
        "pix",
        "parte_segurada",
        "parte_bloqueada",
        "psi_da_probabilidade",
        "variavel_que_mais_mudou",
        "psi_dela",
        "aprovados",
        "esperada",
        "conhecida",
        "razao",
        "razao_acumulada",
    ]


def test_o_monitor_junta_a_deriva_as_saidas_e_a_perda(semanas: pl.DataFrame) -> None:
    nomes = {variavel.nome for variavel in ENTRADA}

    assert set(semanas["variavel_que_mais_mudou"]) <= nomes
    assert (semanas["psi_dela"] >= 0).all()
    assert (semanas["psi_da_probabilidade"] >= 0).all()
    assert ((semanas["parte_segurada"] + semanas["parte_bloqueada"]).is_between(0, 1)).all()
    assert (semanas["aprovados"] <= semanas["pix"]).all()
    assert semanas["razao_acumulada"][-1] == pytest.approx(
        float(semanas["conhecida"].sum()) / float(semanas["esperada"].sum())
    )


def test_a_deriva_e_contra_as_quatro_semanas_antes_do_corte(
    visao: VisaoDoBanco, variaveis: pl.DataFrame, semanas: pl.DataFrame
) -> None:
    pix = janela_de_teste(visao, variaveis, Dobra(CORTE - timedelta(days=REFERENCIA), CORTE + timedelta(weeks=1)))
    modelo = PeloValor()
    probabilidade = pl.Series(modelo.probabilidade_da(modelo.pontuar(pix)))
    antes, na_semana = pix["dia"] < CORTE, pix["dia"] >= CORTE

    assert REFERENCIA == 28
    assert semanas["psi_da_probabilidade"][0] == pytest.approx(
        psi(probabilidade.filter(antes), probabilidade.filter(na_semana))
    )
    assert semanas["psi_dela"][0] == pytest.approx(
        max(psi(pix.filter(antes)[nome], pix.filter(na_semana)[nome]) for nome in (v.nome for v in ENTRADA))
    )


def test_sem_semana_inteira_nao_ha_o_que_monitorar(visao: VisaoDoBanco, variaveis: pl.DataFrame) -> None:
    with pytest.raises(ValueError, match="não há uma semana inteira"):
        semana_a_semana(visao, variaveis, PeloValor(), Custos(), corte=CORTE, ate=date(2025, 9, 5))
