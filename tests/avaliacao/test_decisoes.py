from datetime import date

import polars as pl
import pytest

from antifraude_pix.avaliacao.decisoes import NOMES, POLITICAS, balanco, com_a_politica, custo_de_fato
from antifraude_pix.decisao.custo import Custos

OUTUBRO = date(2025, 10, 1)
CUSTOS = Custos(confirmar=10.0, bloquear_um_honesto=150.0, confirmacao_para=0.7)


def _mes(linhas: list[tuple[float, float, str | None]], modelo: str = "lightgbm") -> pl.DataFrame:
    """Um mês de um modelo: a probabilidade, o valor e o golpe de cada Pix. A pontuação é a probabilidade."""
    probabilidades, valores, golpes = zip(*linhas, strict=True)

    return pl.DataFrame(
        {
            "modelo": modelo,
            "corte": OUTUBRO,
            "pontuacao": probabilidades,
            "probabilidade": probabilidades,
            "valor": valores,
            "golpe": golpes,
        },
        schema_overrides={"golpe": pl.String},
    )


# Quatro Pix que contam a história, e 996 Pix honestos de R$ 50 e 0,01% de chance, que só completam o mês:
# - o golpe de R$ 1.000 com 90%: pelo custo, bloqueado;
# - o golpe de R$ 200 com 20%: pelo custo, segurado (aprovar custa R$ 40; segurar, 10 + 12 = R$ 22);
# - o honesto de R$ 5.000 com 1%: pelo custo, segurado (aprovar custa R$ 50; segurar, R$ 25);
# - o honesto de R$ 300 com 60%: pelo custo, bloqueado (segurar custa 10 + 54 = R$ 64; bloquear, R$ 60).
HISTORIA = [(0.9, 1_000.0, "falsa_central"), (0.2, 200.0, "falso_parente"), (0.01, 5_000.0, None), (0.6, 300.0, None)]
MES = _mes(HISTORIA + [(0.0001, 50.0, None)] * 996)


def _do(politica: str) -> dict[str, float]:
    """Os números do balanço do mês, numa política."""
    linha = balanco(MES, CUSTOS).filter(pl.col("politica") == politica)

    return linha.drop("modelo", "politica").row(0, named=True)


def test_o_custo_de_fato_de_cada_saida_com_e_sem_golpe() -> None:
    # Um golpe e um Pix honesto de R$ 1.000 em cada saída.
    seis = _mes([(0.0, 1_000.0, "venda_falsa"), (0.0, 1_000.0, None)] * 3).with_columns(
        pl.Series("acao", ["aprovar", "aprovar", "segurar", "segurar", "bloquear", "bloquear"])
    )

    assert seis.select(custo_de_fato(CUSTOS).alias("custo"))["custo"].to_list() == pytest.approx(
        [1_000.0, 0.0, 10.0 + 300.0, 10.0, 0.0, 150.0]
    )


def test_pelo_custo_cada_pix_vai_para_a_sua_saida() -> None:
    decididas = com_a_politica(MES, "custo", CUSTOS)

    assert decididas["acao"].cast(pl.String).head(4).to_list() == ["bloquear", "segurar", "segurar", "bloquear"]
    assert decididas["acao"].cast(pl.String).tail(996).unique().to_list() == ["aprovar"]
    # O golpe segurado custa a confirmação e os 30% que ela não para; o honesto segurado, a confirmação.
    assert decididas["custo"].head(4).to_list() == pytest.approx([0.0, 10.0 + 60.0, 10.0, 150.0])


def test_o_balanco_de_cada_politica() -> None:
    assert _do("aprovar_tudo") == pytest.approx(
        {
            "custo": 1_200.0,
            "perdido": 1_200.0,
            "golpes_parados": 0.0,
            "golpes": 2,
            "honestos_segurados": 0,
            "honestos_bloqueados": 0,
            "parte_segurada": 0.0,
            "parte_bloqueada": 0.0,
        }
    )
    # Bloqueia de 50% para cima: o golpe de 90% e o honesto de 60%; o golpe de 20% passa.
    assert _do("limiar") == pytest.approx(
        _do("aprovar_tudo")
        | {
            "custo": 200.0 + 150.0,
            "perdido": 200.0,
            "golpes_parados": 1.0,
            "honestos_bloqueados": 1,
            "parte_bloqueada": 2 / 1_000,
        }
    )
    # Em mil Pix, o alerta segura cinco, os de maior pontuação: os quatro da história e um dos honestos de R$ 50.
    # Os 996 empatam; o empate não cabe, e o alerta segura só os quatro.
    assert _do("alerta") == pytest.approx(
        _do("aprovar_tudo")
        | {
            "custo": 4 * 10.0 + 0.3 * 1_200.0,
            "perdido": 0.3 * 1_200.0,
            "golpes_parados": 2 * 0.7,
            "honestos_segurados": 2,
            "parte_segurada": 4 / 1_000,
        }
    )
    assert _do("custo") == pytest.approx(
        _do("aprovar_tudo")
        | {
            "custo": 0.0 + 70.0 + 10.0 + 150.0,
            "perdido": 60.0,
            "golpes_parados": 1.0 + 0.7,
            "honestos_segurados": 1,
            "honestos_bloqueados": 1,
            "parte_segurada": 2 / 1_000,
            "parte_bloqueada": 2 / 1_000,
        }
    )


def test_cada_modelo_tem_o_seu_balanco() -> None:
    dois = pl.concat([MES, MES.with_columns(pl.lit("regras").alias("modelo"))])
    linhas = balanco(dois, CUSTOS)

    assert linhas.height == 2 * len(POLITICAS)
    assert (
        linhas.filter(pl.col("modelo") == "regras")
        .drop("modelo")
        .equals(linhas.filter(pl.col("modelo") == "lightgbm").drop("modelo"))
    )


def test_cada_politica_tem_um_nome_para_quem_le() -> None:
    assert set(NOMES) == set(POLITICAS)
    assert NOMES["alerta"] == "segurar o alerta de 0,5%"
    assert NOMES["limiar"] == "bloquear de 50% para cima"


def test_a_politica_tem_de_existir() -> None:
    with pytest.raises(ValueError, match="Não conheço a política 'sorteio'"):
        com_a_politica(MES, "sorteio")


def test_o_limiar_bloqueia_a_partir_de_50_por_cento_inclusive() -> None:
    tres = _mes([(0.4999, 100.0, None), (0.5, 100.0, None), (0.5001, 100.0, None)])

    assert com_a_politica(tres, "limiar")["acao"].cast(pl.String).to_list() == ["aprovar", "bloquear", "bloquear"]
