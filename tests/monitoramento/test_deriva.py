import math

import numpy as np
import polars as pl
import pytest

from antifraude_pix.monitoramento.deriva import ESTAVEL, MUDOU, PISO, nivel, por_periodo, psi


def test_a_mesma_distribuicao_nao_muda_nada() -> None:
    valores = pl.Series(np.random.default_rng(1).normal(100, 20, 5_000))

    assert psi(valores, valores) == 0
    assert psi(valores, valores.sample(fraction=1.0, shuffle=True, seed=2)) == 0


def test_a_categoria_que_muda_de_tamanho_da_o_indice_feito_a_mao() -> None:
    referencia = pl.Series(["chave"] * 50 + ["qr"] * 50)
    atual = pl.Series(["chave"] * 80 + ["qr"] * 20)
    # (0,8 - 0,5) ln(0,8 / 0,5) + (0,2 - 0,5) ln(0,2 / 0,5); a faixa do vazio, zero nas duas, não conta.
    esperado = 0.3 * math.log(0.8 / 0.5) + (-0.3) * math.log(0.2 / 0.5)

    assert psi(referencia, atual) == pytest.approx(esperado)


def test_o_numero_vai_nas_faixas_dos_decis_da_referencia() -> None:
    referencia = pl.Series(range(1, 101), dtype=pl.Float64)
    # Metade do atual cai na primeira faixa da referência, e a outra metade, na última.
    atual = pl.Series([1.0] * 50 + [100.0] * 50)
    esperado = 2 * (0.5 - 0.1) * math.log(0.5 / 0.1) + 8 * (PISO - 0.1) * math.log(PISO / 0.1)

    assert psi(referencia, atual) == pytest.approx(esperado)


def test_o_vazio_que_cresce_tambem_e_mudanca() -> None:
    referencia = pl.Series([1.0, 2.0, 3.0, 4.0] * 25)
    # Os quatro valores guardam a proporção entre si; só o vazio muda, de nada para um quinto.
    atual = pl.Series([1.0, 2.0, 3.0, 4.0] * 20 + [None] * 20)
    esperado = 4 * (0.2 - 0.25) * math.log(0.2 / 0.25) + (0.2 - PISO) * math.log(0.2 / PISO)

    assert psi(referencia, atual) == pytest.approx(esperado)
    assert psi(referencia, atual) > MUDOU


def test_o_sim_ou_nao_e_a_categoria_nova_contam_uma_faixa_por_valor() -> None:
    sim_ou_nao = pl.Series([True] * 90 + [False] * 10)

    assert psi(sim_ou_nao, sim_ou_nao) == 0
    assert psi(sim_ou_nao, pl.Series([True] * 50 + [False] * 50)) > MUDOU
    # A categoria que a referência não tem fica com o piso, e o índice não vira infinito.
    assert math.isfinite(psi(pl.Series(["a"] * 10), pl.Series(["a"] * 5 + ["b"] * 5)))


def test_o_numero_de_poucos_valores_nao_repete_o_corte() -> None:
    poucos = pl.Series([0] * 70 + [1] * 20 + [2] * 10)

    assert psi(poucos, poucos) == 0
    assert psi(poucos, pl.Series([0] * 10 + [1] * 20 + [2] * 70)) > MUDOU


def test_sem_valores_nao_ha_o_que_comparar() -> None:
    with pytest.raises(ValueError, match="precisam de valores"):
        psi(pl.Series([], dtype=pl.Float64), pl.Series([1.0]))


@pytest.mark.parametrize(
    ("indice", "palavras"),
    [(0.0, "estável"), (ESTAVEL - 0.001, "estável"), (ESTAVEL, "mudou um pouco"), (MUDOU, "mudou"), (3.0, "mudou")],
)
def test_o_indice_em_palavras(indice: float, palavras: str) -> None:
    assert nivel(indice) == palavras


def test_por_periodo_da_uma_linha_por_periodo_e_coluna() -> None:
    referencia = pl.DataFrame({"valor": [10.0, 20.0, 30.0, 40.0] * 25, "chave": ["a", "b"] * 50})
    atual = pl.DataFrame(
        {
            "semana": ["1ª"] * 100 + ["2ª"] * 100,
            "valor": [10.0, 20.0, 30.0, 40.0] * 25 + [40.0] * 100,
            "chave": ["a", "b"] * 50 + ["a"] * 100,
        }
    )
    indices = por_periodo(referencia, atual, ["valor", "chave"], "semana")

    assert indices.columns == ["semana", "variavel", "psi"]
    assert indices.select("semana", "variavel").rows() == [
        ("1ª", "valor"),
        ("1ª", "chave"),
        ("2ª", "valor"),
        ("2ª", "chave"),
    ]
    assert indices.filter(pl.col("semana") == "1ª")["psi"].to_list() == [0, 0]
    assert (indices.filter(pl.col("semana") == "2ª")["psi"] > MUDOU).all()
