from collections.abc import Mapping

import polars as pl

from antifraude_pix.modelos.regras import REGRAS, acionadas, pontos
from antifraude_pix.variaveis.catalogo import NOMES

CALMO: dict[str, object] = {
    "minutos_desde_o_aparelho_novo": None,
    "valor": 80.0,
    "contestacoes_do_recebedor": 0,
    "recebedor_novo": False,
    "valor_sobre_a_mediana": 1.0,
    "parte_do_saldo": 0.05,
    "pix_na_ultima_hora": 0,
    "parte_do_limite_noturno": None,
}
"""Um Pix de R$ 80, do aparelho de sempre, para quem o cliente já pagou, de dia: nenhuma regra vale."""

ESQUEMA = {
    "minutos_desde_o_aparelho_novo": pl.Float64,
    "valor": pl.Float64,
    "contestacoes_do_recebedor": pl.Int32,
    "recebedor_novo": pl.Boolean,
    "valor_sobre_a_mediana": pl.Float64,
    "parte_do_saldo": pl.Float64,
    "pix_na_ultima_hora": pl.Int32,
    "parte_do_limite_noturno": pl.Float64,
}


def _pontos(*mudancas: Mapping[str, object]) -> list[float]:
    return [
        float(ponto) for ponto in pontos(pl.DataFrame([CALMO | dict(mudanca) for mudanca in mudancas], schema=ESQUEMA))
    ]


def test_cada_regra_soma_os_pontos_dela() -> None:
    assert _pontos(
        {},
        {"minutos_desde_o_aparelho_novo": 30.0, "valor": 500.0},
        {"contestacoes_do_recebedor": 1},
        {"recebedor_novo": True, "valor_sobre_a_mediana": 6.0},
        {"parte_do_saldo": 0.6, "valor": 300.0},
        {"pix_na_ultima_hora": 2},
        {"parte_do_limite_noturno": 0.9},
    ) == [0, 3, 3, 2, 2, 1, 1]


def test_cada_regra_tem_a_sua_borda() -> None:
    assert _pontos(
        # O aparelho novo com Pix pequeno, e o aparelho de dois dias atrás.
        {"minutos_desde_o_aparelho_novo": 30.0, "valor": 150.0},
        {"minutos_desde_o_aparelho_novo": 2 * 24 * 60.0, "valor": 500.0},
        # O valor fora do hábito para quem o cliente já pagou, e metade do saldo num Pix pequeno.
        {"recebedor_novo": False, "valor_sobre_a_mediana": 6.0},
        {"parte_do_saldo": 0.6, "valor": 150.0},
        # O segundo Pix da hora, e o noturno de três quartos do limite.
        {"pix_na_ultima_hora": 1},
        {"parte_do_limite_noturno": 0.75},
    ) == [0, 0, 0, 0, 0, 0]


def test_tudo_junto_soma_tudo_e_o_vazio_nao_aciona_nada() -> None:
    tudo = {
        "minutos_desde_o_aparelho_novo": 5.0,
        "valor": 900.0,
        "contestacoes_do_recebedor": 2,
        "recebedor_novo": True,
        "valor_sobre_a_mediana": 20.0,
        "parte_do_saldo": 0.9,
        "pix_na_ultima_hora": 3,
        "parte_do_limite_noturno": 0.9,
    }
    vazio = dict.fromkeys(CALMO)

    assert _pontos(tudo, vazio) == [12, 0]
    assert not acionadas(pl.DataFrame([vazio], schema=ESQUEMA)).row(0, named=True)["aparelho_novo"]


def test_as_regras_tem_nome_frase_e_usam_variaveis_que_existem() -> None:
    assert len({regra.nome for regra in REGRAS}) == len(REGRAS)
    assert all(regra.descricao.endswith(".") and regra.pontos > 0 for regra in REGRAS)
    assert {coluna for regra in REGRAS for coluna in regra.condicao.meta.root_names()} <= set(NOMES)
