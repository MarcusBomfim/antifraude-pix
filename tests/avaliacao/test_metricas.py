from datetime import date

import polars as pl
import pytest

from antifraude_pix.avaliacao.metricas import com_o_alerta, confiabilidade, curva_do_alerta, por_golpe, resumo

OUTUBRO, NOVEMBRO = date(2025, 10, 1), date(2025, 11, 1)

ESQUEMA = {
    "modelo": pl.String,
    "corte": pl.Date,
    "pontuacao": pl.Float64,
    "probabilidade": pl.Float64,
    "valor": pl.Float64,
    "golpe": pl.String,
}


def _mes(
    pontuacoes: list[float],
    golpes: dict[int, str] | None = None,
    valores: dict[int, float] | None = None,
    **outros: object,
) -> pl.DataFrame:
    """Um mês de um modelo: a pontuação de cada Pix, que também é a probabilidade, e os golpes pelo índice."""
    golpes, valores = golpes or {}, valores or {}

    return pl.DataFrame(
        {
            "modelo": outros.get("modelo", "lightgbm"),
            "corte": outros.get("corte", OUTUBRO),
            "pontuacao": pontuacoes,
            "probabilidade": [pontuacao / 1_000 for pontuacao in pontuacoes],
            "valor": [valores.get(indice, 50.0) for indice in range(len(pontuacoes))],
            "golpe": [golpes.get(indice) for indice in range(len(pontuacoes))],
        },
        schema=ESQUEMA,
    )


def test_o_alerta_segura_o_topo_de_cada_mes() -> None:
    mil = _mes([float(pontuacao) for pontuacao in range(1_000)])
    alertados = com_o_alerta(mil).filter(pl.col("alertado"))["pontuacao"].sort().to_list()

    assert alertados == [995.0, 996.0, 997.0, 998.0, 999.0]

    # Dois meses de 400 Pix: dois de cada mês, e não os quatro maiores dos dois juntos.
    dois_meses = pl.concat(
        [_mes([float(p) for p in range(400)]), _mes([float(p) + 1_000 for p in range(400)], corte=NOVEMBRO)]
    )
    por_mes = com_o_alerta(dois_meses).filter(pl.col("alertado")).group_by("corte").len().sort("corte")

    assert por_mes["len"].to_list() == [2, 2]


def test_o_alerta_arredonda_para_baixo() -> None:
    # Em 999 Pix, 0,5% são 4,995 Pix: cabem quatro.
    assert com_o_alerta(_mes([float(pontuacao) for pontuacao in range(999)]))["alertado"].sum() == 4


def test_o_empate_que_nao_cabe_inteiro_fica_de_fora() -> None:
    # Cabem cinco: os quatro de 10 pontos entram; os três de 9, empatados, não cabem todos, e nenhum entra.
    pontuacoes = [10.0] * 4 + [9.0] * 3 + [0.0] * 993

    assert com_o_alerta(_mes(pontuacoes))["alertado"].sum() == 4


def test_o_alerta_e_uma_parte_dos_pix() -> None:
    with pytest.raises(ValueError, match="entre 0 e 1"):
        com_o_alerta(_mes([1.0, 2.0]), alerta=0.0)


def test_o_resumo_conta_o_topo_da_fila() -> None:
    # Mil Pix, quatro golpes: o 1º, o 3º, o 6º e o 500º da fila. Cabem cinco no alerta.
    lugares = [1, 3, 6, 500]
    golpes = {1_000 - lugar: "falso_parente" for lugar in lugares}
    valores = {1_000 - lugar: valor for lugar, valor in zip(lugares, [100.0, 200.0, 300.0, 400.0], strict=True)}
    linha = resumo(_mes([float(p) for p in range(1_000)], golpes, valores)).row(0, named=True)

    # A precisão em cada golpe: 1/1, 2/3, 3/6 e 4/500.
    assert linha["precisao_media"] == pytest.approx((1 + 2 / 3 + 3 / 6 + 4 / 500) / 4)
    # Dos 996 legítimos, cada golpe fica acima de 996, 995, 993 e 500.
    assert linha["area_roc"] == pytest.approx((996 + 995 + 993 + 500) / (4 * 996))
    assert (linha["recall_no_alerta"], linha["precisao_no_alerta"], linha["legitimos_por_golpe_pego"]) == (
        0.5,
        0.4,
        1.5,
    )
    # Pegou os golpes de R$ 100 e R$ 200, dos R$ 1.000 de golpe.
    assert linha["recall_em_reais"] == pytest.approx(0.3)
    assert linha["golpes"] == 4
    assert linha["golpes_previstos"] == pytest.approx(sum(range(1_000)) / 1_000)


def test_o_modelo_que_nao_pega_nenhum_golpe_no_alerta_nao_tem_legitimos_por_golpe() -> None:
    linha = resumo(_mes([float(p) for p in range(1_000)], {0: "venda_falsa"})).row(0, named=True)

    assert (linha["recall_no_alerta"], linha["legitimos_por_golpe_pego"]) == (0.0, None)


def test_cada_golpe_e_medido_contra_os_legitimos() -> None:
    # A invasão é o 1º e o 2º da fila; a venda falsa, o 3º e o 900º.
    golpes = {999: "invasao_de_conta", 998: "invasao_de_conta", 997: "venda_falsa", 100: "venda_falsa"}
    tabela = por_golpe(_mes([float(p) for p in range(1_000)], golpes))
    linhas = {linha["golpe"]: linha for linha in tabela.iter_rows(named=True)}

    assert linhas["invasao_de_conta"]["precisao_media"] == pytest.approx(1.0)
    assert linhas["invasao_de_conta"]["recall_no_alerta"] == 1.0
    # Sem a invasão, a venda falsa é o 1º e o 898º: precisões 1/1 e 2/898.
    assert linhas["venda_falsa"]["precisao_media"] == pytest.approx((1 + 2 / 898) / 2)
    assert linhas["venda_falsa"]["recall_no_alerta"] == 0.5


def test_a_confiabilidade_compara_o_previsto_com_o_que_aconteceu() -> None:
    meses = pl.DataFrame(
        {
            "modelo": "lightgbm",
            "corte": OUTUBRO,
            "pontuacao": [0.0] * 5,
            "probabilidade": [0.00005, 0.00005, 0.5, 0.7, 0.1],
            "valor": 50.0,
            "golpe": [None, None, "falsa_central", None, None],
        },
        schema=ESQUEMA,
    )
    faixas = {linha["faixa"]: linha for linha in confiabilidade(meses).iter_rows(named=True)}

    assert faixas["0 a 0.0001"]["observada"] == 0.0
    assert faixas["0 a 0.0001"]["pix"] == 2
    # A faixa é fechada à esquerda: o Pix de exatamente 10% fica na de 10% a 100%.
    assert faixas["0.1 a 1"]["pix"] == 3
    assert faixas["0.1 a 1"]["prevista"] == pytest.approx(1.3 / 3)
    assert faixas["0.1 a 1"]["observada"] == pytest.approx(1 / 3)


def test_a_curva_do_alerta_bate_com_o_alerta_de_cada_mes() -> None:
    # Outubro: golpe no 1º lugar e golpe empatado no 3º e no 4º. Novembro: golpe no 2º lugar. Mil Pix em cada mês.
    outubro = _mes([10.0, 9.0, 8.0, 8.0] + [0.0] * 996, {0: "falso_parente", 2: "falso_parente"})
    novembro = _mes([10.0, 9.0] + [0.0] * 998, {1: "venda_falsa"}, corte=NOVEMBRO)
    meses = pl.concat([outubro, novembro])
    curva = curva_do_alerta(meses)

    # O empatado só é pego quando cabem os dois empatados: com quatro Pix segurados no mês.
    assert curva["parte_segurada"].to_list() == [0.001, 0.002, 0.004]
    assert curva["recall"].to_list() == pytest.approx([1 / 3, 2 / 3, 1.0])

    for parte, recall in zip(curva["parte_segurada"], curva["recall"], strict=True):
        alertados = com_o_alerta(meses, alerta=parte)
        assert alertados.filter(pl.col("golpe").is_not_null())["alertado"].mean() == pytest.approx(recall)
