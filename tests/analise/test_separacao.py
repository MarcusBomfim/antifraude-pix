import polars as pl
import pytest

from antifraude_pix.analise.separacao import ANALISADAS, forca, forcas
from antifraude_pix.dados.banco import VisaoDoBanco
from antifraude_pix.simulacao.golpes import GOLPES, Golpe
from antifraude_pix.simulacao.simulador import Simulacao
from antifraude_pix.variaveis.calculo import calcular

G, L = True, False
"""Golpe e legítimo, para os exemplos caberem numa linha."""


def _forca(valores: list[float | None], golpe: list[bool]) -> float:
    return forca(pl.Series(valores, dtype=pl.Float64), pl.Series(golpe))


def test_a_variavel_que_separa_tudo_tem_forca_um_nos_dois_sentidos() -> None:
    assert _forca([1, 2, 3, 4], [L, L, G, G]) == 1.0
    assert _forca([4, 3, 2, 1], [L, L, G, G]) == 1.0


def test_a_variavel_que_nao_separa_nada_tem_forca_meio() -> None:
    assert _forca([7, 7, 7, 7], [L, G, L, G]) == 0.5


def test_a_forca_conta_os_pares_e_o_empate_vale_meio() -> None:
    # Golpes 3 e 5, legítimos 1, 3 e 4: dos seis pares, o golpe fica acima em quatro e empata em um.
    assert _forca([3, 5, 1, 3, 4], [G, G, L, L, L]) == pytest.approx(4.5 / 6)


def test_o_vazio_conta_como_o_menor_valor_de_todos() -> None:
    # Golpes vazio e 5, legítimos vazio, 1 e 2: o vazio empata com o vazio e fica abaixo do resto.
    # Se o vazio fosse o maior, o golpe vazio ficaria acima de 1 e de 2, e a força seria 4,5 / 6.
    assert _forca([None, 5, None, 1, 2], [G, G, L, L, L]) == pytest.approx(3.5 / 6)


def test_sem_golpe_ou_sem_legitimo_nao_ha_forca() -> None:
    with pytest.raises(ValueError, match="pede Pix de golpe e Pix legítimos"):
        _forca([1, 2], [L, L])

    with pytest.raises(ValueError, match="pede Pix de golpe e Pix legítimos"):
        _forca([1, 2], [G, G])


def test_o_primeiro_mes_fica_de_fora_porque_ali_todo_recebedor_e_novo(simulacao: Simulacao) -> None:
    variaveis = calcular(VisaoDoBanco.da(simulacao))
    dias = simulacao.transacoes.select(
        "id", (pl.col("momento").dt.date() - simulacao.config.inicio).dt.total_days().alias("dia")
    )

    def forca_do_valor_trocado_nos(escolhidos: pl.Expr) -> pl.DataFrame:
        trocados = dias.filter(escolhidos)["id"].implode()
        valor = pl.col("valor")
        com_o_valor_trocado = variaveis.with_columns(
            pl.when(pl.col("id").is_in(trocados)).then(-valor).otherwise(valor).alias("valor")
        )
        return forcas(simulacao, com_o_valor_trocado).filter(pl.col("variavel") == "valor")

    forca_do_valor = forcas(simulacao, variaveis).filter(pl.col("variavel") == "valor")
    dia = pl.col("dia")

    # Trocar o sinal do valor nos primeiros trinta dias não muda nada; no trigésimo primeiro, muda.
    assert forca_do_valor_trocado_nos(dia < 30).equals(forca_do_valor)
    assert not forca_do_valor_trocado_nos(dia == 30).equals(forca_do_valor)


def test_cada_golpe_tem_o_seu_rastro_e_nenhuma_variavel_pega_o_falso_parente_sozinha(simulacao: Simulacao) -> None:
    tabela = forcas(simulacao, calcular(VisaoDoBanco.da(simulacao)))
    forca_de = {(variavel, golpe): valor for variavel, golpe, valor in tabela.iter_rows()}
    outros = [golpe.value for golpe in GOLPES if golpe is not Golpe.INVASAO_DE_CONTA]

    assert tabela.height == len(ANALISADAS) * len(GOLPES)

    # O aparelho da quadrilha só denuncia a invasão; o preço de anúncio, só a venda falsa.
    assert forca_de["outros_clientes_no_aparelho", "invasao_de_conta"] > 0.85
    assert all(forca_de["outros_clientes_no_aparelho", golpe] < 0.55 for golpe in outros)
    assert forca_de["termina_em_90_centavos", "venda_falsa"] > 0.95
    assert forca_de["termina_em_90_centavos", "falso_parente"] < 0.55

    # O falso parente sai do aparelho de sempre, para uma pessoa, em valor que o autônomo também cobra.
    assert max(forca_de[variavel, "falso_parente"] for variavel in ANALISADAS) < 0.96
