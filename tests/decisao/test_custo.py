import polars as pl
import pytest
from hypothesis import given
from hypothesis import strategies as st

from antifraude_pix.decisao.custo import ACAO, ACOES, Acao, Custos, acao, acao_de, custos_esperados, decidir

CUSTOS = Custos(confirmar=10.0, bloquear_um_honesto=150.0, confirmacao_para=0.7)


def _decisoes(probabilidades: list[float], valores: list[float], custos: Custos = CUSTOS) -> list[str]:
    pix = pl.DataFrame({"probabilidade": probabilidades, "valor": valores})

    return decidir(pix, custos)["acao"].cast(pl.String).to_list()


def test_cada_pix_vai_para_a_saida_de_menor_custo_esperado() -> None:
    # - R$ 50 com 0,1% de chance: aprovar custa R$ 0,05; segurar, R$ 10,02; bloquear, R$ 149,85;
    # - R$ 5.000 com 1%: aprovar custa R$ 50; segurar, 10 + 15 = R$ 25; bloquear, R$ 148,50;
    # - R$ 1.000 com 90%: aprovar custa R$ 900; segurar, 10 + 270 = R$ 280; bloquear, R$ 15.
    assert _decisoes([0.001, 0.01, 0.9], [50.0, 5_000.0, 1_000.0]) == ["aprovar", "segurar", "bloquear"]


def test_o_limiar_depende_do_valor() -> None:
    # A mesma chance de 1%: o Pix de R$ 50 passa, o de R$ 5.000 espera a confirmação.
    assert _decisoes([0.01, 0.01], [50.0, 5_000.0]) == ["aprovar", "segurar"]
    # A mesma chance de 40%: o de R$ 100 espera; o de R$ 2.000 é bloqueado.
    assert _decisoes([0.4, 0.4], [100.0, 2_000.0]) == ["segurar", "bloquear"]


def test_no_empate_fica_a_saida_mais_branda() -> None:
    custos = Custos(confirmar=10.0, bloquear_um_honesto=150.0, confirmacao_para=0.5)
    # 25% de R$ 80: aprovar custa R$ 20, e segurar, 10 + 10 = R$ 20.
    # Metade de chance em R$ 260: segurar custa 10 + 65 = R$ 75, e bloquear, R$ 75.
    assert _decisoes([0.25, 0.5], [80.0, 260.0], custos) == ["aprovar", "segurar"]
    # O mesmo empate, num Pix só.
    assert [acao_de(0.25, 80.0, custos), acao_de(0.5, 260.0, custos)] == [Acao.APROVAR, Acao.SEGURAR]


def test_os_custos_esperados_sao_os_da_conta() -> None:
    pix = pl.DataFrame({"probabilidade": [0.2], "valor": [500.0]})
    esperado = custos_esperados(pl.col("probabilidade"), pl.col("valor"), CUSTOS)
    linha = pix.select(saida.alias(nome) for nome, saida in esperado.items()).row(0, named=True)

    assert linha == pytest.approx({"aprovar": 100.0, "segurar": 10.0 + 30.0, "bloquear": 120.0})


def test_a_decisao_e_uma_coluna_a_mais_de_categoria_fechada() -> None:
    pix = pl.DataFrame({"id": [7, 8], "probabilidade": [0.0, 1.0], "valor": [10.0, 10.0]})
    decididos = decidir(pix)

    assert decididos.columns == ["id", "probabilidade", "valor", "acao"]
    assert decididos.schema["acao"] == ACAO
    assert ACOES == (Acao.APROVAR, Acao.SEGURAR, Acao.BLOQUEAR)
    assert decididos["acao"].cast(pl.String).to_list() == ["aprovar", "bloquear"]


@pytest.mark.parametrize(
    ("confirmar", "bloquear", "para", "mensagem"),
    [
        (0.0, 150.0, 0.7, "positivo"),
        (10.0, 10.0, 0.7, "custa mais que confirmar"),
        (10.0, 150.0, 0.0, "maior que 0 e até 1"),
        (10.0, 150.0, 1.2, "maior que 0 e até 1"),
    ],
)
def test_os_custos_tem_de_fazer_sentido(confirmar: float, bloquear: float, para: float, mensagem: str) -> None:
    with pytest.raises(ValueError, match=mensagem):
        Custos(confirmar, bloquear, para)


_CHANCE = st.floats(0.0, 1.0)
_VALOR = st.floats(0.01, 100_000.0)


@given(_CHANCE, _VALOR)
def test_a_saida_escolhida_custa_o_minimo(probabilidade: float, valor: float) -> None:
    pix = pl.DataFrame({"probabilidade": [probabilidade], "valor": [valor]})
    esperado = custos_esperados(pl.col("probabilidade"), pl.col("valor"), CUSTOS)
    linha = pix.select(
        acao(pl.col("probabilidade"), pl.col("valor"), CUSTOS).cast(pl.String).alias("acao"),
        *(saida.alias(nome) for nome, saida in esperado.items()),
    ).row(0, named=True)

    assert linha[linha["acao"]] == min(linha[saida.value] for saida in ACOES)


@given(_CHANCE, _VALOR)
def test_a_saida_de_um_pix_so_e_a_da_tabela(probabilidade: float, valor: float) -> None:
    assert acao_de(probabilidade, valor, CUSTOS).value == _decisoes([probabilidade], [valor])[0]


@given(_CHANCE, _CHANCE, _VALOR)
def test_mais_chance_de_golpe_nunca_deixa_a_saida_mais_branda(uma: float, outra: float, valor: float) -> None:
    menor, maior = sorted([uma, outra])
    saidas = _decisoes([menor, maior], [valor, valor])

    assert ACOES.index(Acao(saidas[0])) <= ACOES.index(Acao(saidas[1]))


@given(_CHANCE, _VALOR, _VALOR)
def test_mais_dinheiro_em_jogo_nunca_deixa_a_saida_mais_branda(probabilidade: float, um: float, outro: float) -> None:
    menor, maior = sorted([um, outro])
    saidas = _decisoes([probabilidade, probabilidade], [menor, maior])

    assert ACOES.index(Acao(saidas[0])) <= ACOES.index(Acao(saidas[1]))
