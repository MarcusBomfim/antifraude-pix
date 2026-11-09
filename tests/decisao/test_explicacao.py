import ast
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from antifraude_pix.dados.banco import VisaoDoBanco
from antifraude_pix.decisao.custo import ACAO, Acao, decidir
from antifraude_pix.decisao.explicacao import Razao, chance, contribuicoes, escolher, explicar, razoes, texto
from antifraude_pix.modelos.preparo import ENTRADA
from antifraude_pix.modelos.regras import REGRAS
from antifraude_pix.modelos.rotulo import Dobra, janela_de_teste, janela_de_treino
from antifraude_pix.modelos.treino import MODELOS, Calibrado, Regras, treinar
from antifraude_pix.variaveis.calculo import calcular
from tests.apoio import simulacao_media

CORTE = date(2025, 9, 1)


@pytest.fixture(scope="module")
def setembro() -> pl.DataFrame:
    visao = VisaoDoBanco.da(simulacao_media())
    return janela_de_teste(visao, calcular(visao), Dobra(CORTE, date(2025, 9, 29)))


@pytest.fixture(scope="module")
def treinados() -> dict[str, Calibrado]:
    visao = VisaoDoBanco.da(simulacao_media())
    janela = janela_de_treino(visao, calcular(visao), CORTE)

    return {nome: treinar(nome, janela, CORTE) for nome in MODELOS}


@pytest.fixture(scope="module")
def decididos(setembro: pl.DataFrame, treinados: dict[str, Calibrado]) -> pl.DataFrame:
    modelo = treinados["lightgbm"]

    return decidir(setembro.with_columns(pl.Series("probabilidade", modelo.probabilidade(setembro))))


@pytest.mark.parametrize("nome", list(MODELOS))
def test_a_base_mais_as_contribuicoes_e_a_pontuacao(
    nome: str, setembro: pl.DataFrame, treinados: dict[str, Calibrado]
) -> None:
    modelo = treinados[nome]
    pesos = contribuicoes(modelo, setembro)

    assert pesos.height == setembro.height
    assert np.allclose(pesos.sum_horizontal().to_numpy(), modelo.pontuar(setembro), rtol=0, atol=1e-9)


def test_cada_modelo_explica_pelas_suas_variaveis_e_nunca_pela_idade(
    setembro: pl.DataFrame, treinados: dict[str, Calibrado]
) -> None:
    nomes = [variavel.nome for variavel in ENTRADA]

    assert contribuicoes(treinados["lightgbm"], setembro.head(5)).columns == [*nomes, "base"]
    assert contribuicoes(treinados["logistica"], setembro.head(5)).columns == [*nomes, "base"]
    assert contribuicoes(treinados["regras"], setembro.head(5)).columns == [*(r.nome for r in REGRAS), "base"]


def test_as_razoes_sao_as_que_mais_empurraram_para_o_golpe(
    decididos: pl.DataFrame, treinados: dict[str, Calibrado]
) -> None:
    modelo = treinados["lightgbm"]
    segurados = decididos.filter(pl.col("acao") != Acao.APROVAR.value)
    pesos = contribuicoes(modelo, segurados).drop("base")

    todas = razoes(modelo, segurados)

    assert segurados.height > 0
    # Nas árvores, quase todo Pix segurado tem três variáveis empurrando para o golpe.
    assert sum(len(as_razoes) == 3 for as_razoes in todas) / len(todas) > 0.9

    for as_razoes, linha in zip(todas, pesos.iter_rows(named=True), strict=True):
        assert 1 <= len(as_razoes) <= 3
        assert all(razao.peso > 0 for razao in as_razoes)
        assert [razao.peso for razao in as_razoes] == sorted((razao.peso for razao in as_razoes), reverse=True)
        assert len({razao.frase for razao in as_razoes}) == len(as_razoes)
        # A primeira razão é a variável que mais empurrou.
        assert as_razoes[0].nome == max(linha, key=lambda nome: linha[nome])


def test_na_invasao_de_conta_a_razao_quase_sempre_e_o_aparelho(
    decididos: pl.DataFrame, treinados: dict[str, Calibrado]
) -> None:
    invasoes = simulacao_media().verdade.fraudes.filter(pl.col("golpe") == "invasao_de_conta")
    pegas = decididos.join(invasoes.select(pl.col("transacao").alias("id")), on="id", how="semi").filter(
        pl.col("acao") != Acao.APROVAR.value
    )
    do_aparelho = [
        any(razao.nome in {"minutos_desde_o_aparelho_novo", "outros_clientes_no_aparelho"} for razao in as_razoes)
        for as_razoes in razoes(treinados["lightgbm"], pegas)
    ]

    # O segundo Pix da invasão para a mesma laranja tem outra razão mais forte: a laranja que o cliente pagou há
    # minutos. Com as razões tiradas ao contrário, o aparelho quase nunca apareceria.
    assert pegas.height >= 10
    assert sum(do_aparelho) / len(do_aparelho) >= 0.8


def test_a_explicacao_de_cada_pix_decidido(decididos: pl.DataFrame, treinados: dict[str, Calibrado]) -> None:
    explicados = explicar(treinados["lightgbm"], decididos)
    aprovados = explicados.filter(pl.col("acao") == Acao.APROVAR.value)
    bloqueados = explicados.filter(pl.col("acao") == Acao.BLOQUEAR.value)

    assert explicados.columns == [*decididos.columns, "explicacao"]
    assert explicados.drop("explicacao").equals(decididos)
    assert aprovados["explicacao"].str.starts_with("Aprovado, com ").all()
    assert not aprovados["explicacao"].str.contains(":").any()
    assert bloqueados.height > 0
    assert bloqueados["explicacao"].str.contains(r"^Bloqueado, com \d+(,\d)?% de chance de golpe: .+\.$").all()


def test_as_regras_se_explicam_pelas_regras_que_valeram(setembro: pl.DataFrame) -> None:
    regras = Calibrado("regras", Regras(), inclinacao=1.0, intercepto=-5.0)
    # Um Pix de R$ 500, de um aparelho que entrou há 10 minutos, para um recebedor já contestado.
    pix = setembro.head(1).with_columns(
        pl.lit(500.0).alias("valor"),
        pl.lit(10.0).alias("minutos_desde_o_aparelho_novo"),
        pl.lit(2, dtype=pl.Int32).alias("contestacoes_do_recebedor"),
        pl.lit(0.1).alias("parte_do_saldo"),
        pl.lit(0, dtype=pl.Int32).alias("pix_na_ultima_hora"),
        pl.lit(None, dtype=pl.Float64).alias("parte_do_limite_noturno"),
        pl.lit(value=False).alias("recebedor_novo"),
    )
    (as_razoes,) = razoes(regras, pix)

    assert [razao.nome for razao in as_razoes] == ["aparelho_novo", "recebedor_contestado"]
    assert texto(Acao.BLOQUEAR, 0.73, as_razoes) == (
        "Bloqueado, com 73% de chance de golpe: o Pix de mais de R$ 200 sai de um aparelho que entrou na conta há "
        "menos de um dia; e o recebedor já teve um Pix contestado por um cliente do banco."
    )


def test_o_texto_com_uma_duas_ou_tres_razoes() -> None:
    a, b, c = (Razao(nome, 1.0, frase) for nome, frase in (("a", "uma coisa"), ("b", "outra"), ("c", "mais uma")))

    assert texto(Acao.APROVAR, 0.0002, ()) == "Aprovado, com menos de 0,1% de chance de golpe."
    assert (
        texto(Acao.SEGURAR, 0.042, (a,))
        == "Segurado para confirmar com o cliente, com 4,2% de chance de golpe: uma coisa."
    )
    assert texto(Acao.SEGURAR, 0.042, (a, b)) == (
        "Segurado para confirmar com o cliente, com 4,2% de chance de golpe: uma coisa; e outra."
    )
    assert (
        texto(Acao.BLOQUEAR, 0.97, (a, b, c)) == "Bloqueado, com 97% de chance de golpe: uma coisa; outra; e mais uma."
    )


@pytest.mark.parametrize(
    ("probabilidade", "esperado"),
    [(0.0, "menos de 0,1%"), (0.001, "0,1%"), (0.0423, "4,2%"), (0.0999, "10,0%"), (0.1, "10%"), (0.973, "97%")],
)
def test_a_chance_para_quem_le(probabilidade: float, esperado: str) -> None:
    assert chance(probabilidade) == esperado


def test_a_decisao_nao_le_a_verdade() -> None:
    # O pacote da decisão roda no banco, Pix a Pix: não importa o simulador, a gravação, as análises nem a avaliação.
    pasta = Path(__file__).parents[2] / "src" / "antifraude_pix" / "decisao"
    proibidos = (
        "antifraude_pix.simulacao",
        "antifraude_pix.dados.gravacao",
        "antifraude_pix.analise",
        "antifraude_pix.avaliacao",
    )

    for arquivo in pasta.glob("*.py"):
        arvore = ast.parse(arquivo.read_text(encoding="utf-8"))
        importados = [no.module or "" for no in ast.walk(arvore) if isinstance(no, ast.ImportFrom)] + [
            nome.name for no in ast.walk(arvore) if isinstance(no, ast.Import) for nome in no.names
        ]
        nomes = {no.id for no in ast.walk(arvore) if isinstance(no, ast.Name)} | {
            no.attr for no in ast.walk(arvore) if isinstance(no, ast.Attribute)
        }

        assert not [modulo for modulo in importados if modulo.startswith(proibidos)], arquivo.name
        assert not [nome for nome in nomes if "verdade" in nome], arquivo.name


def test_a_escolha_das_razoes_num_exemplo_a_mao() -> None:
    # O recebedor novo e as vezes que o cliente pagou dizem a mesma coisa: viram uma razão só, e a terceira vaga
    # fica com a próxima. A hora puxou para o golpe, mas menos que as outras; o aparelho puxou para o legítimo.
    pesos = {
        "recebedor_novo": 2.0,
        "vezes_que_pagou_o_recebedor": 1.5,
        "valor": 1.0,
        "valor_redondo": 0.8,
        "hora": 0.1,
        "minutos_desde_o_aparelho_novo": -3.0,
    }
    valores = {
        "recebedor_novo": True,
        "vezes_que_pagou_o_recebedor": 0,
        "valor": 1_500.0,
        "valor_redondo": True,
        "hora": 14,
        "minutos_desde_o_aparelho_novo": None,
    }

    assert [(razao.nome, razao.frase) for razao in escolher(pesos, valores)] == [
        ("recebedor_novo", "o cliente nunca tinha pagado este recebedor"),
        ("valor", "o Pix é de R$ 1.500,00"),
        ("valor_redondo", "o valor é redondo, em dezenas de reais"),
    ]
    assert [razao.nome for razao in escolher(pesos, valores, quantas=5)] == [
        "recebedor_novo",
        "valor",
        "valor_redondo",
        "hora",
    ]


@pytest.mark.parametrize("nome", list(MODELOS))
def test_o_lote_em_que_todo_pix_foi_aprovado_sai_sem_razoes(
    nome: str, setembro: pl.DataFrame, treinados: dict[str, Calibrado]
) -> None:
    # O serviço decide um Pix por vez, e quase todo Pix é aprovado: não há nada a explicar.
    modelo = treinados[nome]
    aprovados = setembro.head(3).with_columns(
        pl.lit(0.0001).alias("probabilidade"), pl.lit("aprovar").cast(ACAO).alias("acao")
    )

    assert contribuicoes(modelo, setembro.head(0)).height == 0
    assert razoes(modelo, setembro.head(0)) == []
    assert (
        explicar(modelo, aprovados)["explicacao"].to_list() == ["Aprovado, com menos de 0,1% de chance de golpe."] * 3
    )
