from datetime import date, timedelta

import lightgbm as lgb
import numpy as np
import polars as pl
import pytest
from sklearn.metrics import average_precision_score

from antifraude_pix.dados.banco import VisaoDoBanco
from antifraude_pix.modelos import regras
from antifraude_pix.modelos.preparo import ENTRADA, categoricas, matriz_das_arvores
from antifraude_pix.modelos.rotulo import Dobra, janela_de_teste, janela_de_treino
from antifraude_pix.modelos.treino import (
    LIGHTGBM,
    MODELOS,
    Calibrado,
    Regras,
    calibrar,
    separar,
    treinar,
    treinar_arvores,
)
from antifraude_pix.variaveis.calculo import calcular
from antifraude_pix.variaveis.catalogo import CATALOGO
from tests.apoio import simulacao_media

CORTE = date(2025, 9, 1)


@pytest.fixture(scope="module")
def visao() -> VisaoDoBanco:
    return VisaoDoBanco.da(simulacao_media())


@pytest.fixture(scope="module")
def variaveis(visao: VisaoDoBanco) -> pl.DataFrame:
    return calcular(visao)


@pytest.fixture(scope="module")
def janela(visao: VisaoDoBanco, variaveis: pl.DataFrame) -> pl.DataFrame:
    return janela_de_treino(visao, variaveis, CORTE)


@pytest.fixture(scope="module")
def setembro(visao: VisaoDoBanco, variaveis: pl.DataFrame) -> pl.DataFrame:
    return janela_de_teste(visao, variaveis, Dobra(CORTE, date(2025, 9, 29)))


@pytest.fixture(scope="module")
def treinados(janela: pl.DataFrame) -> dict[str, Calibrado]:
    return {nome: treinar(nome, janela, CORTE) for nome in MODELOS}


def test_a_calibracao_e_um_sorteio_entre_os_pix_maduros_que_o_treino_nao_ve(janela: pl.DataFrame) -> None:
    ajuste, calibracao = separar(janela, CORTE, np.random.default_rng(1))
    maduros = janela.filter(pl.col("dia") < CORTE - timedelta(days=28))

    assert ajuste.height + calibracao.height == janela.height
    assert ajuste.join(calibracao, on="id", how="semi").is_empty()
    assert calibracao.filter(pl.col("dia") >= CORTE - timedelta(days=28)).is_empty()
    assert 0.1 < calibracao.height / maduros.height < 0.3


def test_o_mesmo_treino_com_a_mesma_semente_da_a_mesma_probabilidade(
    janela: pl.DataFrame, setembro: pl.DataFrame, treinados: dict[str, Calibrado]
) -> None:
    for nome in MODELOS:
        assert np.array_equal(
            treinar(nome, janela, CORTE).probabilidade(setembro), treinados[nome].probabilidade(setembro)
        )


def test_a_probabilidade_cresce_com_a_pontuacao_e_fica_de_zero_a_um(
    setembro: pl.DataFrame, treinados: dict[str, Calibrado]
) -> None:
    for modelo in treinados.values():
        probabilidade = modelo.probabilidade(setembro)

        assert modelo.inclinacao > 0
        # O golpe de cara mais clara chega a 1 na conta de ponto flutuante: é o teto, e não um erro.
        assert ((probabilidade >= 0) & (probabilidade <= 1)).all()
        ordem = np.argsort(modelo.pontuar(setembro), kind="stable")
        assert (np.diff(probabilidade[ordem]) >= 0).all()


def test_na_calibracao_a_soma_das_probabilidades_e_o_numero_de_golpes(
    janela: pl.DataFrame, treinados: dict[str, Calibrado]
) -> None:
    # A propriedade da reta de Platt: no que ela ajustou, a média prevista é a média observada.
    _, calibracao = separar(janela, CORTE, np.random.default_rng(2026))

    for modelo in treinados.values():
        assert modelo.probabilidade(calibracao).sum() == pytest.approx(calibracao["rotulo"].sum(), rel=0.02)


def test_as_regras_pontuam_pelos_pontos(setembro: pl.DataFrame, treinados: dict[str, Calibrado]) -> None:
    assert isinstance(treinados["regras"].pontuador, Regras)
    assert np.array_equal(treinados["regras"].pontuar(setembro), regras.pontos(setembro))


def test_os_modelos_que_aprendem_ganham_das_regras_contra_a_verdade(
    setembro: pl.DataFrame, treinados: dict[str, Calibrado]
) -> None:
    golpes = simulacao_media().verdade.fraudes.select(
        pl.col("transacao").alias("id"), pl.lit(value=True).alias("golpe")
    )
    alvo = setembro.join(golpes, on="id", how="left")["golpe"].fill_null(value=False).to_numpy()
    precisao = {
        nome: average_precision_score(alvo, modelo.probabilidade(setembro)) for nome, modelo in treinados.items()
    }

    assert precisao["lightgbm"] > 0.8
    assert precisao["logistica"] > 0.8
    assert precisao["regras"] < min(precisao["logistica"], precisao["lightgbm"]) - 0.2


def test_a_calibracao_pede_golpes_e_legitimos(janela: pl.DataFrame) -> None:
    so_legitimos = janela.filter(~pl.col("rotulo")).head(500)

    with pytest.raises(ValueError, match="pede golpes e legítimos"):
        calibrar("regras", Regras(), so_legitimos)


def test_o_peso_devolve_aos_legitimos_a_proporcao_que_eles_tem(janela: pl.DataFrame) -> None:
    # As árvores veem um legítimo em cada dez. Sem o peso dez, um Pix comum sairia com dez vezes mais cara de golpe.
    ajuste, _ = separar(janela, CORTE, np.random.default_rng(1))
    rotulo = ajuste["rotulo"].to_numpy()
    sorteio = np.random.default_rng(3)
    amostra = rotulo | (sorteio.random(rotulo.size) < 0.1)
    sem_peso = lgb.LGBMClassifier(
        **LIGHTGBM,
        random_state=int(sorteio.integers(2**31)),
        n_jobs=4,
        deterministic=True,
        force_col_wise=True,
        verbose=-1,
    )
    sem_peso.fit(matriz_das_arvores(ajuste)[amostra], rotulo[amostra], categorical_feature=list(categoricas()))
    com_peso = treinar_arvores(ajuste, np.random.default_rng(3), ENTRADA).modelo

    def tipica(modelo: lgb.LGBMClassifier) -> float:
        chance = np.asarray(modelo.predict_proba(matriz_das_arvores(ajuste)), dtype=np.float64)
        return float(np.median(chance[:, 1]))

    assert tipica(com_peso) < tipica(sem_peso) / 3


def test_nenhum_modelo_olha_a_idade(setembro: pl.DataFrame, treinados: dict[str, Calibrado]) -> None:
    # Todos os clientes com 90 anos, ou com 18: a pontuação de cada Pix não muda.
    for modelo in treinados.values():
        pontuacao = modelo.pontuar(setembro)

        for idade in (18, 90):
            outra = setembro.with_columns(pl.lit(idade, dtype=pl.Int16).alias("idade"))
            assert np.array_equal(modelo.pontuar(outra), pontuacao)


@pytest.mark.parametrize("nome", ["logistica", "lightgbm"])
def test_para_medir_o_modelo_pode_receber_a_idade(nome: str, janela: pl.DataFrame, setembro: pl.DataFrame) -> None:
    com_a_idade = treinar(nome, janela, CORTE, entrada=CATALOGO)
    noventa = setembro.with_columns(pl.lit(90, dtype=pl.Int16).alias("idade"))

    assert not np.array_equal(com_a_idade.pontuar(noventa), com_a_idade.pontuar(setembro))


def test_a_probabilidade_tem_teto_e_piso_sem_estourar_a_conta() -> None:
    reta = Calibrado("regras", Regras(), inclinacao=1.0, intercepto=0.0)

    assert reta.probabilidade_da(np.array([-1e4, 0.0, 1e4])).tolist() == pytest.approx([0.0, 0.5, 1.0])
