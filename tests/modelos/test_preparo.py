from datetime import date

import numpy as np
import polars as pl
import pytest

from antifraude_pix.dados.banco import VisaoDoBanco
from antifraude_pix.modelos.preparo import ENTRADA, PROTEGIDAS, PreparoDaLogistica, categoricas, matriz_das_arvores
from antifraude_pix.modelos.rotulo import janela_de_treino
from antifraude_pix.variaveis.calculo import calcular
from antifraude_pix.variaveis.catalogo import CATALOGO, NOMES, Tipo
from tests.apoio import simulacao_media


@pytest.fixture(scope="module")
def janela() -> pl.DataFrame:
    visao = VisaoDoBanco.da(simulacao_media())
    return janela_de_treino(visao, calcular(visao), date(2025, 9, 1))


@pytest.fixture(scope="module")
def preparo(janela: pl.DataFrame) -> PreparoDaLogistica:
    return PreparoDaLogistica.ajustado(janela)


def test_a_logistica_recebe_cada_numero_com_o_vazio_ao_lado_e_a_hora_em_casas(preparo: PreparoDaLogistica) -> None:
    colunas = preparo.colunas
    numeros = [v.nome for v in ENTRADA if v.tipo is Tipo.NUMERO and v.nome not in {"hora", "dia_da_semana"}]
    sim_ou_nao = [v.nome for v in ENTRADA if v.tipo is Tipo.SIM_OU_NAO]

    assert len(colunas) == 2 * len(numeros) + len(sim_ou_nao) + 24 + 7 + (3 + 1) + (5 + 1)
    assert {"valor", "valor vazia", "hora=0", "hora=23", "dia_da_semana=6", "chave=cnpj", "chave vazia"} <= set(colunas)
    assert "hora" not in colunas
    assert len(set(colunas)) == len(colunas)


def test_o_treino_sai_padronizado_pelas_contas_dele_mesmo(janela: pl.DataFrame, preparo: PreparoDaLogistica) -> None:
    matriz = preparo.matriz(janela)
    variam = preparo.desvio != 1.0

    assert matriz.shape == (janela.height, len(preparo.colunas))
    assert np.allclose(matriz.mean(axis=0), 0.0, atol=1e-9)
    assert np.allclose(matriz.std(axis=0)[variam], 1.0)


def test_outro_pix_e_padronizado_com_a_media_e_o_desvio_do_treino(
    janela: pl.DataFrame, preparo: PreparoDaLogistica
) -> None:
    valor = preparo.colunas.index("valor")
    em_log = np.log1p(janela["valor"].to_numpy())

    assert preparo.media[valor] == pytest.approx(em_log.mean())
    assert preparo.desvio[valor] == pytest.approx(em_log.std())

    # Um Pix de R$ 10.000, fora de tudo o que o treino viu, não é empurrado para o meio.
    grande = janela.head(1).with_columns(pl.lit(10_000.0).alias("valor"))

    assert preparo.matriz(grande)[0, valor] == pytest.approx((np.log1p(10_000) - em_log.mean()) / em_log.std())


def test_o_vazio_vira_zero_com_o_indicador_aceso(janela: pl.DataFrame, preparo: PreparoDaLogistica) -> None:
    # O aparelho de sempre não tem data de entrada. Sem o indicador, ele viraria o aparelho que acabou de entrar.
    de_sempre = janela.filter(pl.col("minutos_desde_o_aparelho_novo").is_null()).head(1)
    minutos = preparo.colunas.index("minutos_desde_o_aparelho_novo")
    vazia = preparo.colunas.index("minutos_desde_o_aparelho_novo vazia")
    linha = preparo.matriz(de_sempre)[0]

    assert de_sempre.height == 1
    assert linha[minutos] == pytest.approx((0.0 - preparo.media[minutos]) / preparo.desvio[minutos])
    assert linha[vazia] == pytest.approx((1.0 - preparo.media[vazia]) / preparo.desvio[vazia])


def test_a_hora_acende_so_a_casa_dela(janela: pl.DataFrame, preparo: PreparoDaLogistica) -> None:
    das_tres = janela.filter(pl.col("hora") == 3).head(1)
    casas = [preparo.colunas.index(f"hora={hora}") for hora in range(24)]
    brutas = preparo.matriz(das_tres)[0, casas] * preparo.desvio[casas] + preparo.media[casas]

    assert np.allclose(brutas, [1.0 if hora == 3 else 0.0 for hora in range(24)])


def test_a_categoria_tem_de_ser_fechada(janela: pl.DataFrame) -> None:
    with pytest.raises(TypeError, match="categoria fechada"):
        PreparoDaLogistica.ajustado(janela.with_columns(pl.col("chave").cast(pl.String)))


def test_a_entrada_e_o_catalogo_sem_a_idade(preparo: PreparoDaLogistica) -> None:
    nomes = [variavel.nome for variavel in ENTRADA]

    assert PROTEGIDAS == ("idade",)
    assert nomes == [nome for nome in NOMES if nome != "idade"]
    assert not any(coluna.startswith("idade") for coluna in preparo.colunas)


def test_as_arvores_recebem_o_numero_como_esta_e_o_vazio_como_vazio(janela: pl.DataFrame) -> None:
    matriz = matriz_das_arvores(janela)
    nomes = [variavel.nome for variavel in ENTRADA]
    valor, mediana, chave = nomes.index("valor"), nomes.index("mediana_de_90_dias"), nomes.index("chave")

    assert matriz.shape == (janela.height, len(ENTRADA))
    assert np.array_equal(matriz[:, valor], janela["valor"].to_numpy())
    assert np.isnan(matriz[:, mediana]).sum() == janela["mediana_de_90_dias"].null_count()
    assert np.array_equal(
        matriz[:, chave], janela["chave"].to_physical().cast(pl.Float64).fill_null(np.nan).to_numpy(), equal_nan=True
    )
    assert (nomes.index("iniciacao"), nomes.index("chave")) == categoricas()
    # A idade vem depois delas no catálogo: com ela ou sem ela, as categorias ficam no mesmo lugar.
    assert categoricas(CATALOGO) == (NOMES.index("iniciacao"), NOMES.index("chave"))
    assert matriz_das_arvores(janela, CATALOGO).shape == (janela.height, len(CATALOGO))
