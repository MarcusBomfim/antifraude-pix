import polars as pl

from antifraude_pix.dados.banco import VisaoDoBanco
from antifraude_pix.simulacao.simulador import Simulacao
from antifraude_pix.variaveis.calculo import calcular
from antifraude_pix.variaveis.catalogo import CATALOGO, NOMES, Grupo, Tipo


def test_cada_variavel_tem_um_nome_so_e_uma_frase_em_portugues() -> None:
    assert len(set(NOMES)) == len(NOMES)
    assert all(variavel.descricao[0].isupper() and variavel.descricao.endswith(".") for variavel in CATALOGO)
    assert {variavel.grupo for variavel in CATALOGO} == set(Grupo)


def test_o_tipo_do_catalogo_e_o_tipo_da_coluna(simulacao: Simulacao) -> None:
    calculadas = calcular(VisaoDoBanco.da(simulacao).ate(simulacao.config.inicio.replace(day=10)))

    for variavel in CATALOGO:
        tipo = calculadas.schema[variavel.nome]

        match variavel.tipo:
            case Tipo.NUMERO:
                assert tipo.is_numeric(), variavel.nome
            case Tipo.SIM_OU_NAO:
                assert tipo == pl.Boolean, variavel.nome
            case Tipo.CATEGORIA:
                assert isinstance(tipo, pl.Enum), variavel.nome
