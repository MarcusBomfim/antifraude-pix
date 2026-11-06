import polars as pl
import pytest

from antifraude_pix.dados.banco import VisaoDoBanco
from antifraude_pix.decisao.frases import FRASES, contagem, frase, porcentagem, reais, tempo, vezes
from antifraude_pix.modelos.preparo import ENTRADA
from antifraude_pix.variaveis.calculo import calcular
from tests.apoio import simulacao_media


def test_cada_variavel_que_o_modelo_recebe_tem_a_sua_frase_e_a_idade_nao_tem() -> None:
    assert list(FRASES) == [variavel.nome for variavel in ENTRADA]
    assert "idade" not in FRASES

    with pytest.raises(ValueError, match="não entra no modelo"):
        frase("idade", 72)


@pytest.mark.parametrize(
    ("valor", "esperado"),
    [(1_250.0, "R$ 1.250,00"), (0.5, "R$ 0,50"), (1_234_567.891, "R$ 1.234.567,89"), (59.9, "R$ 59,90")],
)
def test_o_dinheiro_como_se_escreve(valor: float, esperado: str) -> None:
    assert reais(valor) == esperado


@pytest.mark.parametrize(
    ("fracao", "esperado"), [(0.64, "64%"), (1.0, "100%"), (0.004, "0,4%"), (0.0, "0%"), (2.5, "250%")]
)
def test_a_porcentagem(fracao: float, esperado: str) -> None:
    assert porcentagem(fracao) == esperado


@pytest.mark.parametrize(
    ("quantas", "esperado"),
    [(1.0, "1 vez"), (1.04, "1 vez"), (2.0, "2 vezes"), (2.5, "2,5 vezes"), (9.96, "10 vezes"), (12.4, "12 vezes")],
)
def test_as_vezes(quantas: float, esperado: str) -> None:
    assert vezes(quantas) == esperado


@pytest.mark.parametrize(
    ("minutos", "esperado"),
    [
        (0.5, "menos de um minuto"),
        (1.0, "1 minuto"),
        (59.9, "59 minutos"),
        (60.0, "1 hora"),
        (47 * 60 + 59, "47 horas"),
        (48 * 60, "2 dias"),
        (90 * 24 * 60, "90 dias"),
    ],
)
def test_o_tempo_na_unidade_que_se_fala(minutos: float, esperado: str) -> None:
    assert tempo(minutos) == esperado


def test_o_singular_e_o_plural() -> None:
    assert contagem(1, "outro cliente", "outros clientes") == "1 outro cliente"
    assert contagem(3, "outro cliente", "outros clientes") == "3 outros clientes"


@pytest.mark.parametrize(
    ("variavel", "valor", "esperado"),
    [
        ("valor", 1_250.0, "o Pix é de R$ 1.250,00"),
        ("hora", 3, "o Pix foi feito entre 3h e 4h"),
        ("hora", 23, "o Pix foi feito entre 23h e 0h"),
        ("dia_da_semana", 6, "o Pix foi feito num domingo"),
        ("dia_da_semana", 0, "o Pix foi feito numa segunda-feira"),
        ("iniciacao", "manual", "o Pix foi feito digitando a agência e a conta"),
        ("chave", "aleatoria", "o recebedor recebe por uma chave aleatória, que não diz de quem é"),
        ("chave", None, "o Pix não usou chave"),
        ("recebedor_pj", True, "o recebedor é uma empresa"),
        ("valor_redondo", False, "o valor não é redondo"),
        ("parte_do_saldo", 0.64, "o Pix leva 64% do saldo"),
        ("parte_do_limite_noturno", 0.9, "o Pix leva 90% do limite noturno"),
        ("parte_do_limite_noturno", None, "o Pix é de dia, fora do limite noturno"),
        ("dias_de_conta", 1, "o cliente tem conta há 1 dia"),
        ("pix_na_ultima_hora", 0, "o cliente não fez nenhum Pix na hora anterior"),
        ("pix_na_ultima_hora", 3, "o cliente fez 3 Pix na hora anterior"),
        ("valor_nas_ultimas_24h", 0.0, "o cliente não mandou nada por Pix nas 24 horas anteriores"),
        ("minutos_desde_o_ultimo_pix", None, "é o primeiro Pix do cliente"),
        ("minutos_desde_o_ultimo_pix", 4.2, "o Pix anterior do cliente foi há 4 minutos"),
        ("mediana_de_90_dias", None, "o cliente não fez Pix nos 90 dias anteriores"),
        ("valor_sobre_a_mediana", 8.0, "o valor é 8 vezes o Pix típico do cliente"),
        ("valor_sobre_a_mediana", 0.5, "o valor é 50% do Pix típico do cliente"),
        ("valor_sobre_o_maior_de_90_dias", 3.0, "o valor passa do maior Pix do cliente em 90 dias: é 3 vezes ele"),
        ("valor_sobre_o_maior_de_90_dias", 1.0, "o valor é 100% do maior Pix do cliente em 90 dias"),
        ("pix_noturnos_em_90_dias", 1, "o cliente fez 1 Pix noturno nos 90 dias anteriores"),
        ("vezes_que_pagou_o_recebedor", 0, "o cliente nunca tinha pagado este recebedor"),
        ("vezes_que_pagou_o_recebedor", 1, "o cliente já tinha pagado este recebedor 1 vez"),
        ("recebedor_novo", True, "o cliente nunca tinha pagado este recebedor"),
        ("dias_desde_o_primeiro_pix_ao_recebedor", None, "é o primeiro Pix do cliente para este recebedor"),
        (
            "dias_desde_o_primeiro_pix_ao_recebedor",
            30.5,
            "o primeiro Pix do cliente para este recebedor foi há 30 dias",
        ),
        ("pix_recebidos_em_24h", 9, "o recebedor recebeu 9 Pix de clientes do banco nas 24 horas anteriores"),
        ("pagadores_novos_em_24h", 1, "1 cliente do banco pagou o recebedor pela primeira vez nas 24 horas anteriores"),
        (
            "pagadores_novos_em_7_dias",
            4,
            "4 clientes do banco pagaram o recebedor pela primeira vez nos 7 dias anteriores",
        ),
        ("contestacoes_do_recebedor", 2, "2 Pix para este recebedor já tinham sido contestados como golpe"),
        ("minutos_desde_o_aparelho_novo", 12.0, "o aparelho entrou na conta há 12 minutos"),
        ("minutos_desde_o_aparelho_novo", None, "o Pix saiu do aparelho de sempre"),
        ("outros_clientes_no_aparelho", 3, "o aparelho já tinha entrado na conta de 3 outros clientes"),
    ],
)
def test_a_frase_de_cada_variavel_com_o_valor(variavel: str, valor: object, esperado: str) -> None:
    assert frase(variavel, valor) == esperado


def test_toda_variavel_tem_frase_para_todo_valor_que_aparece_nos_dados() -> None:
    # Cada valor de cada variável que aparece numa simulação, com o vazio, o zero e o extremo.
    variaveis = calcular(VisaoDoBanco.da(simulacao_media()))

    for variavel in ENTRADA:
        for valor in variaveis[variavel.nome].unique().to_list():
            texto = frase(variavel.nome, valor)

            assert texto
            assert texto == texto.strip()
            assert not any(estranho in texto for estranho in ("None", "nan", "inf", "{", "_"))
            assert texto[0] == texto[0].lower()


def test_a_frase_le_o_valor_como_o_polars_entrega() -> None:
    linha = pl.DataFrame({"chave": ["cpf"]}, schema={"chave": pl.Enum(["cpf", "cnpj"])}).row(0, named=True)

    assert frase("chave", linha["chave"]) == "o recebedor recebe pelo CPF"
