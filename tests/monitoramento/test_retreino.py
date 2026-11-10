import polars as pl
import pytest

from antifraude_pix.monitoramento.retreino import ACASO, IDOSO, MARGEM, comparar

HONESTOS, GOLPES = 200, 20


def _pix(probabilidade_do_golpe: list[float], probabilidade_do_honesto: float = 0.001) -> pl.DataFrame:
    """Vinte golpes contestados e duzentos Pix honestos de R$ 100, com a probabilidade de cada um."""
    return pl.DataFrame(
        {
            "id": range(GOLPES + HONESTOS),
            "valor": [100.0] * (GOLPES + HONESTOS),
            "probabilidade": probabilidade_do_golpe + [probabilidade_do_honesto] * HONESTOS,
            "golpe": ["contestado"] * GOLPES + [None] * HONESTOS,
        }
    )


def _idades(idade: int = 40) -> pl.DataFrame:
    return pl.DataFrame({"id": range(GOLPES + HONESTOS), "idade": [idade] * (GOLPES + HONESTOS)})


# O campeão deixa passar metade dos golpes, com a probabilidade de um Pix honesto; o desafiante pega todos.
CAMPEAO = _pix([0.9] * 10 + [0.001] * 10)
DESAFIANTE = _pix([0.9] * 20)


def test_o_desafiante_que_pega_mais_e_custa_menos_ganha() -> None:
    veredito = comparar(CAMPEAO, DESAFIANTE, _idades())

    assert veredito.promover
    assert veredito.razoes == ()
    assert veredito.desafiante.precisao_media > veredito.campeao.precisao_media + MARGEM
    assert veredito.desafiante.custo < veredito.campeao.custo
    assert (veredito.campeao.golpes_pegos, veredito.desafiante.golpes_pegos, veredito.desafiante.golpes) == (10, 20, 20)


def test_o_desafiante_igual_ao_campeao_nao_ganha_por_ser_novo() -> None:
    # Os dez golpes empatados com os duzentos honestos: 0,5 de recall com precisão 1, e 0,5 com 20 / 220.
    veredito = comparar(CAMPEAO, CAMPEAO, _idades())

    assert not veredito.promover
    assert veredito.razoes == ("a precisão média dele, 0,545, não passa a do campeão, 0,545, por 0,01",)


def test_o_desafiante_que_ordena_melhor_mas_segura_os_honestos_custa_mais() -> None:
    # Ele põe todo golpe na frente, mas dá 30% a todo Pix honesto: segura cada um, a R$ 10 a confirmação.
    # O campeão custa os dez golpes que deixa passar, de R$ 100 cada; o desafiante, duzentas confirmações.
    caro = _pix([0.9] * 20, probabilidade_do_honesto=0.3)
    veredito = comparar(CAMPEAO, caro, _idades())

    assert not veredito.promover
    assert veredito.desafiante.precisao_media == 1.0
    assert veredito.razoes == ("as decisões dele custam mais: R$ 2.000 contra R$ 1.000",)


def _bloqueia_honestos(quantos: int) -> pl.DataFrame:
    """O desafiante que pega todos os golpes e bloqueia os primeiros honestos, com uma chance que manda bloquear."""
    return DESAFIANTE.with_columns(
        pl.when(pl.col("id").is_between(GOLPES, GOLPES + quantos - 1))
        .then(pl.lit(0.97))
        .otherwise(pl.col("probabilidade"))
        .alias("probabilidade")
    )


@pytest.mark.parametrize(
    ("quantos", "razoes"),
    [
        (4, ()),
        (
            5,
            (
                "ele bloqueia 5 Pix honestos de clientes de 65 anos ou mais que o campeão não bloqueia, contra 0 do "
                "contrário: mais do que o acaso explica",
            ),
        ),
    ],
)
def test_o_bloqueio_dos_mais_velhos_so_pesa_quando_o_acaso_nao_explica(quantos: int, razoes: tuple[str, ...]) -> None:
    # Só o desafiante bloqueia: b honestos contra 0. O acaso explica até b = 2 * raiz(b), isto é, até 4. Mesmo
    # bloqueando, ele ainda custa menos e ordena melhor que o campeão: a única regra em jogo é a dos mais velhos.
    veredito = comparar(CAMPEAO, _bloqueia_honestos(quantos), _idades(IDOSO))

    assert ACASO == 2.0
    assert (veredito.so_o_desafiante_bloqueia, veredito.so_o_campeao_bloqueia) == (quantos, 0)
    assert veredito.desafiante.idosos_honestos_bloqueados == quantos
    assert veredito.razoes == razoes


def test_os_mais_novos_bloqueados_nao_entram_na_regra_dos_mais_velhos() -> None:
    veredito = comparar(CAMPEAO, _bloqueia_honestos(5), _idades(IDOSO - 1))

    assert veredito.so_o_desafiante_bloqueia == 0
    assert not any("65 anos ou mais" in razao for razao in veredito.razoes)


def test_os_dois_tem_de_ser_medidos_nos_mesmos_pix() -> None:
    with pytest.raises(ValueError, match="nos mesmos Pix"):
        comparar(CAMPEAO, DESAFIANTE.head(100), _idades())


def test_sem_a_idade_de_algum_cliente_a_regra_dos_mais_velhos_nao_tem_como_valer() -> None:
    with pytest.raises(ValueError, match="Falta a idade"):
        comparar(CAMPEAO, DESAFIANTE, _idades().head(10))
