import polars as pl
import pytest

from antifraude_pix.avaliacao.justica import FAIXA, faixa_de_idade, por_idade


def test_as_faixas_de_idade_e_as_bordas() -> None:
    idades = pl.DataFrame({"idade": [18, 34, 35, 54, 55, 64, 65, 98]})
    faixas = idades.select(faixa_de_idade(pl.col("idade")).alias("faixa"))["faixa"].cast(pl.String).to_list()

    assert faixas == ["18 a 34", "18 a 34", "35 a 54", "35 a 54", "55 a 64", "55 a 64", "65 ou mais", "65 ou mais"]
    assert FAIXA.categories.to_list() == ["18 a 34", "35 a 54", "55 a 64", "65 ou mais"]


def _pix(
    idade: int, acao: str, golpe: str | None = None, valor: float = 100.0, probabilidade: float = 0.0
) -> dict[str, object]:
    return {
        "modelo": "lightgbm",
        "acao": acao,
        "probabilidade": probabilidade,
        "valor": valor,
        "golpe": golpe,
        "idade": idade,
    }


def test_os_dois_erros_em_cada_faixa() -> None:
    # Os mais novos: dois golpes, um pego; vinte mil Pix honestos, dois segurados e um bloqueado.
    # Os mais velhos: um golpe de R$ 900, pego, e um de R$ 100, que passou; dez mil Pix honestos, quatro segurados.
    pix = (
        [_pix(20, "segurar", "venda_falsa", probabilidade=0.6), _pix(30, "aprovar", "venda_falsa", probabilidade=0.2)]
        + [_pix(25, "segurar")] * 2
        + [_pix(25, "bloquear")]
        + [_pix(25, "aprovar")] * 19_997
        + [_pix(70, "bloquear", "falso_parente", valor=900.0, probabilidade=0.9)]
        + [_pix(80, "aprovar", "falsa_central", valor=100.0)]
        + [_pix(70, "segurar")] * 4
        + [_pix(70, "aprovar")] * 9_996
    )
    linhas = por_idade(pl.DataFrame(pix, schema_overrides={"golpe": pl.String}))
    jovens, velhos = (linhas.row(indice, named=True) for indice in range(2))

    assert linhas["faixa"].cast(pl.String).to_list() == ["18 a 34", "65 ou mais"]
    assert jovens == pytest.approx(
        jovens
        | {
            "golpes": 2,
            "golpes_pegos": 1,
            "recall": 0.5,
            "recall_em_reais": 0.5,
            "honestos": 20_000,
            "honestos_segurados": 1.0,
            "honestos_bloqueados": 0.5,
            "golpes_previstos": 0.8,
        }
    )
    assert velhos == pytest.approx(
        velhos
        | {
            "golpes": 2,
            "golpes_pegos": 1,
            "recall": 0.5,
            "recall_em_reais": 0.9,
            "honestos": 10_000,
            "honestos_segurados": 4.0,
            "honestos_bloqueados": 0.0,
            "golpes_previstos": 0.9,
        }
    )


def test_os_modelos_ficam_na_ordem_em_que_vem_e_as_faixas_da_mais_nova_para_a_mais_velha() -> None:
    pix = [
        _pix(70, "aprovar", "venda_falsa") | {"modelo": "regras"},
        _pix(20, "aprovar", "venda_falsa") | {"modelo": "regras"},
        _pix(70, "aprovar", "venda_falsa"),
        _pix(20, "aprovar", "venda_falsa"),
    ]
    linhas = por_idade(pl.DataFrame(pix, schema_overrides={"golpe": pl.String}))

    assert linhas.select("modelo", pl.col("faixa").cast(pl.String)).rows() == [
        ("regras", "18 a 34"),
        ("regras", "65 ou mais"),
        ("lightgbm", "18 a 34"),
        ("lightgbm", "65 ou mais"),
    ]


def test_o_cliente_e_adulto() -> None:
    with pytest.raises(ValueError, match="adultos"):
        por_idade(pl.DataFrame([_pix(17, "aprovar", "venda_falsa")], schema_overrides={"golpe": pl.String}))
