import polars as pl

from antifraude_pix.simulacao.golpes import Golpe
from antifraude_pix.simulacao.simulador import Simulacao


def _por_ataque(simulacao: Simulacao) -> pl.DataFrame:
    """Cada ataque, com o golpe, quando terminou e quando a vítima reclamou — se reclamou."""
    return (
        simulacao.verdade.fraudes.join(
            simulacao.transacoes.select(pl.col("id").alias("transacao"), "momento"), on="transacao"
        )
        .join(simulacao.contestacoes, on="transacao", how="left")
        .group_by("ataque")
        .agg(
            pl.col("golpe").first(),
            pl.col("momento").max().alias("ultimo"),
            pl.col("contestada_em").min().alias("primeira"),
            pl.col("contestada_em").max().alias("ultima"),
            pl.col("contestada_em").is_null().any().alias("alguma_sem"),
            pl.col("contestada_em").is_not_null().any().alias("alguma_com"),
        )
    )


def test_a_vitima_contesta_o_ataque_inteiro_de_uma_vez(simulacao: Simulacao) -> None:
    ataques = _por_ataque(simulacao)

    assert ataques.filter(pl.col("alguma_sem") & pl.col("alguma_com")).height == 0
    assert ataques.filter(pl.col("primeira") != pl.col("ultima")).height == 0


def test_cada_golpe_e_contestado_na_proporcao_dele_e_parte_nunca_e(simulacao: Simulacao) -> None:
    ataques = _por_ataque(simulacao)

    # As chances da ADR-0004, escritas aqui, e não lidas do simulador: se o código mudar, o teste avisa.
    chances = {
        Golpe.INVASAO_DE_CONTA: 0.95,
        Golpe.FALSA_CENTRAL: 0.90,
        Golpe.FALSO_PARENTE: 0.85,
        Golpe.VENDA_FALSA: 0.60,
    }

    for golpe, chance in chances.items():
        dele = ataques.filter(pl.col("golpe") == golpe.value)
        contestados = int(dele["alguma_com"].sum()) / dele.height

        assert abs(contestados - chance) < 0.15, golpe

    # Sem contestação, o banco nunca fica sabendo: é o rótulo que falta.
    assert ataques.filter(~pl.col("alguma_com")).height > 0


def test_a_invasao_e_contestada_logo_e_a_venda_falsa_depois_de_dias(simulacao: Simulacao) -> None:
    demoras = (
        _por_ataque(simulacao)
        .filter(pl.col("alguma_com"))
        .group_by("golpe")
        .agg(((pl.col("primeira") - pl.col("ultimo")).dt.total_minutes() / 60).median().alias("horas"))
    )

    def horas(golpe: Golpe) -> float:
        return float(demoras.filter(pl.col("golpe") == golpe.value)["horas"].item())

    assert horas(Golpe.INVASAO_DE_CONTA) < 24
    assert horas(Golpe.VENDA_FALSA) > 3 * 24
    assert horas(Golpe.INVASAO_DE_CONTA) < horas(Golpe.FALSO_PARENTE) < horas(Golpe.VENDA_FALSA)


def test_ha_pix_legitimo_contestado_mas_raro(simulacao: Simulacao) -> None:
    legitimos_contestados = simulacao.contestacoes.join(simulacao.verdade.fraudes, on="transacao", how="anti").height

    # Cinquenta a cada milhão, como diz a ADR-0004; o triplo, para a sorte.
    assert legitimos_contestados <= 150 * simulacao.transacoes.height / 1_000_000
