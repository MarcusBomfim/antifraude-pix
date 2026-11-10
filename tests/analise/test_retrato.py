from dataclasses import replace
from datetime import date
from pathlib import Path

import polars as pl
import pytest

from antifraude_pix.analise import figuras, retrato
from antifraude_pix.modelos.rotulo import Dobra
from antifraude_pix.simulacao.golpes import GOLPES
from antifraude_pix.simulacao.simulador import Simulacao, simular
from tests.apoio import MEDIA, cartoes_inventados


def test_cada_grupo_tem_as_24_horas_e_elas_somam_tudo(simulacao: Simulacao) -> None:
    horas = retrato.por_hora(simulacao)

    assert horas["grupo"].unique(maintain_order=True).to_list() == list(retrato.GRUPOS)
    assert horas.group_by("grupo").agg(pl.len().alias("horas"), pl.col("parte").sum())["horas"].to_list() == [24] * 5
    assert horas.group_by("grupo").agg(pl.col("parte").sum())["parte"].to_list() == pytest.approx([1.0] * 5)


def test_a_invasao_e_da_madrugada_e_a_falsa_central_do_horario_comercial(simulacao: Simulacao) -> None:
    horas = retrato.por_hora(simulacao)

    def parte(grupo: str, de: int, ate: int) -> float:
        return float(horas.filter((pl.col("grupo") == grupo) & pl.col("hora").is_between(de, ate))["parte"].sum())

    assert parte("invasão de conta", 0, 4) > 5 * parte(retrato.LEGITIMO, 0, 4)
    # A ligação começa quase sempre no horário comercial, e os Pix dela seguem por até hora e meia.
    assert parte("falsa central", 9, 18) > 0.9


def test_o_golpe_pede_mais_que_o_pix_comum(simulacao: Simulacao) -> None:
    faixas = {linha["grupo"]: linha for linha in retrato.faixas_de_valor(simulacao).iter_rows(named=True)}

    assert list(faixas) == list(retrato.GRUPOS)
    assert faixas["falso parente"]["mediana"] > 10 * faixas[retrato.LEGITIMO]["mediana"]
    assert all(
        linha["p10"] <= linha["p25"] <= linha["mediana"] <= linha["p75"] <= linha["p90"] for linha in faixas.values()
    )


def test_a_contestacao_so_cresce_e_para_onde_o_golpe_para(simulacao: Simulacao) -> None:
    demora = retrato.demora_da_contestacao(simulacao)

    assert sorted(demora["golpe"].unique().to_list()) == sorted(golpe.value for golpe in GOLPES)

    for golpe in GOLPES:
        curva = demora.filter(pl.col("golpe") == golpe.value)["contestados"].to_list()
        assert curva == sorted(curva)
        assert curva[-1] < 1

    venda_falsa = demora.filter((pl.col("golpe") == "venda_falsa") & (pl.col("dias") == 1))["contestados"].item()
    assert venda_falsa < 0.05


def test_a_laranja_recebe_de_mais_desconhecidos_que_a_pessoa_comum(simulacao: Simulacao) -> None:
    pagadores = retrato.pagadores_novos_por_recebedor(simulacao)

    def tres_ou_mais(tipo: str) -> float:
        dele = pagadores.filter((pl.col("tipo") == tipo) & pl.col("faixa").is_in(["3 a 5", "6 a 10", "mais de 10"]))
        return float(dele["parte"].sum())

    assert tres_ou_mais("conta laranja") > 3 * tres_ou_mais("pessoa")


def test_os_rotulos_proximos_se_afastam_sem_mudar_de_ordem() -> None:
    afastados = figuras._afastados({"a": 0.91, "b": 0.90, "c": 0.5}, distancia=0.05)

    assert afastados == pytest.approx({"c": 0.5, "b": 0.90, "a": 0.95})


def test_as_figuras_sao_gravadas_em_png(simulacao: Simulacao, tmp_path: Path) -> None:
    # A simulação média termina em setembro: o teste dos modelos é setembro, sem o mês de validação, e o monitor
    # acompanha as quatro semanas de setembro, com a virada no primeiro dia dele.
    cartoes = cartoes_inventados(tmp_path / "cartoes")
    virada = simular(replace(MEDIA, virada=date(2025, 9, 1)))
    gravadas = figuras.gerar(simulacao, tmp_path, (Dobra(date(2025, 9, 1), date(2025, 9, 29)),), cartoes, virada)

    assert [figura.name for figura in gravadas] == [
        "horas.png",
        "valores.png",
        "contestacao.png",
        "recebedores.png",
        "separacao.png",
        "golpes_pegos.png",
        "golpes_por_modelo.png",
        "calibracao.png",
        "rotulo_atrasado.png",
        "mapa_da_decisao.png",
        "custo_das_politicas.png",
        "erros_por_idade.png",
        "fraudes_nos_cartoes.png",
        "monitoramento.png",
    ]
    assert all(figura.read_bytes().startswith(b"\x89PNG") for figura in gravadas)


def test_o_monitor_da_figura_tem_a_falsa_central_pega_em_cada_semana(simulacao: Simulacao) -> None:
    semanas = figuras.monitorado(simulacao, date(2025, 9, 1))

    assert semanas["semana"].to_list() == [date(2025, 9, dia) for dia in (1, 8, 15, 22)]
    assert semanas["falsa_central_pega"].is_between(0, 1).all()
    # Sem a virada, a falsa central é a de sempre, e o campeão pega a maior parte dela.
    assert semanas["falsa_central_pega"].mean() > 0.5  # type: ignore[operator]
