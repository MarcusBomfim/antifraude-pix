from datetime import date, datetime, time
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl
import pytest

from antifraude_pix.dados.banco import VisaoDoBanco
from antifraude_pix.dados.esquema import APARELHOS_NOVOS, CONTESTACOES, FUSO, MOMENTO
from antifraude_pix.monitoramento.rotulo import (
    CONTESTADO,
    chegada,
    conhecidos,
    parte_que_chegou,
    perda_por_semana,
    razao,
)

ATE = date(2025, 9, 15)  # uma segunda-feira


def _as(dia: date, hora: int = 12) -> datetime:
    return datetime.combine(dia, time(hora), tzinfo=ZoneInfo(FUSO))


def _visao(pix: dict[int, datetime], contestacoes: dict[int, datetime]) -> VisaoDoBanco:
    """Uma visão do banco feita à mão, só com o que o rótulo usa: o momento de cada Pix e de cada contestação."""
    return VisaoDoBanco(
        clientes=pl.DataFrame(),
        transacoes=pl.DataFrame(
            {"id": list(pix), "momento": list(pix.values())}, schema={"id": pl.Int64, "momento": MOMENTO}
        ),
        aparelhos_novos=pl.DataFrame(schema=APARELHOS_NOVOS),
        contestacoes=pl.DataFrame(
            {"transacao": list(contestacoes), "contestada_em": list(contestacoes.values())}, schema=CONTESTACOES
        ),
    )


def _curva(parte: list[float]) -> pl.DataFrame:
    return pl.DataFrame({"dias": range(len(parte)), "parte": parte}, schema={"dias": pl.Int32, "parte": pl.Float64})


def test_a_chegada_conta_so_os_pix_que_ja_tiveram_todo_o_prazo() -> None:
    antigo = date(2025, 5, 1)
    visao = _visao(
        {1: _as(antigo, 0), 2: _as(antigo, 0), 3: _as(antigo, 0), 4: _as(antigo, 0), 5: _as(date(2025, 9, 1))},
        {
            1: _as(date(2025, 5, 1)),  # meio dia depois
            2: _as(date(2025, 5, 3), 0),  # dois dias depois
            3: _as(date(2025, 5, 3), 0),
            4: _as(date(2025, 5, 11), 0),  # dez dias depois
            5: _as(date(2025, 9, 2)),  # o Pix recente: ainda não teve o prazo inteiro
        },
    )
    curva = chegada(visao, ATE)

    assert curva["dias"].to_list() == list(range(81))
    assert curva.filter(pl.col("dias").is_in([0, 1, 2, 9, 10, 80]))["parte"].to_list() == [0, 0.25, 0.75, 0.75, 1, 1]


def test_a_contestacao_que_ainda_nao_chegou_nao_entra_na_chegada() -> None:
    visao = _visao({1: _as(date(2025, 5, 1)), 2: _as(date(2025, 5, 1))}, {1: _as(date(2025, 5, 2)), 2: _as(ATE)})

    assert chegada(visao, ATE)["parte"][1] == 1.0


def test_sem_contestacao_antiga_nao_ha_chegada_a_medir() -> None:
    visao = _visao({1: _as(date(2025, 9, 1))}, {1: _as(date(2025, 9, 2))})

    with pytest.raises(ValueError, match="mais de 80 dias antes de 2025-09-15"):
        chegada(visao, ATE)


def test_a_parte_que_chegou_fica_entre_os_pontos_da_curva() -> None:
    curva = _curva([0.0, 0.5, 0.9, 1.0])

    assert parte_que_chegou(curva, [-3.0, 0.0, 0.5, 1.5, 2.0, 10.0]).tolist() == [0, 0, 0.25, 0.7, 0.9, 1]


def test_os_conhecidos_sao_os_contestados_antes_do_dia_com_o_rotulo_do_banco() -> None:
    visao = _visao(
        {1: _as(date(2025, 9, 1)), 2: _as(date(2025, 9, 2)), 3: _as(date(2025, 9, 3))},
        {1: _as(date(2025, 9, 14), 23), 2: _as(ATE, 0)},
    )

    assert conhecidos(visao, ATE).rows() == [(1, CONTESTADO)]


def _decididos() -> tuple[VisaoDoBanco, pl.DataFrame]:
    """Cinco Pix em duas semanas: dois aprovados e conhecidos de um jeito, um segurado, um aprovado depois do dia."""
    momentos = {
        1: _as(date(2025, 9, 2)),
        2: _as(date(2025, 9, 3)),
        3: _as(date(2025, 9, 4)),
        4: _as(date(2025, 9, 9)),
        5: _as(ATE),
    }
    # O 1 já é conhecido no dia; o 4 só vai ser contestado depois dele.
    visao = _visao(momentos, {1: _as(date(2025, 9, 5)), 4: _as(date(2025, 9, 20))})
    decididos = pl.DataFrame(
        {
            "id": list(momentos),
            "momento": list(momentos.values()),
            "valor": [1_000.0, 500.0, 200.0, 300.0, 9_999.0],
            "probabilidade": [0.01, 0.02, 0.5, 0.1, 0.9],
            "acao": ["aprovar", "aprovar", "segurar", "aprovar", "aprovar"],
        },
        schema_overrides={"momento": MOMENTO},
    )

    return visao, decididos


def test_a_perda_de_cada_semana_conta_so_os_aprovados_de_antes_do_dia() -> None:
    visao, decididos = _decididos()
    semanas = perda_por_semana(decididos, visao, ATE, _curva([1.0] * 81))

    assert semanas.rows() == [
        (date(2025, 9, 1), 2, 0.01 * 1_000 + 0.02 * 500, 1_000.0, 1_000 / 20),
        (date(2025, 9, 8), 1, 0.1 * 300, 0.0, 0.0),
    ]
    assert semanas.columns == ["semana", "aprovados", "esperada", "conhecida", "razao"]
    assert razao(semanas) == 1_000 / 50


def test_a_perda_esperada_desconta_o_rotulo_que_ainda_nao_chegou() -> None:
    visao, decididos = _decididos()
    # A curva sobe um oitenta avos por dia: o Pix 4, de 5,5 dias, tem 5,5 / 80 do rótulo.
    semanas = perda_por_semana(decididos, visao, ATE, _curva(list(np.arange(81) / 80)))

    assert semanas["esperada"][1] == pytest.approx(0.1 * 300 * 5.5 / 80)


def test_sem_perda_esperada_nao_ha_razao() -> None:
    semanas = pl.DataFrame({"esperada": [0.0], "conhecida": [10.0]})

    with pytest.raises(ValueError, match="alguma perda esperada"):
        razao(semanas)
