import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Any

import polars as pl
import pytest
import uvicorn
from fastapi.testclient import TestClient
from mlflow.tracking import MlflowClient

from antifraude_pix.cartoes import base as base_dos_cartoes
from antifraude_pix.cli import principal
from antifraude_pix.monitoramento.rotulo import ALARME
from antifraude_pix.rastreio.experimentos import EXPERIMENTO, VARIANTES, endereco
from antifraude_pix.rastreio.registro import CAMPEAO, DESAFIANTE, NOME, VEREDITO
from antifraude_pix.variaveis.catalogo import NOMES
from tests.apoio import cartoes_inventados


def test_simular_grava_os_dados_e_conta_o_que_saiu(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    saida = principal(
        ["simular", "--clientes", "80", "--dias", "20", "--golpes-por-mil", "120", "--saida", str(tmp_path)]
    )
    impresso = capsys.readouterr().out

    assert saida == 0
    assert impresso.startswith("80 clientes, 20 dias a partir de 01/07/2025, semente 2026.\n")
    assert "Pix;" in impresso
    assert f"Gravado em {tmp_path}" in impresso
    assert (tmp_path / "banco" / "transacoes.parquet").exists()


def test_validar_confere_o_que_simular_gravou(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    principal(["simular", "--clientes", "30", "--dias", "10", "--saida", str(tmp_path)])
    capsys.readouterr()

    assert principal(["validar", str(tmp_path)]) == 0
    assert capsys.readouterr().out.endswith("Os dados seguem o contrato.\n")


def test_a_configuracao_invalida_e_recusada_com_o_motivo(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert principal(["simular", "--clientes", "0", "--saida", str(tmp_path)]) == 2
    assert capsys.readouterr().err == "antifraude-pix simular: O número de clientes tem de ser positivo.\n"
    assert not (tmp_path / "banco").exists()


def test_simular_com_a_virada_grava_o_dia_dela(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    pedido = ["simular", "--clientes", "30", "--dias", "10", "--virada", "2025-07-05", "--saida", str(tmp_path)]

    assert principal(pedido) == 0
    assert (
        json.loads((tmp_path / "manifesto.json").read_text(encoding="utf-8"))["configuracao"]["virada"] == "2025-07-05"
    )


def test_a_virada_fora_da_simulacao_e_recusada_com_o_motivo(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    pedido = ["simular", "--clientes", "30", "--dias", "10", "--virada", "2025-07-11", "--saida", str(tmp_path)]

    assert principal(pedido) == 2
    assert capsys.readouterr().err == "antifraude-pix simular: A virada tem de cair dentro da simulação.\n"


def test_validar_a_pasta_que_nao_existe_falha_com_o_motivo(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert principal(["validar", str(tmp_path / "nenhuma")]) == 1
    assert "antifraude-pix validar: não consegui ler" in capsys.readouterr().err


def test_o_comando_que_nao_existe_e_recusado_pelo_leitor() -> None:
    with pytest.raises(SystemExit) as saida:
        principal(["treinar"])

    assert saida.value.code == 2


def test_variaveis_le_o_banco_e_grava_uma_linha_por_pix(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    principal(["simular", "--clientes", "40", "--dias", "15", "--saida", str(tmp_path)])
    capsys.readouterr()

    assert principal(["variaveis", str(tmp_path)]) == 0
    impresso = capsys.readouterr().out
    variaveis = pl.read_parquet(tmp_path / "variaveis.parquet")
    pix = pl.read_parquet(tmp_path / "banco" / "transacoes.parquet")

    assert impresso.startswith(f"33 variáveis de {pix.height} Pix, gravadas em {tmp_path / 'variaveis.parquet'}")
    assert variaveis["id"].equals(pix["id"])
    assert variaveis.columns == ["id", *NOMES]


def test_variaveis_grava_onde_foi_pedido(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    dados, destino = tmp_path / "dados", tmp_path / "outra" / "pasta" / "variaveis-de-teste.parquet"
    principal(["simular", "--clientes", "30", "--dias", "10", "--saida", str(dados)])

    assert principal(["variaveis", str(dados), "--saida", str(destino)]) == 0
    assert f"gravadas em {destino}" in capsys.readouterr().out
    assert pl.read_parquet(destino).columns == ["id", *NOMES]
    assert not (dados / "variaveis.parquet").exists()


def test_variaveis_da_pasta_que_nao_existe_falha_com_o_motivo(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert principal(["variaveis", str(tmp_path / "nenhuma")]) == 1
    assert "antifraude-pix variaveis: não consegui ler" in capsys.readouterr().err


def test_avaliar_treina_mes_a_mes_e_mede_contra_a_verdade(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    # De 1º de julho a 2 de novembro: aquecimento, agosto de treino, setembro de validação, outubro e novembro de teste.
    principal(["simular", "--clientes", "150", "--dias", "125", "--golpes-por-mil", "200", "--saida", str(tmp_path)])
    capsys.readouterr()

    assert principal(["avaliar", str(tmp_path)]) == 0
    impresso = capsys.readouterr().out
    pontuacoes = pl.read_parquet(tmp_path / "avaliacao.parquet")

    assert impresso.startswith("Teste de 01/10/2025 a 02/11/2025, em 2 meses: ")
    assert all(modelo in impresso for modelo in ("as regras de hoje", "a regressão logística", "o LightGBM"))
    assert pontuacoes["modelo"].unique(maintain_order=True).to_list() == ["regras", "logistica", "lightgbm"]
    assert pontuacoes["corte"].unique().sort().to_list() == [date(2025, 10, 1), date(2025, 11, 1)]


def test_avaliar_dados_curtos_demais_falha_com_o_motivo(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    principal(["simular", "--clientes", "30", "--dias", "60", "--saida", str(tmp_path)])
    capsys.readouterr()

    assert principal(["avaliar", str(tmp_path)]) == 1
    assert "não chegam a um mês de teste" in capsys.readouterr().err
    assert not (tmp_path / "avaliacao.parquet").exists()


def test_avaliar_a_pasta_que_nao_existe_falha_com_o_motivo(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert principal(["avaliar", str(tmp_path / "nenhuma")]) == 1
    assert "antifraude-pix avaliar: não consegui ler" in capsys.readouterr().err


def test_decidir_decide_explica_e_mede_o_custo_e_os_erros_por_idade(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    principal(["simular", "--clientes", "150", "--dias", "125", "--golpes-por-mil", "200", "--saida", str(tmp_path)])
    capsys.readouterr()

    assert principal(["decidir", str(tmp_path)]) == 0
    impresso = capsys.readouterr().out
    decisoes = pl.read_parquet(tmp_path / "decisoes.parquet")
    nao_aprovados = decisoes.filter(pl.col("acao") != "aprovar")

    assert impresso.startswith("Decisões de 01/10/2025 a 02/11/2025, em 2 meses: ")
    assert all(
        politica in impresso
        for politica in ("aprovar tudo", "bloquear de 50% para cima", "segurar o alerta de 0,5%", "decidir pelo custo")
    )
    assert all(faixa in impresso for faixa in ("18 a 34", "35 a 54", "55 a 64", "65 ou mais"))
    assert decisoes.columns == [
        "id",
        "corte",
        "modelo",
        "pontuacao",
        "probabilidade",
        "valor",
        "idade",
        "acao",
        "explicacao",
        "golpe",
    ]
    assert decisoes["corte"].unique().sort().to_list() == [date(2025, 10, 1), date(2025, 11, 1)]
    assert nao_aprovados.height > 0
    assert nao_aprovados["explicacao"].str.contains(" de chance de golpe: ").all()


def test_decidir_dados_curtos_demais_falha_com_o_motivo(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    principal(["simular", "--clientes", "30", "--dias", "60", "--saida", str(tmp_path)])
    capsys.readouterr()

    assert principal(["decidir", str(tmp_path)]) == 1
    assert "antifraude-pix decidir: os dados de" in capsys.readouterr().err
    assert not (tmp_path / "decisoes.parquet").exists()


def test_decidir_a_pasta_que_nao_existe_falha_com_o_motivo(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert principal(["decidir", str(tmp_path / "nenhuma")]) == 1
    assert "antifraude-pix decidir: não consegui ler" in capsys.readouterr().err


def test_experimentos_grava_uma_corrida_por_modelo_de_cada_variante(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    principal(["simular", "--clientes", "150", "--dias", "125", "--golpes-por-mil", "200", "--saida", str(tmp_path)])
    capsys.readouterr()

    assert principal(["experimentos", str(tmp_path)]) == 0
    impresso = capsys.readouterr().out
    corridas = MlflowClient(endereco(tmp_path)).search_runs(
        [MlflowClient(endereco(tmp_path)).get_experiment_by_name(EXPERIMENTO).experiment_id]  # type: ignore[union-attr]
    )

    assert impresso.startswith("As variantes, com a precisão média de cada modelo nos meses de teste:")
    assert all(f"- {variante.nome}: " in impresso for variante in VARIANTES)
    assert "uv run mlflow ui --backend-store-uri sqlite:///" in impresso
    assert len(corridas) == sum(len(variante.modelos) for variante in VARIANTES)


def test_registrar_grava_o_campeao_e_a_ficha(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    principal(["simular", "--clientes", "150", "--dias", "125", "--golpes-por-mil", "200", "--saida", str(tmp_path)])
    capsys.readouterr()

    assert principal(["registrar", str(tmp_path), "--ficha", str(tmp_path / "docs" / "ficha.md")]) == 0
    impresso = capsys.readouterr().out
    campeao = MlflowClient(endereco(tmp_path)).get_model_version_by_alias(NOME, CAMPEAO)

    assert "O modelo antifraude-pix, versão 1, registrado no MLflow com o apelido campeao." in impresso
    assert (tmp_path / "docs" / "ficha.md").read_text(encoding="utf-8") == campeao.description


def test_cartoes_baixa_confere_e_roda_o_mesmo_processo(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    origem = cartoes_inventados(tmp_path / "origem")
    monkeypatch.setattr(base_dos_cartoes, "ENDERECO", origem.as_uri())
    monkeypatch.setattr(base_dos_cartoes, "MD5", hashlib.md5(origem.read_bytes(), usedforsecurity=False).hexdigest())

    assert principal(["cartoes", str(tmp_path / "dados")]) == 0
    impresso = capsys.readouterr().out

    assert impresso.startswith("Baixando a base de cartões da ULB do OpenML (151 MB)")
    assert all(nome in impresso for nome in ("o valor sozinho", "a regressão logística", "o LightGBM"))
    assert (tmp_path / "dados" / "cartoes" / base_dos_cartoes.ARQUIVO).exists()
    cliente = MlflowClient(endereco(tmp_path / "dados"))
    dos_cartoes = cliente.get_experiment_by_name("cartoes-ulb")
    assert dos_cartoes is not None
    assert len(cliente.search_runs([dos_cartoes.experiment_id])) == 3


def test_cartoes_com_o_arquivo_que_nao_confere_falha_com_o_motivo(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(base_dos_cartoes, "ENDERECO", cartoes_inventados(tmp_path / "origem").as_uri())

    assert principal(["cartoes", str(tmp_path / "dados")]) == 1
    assert "antifraude-pix cartoes: não consegui a base de cartões" in capsys.readouterr().err


def test_registrar_a_pasta_que_nao_existe_falha_com_o_motivo(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert principal(["registrar", str(tmp_path / "nenhuma")]) == 1
    assert "antifraude-pix registrar: não consegui ler" in capsys.readouterr().err


def test_latencia_mede_por_dentro_e_pela_api(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    principal(["simular", "--clientes", "150", "--dias", "125", "--golpes-por-mil", "200", "--saida", str(tmp_path)])
    principal(["registrar", str(tmp_path)])
    capsys.readouterr()

    assert principal(["latencia", str(tmp_path), "--pix", "40"]) == 0
    impresso = capsys.readouterr().out

    assert "Quem decide é o antifraude-pix, versão 1." in impresso
    assert "Os Pix de 01/11/2025 em diante" in impresso
    assert all(
        medida in impresso for medida in ("as variáveis", "a decisão", "dentro do serviço", "a ida e volta pela API")
    )


def test_servir_sobe_o_estado_e_o_campeao(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    principal(["simular", "--clientes", "150", "--dias", "125", "--golpes-por-mil", "200", "--saida", str(tmp_path)])
    principal(["registrar", str(tmp_path)])
    capsys.readouterr()
    servidas: list[Any] = []
    monkeypatch.setattr(uvicorn, "run", lambda app, **opcoes: servidas.append((app, opcoes)))

    assert principal(["servir", str(tmp_path), "--porta", "8123"]) == 0
    ((app, opcoes),) = servidas
    saude = TestClient(app).get("/saude").json()

    assert "A API está em http://127.0.0.1:8123" in capsys.readouterr().out
    assert opcoes["port"] == 8123
    assert saude["versao"] == "1"
    assert saude["pix_no_estado"] == pl.read_parquet(tmp_path / "banco" / "transacoes.parquet").height


def test_servir_sem_campeao_falha_com_o_motivo(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    principal(["simular", "--clientes", "30", "--dias", "10", "--saida", str(tmp_path)])
    capsys.readouterr()

    assert principal(["servir", str(tmp_path)]) == 1
    assert "antifraude-pix servir: não consegui subir o serviço" in capsys.readouterr().err


def test_monitorar_e_retreinar_acompanham_o_campeao_e_o_poem_contra_o_desafiante(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    principal(
        [
            "simular",
            "--clientes",
            "150",
            "--dias",
            "160",
            "--golpes-por-mil",
            "200",
            "--virada",
            "2025-11-01",
            "--saida",
            str(tmp_path),
        ]
    )
    capsys.readouterr()

    # Antes de 15 de outubro não termina nenhum mês de teste: o primeiro vai de 1º de outubro a 1º de novembro.
    assert principal(["registrar", str(tmp_path), "--corte", "2025-10-15"]) == 1
    assert "não chegam a um mês de teste antes de 15/10/2025" in capsys.readouterr().err
    assert principal(["monitorar", str(tmp_path)]) == 1
    assert "antifraude-pix monitorar: não consegui ler os dados e o campeão" in capsys.readouterr().err

    assert principal(["registrar", str(tmp_path), "--corte", "2025-11-01"]) == 0
    assert "Treinado com o que o banco sabia na meia-noite de 01/11/2025." in capsys.readouterr().out
    assert principal(["monitorar", str(tmp_path), "--ate", "2025-11-07"]) == 1
    assert "não há uma semana inteira de Pix para acompanhar" in capsys.readouterr().err

    assert principal(["monitorar", str(tmp_path), "--ate", "2025-12-01"]) == 0
    impresso = capsys.readouterr().out
    semanas = pl.read_parquet(tmp_path / "monitoramento.parquet")

    assert "O campeão é o antifraude-pix, versão 1, treinado com o que o banco sabia em 01/11/2025." in impresso
    assert "Cada semana inteira de 03/11/2025 a 30/11/2025, contra as 4 semanas antes do corte" in impresso
    assert semanas["semana"].to_list() == [
        date(2025, 11, 3),
        date(2025, 11, 10),
        date(2025, 11, 17),
        date(2025, 11, 24),
    ]
    assert ("Alarme: o golpe passa mais" in impresso) is (semanas["razao_acumulada"][-1] >= ALARME)
    assert ("Dentro do esperado" in impresso) is (semanas["razao_acumulada"][-1] < ALARME)

    # O desafiante de 1º de dezembro seria treinado em 17 de novembro; o de 10 de novembro, antes do campeão.
    assert principal(["retreinar", str(tmp_path), "--ate", "2025-11-10"]) == 1
    assert "ele não saberia nada a mais" in capsys.readouterr().err

    assert principal(["retreinar", str(tmp_path), "--ate", "2025-12-01"]) == 0
    impresso = capsys.readouterr().out
    cliente = MlflowClient(endereco(tmp_path))
    desafiante = cliente.get_model_version_by_alias(NOME, DESAFIANTE)
    promovido = desafiante.tags[VEREDITO] == "promovido"

    assert "O desafiante: versão 2, treinado com o que o banco sabia em 17/11/2025" in impresso
    assert "Os dois decidiram os mesmos" in impresso
    assert str(desafiante.version) == "2"
    assert (str(cliente.get_model_version_by_alias(NOME, CAMPEAO).version) == "2") is promovido
    assert ("O desafiante ganhou: a versão 2" in impresso) is promovido
    assert ("O campeão continua, e a versão 2 fica como desafiante" in impresso) is not promovido
