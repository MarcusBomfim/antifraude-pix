import json
from pathlib import Path

import pytest

from antifraude_pix.dados.gravacao import MANIFESTO, gravar, ler
from antifraude_pix.simulacao.simulador import Simulacao


def test_o_que_se_grava_e_o_que_se_le(simulacao: Simulacao, tmp_path: Path) -> None:
    gravar(simulacao, tmp_path)
    lida = ler(tmp_path)

    assert lida.config == simulacao.config
    assert lida.transacoes.equals(simulacao.transacoes)
    assert lida.clientes.equals(simulacao.clientes)
    assert lida.aparelhos_novos.equals(simulacao.aparelhos_novos)
    assert lida.contestacoes.equals(simulacao.contestacoes)
    assert lida.verdade.fraudes.equals(simulacao.verdade.fraudes)
    assert lida.verdade.ataques.equals(simulacao.verdade.ataques)
    assert lida.verdade.laranjas.equals(simulacao.verdade.laranjas)


def test_o_banco_e_a_verdade_ficam_em_pastas_separadas_e_o_manifesto_diz_de_onde_vieram(
    simulacao: Simulacao, tmp_path: Path
) -> None:
    manifesto = gravar(simulacao, tmp_path)

    assert sorted(path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*.parquet")) == [
        "banco/aparelhos_novos.parquet",
        "banco/clientes.parquet",
        "banco/contestacoes.parquet",
        "banco/transacoes.parquet",
        "verdade/ataques.parquet",
        "verdade/fraudes.parquet",
        "verdade/laranjas.parquet",
    ]
    assert json.loads((tmp_path / MANIFESTO).read_text(encoding="utf-8")) == manifesto
    assert manifesto["formato"] == 1
    assert manifesto["simulador"] == "antifraude-pix 0.1.0"
    assert manifesto["configuracao"]["inicio"] == "2025-07-01"
    assert manifesto["tabelas"]["banco/transacoes.parquet"]["linhas"] == simulacao.transacoes.height


def test_os_mesmos_dados_dao_os_mesmos_arquivos_byte_a_byte(simulacao: Simulacao, tmp_path: Path) -> None:
    uma = gravar(simulacao, tmp_path / "uma")
    outra = gravar(simulacao, tmp_path / "outra")

    assert uma["tabelas"] == outra["tabelas"]


def test_o_arquivo_mexido_depois_de_gravado_e_recusado(simulacao: Simulacao, tmp_path: Path) -> None:
    gravar(simulacao, tmp_path)

    with (tmp_path / "banco" / "transacoes.parquet").open("ab") as arquivo:
        arquivo.write(b"\0")

    with pytest.raises(ValueError, match=r"O arquivo banco/transacoes.parquet mudou depois de gravado"):
        ler(tmp_path)


def test_a_pasta_de_outro_formato_e_recusada(simulacao: Simulacao, tmp_path: Path) -> None:
    manifesto = gravar(simulacao, tmp_path)
    (tmp_path / MANIFESTO).write_text(json.dumps({**manifesto, "formato": 2}), encoding="utf-8")

    with pytest.raises(ValueError, match="está no formato 2, e este simulador lê o 1"):
        ler(tmp_path)
