"""Os dados em disco: o que o banco vê numa pasta, a verdade em outra, e o manifesto que diz de onde vieram (ADR-0004).

```
dados/
  manifesto.json        a configuração, a versão do simulador, e as linhas e o SHA-256 de cada tabela
  banco/                o que o modelo pode ler
    clientes.parquet
    transacoes.parquet
    aparelhos_novos.parquet
    contestacoes.parquet
  verdade/              o que só o simulador sabe: para medir, e nunca para treinar
    fraudes.parquet
    ataques.parquet
    laranjas.parquet
```

As duas pastas separadas tornam o vazamento visível: o código que treina lê de
`banco/`, e um `verdade/` nele chama a atenção em qualquer revisão.
"""

import hashlib
import json
from dataclasses import asdict
from datetime import date
from importlib.metadata import version
from pathlib import Path
from typing import Any

import polars as pl

from antifraude_pix.dados.banco import VisaoDoBanco
from antifraude_pix.dados.esquema import DA_VERDADE, DO_BANCO
from antifraude_pix.simulacao.configuracao import Configuracao
from antifraude_pix.simulacao.simulador import Simulacao, Verdade

BANCO = "banco"
VERDADE = "verdade"
MANIFESTO = "manifesto.json"
FORMATO = 1
"""A versão do formato da pasta: muda quando uma tabela ganha, perde ou muda uma coluna."""


def gravar(simulacao: Simulacao, pasta: Path) -> dict[str, Any]:
    """Grava a simulação na pasta e devolve o manifesto. Os mesmos dados dão os mesmos arquivos, byte a byte."""
    tabelas = {
        **{f"{BANCO}/{nome}.parquet": tabela for nome, tabela in _do_banco(simulacao).items()},
        **{f"{VERDADE}/{nome}.parquet": tabela for nome, tabela in _da_verdade(simulacao.verdade).items()},
    }
    arquivos = {}

    for caminho, tabela in tabelas.items():
        destino = pasta / caminho
        destino.parent.mkdir(parents=True, exist_ok=True)
        tabela.write_parquet(destino, compression="zstd", statistics=True)
        arquivos[caminho] = {"linhas": tabela.height, "sha256": hashlib.sha256(destino.read_bytes()).hexdigest()}

    manifesto = {
        "formato": FORMATO,
        "simulador": f"antifraude-pix {version('antifraude-pix')}",
        # Sem a virada, o manifesto fica como era antes dela: a mesma simulação dá os mesmos bytes (ADR-0019).
        "configuracao": {
            chave: _json(valor)
            for chave, valor in asdict(simulacao.config).items()
            if not (chave == "virada" and valor is None)
        },
        "tabelas": arquivos,
    }
    (pasta / MANIFESTO).write_text(json.dumps(manifesto, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    return manifesto


def ler(pasta: Path) -> Simulacao:
    """Lê a simulação gravada, conferindo cada arquivo pelo SHA-256 do manifesto."""
    manifesto = _manifesto_conferido(pasta, ate_a_verdade=True)

    configuracao = dict(manifesto["configuracao"])
    configuracao["inicio"] = date.fromisoformat(configuracao["inicio"])

    # A virada entrou na Parte 7: o manifesto da simulação sem ela não a tem.
    if configuracao.get("virada") is not None:
        configuracao["virada"] = date.fromisoformat(configuracao["virada"])

    def tabela(parte: str, nome: str) -> pl.DataFrame:
        return pl.read_parquet(pasta / parte / f"{nome}.parquet")

    return Simulacao(
        config=Configuracao(**configuracao),
        clientes=tabela(BANCO, "clientes"),
        transacoes=tabela(BANCO, "transacoes"),
        aparelhos_novos=tabela(BANCO, "aparelhos_novos"),
        contestacoes=tabela(BANCO, "contestacoes"),
        verdade=Verdade(
            fraudes=tabela(VERDADE, "fraudes"),
            ataques=tabela(VERDADE, "ataques"),
            laranjas=tabela(VERDADE, "laranjas"),
        ),
    )


def ler_o_banco(pasta: Path) -> VisaoDoBanco:
    """Lê só o que o banco vê: a pasta da verdade nem é aberta (ADR-0005)."""
    _manifesto_conferido(pasta, ate_a_verdade=False)

    return VisaoDoBanco(**{nome: pl.read_parquet(pasta / BANCO / f"{nome}.parquet") for nome in DO_BANCO})


def _manifesto_conferido(pasta: Path, *, ate_a_verdade: bool) -> dict[str, Any]:
    """O manifesto da pasta, conferidos o formato e o SHA-256 de cada arquivo.

    Os arquivos da verdade só são conferidos quando vão ser lidos.
    """
    manifesto: dict[str, Any] = json.loads((pasta / MANIFESTO).read_text(encoding="utf-8"))

    if manifesto["formato"] != FORMATO:
        raise ValueError(f"A pasta {pasta} está no formato {manifesto['formato']}, e este simulador lê o {FORMATO}.")

    for caminho, dados in manifesto["tabelas"].items():
        if not ate_a_verdade and not caminho.startswith(f"{BANCO}/"):
            continue

        if hashlib.sha256((pasta / caminho).read_bytes()).hexdigest() != dados["sha256"]:
            raise ValueError(f"O arquivo {caminho} mudou depois de gravado: o SHA-256 não é o do manifesto.")

    return manifesto


def _do_banco(simulacao: Simulacao) -> dict[str, pl.DataFrame]:
    tabelas = {
        "clientes": simulacao.clientes,
        "transacoes": simulacao.transacoes,
        "aparelhos_novos": simulacao.aparelhos_novos,
        "contestacoes": simulacao.contestacoes,
    }
    assert tabelas.keys() == DO_BANCO.keys()

    return tabelas


def _da_verdade(verdade: Verdade) -> dict[str, pl.DataFrame]:
    tabelas = {"fraudes": verdade.fraudes, "ataques": verdade.ataques, "laranjas": verdade.laranjas}
    assert tabelas.keys() == DA_VERDADE.keys()

    return tabelas


def _json(valor: object) -> object:
    return valor.isoformat() if isinstance(valor, date) else valor
