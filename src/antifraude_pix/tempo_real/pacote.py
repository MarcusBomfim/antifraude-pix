"""O campeão empacotado: o modelo do registro numa pasta que roda em qualquer máquina (ADR-0023).

O registro do MLflow guarda o lugar de cada artefato pelo caminho absoluto da
máquina que treinou, `file:///C:/Users/...`, e num contêiner Linux esse caminho
não existe. Mais do que isso, o serviço no ar não deveria ler o registro: se
alguém promove outro modelo, o que decide os Pix mudaria debaixo dele, sem
ninguém subir nada.

`empacotar` copia o modelo de um apelido para uma pasta, com o `pacote.json`,
que diz qual modelo é: o nome, a versão, o apelido e o corte do treino. `abrir`
carrega o decisor dessa pasta, sem o registro. Trocar o modelo do serviço é
empacotar de novo e subir o serviço de novo, e a versão vai na resposta de
cada Pix.

A cópia tem um conserto. O `MLmodel` guarda o arquivo do modelo como código
pelo caminho de quem registrou, `C:\\Users\\...\\modelo_do_pix.py`, e o MLflow
o acha pelo nome, cortando o caminho nas barras da máquina onde roda. No Linux,
a barra invertida não corta nada, e o modelo não abre. O pacote guarda só o nome.
"""

import json
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path, PureWindowsPath

import mlflow
from mlflow.models import Model
from mlflow.tracking import MlflowClient

from antifraude_pix.rastreio.experimentos import endereco
from antifraude_pix.rastreio.registro import CAMPEAO, NOME
from antifraude_pix.tempo_real.servico import Decisor, decisor_de

FICHA = "pacote.json"
"""O arquivo, dentro do pacote, que diz qual modelo ele tem."""


@dataclass(frozen=True)
class Pacote:
    """O modelo que o pacote tem: o nome e a versão no registro, o apelido de quando foi empacotado, e o corte.

    O corte é o dia de onde o treino parou de ler, AAAA-MM-DD. O modelo registrado antes da Parte 7 não o guardou.
    """

    nome: str
    versao: str
    apelido: str
    corte: str | None


def empacotar(pasta_dos_dados: Path, destino: Path, apelido: str = CAMPEAO) -> Pacote:
    """Copia o modelo do apelido para a pasta de destino, trocando o pacote que estiver lá."""
    mlflow.set_tracking_uri(endereco(pasta_dos_dados))
    cliente = MlflowClient()
    versao = cliente.get_model_version_by_alias(NOME, apelido)
    corte = cliente.get_run(str(versao.run_id)).data.params.get("treino.corte")
    pacote = Pacote(NOME, str(versao.version), apelido, corte)
    destino.parent.mkdir(parents=True, exist_ok=True)

    # O pacote novo nasce ao lado e só então toma o lugar do antigo: quem abrir a pasta acha um ou o outro inteiro.
    with tempfile.TemporaryDirectory(dir=destino.parent) as temporaria:
        novo = Path(temporaria) / "pacote"
        mlflow.artifacts.download_artifacts(f"models:/{NOME}@{apelido}", dst_path=str(novo))
        _so_o_nome_do_codigo(novo)
        (novo / FICHA).write_text(
            json.dumps(asdict(pacote), ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
        )

        if destino.exists():
            shutil.rmtree(destino)

        novo.rename(destino)

    return pacote


def _so_o_nome_do_codigo(pasta: Path) -> None:
    """Troca, no `MLmodel`, o caminho do arquivo do modelo como código pelo nome dele, que vale em qualquer máquina."""
    caminho = pasta / "MLmodel"
    modelo = Model.load(str(caminho))
    funcao = modelo.flavors.get("python_function", {})

    if funcao.get("model_code_path"):
        # O caminho do Windows corta nas duas barras; o do Linux, na barra comum: o nome sai certo dos dois.
        funcao["model_code_path"] = PureWindowsPath(funcao["model_code_path"]).name
        modelo.save(str(caminho))


def abrir(pasta_do_pacote: Path) -> tuple[Decisor, Pacote]:
    """O decisor do pacote e o que ele é, sem o registro do MLflow."""
    ficha = pasta_do_pacote / FICHA

    if not ficha.is_file():
        raise FileNotFoundError(
            f"{pasta_do_pacote} não é um pacote: falta o {FICHA}. Empacote o campeão com `antifraude-pix empacotar`."
        )

    pacote = Pacote(**json.loads(ficha.read_text(encoding="utf-8")))

    return decisor_de(mlflow.pyfunc.load_model(str(pasta_do_pacote))), pacote
