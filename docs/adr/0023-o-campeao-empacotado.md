# ADR-0023 — O campeão empacotado

**Status:** aceita

## Contexto

Até a Parte 7, o serviço carregava o campeão direto do registro do MLflow (ADR-0017). Num contêiner, isso não funciona, por dois motivos:
- **O registro guarda caminhos da máquina que treinou.** O lugar de cada artefato vai para o banco SQLite como caminho absoluto, `file:///C:/Users/...`. Num contêiner Linux, esse caminho não existe.
- **O serviço no ar não deveria ler o registro.** Se alguém promove outro modelo (ADR-0021), o que decide os Pix mudaria debaixo do serviço, sem ninguém subir nada e sem registro de quando mudou.

A primeira tentativa no contêiner achou um terceiro problema. O `MLmodel` guarda o arquivo do modelo como código pelo caminho de quem registrou, `C:\Users\...\modelo_do_pix.py`. O MLflow acha esse arquivo pelo nome, com `os.path.basename`, que corta nas barras da máquina onde roda. No Linux, a barra invertida não corta nada: o MLflow emenda o caminho do Windows inteiro na pasta do modelo e não acha o arquivo. Todo modelo como código registrado no Windows tem esse defeito.

## Decisão

- **`antifraude-pix empacotar` copia o modelo de um apelido para uma pasta** (`tempo_real/pacote.py`), por padrão `dados/campeao`. Junto vai o `pacote.json`, que diz qual modelo é: o nome, a versão, o apelido e o corte do treino. O modelo registrado antes da Parte 7 não tem o corte, e o campo fica vazio.
- **O pacote guarda só o nome do arquivo do código.** No `MLmodel` copiado, o `model_code_path` vira `modelo_do_pix.py`, pela própria classe `Model` do MLflow. O caminho do Windows é cortado com `PureWindowsPath`, que entende as duas barras.
- **O pacote novo nasce ao lado e só então toma o lugar do antigo.** Quem abrir a pasta acha um ou o outro inteiro, e nada da troca fica para trás.
- **`servir --pacote` abre o pacote no lugar do registro.** O decisor sai do mesmo jeito que do registro: o caminho de um Pix só, se o modelo tem; senão, o `predict` do MLflow.
- **Trocar o modelo do serviço é empacotar de novo e subir o serviço de novo.** A versão vai na resposta de cada Pix e em cada métrica.

## Consequências

- **O contêiner decide sem o registro.** Com o estado de 1.981.634 Pix e o campeão versão 3 do pacote, ele fica saudável em uns 60 s. Fora do Docker são 26 s: a diferença é ler os dados pelo volume do Windows.
- **Os testes conferem o pacote fora da máquina do registro.** O pacote é aberto depois de apagados os artefatos para onde o registro aponta, e decide igual ao modelo treinado no mesmo corte. O `MLmodel` do pacote acha o código pelo nome. A pasta que não é pacote diz como fazer um.
- **O pacote tem 2,4 MB**: as árvores em texto, os ajustes em JSON, os tipos e a cópia do código.
- **O pacote é tirado do registro, e não o substitui.** O registro continua guardando as versões, os vereditos e as fichas; o pacote é o que vai para o ar.
