# ADR-0025 — A imagem, a entrega contínua e o painel

**Status:** aceita

## Contexto

Até a Parte 7, o serviço rodava na máquina de quem tem o projeto, com o `uv`. Falta o caminho até o ar:
- uma imagem que roda igual em qualquer lugar;
- um CI que só entrega a imagem que passou;
- um painel onde quem cuida do serviço vê, de relance, se ele está decidindo, quanto, e em quanto tempo.

## Decisão

- **A imagem leva o código e as dependências do `uv.lock`, e nada dos dados** (`Dockerfile`):
  - a construção em dois estágios: o primeiro instala o ambiente com o `uv sync --locked --no-dev` e o pacote sem depender da pasta `src`; o segundo copia só o ambiente;
  - as dependências numa camada própria, que só muda com o `uv.lock`;
  - o mesmo Python do projeto, 3.13.18, na imagem slim, com o OpenMP que o LightGBM usa;
  - um usuário sem poder, a saúde pelo `/saude`, com três minutos para o estado subir;
  - as imagens de base fixadas pelo digest, além da tag;
  - o `.dockerignore` como lista do que entra: o pyproject, o `uv.lock`, a versão do Python, a licença e o código.
- **Os dados entram como volume, só para leitura, em `/dados`**: a história do banco e o pacote do campeão (ADR-0023). O comando da imagem serve o pacote, para fora do contêiner.
- **O `compose.yaml` sobe o serviço e o que o observa, numa máquina só**:
  - o coletor do OpenTelemetry recebe da API os traços e as métricas, pelo OTLP;
  - o Prometheus guarda as métricas, e o Jaeger, os traços;
  - o Grafana mostra o painel.

  A API guarda um traço a cada dez pedidos, que é quase de graça (ADR-0022), e as métricas contam todos. As portas abrem só para a própria máquina: o Grafana deixa ver o painel sem senha, e nada ali é para a rede. As versões de cada imagem são fixadas, e de pelo menos duas semanas antes.
- **O coletor é a segunda barreira do dado pessoal.** Um `transform` deixa em cada span só os atributos da lista da telemetria (ADR-0022), e um teste confere que as duas listas são a mesma. A primeira tentativa usou o `redaction`, que também apagou os atributos do recurso, e o Jaeger recebeu os traços sem o nome do serviço.
- **O CI ganha o job da imagem, depois da verificação**:
  - a fumaça (`implantacao/fumaca.sh`) grava dados pequenos, registra e empacota o campeão, constrói a imagem, sobe o contêiner até ele ficar saudável, e manda pela reprodução os Pix dos dois últimos dias (ADR-0024);
  - no push para a `main`, a imagem que passou na fumaça vai para o GitHub Container Registry, com o commit no nome e como a última;
  - só esse job pode escrever no registro de imagens.

  A fumaça é a mesma em qualquer máquina com o Docker e o uv.
- **O painel vem do repositório** (`implantacao/grafana/painel.json`), e não da tela:
  - no alto, os números de agora: Pix por segundo, a parte segurada e a bloqueada, o p99 da decisão, os segurados esperando o cliente, e a versão do modelo;
  - embaixo, o tempo de cada um. Os aprovados ficam num gráfico, e os segurados e bloqueados em outro: as escalas diferem em centenas de vezes, e um eixo duplo enganaria;
  - cada ação tem a mesma cor em todo gráfico, e todo gráfico tem legenda;
  - o tempo da decisão por etapa, o dinheiro segurado e bloqueado (só a soma), a resposta do cliente ao segurado, a chance de golpe dos aprovados, e os pedidos recusados.

  Um teste confere que o painel consulta todas as métricas do serviço e só elas, e só pelos atributos da telemetria.

## Consequências

- **A imagem tem 1,5 GB.** Quase tudo é do MLflow completo, que traz o scikit-learn, o SciPy e o matplotlib. O serviço só precisa abrir o modelo, e o `mlflow-skinny` cortaria a imagem. Mas o projeto fixa o `mlflow` inteiro, e a troca pede outra medida.
- **O contêiner precisa de uns 2,7 GB de memória** para o estado de dois milhões de Pix.
- **A fumaça levou 1 min 11 s na máquina de desenvolvimento**, com as camadas da imagem em cache. No CI, a imagem é construída do zero a cada vez: a primeira construção aqui levou 1 min 47 s.
- **O painel mostra a operação, e não a qualidade do modelo.** O campeão de `dados` foi treinado com tudo, dezembro inclusive. A qualidade é a do monitor da Parte 7, com o rótulo que chega depois.
- **O Grafana sem senha e as portas na própria máquina são para estudo.** Num banco, o painel fica atrás do login da empresa, e o coletor, numa rede só dele.
