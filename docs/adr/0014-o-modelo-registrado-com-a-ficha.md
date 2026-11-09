# ADR-0014 — O modelo registrado, com a sua ficha

**Status:** aceita

## Contexto

Até a Parte 4, o modelo só existia dentro da avaliação: treinado em cada mês, medido, e jogado fora. A API da Parte 6 precisa carregar um modelo pronto, e a Parte 7 precisa trocá-lo por outro quando o outro for melhor.

As duas coisas pedem um modelo com nome, versão e apelido, e não um arquivo solto numa pasta. Quem for usá-lo, revisá-lo ou trocá-lo precisa saber o que ele faz, com que dados aprendeu, como foi medido e onde erra. É o que a ficha do modelo responde, a *model card* de Mitchell e outros (2019).

## Decisão

- **O modelo registrado é o LightGBM treinado com tudo o que o banco sabia** na véspera de 1º de janeiro de 2026. São 1.952.800 Pix, dos quais 1.470 o banco já sabia que eram golpe, e a calibração é em 317.734 Pix maduros que o modelo não viu (`rastreio/registro.py`).
- **Um pacote só faz tudo** (`rastreio/modelo_do_pix.py`):
  - entram as 32 variáveis de um Pix;
  - saem a pontuação, a probabilidade calibrada, a decisão pelo custo (ADR-0010) e a explicação em português (ADR-0012).

  Quem usa não junta as peças.
- **O modelo é código, e não pickle.** O MLflow carrega o arquivo `modelo_do_pix.py`, e o que o modelo aprendeu vai em formatos que se leem sem executar nada:
  - as árvores, no texto em que o próprio LightGBM as grava;
  - a reta da calibração, os custos e a lista das variáveis, num JSON;
  - o tipo de cada variável, numa tabela vazia em Parquet, que guarda até as categorias.

  O pickle executa código ao ser carregado, e o próprio MLflow recomenda o modelo como código. Um teste confere que o modelo carregado do registro dá a mesma probabilidade que o modelo treinado, até a 12ª casa, e a mesma decisão.
- **A assinatura diz o que entra e o que sai**: as 32 variáveis com o tipo de cada uma, e quais podem vir vazias. O minuto desde o aparelho novo, por exemplo, vem vazio quando o aparelho é o de sempre. Ela sai de um exemplo que mostra cada variável cheia e, quando pode, vazia.
- **O registro guarda o código do pacote e a versão exata de cada biblioteca**, para o modelo carregar igual em outra máquina.
- **O modelo tem nome e apelido**: `antifraude-pix`, com o apelido `campeao` na versão que decide. A API carrega `models:/antifraude-pix@campeao`, e não a versão 1. Trocar o modelo é mudar o apelido.
- **A ficha sai das medidas**, e não de um texto escrito à mão (`rastreio/ficha.py`). Ela traz:
  - o que o modelo faz, e para que ele não serve;
  - os dados, a entrada e o modelo;
  - a medida mês a mês, e cada golpe;
  - a decisão, os erros por idade, e onde ele erra;
  - como usar.

  Ela é a descrição da versão no registro, um artefato da corrida, e o arquivo [`docs/ficha-do-modelo.md`](../ficha-do-modelo.md).

## Consequências

- **Carregar o modelo é uma linha**: `mlflow.pyfunc.load_model("models:/antifraude-pix@campeao")`.
- **A medida da ficha é a do processo, e não a do modelo registrado.** O modelo registrado aprendeu com todos os meses, e não sobra mês para testá-lo. A ficha mostra o mesmo processo refeito em cada mês de teste, que é a melhor estimativa de como ele vai se sair em janeiro.
- **Explicar um lote todo aprovado quebrava.** O LightGBM recusa uma matriz sem linhas, e a explicação pedia as razões de nenhum Pix. Na API, que decide um Pix por vez e quase sempre aprova, isso aconteceria quase sempre. Foi corrigido, com um teste para cada modelo.
- **Registrar de novo cria a versão seguinte e leva o apelido para ela.** Na Parte 7, o candidato só ganha o apelido depois de ganhar do campeão na mesma medida.
