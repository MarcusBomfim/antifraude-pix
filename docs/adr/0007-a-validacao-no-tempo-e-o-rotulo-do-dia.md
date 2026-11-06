# ADR-0007 — A validação no tempo, e o rótulo do dia: o modelo aprende com o que o banco sabia

**Status:** aceita

## Contexto

O jeito comum de medir um modelo é sortear uma parte dos dados para o teste. Num modelo de fraude, isso mede errado, e para melhor:
- o golpe de dezembro ensina o modelo a pegar o de outubro, e a mesma laranja aparece no treino e no teste;
- o rótulo do treino é o de hoje, com as contestações que chegaram semanas depois de cada Pix. No dia em que o banco treina, ele ainda não as tem.

O rótulo é o problema maior. O banco só sabe que um Pix foi golpe quando a vítima contesta (ADR-0004):
- a invasão de conta, em horas;
- a venda falsa, em semanas, e quase metade nunca.

No dia do treino, o golpe da última semana parece legítimo, e o golpe que ninguém contesta parece legítimo para sempre.

## Decisão

- **Uma dobra por mês, com o treino crescendo.** Em cada corte, no primeiro dia do mês, o modelo treina com tudo o que veio antes e é testado no mês do corte. Na simulação padrão, de julho a dezembro:
  - os primeiros 30 dias são o aquecimento: ali, todo recebedor é novo, porque a história começa ali, e nenhum Pix deles entra no treino nem no teste;
  - agosto é o primeiro mês de treino;
  - setembro é a validação, o mês em que os parâmetros dos modelos foram escolhidos (ADR-0008), e não entra no teste;
  - outubro, novembro e dezembro são o teste: três modelos para cada mês, cada um treinado na véspera.
- **O treino aprende com o rótulo do dia.** O Pix é golpe se foi contestado antes do corte, e mais nada: a janela de treino corta a visão do banco no corte (`visao.ate`, ADR-0005), e não toca na verdade do simulador.
- **O treino não espera o rótulo amadurecer.** Esperar deixaria de fora os dias mais recentes, cujos golpes ainda não foram contestados. A espera ficou como parâmetro, e foi medida com o LightGBM, nos três meses de teste:

  | O treino aprende com | Precisão média | Na venda falsa |
  |---|--:|--:|
  | o rótulo do dia, sem esperar | 0,934 | 0,787 |
  | o rótulo do dia, esperando 14 dias | 0,932 | 0,789 |
  | o rótulo do dia, esperando 28 dias | 0,929 | 0,792 |
  | a verdade, que nenhum banco tem | 0,949 | 0,864 |

  Esperar não paga o que custa: o golpe mais recente vale mais que o rótulo mais limpo. A distância para a verdade é o preço do rótulo atrasado, e quase todo ele está na venda falsa.
- **A medida é contra a verdade do simulador.** Num banco, seria o rótulo maduro, meses depois. Aqui é a verdade, que só a avaliação lê: o pacote dos modelos não importa o simulador, e um teste confere isso.

## Consequências

- **O modelo aprende com golpe marcado como legítimo**: o da última semana e o nunca contestado. Ele aprende assim mesmo, porque o golpe novo se parece com o antigo que já foi contestado.
- **A venda falsa paga o preço**: é o golpe que o banco aprende menos, porque é o que ele fica sabendo mais tarde. É a primeira candidata a outra fonte de rótulo, como a denúncia de quem recebeu ou a marcação de fraude no DICT.
- **A avaliação não vaza, e um teste prova isso de ponta a ponta**: cortar a simulação no meio do mês de teste não muda a pontuação de nenhum Pix anterior ao corte.
- **O treino de cada mês é feito do zero.** Na Parte 7, o monitoramento compara o modelo do mês com o candidato, e decide quando trocar.
