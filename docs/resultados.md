# Resultados da execução real

Executado em **06/10/2026**, Python **3.12.14**, Neo4j Community **5.26.0**. Os dados são MovieLens `ml-latest-small`, obtidos de GroupLens. O conjunto “latest” pode mudar: os hashes em `resultados/movielens-small.json` identificam exatamente os CSVs usados, em vez de depender apenas do nome do download.

Parâmetros fixos: seed **42**, K **20**, mínimo de **3** coavaliadores, top-**10** e relevância **nota ≥ 4**. Não houve busca de hiperparâmetros usando o teste.

## Dados e grafo

| Medida | Valor |
| --- | ---: |
| Filmes no catálogo | 9.742 |
| Usuários com avaliações de treino | 610 |
| Avaliações totais | 100.836 |
| Treino | 80.669 |
| Teste | 20.167 |
| Similaridades válidas, um sentido por par | 1.658.747 |
| Comédias avaliadas com nota ≥ 4 pelo usuário 1, no treino | 55 |

O conjunto inteiro de treino foi importado no Neo4j. As contagens reais de filmes, avaliações e relações `SIMILAR` foram comparadas com o manifesto. O top-10 do usuário 1 calculado em Cypher teve os mesmos IDs, ordem e notas que Python, com tolerância de `1e-10` nas notas. A execução sintética também conferiu usuários sem candidatos e usuário desconhecido.

Evidência: [`neo4j-validacao.json`](../resultados/neo4j-validacao.json), que inclui contagens e os filmes retornados. O tempo observado da importação e consultas foi 133,376 segundos; não é uma promessa de desempenho em outros computadores.

## Previsão das notas de teste

| Métrica | Valor |
| --- | ---: |
| Precisão (previsão ≥ 4) | 74,1777% |
| Recall (nota real ≥ 4) | 26,2454% |
| F1 | 38,7724% |
| MAE | 0,811236 |
| RMSE | 1,084411 |
| Cobertura KNN | 90,0630% |
| Previsões por KNN | 18.163 |
| Fallback pela média do filme | 1.221 |
| Fallback pela média do usuário | 783 |

Matriz de confusão:

| Nota real / previsão | Previsão ≥ 4 | Previsão < 4 |
| --- | ---: | ---: |
| Real ≥ 4 | TP = 2.571 | FN = 7.225 |
| Real < 4 | FP = 895 | TN = 9.476 |

O modelo tem precisão razoável entre as notas classificadas como positivas, mas deixa de recuperar muitos positivos. A avaliação inclui os fallbacks; 90,0630% dos casos tiveram vizinhos válidos para KNN.

## Ranking no catálogo completo de treino

| Métrica | Valor |
| --- | ---: |
| Usuários com relevantes no teste | 593 |
| Usuários sem relevantes no teste | 17 |
| Recomendações retornadas | 5.930 |
| Filmes relevantes no teste | 9.796 |
| Acertos no top-10 | 3 |
| Precisão@10 macro | 0,050590% |
| Recall@10 macro | 0,010063% |
| Precisão@10 micro | 0,050590% |
| Recall@10 micro | 0,030625% |

**O ranking teve desempenho baixo.** A precisão de 74,18% das notas não deve ser apresentada como precisão do top-10. O resultado usa todos os candidatos de treino, e não apenas os filmes conhecidos no teste.

Uma limitação da fórmula bruta exigida é que um candidato com um único vizinho positivo pode receber exatamente a nota desse vizinho. Isso favorece notas extremas e muitos empates em 5, inclusive em filmes pouco avaliados. O suporte mínimo de três coavaliadores para uma relação não garante três vizinhos no histórico do usuário-alvo. O clipping e o tratamento de correlações negativas também afetam a ordenação. Essas são explicações possíveis para o resultado, não causas isoladas demonstradas por ablação.

O teste contém apenas uma parte das preferências observadas: um filme fora das avaliações retidas não pode ser confirmado como irrelevante. Ainda assim, os três acertos mostram que esta configuração não recuperou bem os relevantes conhecidos. Melhorias futuras poderiam exigir maior evidência por previsão, regularizar similaridades ou usar regressão centrada em médias; seriam variantes a comparar em validação separada, e não resultados medidos nesta entrega. Mantivemos a fórmula e registramos a limitação sem ajustar parâmetros para favorecer o teste.

## Validação e reprodução

**19 testes passaram**, incluindo integração real com Neo4j. Os testes verificam leitura, split, Pearson manual, algoritmo esparso contra fórmula direta, variância zero, sinais, K, escala, fallbacks, ranking sem vistos, métricas conhecidas, ausência de vazamento, integridade CSV/JSON, requisições parametrizadas, erros HTTP/JSON e equivalência Python/Cypher.

Para reproduzir, siga o README e execute:

```sh
python -m recomendador preparar --dados data/ml-latest-small --seed 42 --k 20 --min-comuns 3
python -m recomendador avaliar --modelo artefatos --top 10 --limiar 4 --saida resultados/avaliacao-local.json
python -m recomendador importar --modelo artefatos --execucao ml42 --lote 5000
python -m recomendador consultar comedias --execucao ml42 --usuario 1
python -m recomendador consultar recomendar --execucao ml42 --usuario 1 --top 10
```

[`movielens-small.json`](../resultados/movielens-small.json) preserva também os resultados de ranking por usuário, a matriz de confusão, os parâmetros e os hashes. A configuração de GitHub Actions verifica a base sintética e integração, sem precisar baixar MovieLens a cada commit.
