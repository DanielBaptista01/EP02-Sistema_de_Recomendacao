**Nomes:**Daniel Santos Baptista e Isaias Maia de Oliveira

# EP02 — Sistema de Recomendação de Filmes

Filtragem colaborativa **item-item** com MovieLens, **Pearson**, regressão **KNN** e banco de grafos **Neo4j**. A implementação usa **somente a biblioteca padrão do Python**: CSVs, cálculos, métricas e comunicação HTTP foram implementados neste projeto, sem pandas, NumPy, scikit-learn, Surprise, APOC, GDS ou driver externo.

Enunciado: [Sistemas de Recomendação de Filmes](https://crivelaro.notion.site/Sistemas-de-Recomenda-o-de-Filmes-287a6ec0abcd8096ae60c199e1e6278d). Consulte a [modelagem e metodologia](docs/metodologia.md), os [resultados reais](docs/resultados.md) e as [referências e autoria](docs/referencias.md).

## Critérios atendidos

| Exigência | Implementação |
| --- | --- |
| Apenas `movies.csv` e `ratings.csv` | Leitura e validação em `recomendador/dados.py` |
| Nós, arestas e atributos | [Modelagem](docs/metodologia.md#modelagem-do-grafo), com constraints |
| Script de inserção em Python | Comando `importar`, com lotes e parâmetros |
| Pearson entre filmes | Algoritmo esparso em `recomendador/modelo.py` |
| Similaridades no grafo | Relações `SIMILAR` com correlação e usuários comuns |
| Regressão KNN | Soma ponderada por Pearson e denominador absoluto |
| Consultas e agregações em Python | Comédias com nota ≥ 4, estatísticas e top-N em Cypher |
| Treino 80%, teste 20% | Split reproduzível, sem vazamento de notas |
| Precisão e recall | Métricas das notas e de ranking, com resultados registrados |
| Sem bibliotecas não autorizadas | Nenhuma dependência Python de terceiros |

## Pré-requisitos

- Python **3.11+** (validado com 3.12).
- Neo4j Community **5.26.0** local, por Desktop, instalação oficial ou Docker. A **Query API** precisa estar em `http://localhost:7474`.
- [MovieLens Small](https://grouplens.org/datasets/movielens/latest/): descompacte `ml-latest-small.zip` e coloque **apenas `movies.csv` e `ratings.csv`** em `data/ml-latest-small/`.

Não é necessário instalar pacotes com `pip`. Os CSVs completos e os artefatos grandes ficam fora do Git. Os testes incluem uma base **sintética própria**, e `resultados/` contém evidências da execução real.

## Executar

Abra o terminal na raiz do repositório. Se seu sistema usa `python3`, substitua `python` nos comandos.

### 1. Treinar e avaliar

```sh
python -m recomendador preparar --dados data/ml-latest-small --saida artefatos --seed 42 --k 20 --min-comuns 3
python -m recomendador avaliar --modelo artefatos --top 10 --limiar 4 --saida resultados/avaliacao-local.json
python -m recomendador recomendar --modelo artefatos --usuario 1 --top 10
python -m recomendador prever --modelo artefatos --usuario 1 --filme 318
```

`preparar` gera `movies.csv`, `ratings.csv` (**somente treino**), `teste.csv`, `similaridades.csv` e `manifesto.json`, com parâmetros, contagens e hashes. Nenhuma nota de teste entra no treinamento. Uma saída existente exige outra pasta ou `--sobrescrever`.

As recomendações locais mostram nota, método e vizinhos. Filmes já avaliados **no treino** não são recomendados. Essa etapa permite validar os cálculos antes de iniciar o banco; a implementação inclui a etapa Neo4j a seguir.

### 2. Configurar o Neo4j

No Neo4j Desktop, inicie uma instância 5.26 e configure a senha do usuário `neo4j`. Se usar Docker, defina a senha antes de executar `docker compose up -d`.

**PowerShell 7 / Windows:**

```powershell
$env:NEO4J_PASSWORD = Read-Host 'Senha do Neo4j' -MaskInput
$env:NEO4J_HTTP_URL = 'http://localhost:7474'
$env:NEO4J_USER = 'neo4j'
$env:NEO4J_DATABASE = 'neo4j'
# Somente se estiver usando Docker:
docker compose up -d
```

No Windows PowerShell 5.1, substitua a primeira linha por:

```powershell
$senha = Read-Host 'Senha do Neo4j' -AsSecureString
$env:NEO4J_PASSWORD = [System.Net.NetworkCredential]::new('', $senha).Password
```

**Bash / Linux / macOS:**

```bash
read -rsp 'Senha do Neo4j: ' NEO4J_PASSWORD
export NEO4J_PASSWORD
export NEO4J_HTTP_URL=http://localhost:7474
export NEO4J_USER=neo4j
export NEO4J_DATABASE=neo4j
# Somente se estiver usando Docker:
docker compose up -d
```

Espere o Neo4j estar pronto. Abra `http://localhost:7474` para inspecionar o grafo no navegador. A senha é lida do ambiente e não é gravada no projeto.

### 3. Inserir e consultar o grafo

```sh
python -m recomendador importar --modelo artefatos --execucao ml42 --lote 5000
python -m recomendador consultar comedias --execucao ml42 --usuario 1 --limiar 4
python -m recomendador consultar estatisticas --execucao ml42 --top 10
python -m recomendador consultar recomendar --execucao ml42 --usuario 1 --top 10
```

A importação cria filmes, usuários, avaliações de treino e similaridades em uma execução isolada, sem remover outros dados. Uma execução existente é preservada: para reimportar, escolha outro identificador, por exemplo `ml42_v2`. Uma importação interrompida permanece incompleta e não pode ser consultada como modelo pronto.

`consultar recomendar` calcula o KNN **no Neo4j, em Cypher**, a partir das relações importadas. Não lê o ranking do modelo local. Os testes verificam equivalência com a versão Python. As contagens das consultas descrevem o **grafo de treino**; as notas de teste ficam separadas em disco.

### 4. Testes

```sh
python -m unittest discover -v
```

Para incluir a integração real com seu Neo4j local, configure `NEO4J_PASSWORD` e:

```powershell
# PowerShell
$env:EP02_TEST_NEO4J = '1'
python -m unittest discover -v
```

```bash
# Bash
EP02_TEST_NEO4J=1 python -m unittest discover -v
```

O teste real cria uma execução sintética com UUID e limpa **somente seus próprios dados de teste** ao terminar. Verifica inserção, agregações, comédias e equivalência Python/Cypher. O GitHub Actions também está configurado para executar essa integração.

## Arquivos

| Caminho | Responsabilidade |
| --- | --- |
| `recomendador/dados.py` | CSVs, validação e split |
| `recomendador/modelo.py` | Pearson, KNN, fallbacks e ranking |
| `recomendador/avaliacao.py` | Precisão, recall, MAE e RMSE |
| `recomendador/artefatos.py` | CSV/JSON e integridade |
| `recomendador/neo4j.py` | Cliente HTTP, schema, importação e Cypher |
| `recomendador/__main__.py` | Interface de linha de comando |
| `tests/` | Testes unitários, base sintética e integração |
| `docs/` e `resultados/` | Metodologia, referências e evidências |
| `compose.yaml` | Neo4j local opcional |

Use `python -m recomendador --help` ou `python -m recomendador <comando> --help`. Para HTTP 401, confira a senha. Para HTTP 404, confira versão, database e Query API. `NEO4J_HTTP_URL` é uma origem HTTP, não `bolt://localhost:7687`. Para servidores remotos, use HTTPS.
