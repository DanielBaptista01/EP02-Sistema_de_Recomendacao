"""Cliente da Query API Neo4j 5.26+, usando apenas urllib (sem driver externo)."""

import base64
import json
import os
import re
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlsplit
from urllib.request import Request, urlopen


class ErroNeo4j(RuntimeError):
    pass


class ClienteNeo4j:
    def __init__(self, url=None, usuario=None, senha=None, database=None, timeout=120):
        url = url or os.environ.get("NEO4J_HTTP_URL", "http://localhost:7474")
        usuario = usuario or os.environ.get("NEO4J_USER", "neo4j")
        senha = senha if senha is not None else os.environ.get("NEO4J_PASSWORD")
        database = database or os.environ.get("NEO4J_DATABASE", "neo4j")
        partes = urlsplit(url)
        if partes.scheme not in ("http", "https") or not partes.hostname or partes.username:
            raise ValueError("NEO4J_HTTP_URL deve ser uma URL http(s) sem credenciais")
        if partes.query or partes.fragment or partes.path not in ("", "/"):
            raise ValueError("NEO4J_HTTP_URL deve conter apenas origem e porta")
        if partes.scheme == "http" and partes.hostname not in ("localhost", "127.0.0.1", "::1"):
            raise ValueError("use HTTPS ao acessar um Neo4j remoto")
        if not senha:
            raise ValueError("defina NEO4J_PASSWORD com a senha do seu Neo4j")
        self.endpoint = f"{url.rstrip('/')}/db/{quote(database, safe='')}/query/v2"
        self.authorization = "Basic " + base64.b64encode(f"{usuario}:{senha}".encode()).decode()
        self.timeout = timeout

    def executar(self, cypher, parametros=None):
        corpo = json.dumps({"statement": cypher, "parameters": parametros or {}},
                           allow_nan=False).encode()
        pedido = Request(self.endpoint, data=corpo, method="POST", headers={
            "Authorization": self.authorization, "Content-Type": "application/json",
            "Accept": "application/json"})
        try:
            with urlopen(pedido, timeout=self.timeout) as resposta:
                resultado = json.load(resposta)
        except HTTPError as erro:
            # Não ecoa corpo da requisição nem credenciais.
            raise ErroNeo4j(f"Neo4j respondeu HTTP {erro.code}; verifique endereço, senha e Query API") from None
        except (URLError, TimeoutError, OSError) as erro:
            raise ErroNeo4j(f"não foi possível acessar Neo4j: {erro.reason if isinstance(erro, URLError) else type(erro).__name__}") from None
        except (ValueError, UnicodeError):
            raise ErroNeo4j("Neo4j devolveu uma resposta JSON inválida") from None
        if resultado.get("errors"):
            erros = "; ".join(e.get("code", "erro") + ": " + e.get("message", "")
                              for e in resultado["errors"])
            raise ErroNeo4j(erros)
        dados = resultado.get("data", {})
        return [dict(zip(dados.get("fields", []), valores)) for valores in dados.get("values", [])]


def validar_execucao(execucao):
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", execucao):
        raise ValueError("execução deve ter 1–64 letras ASCII, números, '_' ou '-'")


def lotes(linhas, tamanho):
    if tamanho < 1:
        raise ValueError("tamanho do lote deve ser positivo")
    lote = []
    for linha in linhas:
        lote.append(linha)
        if len(lote) == tamanho:
            yield lote
            lote = []
    if lote:
        yield lote


SCHEMA = (
    "CREATE CONSTRAINT ep02_execucao IF NOT EXISTS FOR (n:EP02Execucao) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT ep02_usuario IF NOT EXISTS FOR (n:EP02Usuario) REQUIRE (n.execucao, n.userId) IS UNIQUE",
    "CREATE CONSTRAINT ep02_filme IF NOT EXISTS FOR (n:EP02Filme) REQUIRE (n.execucao, n.movieId) IS UNIQUE",
)


def importar(cliente, modelo, manifesto, execucao, tamanho_lote=1000):
    validar_execucao(execucao)
    if tamanho_lote < 1:
        raise ValueError("tamanho do lote deve ser positivo")
    existente = cliente.executar("MATCH (e:EP02Execucao {id: $run}) RETURN e.pronto AS pronto",
                                 {"run": execucao})
    if existente:
        raise ValueError("execução já existe; escolha outro --execucao para preservar os dados anteriores")
    for cypher in SCHEMA:
        cliente.executar(cypher)
    parametros = {"run": execucao, "k": modelo.k, "min_comuns": modelo.min_comuns,
                  "media": modelo.media_global, "seed": manifesto["seed"],
                  "assinatura": manifesto["sha256_treino"]}
    cliente.executar("CREATE (e:EP02Execucao {id: $run, pronto: false, k: $k, "
                     "minComuns: $min_comuns, mediaGlobal: $media, seed: $seed, "
                     "sha256Treino: $assinatura})", parametros)

    def inserir(cypher, linhas):
        for lote in lotes(linhas, tamanho_lote):
            cliente.executar(cypher, {"run": execucao, "rows": lote})

    inserir("UNWIND $rows AS row MERGE (f:EP02Filme {execucao: $run, movieId: row.id}) "
            "SET f.titulo = row.titulo, f.generos = row.generos, f.mediaTreino = row.media, "
            "f.quantidadeTreino = row.quantidade",
            ({"id": f.movie_id, "titulo": f.titulo, "generos": list(f.generos),
              "media": modelo.medias.get(f.movie_id),
              "quantidade": len(modelo.por_filme.get(f.movie_id, {}))}
             for f in modelo.filmes.values()))
    inserir("UNWIND $rows AS row MERGE (u:EP02Usuario {execucao: $run, userId: row.id})",
            ({"id": u} for u in sorted(modelo.por_usuario)))
    inserir("UNWIND $rows AS row MATCH (u:EP02Usuario {execucao: $run, userId: row.user}) "
            "MATCH (f:EP02Filme {execucao: $run, movieId: row.movie}) "
            "MERGE (u)-[r:AVALIOU]->(f) SET r.nota = row.nota, r.timestamp = row.timestamp",
            ({"user": a.user_id, "movie": a.movie_id, "nota": a.nota, "timestamp": a.timestamp}
             for a in modelo.treino))
    inserir("UNWIND $rows AS row MATCH (i:EP02Filme {execucao: $run, movieId: row.origem}) "
            "MATCH (j:EP02Filme {execucao: $run, movieId: row.destino}) "
            "MERGE (i)-[s:SIMILAR]->(j) SET s.correlacao = row.correlacao, s.comuns = row.comuns",
            modelo.arestas())
    cliente.executar("MATCH (e:EP02Execucao {id: $run}) SET e.pronto = true", {"run": execucao})
    return {"execucao": execucao, "filmes": len(modelo.filmes),
            "usuarios": len(modelo.por_usuario), "avaliacoes": len(modelo.treino),
            "similaridades": manifesto["similaridades"]}


def verificar_pronto(cliente, execucao):
    validar_execucao(execucao)
    linhas = cliente.executar("MATCH (e:EP02Execucao {id: $run}) RETURN e.pronto AS pronto, e.k AS k",
                               {"run": execucao})
    if not linhas or not linhas[0]["pronto"]:
        raise ValueError("execução não existe ou importação não foi concluída")
    return linhas[0]


COMEDIAS = """
MATCH (u:EP02Usuario {execucao: $run, userId: $user})-[r:AVALIOU]->(f:EP02Filme)
WHERE 'Comedy' IN f.generos AND r.nota >= $limiar
RETURN count(f) AS quantidade, collect({movie_id: f.movieId, titulo: f.titulo, nota: r.nota}) AS filmes
"""

ESTATISTICAS = """
MATCH (f:EP02Filme {execucao: $run})
OPTIONAL MATCH (:EP02Usuario {execucao: $run})-[r:AVALIOU]->(f)
RETURN f.movieId AS movie_id, f.titulo AS titulo, count(r) AS avaliacoes, avg(r.nota) AS media
ORDER BY avaliacoes DESC, movie_id ASC LIMIT $n
"""

RECOMENDACOES = """
MATCH (e:EP02Execucao {id: $run, pronto: true})
MATCH (i:EP02Filme {execucao: $run})
WHERE i.quantidadeTreino > 0
  AND NOT EXISTS { MATCH (:EP02Usuario {execucao: $run, userId: $user})-[:AVALIOU]->(i) }
CALL (i) {
  OPTIONAL MATCH (i)-[s:SIMILAR]-(j:EP02Filme)<-[r:AVALIOU]-
                 (:EP02Usuario {execucao: $run, userId: $user})
  WITH j, s, r ORDER BY s.correlacao DESC, s.comuns DESC, j.movieId ASC LIMIT $k
  RETURN sum(s.correlacao * r.nota) AS numerador,
         sum(abs(s.correlacao)) AS denominador, count(s) AS vizinhos
}
WITH i, vizinhos, CASE WHEN denominador > 0.000000000001
  THEN numerador / denominador ELSE i.mediaTreino END AS bruta,
  CASE WHEN denominador > 0.000000000001 THEN 'knn' ELSE 'media_filme' END AS metodo
RETURN i.movieId AS movie_id, i.titulo AS titulo,
       CASE WHEN bruta < 0.5 THEN 0.5 WHEN bruta > 5.0 THEN 5.0 ELSE bruta END AS nota,
       metodo, vizinhos, i.quantidadeTreino AS avaliacoes_treino
ORDER BY nota DESC, avaliacoes_treino DESC, movie_id ASC LIMIT $n
"""
