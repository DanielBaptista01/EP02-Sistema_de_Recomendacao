import json
import os
import tempfile
import threading
import unittest
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from recomendador.artefatos import salvar_modelo
from recomendador.dados import carregar_dados
from recomendador.modelo import Modelo
from recomendador.neo4j import (ClienteNeo4j, ErroNeo4j, importar, verificar_pronto,
                               COMEDIAS, RECOMENDACOES, ESTATISTICAS, lotes)
from tests.test_dados import FIXTURES


class TestHTTP(unittest.TestCase):
    def test_query_api_parametros_json_e_erros_202(self):
        pedidos = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                dados = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                pedidos.append((self.path, dados, self.headers.get("Authorization")))
                self.send_response(202)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                resposta = ({"errors": [{"code": "Neo.TestError", "message": "consulta inválida"}]}
                            if dados["statement"] == "ERRO" else
                            {"data": {"fields": ["numero"], "values": [[42]]}})
                self.wfile.write(json.dumps(resposta).encode())

            def log_message(self, *args):
                pass

        servidor = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=servidor.serve_forever, daemon=True)
        thread.start()
        try:
            cliente = ClienteNeo4j(f"http://127.0.0.1:{servidor.server_port}", senha="somente-teste")
            self.assertEqual(cliente.executar("RETURN $x", {"x": "'não concatenar'"}), [{"numero": 42}])
            self.assertEqual(pedidos[0][0], "/db/neo4j/query/v2")
            self.assertEqual(pedidos[0][1]["parameters"], {"x": "'não concatenar'"})
            self.assertTrue(pedidos[0][2].startswith("Basic "))
            with self.assertRaisesRegex(ErroNeo4j, "Neo.TestError"):
                cliente.executar("ERRO")
        finally:
            servidor.shutdown()
            servidor.server_close()
            thread.join()

    def test_lotes_e_validacao_endereco(self):
        self.assertEqual(list(lotes(range(5), 2)), [[0, 1], [2, 3], [4]])
        with self.assertRaises(ValueError):
            list(lotes([], 0))
        for url in ("bolt://localhost:7687", "http://example.org", "http://a:b@localhost", "http://localhost/?x=1"):
            with self.subTest(url=url), self.assertRaises(ValueError):
                ClienteNeo4j(url, senha="teste")


@unittest.skipUnless(os.environ.get("EP02_TEST_NEO4J") == "1", "defina EP02_TEST_NEO4J=1 para integração real")
class TestIntegracaoNeo4j(unittest.TestCase):
    def test_importacao_consultas_e_equivalencia_python_cypher(self):
        cliente = ClienteNeo4j()
        filmes, treino = carregar_dados(FIXTURES)
        modelo = Modelo(filmes, treino, k=2).calcular_similaridades()
        with tempfile.TemporaryDirectory() as pasta:
            manifesto = salvar_modelo(pasta, modelo, [], 42)
        run = "test_" + uuid.uuid4().hex
        try:
            importar(cliente, modelo, manifesto, run, tamanho_lote=5)
            verificar_pronto(cliente, run)
            parametros = {"run": run, "user": 1, "n": 10, "k": modelo.k, "limiar": 4.0}
            comedias = cliente.executar(COMEDIAS, parametros)
            self.assertEqual(comedias[0]["quantidade"], 2)
            self.assertEqual({f["movie_id"] for f in comedias[0]["filmes"]}, {1, 3})
            estatisticas = cliente.executar(ESTATISTICAS, parametros)
            self.assertEqual(sum(f["avaliacoes"] for f in estatisticas), len(treino))
            for u in (1, 2, 999):
                parametros["user"] = u
                grafo = cliente.executar(RECOMENDACOES, parametros)
                local = modelo.recomendar(u, 10)
                self.assertEqual([f["movie_id"] for f in grafo], [p.movie_id for p in local])
                for f, p in zip(grafo, local):
                    self.assertAlmostEqual(f["nota"], p.nota)
                    self.assertEqual(f["metodo"], p.metodo)
                    self.assertEqual(f["vizinhos"], len(p.vizinhos))
            arestas = cliente.executar("MATCH (:EP02Filme {execucao: $run})-[s:SIMILAR]->() RETURN count(s) AS n",
                                       {"run": run})
            self.assertEqual(arestas[0]["n"], manifesto["similaridades"])
            with self.assertRaises(ValueError):
                importar(cliente, modelo, manifesto, run)
        finally:
            # Somente os dados sintéticos desta execução com UUID, nunca o banco todo.
            cliente.executar("MATCH (n) WHERE (n:EP02Filme OR n:EP02Usuario) AND n.execucao = $run DETACH DELETE n",
                             {"run": run})
            cliente.executar("MATCH (e:EP02Execucao {id: $run}) DELETE e", {"run": run})
