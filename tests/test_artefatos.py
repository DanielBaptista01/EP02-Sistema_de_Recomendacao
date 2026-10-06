import tempfile
import unittest
from pathlib import Path

from recomendador.artefatos import carregar_modelo, carregar_teste, salvar_modelo
from recomendador.dados import carregar_dados, dividir
from recomendador.modelo import Modelo
from tests.test_dados import FIXTURES


class TestArtefatos(unittest.TestCase):
    def test_roundtrip_modelo_e_teste(self):
        filmes, avaliacoes = carregar_dados(FIXTURES)
        treino, teste = dividir(avaliacoes)
        modelo = Modelo(filmes, treino).calcular_similaridades()
        with tempfile.TemporaryDirectory() as pasta:
            manifesto = salvar_modelo(pasta, modelo, teste, 42)
            recuperado, m = carregar_modelo(pasta)
            self.assertEqual(m, manifesto)
            self.assertEqual(list(recuperado.arestas()), list(modelo.arestas()))
            self.assertEqual(recuperado.recomendar(1), modelo.recomendar(1))
            self.assertEqual(carregar_teste(pasta, m), teste)
            with (Path(pasta) / "teste.csv").open("a") as arquivo:
                arquivo.write("999,1,5,1\n")
            with self.assertRaises(ValueError):
                carregar_teste(pasta, m)

    def test_detecta_artefatos_modificados(self):
        filmes, avaliacoes = carregar_dados(FIXTURES)
        treino, teste = dividir(avaliacoes)
        modelo = Modelo(filmes, treino).calcular_similaridades()
        for nome in ("movies.csv", "ratings.csv", "similaridades.csv"):
            with self.subTest(nome=nome), tempfile.TemporaryDirectory() as pasta:
                salvar_modelo(pasta, modelo, teste, 42)
                with (Path(pasta) / nome).open("a") as arquivo:
                    arquivo.write("999,1,5,1\n")
                with self.assertRaises(ValueError):
                    carregar_modelo(pasta)
