import tempfile
import unittest
from pathlib import Path

from recomendador.dados import carregar_dados, dividir, assinatura

FIXTURES = Path(__file__).parent / "fixtures"


class TestDados(unittest.TestCase):
    def test_csv_utf8_multigenero_sem_genero(self):
        filmes, avaliacoes = carregar_dados(FIXTURES)
        self.assertEqual(filmes[3].generos, ("Adventure", "Comedy"))
        self.assertEqual(filmes[7].generos, ())
        self.assertIn("Comédia", filmes[1].titulo)
        self.assertEqual(len(avaliacoes), 33)

    def test_split_80_20_sem_vazamento_e_reproduzivel(self):
        _, avaliacoes = carregar_dados(FIXTURES)
        original = list(avaliacoes)
        treino, teste = dividir(avaliacoes)
        self.assertEqual(len(teste), 7)
        self.assertFalse(set(treino) & set(teste))
        self.assertEqual(set(treino) | set(teste), set(avaliacoes))
        self.assertEqual((treino, teste), dividir(list(reversed(avaliacoes))))
        self.assertEqual(avaliacoes, original)
        self.assertNotEqual(dividir(avaliacoes, seed=43), (treino, teste))
        self.assertEqual(assinatura(avaliacoes), assinatura(reversed(avaliacoes)))

    def test_split_invalido(self):
        for entrada, fracao in (([], .2), ([1], .2), ([1, 2], 0), ([1, 2], 1)):
            with self.assertRaises(ValueError):
                dividir(entrada, fracao)

    def test_validacao_rejeita_csv_invalido(self):
        casos = (
            "1,1,nan,100", "1,1,6,100", "1,99,4,100", "0,1,4,100", "1,1,4,-1",
            "1,1,4,100\n1,1,5,101", "abc,1,4,100",
        )
        with tempfile.TemporaryDirectory() as pasta:
            p = Path(pasta)
            (p / "movies.csv").write_bytes((FIXTURES / "movies.csv").read_bytes())
            for caso in casos:
                with self.subTest(caso=caso):
                    (p / "ratings.csv").write_text("userId,movieId,rating,timestamp\n" + caso,
                                                   encoding="utf-8")
                    with self.assertRaises(ValueError):
                        carregar_dados(p)
            (p / "ratings.csv").write_text("coluna_errada\n1", encoding="utf-8")
            with self.assertRaises(ValueError):
                carregar_dados(p)
