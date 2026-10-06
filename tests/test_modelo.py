import unittest
from dataclasses import replace

from recomendador.dados import Avaliacao, Filme, carregar_dados
from recomendador.modelo import Modelo, pearson, Vizinho
from recomendador.avaliacao import avaliar
from tests.test_dados import FIXTURES


class TestModelo(unittest.TestCase):
    def setUp(self):
        filmes, treino = carregar_dados(FIXTURES)
        self.modelo = Modelo(filmes, treino, k=2, min_comuns=3).calcular_similaridades()

    def test_pearson_conhecido_positivo_negativo_constante(self):
        self.assertAlmostEqual(pearson([(1, 2), (2, 4), (3, 6)]), 1)
        self.assertAlmostEqual(pearson([(1, 6), (2, 4), (3, 2)]), -1)
        self.assertEqual(pearson([(1, 3), (2, 3), (3, 3)]), 0)
        self.assertEqual(pearson([(1, 2)]), 0)
        self.assertEqual(pearson([]), 0)

    def test_sparse_equivale_formula_com_medias_treino(self):
        m = self.modelo
        for i in m.por_filme:
            for j in m.por_filme:
                if i >= j:
                    continue
                comuns = sorted(m.por_filme[i].keys() & m.por_filme[j].keys())
                esperado = pearson([(m.por_filme[i][u], m.por_filme[j][u]) for u in comuns],
                                   m.medias[i], m.medias[j])
                obtido = m.similaridades.get(i, {}).get(j, (0, 0))[0]
                self.assertAlmostEqual(obtido, esperado if len(comuns) >= 3 else 0)
                if obtido:
                    self.assertEqual(m.similaridades[j][i], m.similaridades[i][j])
        self.assertNotIn(6, m.similaridades)  # filme de variância zero

    def test_usa_media_global_do_filme_nao_so_intersecao(self):
        filmes = {i: Filme(i, str(i), ()) for i in (1, 2)}
        treino = [Avaliacao(1, 1, 1, 0), Avaliacao(2, 1, 3, 0), Avaliacao(3, 1, 5, 0),
                  Avaliacao(1, 2, 2, 0), Avaliacao(2, 2, 4, 0)]
        m = Modelo(filmes, treino, min_comuns=2).calcular_similaridades()
        self.assertAlmostEqual(m.similaridades[1][2][0], 1 / (2 ** .5))

    def test_knn_formula_manual_com_sinais_e_k(self):
        p = self.modelo._prever(1, 4, [Vizinho(1, .8, 5, 4), Vizinho(2, -.2, 1, 4)])
        self.assertAlmostEqual(p.nota, 3.8)
        self.assertAlmostEqual(p.nota_bruta, 3.8)
        self.assertEqual(p.metodo, "knn")
        p = self.modelo._prever(1, 4, [Vizinho(1, -.8, 5, 4)])
        self.assertEqual(p.nota, .5)
        self.assertEqual(p.nota_bruta, -5)
        self.modelo.k = 1
        p = self.modelo._prever(1, 4, [Vizinho(1, .2, 5, 4), Vizinho(2, .9, 1, 4)])
        self.assertEqual(p.nota, 1)
        self.assertEqual(p.vizinhos[0].movie_id, 2)

    def test_fallbacks_e_filme_inexistente(self):
        self.assertEqual(self.modelo.prever(1, 6).metodo, "media_filme")
        self.assertEqual(self.modelo.prever(1, 7).metodo, "media_usuario")
        self.assertEqual(self.modelo.prever(999, 7).metodo, "media_global")
        with self.assertRaises(ValueError):
            self.modelo.prever(1, 999)

    def test_top_nao_recomenda_vistos_nem_filme_sem_treino(self):
        m = self.modelo
        top = m.recomendar(1, 99)
        self.assertEqual({p.movie_id for p in top}, {4, 5, 6})
        self.assertTrue(all(.5 <= p.nota <= 5 for p in top))
        self.assertEqual(top, m.recomendar(1, 99))
        self.assertEqual(m.recomendar(2), [])
        self.assertEqual(len(m.recomendar(999, 2)), 2)

    def test_poucos_coavaliadores_nao_geram_aresta(self):
        m = Modelo(self.modelo.filmes, self.modelo.treino, min_comuns=7).calcular_similaridades()
        self.assertEqual(dict(m.similaridades), {})

    def test_parametros_invalidos(self):
        for k, comuns in ((0, 3), (1, 1)):
            with self.assertRaises(ValueError):
                Modelo(self.modelo.filmes, self.modelo.treino, k, comuns)
        with self.assertRaises(ValueError):
            self.modelo.recomendar(1, 0)


class TestAvaliacao(unittest.TestCase):
    def test_confusao_precisao_recall_ranking_e_fallback_sem_vazamento(self):
        filmes = {i: Filme(i, str(i), ()) for i in (1, 2, 3, 4)}
        treino = [Avaliacao(1, i, nota, 0) for i, nota in ((1, 5), (2, 5), (3, 1), (4, 1))]
        teste = [Avaliacao(2, i, nota, 1) for i, nota in ((1, 5), (2, 1), (3, 5), (4, 1))]
        m = Modelo(filmes, treino).calcular_similaridades()
        resultado = avaliar(m, teste, top_n=2)
        notas, ranking = resultado["notas"], resultado["ranking"]
        self.assertEqual([notas[c] for c in ("tp", "fp", "fn", "tn")], [1, 1, 1, 1])
        self.assertEqual(notas["precisao"], .5)
        self.assertEqual(notas["recall"], .5)
        self.assertEqual(notas["mae"], 2)
        self.assertAlmostEqual(notas["rmse"], 8 ** .5)
        self.assertEqual(notas["cobertura_knn"], 0)
        self.assertEqual(ranking["precisao_macro"], .5)
        self.assertEqual(ranking["recall_macro"], .5)
        self.assertEqual(ranking["relevantes"], 2)
        # Alterar rótulos de teste não altera uma única recomendação do modelo.
        top_antes = m.recomendar(2)
        avaliar(m, [replace(a, nota=1) for a in teste])
        self.assertEqual(top_antes, m.recomendar(2))
        with self.assertRaises(ValueError):
            avaliar(m, treino)

    def test_sem_positivos_retornos_zero(self):
        filmes = {1: Filme(1, "1", ())}
        m = Modelo(filmes, [Avaliacao(1, 1, 1, 0)])
        resultado = avaliar(m, [Avaliacao(2, 1, 1, 0)])
        self.assertEqual(resultado["notas"]["precisao"], 0)
        self.assertEqual(resultado["notas"]["recall"], 0)
        self.assertEqual(resultado["ranking"]["usuarios_sem_relevantes"], 1)
        self.assertEqual(resultado["ranking"]["recall_macro"], 0)
