"""Pearson item-item e regressão KNN; nenhum algoritmo delegado a bibliotecas."""

import heapq
import math
from collections import defaultdict
from dataclasses import dataclass
from statistics import fmean


@dataclass(frozen=True)
class Vizinho:
    movie_id: int
    correlacao: float
    nota: float
    comuns: int


@dataclass(frozen=True)
class Previsao:
    movie_id: int
    nota: float
    metodo: str
    vizinhos: tuple[Vizinho, ...] = ()
    nota_bruta: float | None = None


def pearson(pares, media_i=None, media_j=None):
    """Centra nas médias informadas ou, isoladamente, nas médias dos pares comuns."""
    pares = list(pares)
    if len(pares) < 2:
        return 0.0
    if media_i is None:
        media_i = fmean(x for x, _ in pares)
    if media_j is None:
        media_j = fmean(y for _, y in pares)
    numerador = math.fsum((x - media_i) * (y - media_j) for x, y in pares)
    di = math.fsum((x - media_i) ** 2 for x, _ in pares)
    dj = math.fsum((y - media_j) ** 2 for _, y in pares)
    return _normalizar(numerador, di, dj)


def _normalizar(numerador, di, dj):
    if di <= 1e-15 or dj <= 1e-15:
        return 0.0
    return max(-1.0, min(1.0, numerador / math.sqrt(di * dj)))


class Modelo:
    def __init__(self, filmes, treino, k=20, min_comuns=3):
        if k < 1 or min_comuns < 2 or not treino:
            raise ValueError("k >= 1, min_comuns >= 2 e treino não vazio são necessários")
        self.filmes = filmes
        self.treino = list(treino)
        self.k = k
        self.min_comuns = min_comuns
        self.por_usuario = defaultdict(dict)
        self.por_filme = defaultdict(dict)
        for a in treino:
            if a.movie_id not in filmes or (a.movie_id in self.por_usuario[a.user_id]):
                raise ValueError("treino contém filme inexistente ou avaliação duplicada")
            self.por_usuario[a.user_id][a.movie_id] = a.nota
            self.por_filme[a.movie_id][a.user_id] = a.nota
        self.medias = {i: fmean(notas.values()) for i, notas in self.por_filme.items()}
        self.media_global = fmean(a.nota for a in treino)
        self.similaridades = defaultdict(dict)

    def calcular_similaridades(self):
        """Acumula apenas pares coavaliados, um filme por vez (sem matriz densa)."""
        self.similaridades.clear()
        for i in sorted(self.por_filme):
            acumulados = {}
            for u, nota_i in sorted(self.por_filme[i].items()):
                desvio_i = nota_i - self.medias[i]
                for j, nota_j in self.por_usuario[u].items():
                    if j <= i:
                        continue
                    desvio_j = nota_j - self.medias[j]
                    valores = acumulados.setdefault(j, [0, 0.0, 0.0, 0.0])
                    valores[0] += 1
                    valores[1] += desvio_i * desvio_j
                    valores[2] += desvio_i * desvio_i
                    valores[3] += desvio_j * desvio_j
            for j, (n, numerador, di, dj) in acumulados.items():
                if n < self.min_comuns:
                    continue
                correlacao = _normalizar(numerador, di, dj)
                if abs(correlacao) > 1e-12:
                    self.similaridades[i][j] = (correlacao, n)
                    self.similaridades[j][i] = (correlacao, n)
        return self

    def arestas(self):
        for i in sorted(self.similaridades):
            for j, (correlacao, comuns) in sorted(self.similaridades[i].items()):
                if i < j:
                    yield {"origem": i, "destino": j, "correlacao": correlacao, "comuns": comuns}

    def _prever(self, user_id, movie_id, vizinhos):
        # Mais próximos = maior Pearson (não maior valor absoluto).
        melhores = heapq.nsmallest(self.k, vizinhos,
                                  key=lambda v: (-v.correlacao, -v.comuns, v.movie_id))
        denominador = math.fsum(abs(v.correlacao) for v in melhores)
        if denominador > 1e-12:
            bruta = math.fsum(v.correlacao * v.nota for v in melhores) / denominador
            return Previsao(movie_id, max(0.5, min(5.0, bruta)), "knn",
                            tuple(melhores), bruta)
        if movie_id in self.medias:
            return Previsao(movie_id, self.medias[movie_id], "media_filme")
        historico = self.por_usuario.get(user_id, {})
        if historico:
            return Previsao(movie_id, fmean(historico.values()), "media_usuario")
        return Previsao(movie_id, self.media_global, "media_global")

    def prever(self, user_id, movie_id):
        if movie_id not in self.filmes:
            raise ValueError(f"filme {movie_id} não existe no catálogo")
        historico = self.por_usuario.get(user_id, {})
        vizinhos = (Vizinho(j, correlacao, historico[j], comuns)
                    for j, (correlacao, comuns) in self.similaridades.get(movie_id, {}).items()
                    if j in historico)
        return self._prever(user_id, movie_id, vizinhos)

    def recomendar(self, user_id, n=10):
        if n < 1:
            raise ValueError("n deve ser positivo")
        historico = self.por_usuario.get(user_id, {})
        # O catálogo candidato é todo filme com avaliação de TREINO ainda não visto.
        # O teste nunca participa da geração de candidatos ou da ordenação.
        candidatos = defaultdict(list)
        for j, nota in historico.items():
            for i, (correlacao, comuns) in self.similaridades.get(j, {}).items():
                if i not in historico:
                    candidatos[i].append(Vizinho(j, correlacao, nota, comuns))
        previsoes = (self._prever(user_id, i, candidatos.get(i, ()))
                     for i in self.por_filme if i not in historico)
        return heapq.nsmallest(n, previsoes,
                              key=lambda p: (-p.nota, -len(self.por_filme[p.movie_id]), p.movie_id))
