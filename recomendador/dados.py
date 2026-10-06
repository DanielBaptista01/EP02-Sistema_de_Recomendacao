"""Leitura estrita dos dois CSVs e divisão reproduzível sem alterar a entrada."""

import csv
import hashlib
import math
import random
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Filme:
    movie_id: int
    titulo: str
    generos: tuple[str, ...]


@dataclass(frozen=True)
class Avaliacao:
    user_id: int
    movie_id: int
    nota: float
    timestamp: int


def _linhas(path, campos):
    with Path(path).open(encoding="utf-8-sig", newline="") as arquivo:
        leitor = csv.DictReader(arquivo)
        if not set(campos).issubset(leitor.fieldnames or []):
            raise ValueError(f"{path}: cabeçalho deve conter {', '.join(campos)}")
        for numero, linha in enumerate(leitor, 2):
            yield numero, linha


def carregar_dados(pasta):
    pasta = Path(pasta)
    filmes = {}
    for numero, linha in _linhas(pasta / "movies.csv", ("movieId", "title", "genres")):
        try:
            movie_id = int(linha["movieId"])
            titulo = linha["title"].strip()
            generos = tuple(g for g in linha["genres"].split("|")
                            if g and g != "(no genres listed)")
            if movie_id <= 0 or movie_id in filmes or not titulo:
                raise ValueError("ID inválido/duplicado ou título vazio")
            filmes[movie_id] = Filme(movie_id, titulo, generos)
        except (ValueError, TypeError, AttributeError) as erro:
            raise ValueError(f"movies.csv, linha {numero}: {erro}") from erro
    avaliacoes, pares = [], set()
    for numero, linha in _linhas(pasta / "ratings.csv", ("userId", "movieId", "rating", "timestamp")):
        try:
            a = Avaliacao(int(linha["userId"]), int(linha["movieId"]),
                          float(linha["rating"]), int(linha["timestamp"]))
            par = (a.user_id, a.movie_id)
            if a.user_id <= 0 or a.movie_id not in filmes or a.timestamp < 0:
                raise ValueError("usuário, filme ou timestamp inválido")
            if not math.isfinite(a.nota) or not 0.5 <= a.nota <= 5:
                raise ValueError("nota deve estar entre 0.5 e 5")
            if par in pares:
                raise ValueError("avaliação duplicada para usuário/filme")
            pares.add(par)
            avaliacoes.append(a)
        except (ValueError, TypeError) as erro:
            raise ValueError(f"ratings.csv, linha {numero}: {erro}") from erro
    if not filmes or not avaliacoes:
        raise ValueError("movies.csv e ratings.csv não podem estar vazios")
    return filmes, avaliacoes


def dividir(avaliacoes, fracao_teste=0.2, seed=42):
    if not 0 < fracao_teste < 1 or len(avaliacoes) < 2:
        raise ValueError("divisão exige ao menos 2 avaliações e fração entre 0 e 1")
    embaralhadas = sorted(avaliacoes, key=lambda a: (a.user_id, a.movie_id))
    random.Random(seed).shuffle(embaralhadas)
    quantidade = max(1, min(len(embaralhadas) - 1, round(len(embaralhadas) * fracao_teste)))
    return embaralhadas[quantidade:], embaralhadas[:quantidade]


def assinatura(avaliacoes):
    """Fingerprint independente da ordem, para registrar exatamente o split usado."""
    resumo = hashlib.sha256()
    for a in sorted(avaliacoes, key=lambda a: (a.user_id, a.movie_id)):
        resumo.update(f"{a.user_id},{a.movie_id},{a.nota},{a.timestamp}\n".encode())
    return resumo.hexdigest()


def salvar_avaliacoes(path, avaliacoes):
    with Path(path).open("w", encoding="utf-8", newline="") as arquivo:
        escritor = csv.writer(arquivo)
        escritor.writerow(("userId", "movieId", "rating", "timestamp"))
        escritor.writerows((a.user_id, a.movie_id, a.nota, a.timestamp) for a in avaliacoes)
