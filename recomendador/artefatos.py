"""Persistência legível: CSVs separados e manifesto JSON, sem pickle."""

import csv
import hashlib
import json
from pathlib import Path

from .dados import carregar_dados, salvar_avaliacoes, assinatura
from .modelo import Modelo


def salvar_json(path, objeto):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(objeto, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                          encoding="utf-8")


def hash_arquivo(path):
    with Path(path).open("rb") as arquivo:
        return hashlib.file_digest(arquivo, "sha256").hexdigest()


def salvar_modelo(pasta, modelo, teste, seed):
    pasta = Path(pasta)
    pasta.mkdir(parents=True, exist_ok=True)
    salvar_avaliacoes(pasta / "ratings.csv", modelo.treino)
    salvar_avaliacoes(pasta / "teste.csv", teste)
    with (pasta / "movies.csv").open("w", encoding="utf-8", newline="") as arquivo:
        escritor = csv.writer(arquivo)
        escritor.writerow(("movieId", "title", "genres"))
        escritor.writerows((f.movie_id, f.titulo, "|".join(f.generos))
                           for f in sorted(modelo.filmes.values(), key=lambda f: f.movie_id))
    quantidade = 0
    with (pasta / "similaridades.csv").open("w", encoding="utf-8", newline="") as arquivo:
        escritor = csv.DictWriter(arquivo, ("origem", "destino", "correlacao", "comuns"))
        escritor.writeheader()
        for aresta in modelo.arestas():
            escritor.writerow(aresta)
            quantidade += 1
    manifesto = {"formato": 1, "seed": seed, "k": modelo.k, "min_comuns": modelo.min_comuns,
                 "treino": len(modelo.treino), "teste": len(teste),
                 "fracao_teste_real": len(teste) / (len(teste) + len(modelo.treino)),
                 "filmes": len(modelo.filmes), "usuarios_treino": len(modelo.por_usuario),
                 "similaridades": quantidade, "sha256_treino": assinatura(modelo.treino),
                 "sha256_teste": assinatura(teste),
                 "sha256_movies_csv": hash_arquivo(pasta / "movies.csv"),
                 "sha256_similaridades_csv": hash_arquivo(pasta / "similaridades.csv")}
    salvar_json(pasta / "manifesto.json", manifesto)
    return manifesto


def carregar_modelo(pasta):
    pasta = Path(pasta)
    manifesto = json.loads((pasta / "manifesto.json").read_text(encoding="utf-8"))
    if manifesto["formato"] != 1:
        raise ValueError("versão de artefatos não suportada")
    for nome in ("movies", "similaridades"):
        if hash_arquivo(pasta / f"{nome}.csv") != manifesto[f"sha256_{nome}_csv"]:
            raise ValueError(f"{nome}.csv difere do arquivo registrado no manifesto")
    filmes, treino = carregar_dados(pasta)
    if assinatura(treino) != manifesto["sha256_treino"]:
        raise ValueError("ratings.csv difere do treino registrado no manifesto")
    modelo = Modelo(filmes, treino, manifesto["k"], manifesto["min_comuns"])
    with (pasta / "similaridades.csv").open(encoding="utf-8", newline="") as arquivo:
        for linha in csv.DictReader(arquivo):
            i, j = int(linha["origem"]), int(linha["destino"])
            c, n = float(linha["correlacao"]), int(linha["comuns"])
            if i >= j or i not in filmes or j not in filmes or not -1 <= c <= 1:
                raise ValueError("similaridades.csv contém aresta inválida")
            if n < modelo.min_comuns or j in modelo.similaridades[i]:
                raise ValueError("similaridades.csv contém suporte inválido ou aresta duplicada")
            modelo.similaridades[i][j] = (c, n)
            modelo.similaridades[j][i] = (c, n)
    return modelo, manifesto


def carregar_teste(pasta, manifesto):
    # Reutiliza a validação do leitor sem confundir o CSV de teste com o treino.
    from .dados import Avaliacao
    with (Path(pasta) / "teste.csv").open(encoding="utf-8", newline="") as arquivo:
        teste = [Avaliacao(int(r["userId"]), int(r["movieId"]), float(r["rating"]), int(r["timestamp"]))
                 for r in csv.DictReader(arquivo)]
    if assinatura(teste) != manifesto["sha256_teste"]:
        raise ValueError("teste.csv difere do teste registrado no manifesto")
    return teste
