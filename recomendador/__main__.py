"""Interface: python -m recomendador --help."""

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

from .artefatos import carregar_modelo, carregar_teste, salvar_json, salvar_modelo
from .avaliacao import avaliar, apresentar_previsao
from .dados import carregar_dados, dividir
from .modelo import Modelo
from .neo4j import (ClienteNeo4j, ErroNeo4j, importar, verificar_pronto,
                    COMEDIAS, ESTATISTICAS, RECOMENDACOES)


def positivo(texto):
    valor = int(texto)
    if valor < 1:
        raise argparse.ArgumentTypeError("deve ser um inteiro positivo")
    return valor


def parser():
    principal = argparse.ArgumentParser(description="EP02 — MovieLens, Neo4j, Pearson e KNN sem dependências externas")
    comandos = principal.add_subparsers(dest="comando", required=True)
    preparar = comandos.add_parser("preparar", help="divide 80/20, treina e salva os artefatos")
    preparar.add_argument("--dados", type=Path, required=True)
    preparar.add_argument("--saida", type=Path, default=Path("artefatos"))
    preparar.add_argument("--seed", type=int, default=42)
    preparar.add_argument("--k", type=positivo, default=20)
    preparar.add_argument("--min-comuns", type=positivo, default=3)
    preparar.add_argument("--sobrescrever", action="store_true", help="permite atualizar artefatos locais existentes")
    avaliacao = comandos.add_parser("avaliar", help="calcula precisão, recall e erros no teste")
    avaliacao.add_argument("--modelo", type=Path, default=Path("artefatos"))
    avaliacao.add_argument("--top", type=positivo, default=10)
    avaliacao.add_argument("--limiar", type=float, default=4.0)
    avaliacao.add_argument("--saida", type=Path, default=Path("resultados/avaliacao.json"))
    recomendar = comandos.add_parser("recomendar", help="top-N usando artefatos locais de treino")
    recomendar.add_argument("--modelo", type=Path, default=Path("artefatos"))
    recomendar.add_argument("--usuario", type=positivo, required=True)
    recomendar.add_argument("--top", type=positivo, default=10)
    prever = comandos.add_parser("prever", help="nota e explicação dos vizinhos para um filme")
    prever.add_argument("--modelo", type=Path, default=Path("artefatos"))
    prever.add_argument("--usuario", type=positivo, required=True)
    prever.add_argument("--filme", type=positivo, required=True)
    insercao = comandos.add_parser("importar", help="insere treino e similaridades reais no Neo4j")
    insercao.add_argument("--modelo", type=Path, default=Path("artefatos"))
    insercao.add_argument("--execucao", required=True)
    insercao.add_argument("--lote", type=positivo, default=1000)
    consulta = comandos.add_parser("consultar", help="executa Cypher no grafo Neo4j")
    consulta.add_argument("consulta", choices=("comedias", "recomendar", "estatisticas"))
    consulta.add_argument("--execucao", required=True)
    consulta.add_argument("--usuario", type=positivo)
    consulta.add_argument("--top", type=positivo, default=10)
    consulta.add_argument("--limiar", type=float, default=4.0)
    return principal


def executar(args):
    if args.comando == "preparar":
        if args.saida.exists() and any(args.saida.iterdir()) and not args.sobrescrever:
            raise ValueError("pasta de saída não vazia; escolha outra ou use --sobrescrever")
        inicio = time.monotonic()
        filmes, avaliacoes = carregar_dados(args.dados)
        treino, teste = dividir(avaliacoes, seed=args.seed)
        modelo = Modelo(filmes, treino, args.k, args.min_comuns)
        print("Calculando Pearson apenas no treino...", file=sys.stderr)
        modelo.calcular_similaridades()
        manifesto = salvar_modelo(args.saida, modelo, teste, args.seed)
        manifesto["sha256_movies_csv_origem"] = hashlib.sha256((args.dados / "movies.csv").read_bytes()).hexdigest()
        manifesto["sha256_ratings_csv_origem"] = hashlib.sha256((args.dados / "ratings.csv").read_bytes()).hexdigest()
        salvar_json(args.saida / "manifesto.json", manifesto)
        return {**manifesto, "segundos": round(time.monotonic() - inicio, 3)}
    if args.comando == "consultar":
        if args.consulta != "estatisticas" and args.usuario is None:
            raise ValueError("--usuario é obrigatório para comedias e recomendar")
        if not 0.5 <= args.limiar <= 5:
            raise ValueError("--limiar deve estar entre 0.5 e 5")
        cliente = ClienteNeo4j()
        config = verificar_pronto(cliente, args.execucao)
        cypher = {"comedias": COMEDIAS, "recomendar": RECOMENDACOES,
                  "estatisticas": ESTATISTICAS}[args.consulta]
        return cliente.executar(cypher, {"run": args.execucao, "user": args.usuario,
                                        "n": args.top, "limiar": args.limiar, "k": config["k"]})
    modelo, manifesto = carregar_modelo(args.modelo)
    if args.comando == "avaliar":
        teste = carregar_teste(args.modelo, manifesto)
        print("Avaliando notas e ranking no catálogo completo de treino...", file=sys.stderr)
        resultado = avaliar(modelo, teste, args.top, args.limiar)
        resultado["dados"] = manifesto
        salvar_json(args.saida, resultado)
        return {"arquivo": str(args.saida), "notas": resultado["notas"],
                "ranking": {k: v for k, v in resultado["ranking"].items() if k != "por_usuario"}}
    if args.comando == "importar":
        return importar(ClienteNeo4j(), modelo, manifesto, args.execucao, args.lote)
    if args.comando == "prever":
        return apresentar_previsao(modelo, modelo.prever(args.usuario, args.filme))
    return [apresentar_previsao(modelo, p) for p in modelo.recomendar(args.usuario, args.top)]


def main():
    args = parser().parse_args()
    try:
        resultado = executar(args)
        print(json.dumps(resultado, ensure_ascii=False, indent=2, allow_nan=False))
    except (ValueError, OSError, KeyError, ErroNeo4j, csv.Error) as erro:
        print(f"Erro: {erro}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
