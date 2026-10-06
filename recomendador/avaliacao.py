"""Métricas de notas e ranking. Usa exclusivamente o modelo ajustado no treino."""

import math
from collections import Counter, defaultdict
from dataclasses import asdict
from statistics import fmean


def _razao(a, b):
    return a / b if b else 0.0


def avaliar(modelo, teste, top_n=10, limiar=4.0):
    if not teste or top_n < 1 or not 0.5 <= limiar <= 5:
        raise ValueError("teste não vazio, top_n positivo e limiar entre 0.5 e 5 necessários")
    treino_pares = {(a.user_id, a.movie_id) for a in modelo.treino}
    if any((a.user_id, a.movie_id) in treino_pares for a in teste):
        raise ValueError("há vazamento: avaliações de teste também estão no treino")
    tp = fp = fn = tn = 0
    erros, quadrados, metodos = [], [], Counter()
    relevantes = defaultdict(set)
    detalhes = []
    for a in teste:
        p = modelo.prever(a.user_id, a.movie_id)
        erro = p.nota - a.nota
        erros.append(abs(erro))
        quadrados.append(erro * erro)
        metodos[p.metodo] += 1
        positivo, predito = a.nota >= limiar, p.nota >= limiar
        tp += positivo and predito
        fp += not positivo and predito
        fn += positivo and not predito
        tn += not positivo and not predito
        if positivo:
            relevantes[a.user_id].add(a.movie_id)
    hits_total = retornados_total = relevantes_total = 0
    for u in sorted({a.user_id for a in teste}):
        if not relevantes[u]:
            continue
        recomendacoes = modelo.recomendar(u, top_n)
        hits = len({p.movie_id for p in recomendacoes} & relevantes[u])
        precisao = _razao(hits, len(recomendacoes))
        recall = hits / len(relevantes[u])
        hits_total += hits
        retornados_total += len(recomendacoes)
        relevantes_total += len(relevantes[u])
        detalhes.append({"user_id": u, "relevantes_teste": len(relevantes[u]),
                         "retornados": len(recomendacoes), "acertos": hits,
                         "precisao": precisao, "recall": recall})
    precisao, recall = _razao(tp, tp + fp), _razao(tp, tp + fn)
    return {
        "parametros": {"k": modelo.k, "min_comuns": modelo.min_comuns,
                       "top_n": top_n, "limiar_relevancia": limiar},
        "notas": {"quantidade": len(teste), "mae": fmean(erros),
                  "rmse": math.sqrt(fmean(quadrados)), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                  "precisao": precisao, "recall": recall,
                  "f1": _razao(2 * precisao * recall, precisao + recall),
                  "metodos": dict(sorted(metodos.items())),
                  "cobertura_knn": metodos["knn"] / len(teste)},
        "ranking": {"usuarios_avaliados": len(detalhes),
                    "usuarios_sem_relevantes": len({a.user_id for a in teste}) - len(detalhes),
                    "precisao_macro": fmean(d["precisao"] for d in detalhes) if detalhes else 0.0,
                    "recall_macro": fmean(d["recall"] for d in detalhes) if detalhes else 0.0,
                    "precisao_micro": _razao(hits_total, retornados_total),
                    "recall_micro": _razao(hits_total, relevantes_total),
                    "acertos": hits_total, "relevantes": relevantes_total,
                    "retornados": retornados_total,
                    "por_usuario": detalhes},
    }


def apresentar_previsao(modelo, previsao):
    resultado = asdict(previsao)
    resultado["titulo"] = modelo.filmes[previsao.movie_id].titulo
    for vizinho in resultado["vizinhos"]:
        vizinho["titulo"] = modelo.filmes[vizinho["movie_id"]].titulo
    return resultado
