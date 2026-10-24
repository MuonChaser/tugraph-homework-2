"""Additional graph statistics, PageRank comparison, and embedding projection."""
from pathlib import Path
import csv
import gzip
import json
import random

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def trace_pagerank(source, target, n, alpha, max_iter=100, tolerance=1e-12):
    outdegree = np.bincount(source, minlength=n).astype(float)
    dangling = outdegree == 0
    rank = np.full(n, 1.0 / n)
    history = []
    for iteration in range(1, max_iter + 1):
        incoming = np.bincount(target, weights=rank[source] / outdegree[source], minlength=n)
        updated = alpha * (incoming + rank[dangling].sum() / n) + (1.0 - alpha) / n
        residual = float(np.abs(updated - rank).sum())
        history.append((iteration, residual))
        rank = updated
        if residual < tolerance:
            break
    return rank, history


def degree_statistics(values):
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "p99": float(np.percentile(values, 99)),
        "max": int(np.max(values)),
        "zero_count": int(np.count_nonzero(values == 0)),
    }


def walk_statistics(nodes, edges):
    sampled = {int(node): [] for node in nodes}
    node_set = set(sampled)
    for a, b in edges.tolist():
        if a in node_set and b in node_set:
            sampled[a].append(b)
    ordered = sorted(sampled)
    out_sets = {u: set(sampled[u]) for u in ordered}
    rng = random.Random(20261022)
    length_histogram = {}
    pair_count = 0
    for _ in range(4):
        order = ordered[:]
        rng.shuffle(order)
        for start in order:
            walk = [start]
            while len(walk) < 12:
                choices = sampled[walk[-1]]
                if not choices:
                    break
                if len(walk) == 1:
                    nxt = rng.choice(choices)
                else:
                    prev = walk[-2]
                    weights = [1.0 if v == prev or v in out_sets[prev] else .5 for v in choices]
                    nxt = rng.choices(choices, weights=weights, k=1)[0]
                walk.append(nxt)
            length_histogram[len(walk)] = length_histogram.get(len(walk), 0) + 1
            pair_count += sum(min(len(walk), i + 4) - max(0, i - 3) - 1 for i in range(len(walk)))
    return {"sample_edges": sum(map(len, sampled.values())),
            "walk_count": sum(length_histogram.values()),
            "walk_length_histogram": length_histogram, "positive_pairs": pair_count}

def main():
    RESULTS.mkdir(exist_ok=True)
    original = json.loads((RESULTS / "experiment_summary.json").read_text(encoding="utf-8"))
    edges = np.loadtxt(ROOT / "tugraph_export/hyperlink_webpage_webpage.csv",
                       delimiter=",", dtype=np.int32, usecols=(0, 1))
    base = int(edges.min())
    n = int(edges.max()) - base + 1
    source, target = edges[:, 0] - base, edges[:, 1] - base
    indegree = np.bincount(target, minlength=n)
    outdegree = np.bincount(source, minlength=n)
    baseline, baseline_history = trace_pagerank(source, target, n, 0.85)
    baseline_order = np.argsort(baseline)[::-1]
    recorded = original["pagerank"]["top20"]
    assert [int(i + base) for i in baseline_order[:20]] == [r["page_id"] for r in recorded]
    assert max(abs(float(baseline[r["page_id"] - base]) - r["pagerank"]) for r in recorded) < 1e-12
    baseline_top = set(baseline_order[:10].tolist())

    histories = []
    variants = []
    for alpha in (0.75, 0.85, 0.95):
        if alpha == 0.85:
            rank, history = baseline, baseline_history
        else:
            rank, history = trace_pagerank(source, target, n, alpha)
        order = np.argsort(rank)[::-1]
        histories.extend((alpha, iteration, residual) for iteration, residual in history)
        overlap = len(baseline_top.intersection(order[:10].tolist()))
        variants.append({
            "alpha": alpha, "iterations": len(history), "residual": history[-1][1],
            "top_page_id": int(order[0] + base), "top_score": float(rank[order[0]]),
            "top10_overlap_with_alpha085": overlap,
            "top10_jaccard_with_alpha085": overlap / (20 - overlap),
        })
    with (RESULTS / "pagerank_iterations.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["alpha", "iteration", "l1_residual"])
        writer.writerows(histories)

    cumulative = np.cumsum(baseline[baseline_order])
    cuts = sorted(set([1, 3, 10, 100, 1000, int(np.ceil(n * .01)),
                       int(np.ceil(n * .05)), int(np.ceil(n * .10)), n]))
    concentration = [
        {"rank": k, "node_fraction": k / n, "cumulative_score": float(cumulative[k - 1])}
        for k in cuts
    ]
    with (RESULTS / "pagerank_concentration.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["rank", "node_fraction", "cumulative_score"])
        writer.writeheader()
        writer.writerows(concentration)
    with gzip.open(RESULTS / "pagerank_full.csv.gz", "wt", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["page_id", "in_degree", "out_degree", "pagerank"])
        writer.writerows((i + base, int(indegree[i]), int(outdegree[i]), f"{baseline[i]:.15g}") for i in range(n))

    with (RESULTS / "node2vec_embeddings_sample.csv").open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        next(reader)
        rows = list(reader)
    ids = np.asarray([int(row[0]) for row in rows])
    matrix = np.asarray([[float(v) for v in row[1:]] for row in rows])
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    _, singular, components = np.linalg.svd(centered, full_matrices=False)
    variance = singular ** 2 / np.sum(singular ** 2)
    coordinates = centered @ components[:2].T
    query = int(original["node2vec"]["sample_seed_page_id"])
    query_index = int(np.flatnonzero(ids == query)[0])
    unit = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
    similarity = unit @ unit[query_index]
    mask = ids != query
    with (RESULTS / "node2vec_projection.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["page_id", "pc1", "pc2", "cosine_to_89073"])
        writer.writerows((int(node), float(x), float(y), float(sim)) for node, (x, y), sim
                         in zip(ids, coordinates, similarity))
    nearest_ids = [r["page_id"] for r in original["node2vec"]["nearest_pages"]]
    selected = [query] + nearest_ids
    selected_indices = [int(np.flatnonzero(ids == node)[0]) for node in selected]
    selected_unit = unit[selected_indices]
    neighbor_matrix = selected_unit[1:] @ selected_unit[1:].T
    off_diagonal = neighbor_matrix[~np.eye(len(nearest_ids), dtype=bool)]
    projected_distance = np.linalg.norm(coordinates - coordinates[query_index], axis=1)
    training_statistics = walk_statistics(ids, edges)
    assert training_statistics["positive_pairs"] == original["node2vec"]["skipgram_pairs"]
    (RESULTS / "node2vec_training_statistics.json").write_text(json.dumps(training_statistics, indent=2) + "\n", encoding="utf-8")
    with (RESULTS / "node2vec_neighbor_heatmap.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["page_id"] + selected)
        writer.writerows([node] + row.tolist() for node, row in zip(selected, selected_unit @ selected_unit.T))
    analysis = {
        "graph": {
            "vertices": n, "edges": len(edges), "self_loops": int(np.count_nonzero(source == target)),
            "in_degree": degree_statistics(indegree), "out_degree": degree_statistics(outdegree),
        },
        "pagerank": {
            "variants": variants, "concentration": concentration,
            "median_score": float(np.median(baseline)),
            "log_indegree_log_pagerank_pearson": float(np.corrcoef(np.log1p(indegree), np.log(baseline))[0, 1]),
        },
        "embedding": {
            "query_page_id": query, "dimensions": matrix.shape[1], "sample_vertices": len(ids),
            "pc1_explained_variance_ratio": float(variance[0]),
            "pc2_explained_variance_ratio": float(variance[1]),
            "pc12_explained_variance_ratio": float(variance[:2].sum()),
            "nearest10_pairwise_cosine_mean": float(off_diagonal.mean()),
            "nearest10_pairwise_cosine_min": float(off_diagonal.min()),
            "pca_distance_query_to_all_mean": float(projected_distance[mask].mean()),
            "pca_distance_query_to_nearest10_mean": float(projected_distance[selected_indices[1:]].mean()),
            "cosine_to_query": {
                "mean": float(similarity[mask].mean()),
                "median": float(np.median(similarity[mask])),
                "p90": float(np.percentile(similarity[mask], 90)),
                "max": float(similarity[mask].max()),
            },
        },
    }
    (RESULTS / "extended_analysis.json").write_text(json.dumps(analysis, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(analysis, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


