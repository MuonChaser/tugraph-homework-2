"""Render publication figures for the TuGraph graph-analysis report."""
from pathlib import Path
import csv
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager, colors, ticker
from matplotlib.font_manager import FontProperties

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
BLUE, TEAL, ORANGE, INK = "#285F9E", "#008C95", "#D98A2B", "#253044"


def chinese_font(size=10):
    for family in ("SimSun", "Noto Serif CJK SC", "Source Han Serif SC", "Songti SC"):
        try:
            path = font_manager.findfont(FontProperties(family=family), fallback_to_default=False)
            return FontProperties(fname=path, size=size)
        except ValueError:
            continue
    return FontProperties(family="DejaVu Serif", size=size)


def save(figure, name):
    (ROOT / "figures").mkdir(exist_ok=True)
    target = ROOT / "latex_report/figures"
    target.mkdir(parents=True, exist_ok=True)
    figure.tight_layout(pad=1)
    figure.savefig(ROOT / "figures" / f"{name}.png", dpi=220, bbox_inches="tight")
    figure.savefig(target / f"{name}.pdf", bbox_inches="tight")
    plt.close(figure)


def style(axes, grid="y"):
    axes.spines[["top", "right"]].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color("#7D8A99")
        axes.spines[side].set_linewidth(.65)
    axes.tick_params(width=.65, color="#7D8A99", labelcolor=INK, labelsize=9)
    axes.set_axisbelow(True)
    if grid:
        axes.grid(axis=grid, color="#E3E9F0", linewidth=.6)


def main():
    if not (RESULTS / "extended_analysis.json").exists():
        import extended_analysis
        extended_analysis.main()
    summary = json.loads((RESULTS / "experiment_summary.json").read_text(encoding="utf-8"))
    analysis = json.loads((RESULTS / "extended_analysis.json").read_text(encoding="utf-8"))
    full = np.loadtxt(RESULTS / "pagerank_full.csv.gz", delimiter=",", skiprows=1)
    plt.rcParams.update({
        "font.family": ["Times New Roman", "DejaVu Serif"], "font.size": 10,
        "mathtext.fontset": "stix", "pdf.fonttype": 42, "text.color": INK,
        "axes.labelcolor": INK, "axes.titleweight": "normal",
    })
    cn = chinese_font()
    small = chinese_font(8.5)
    top = summary["pagerank"]["top20"][:10]
    palette = colors.LinearSegmentedColormap.from_list("report_rank", [BLUE, TEAL, "#A9D1D8"])
    figure, axes = plt.subplots(figsize=(6.9, 4.2))
    scores = np.asarray([row["pagerank"] for row in top])
    bars = axes.barh(np.arange(10), scores, height=.66, color=palette(np.linspace(0, 1, 10)))
    axes.set_yticks(range(10), [str(row["page_id"]) for row in top])
    axes.invert_yaxis()
    for bar, score in zip(bars, scores):
        axes.text(score + .00015, bar.get_y() + bar.get_height()/2, f"{score:.6f}",
                  va="center", fontsize=8.5, color=INK)
    axes.set_xlim(0, scores.max()*1.23)
    axes.set_xlabel("PageRank 得分", fontproperties=cn)
    axes.set_ylabel("节点编号", fontproperties=cn)
    axes.xaxis.set_major_locator(ticker.MaxNLocator(5))
    style(axes, "x")
    save(figure, "pagerank_top10")

    rank = full[:, 3]
    figure, axes = plt.subplots(figsize=(6.7, 3.8))
    axes.hist(np.log10(rank), bins=45, color=BLUE, alpha=.85, edgecolor="white", linewidth=.5)
    median = analysis["pagerank"]["median_score"]
    axes.axvline(np.log10(median), color=ORANGE, linestyle="--", linewidth=1.4,
                 label=f"中位数 {median:.2e}")
    axes.legend(prop=small, frameon=False, loc="upper right")
    axes.set_xlabel(r"$\log_{10}(\mathrm{PageRank})$")
    axes.set_ylabel("节点数", fontproperties=cn)
    axes.yaxis.set_major_formatter(ticker.FuncFormatter(lambda value, _: f"{value/1000:.0f}k"))
    style(axes)
    save(figure, "pagerank_distribution")

    figure, axes = plt.subplots(figsize=(6.7, 4.1))
    for column, label, color, linestyle in ((1, "入度", BLUE, "-"), (2, "出度", TEAL, "--")):
        degree = full[:, column].astype(int)
        counts = np.bincount(degree)
        survival = np.cumsum(counts[::-1])[::-1] / len(degree)
        x = np.unique(np.geomspace(1, len(counts)-1, 600).astype(int))
        axes.plot(x, survival[x], color=color, linewidth=1.8, linestyle=linestyle, label=label)
    axes.set_xscale("log")
    axes.set_yscale("log")
    axes.set_xlabel("节点度数 k", fontproperties=cn)
    axes.set_ylabel("度数不小于 k 的节点比例", fontproperties=cn)
    axes.legend(prop=cn, frameon=False)
    style(axes, "both")
    save(figure, "degree_distribution")

    with (RESULTS / "pagerank_iterations.csv").open(newline="", encoding="utf-8") as stream:
        histories = list(csv.DictReader(stream))
    figure, axes = plt.subplots(figsize=(6.7, 4.0))
    for alpha, color in ((.75, TEAL), (.85, BLUE), (.95, ORANGE)):
        rows = [r for r in histories if float(r["alpha"]) == alpha]
        x = [int(r["iteration"]) for r in rows]
        y = [float(r["l1_residual"]) for r in rows]
        axes.semilogy(x, y, color=color, linewidth=1.8, label=rf"$\alpha={alpha:.2f}$")
        axes.scatter(x[-1], y[-1], s=24, color=color, zorder=3)
        axes.annotate(f"{y[-1]:.2e}", (x[-1], y[-1]), xytext=(6, 2),
                      textcoords="offset points", color=color, fontsize=8.5)
    axes.axhline(1e-12, color="#8F9AA7", linestyle=":", linewidth=1)
    axes.text(3, 1.8e-12, r"$10^{-12}$", fontsize=8.5, color="#697586")
    axes.set_xlim(0, 124)
    axes.set_xlabel("迭代轮次", fontproperties=cn)
    axes.set_ylabel(r"$L_1$ 残差", fontproperties=cn)
    axes.legend(frameon=False, loc="upper right")
    style(axes)
    save(figure, "pagerank_convergence")

    ordered = np.sort(rank)[::-1]
    cumulative = np.cumsum(ordered)
    x = np.arange(1, len(rank)+1)
    figure, axes = plt.subplots(figsize=(6.7, 3.9))
    axes.plot(x, cumulative, color=BLUE, linewidth=2)
    axes.fill_between(x, cumulative, color=BLUE, alpha=.10)
    for k, offset in ((1000, (-65, 18)), (2820, (-22, 30)), (28191, (10, -24))):
        value = cumulative[k-1]
        axes.scatter(k, value, s=38, color=TEAL, edgecolor="white", linewidth=.8, zorder=3)
        axes.annotate(f"k={k:,}\n{value:.2%}", (k, value), xytext=offset,
                      textcoords="offset points", fontsize=8.5, color=INK,
                      arrowprops={"arrowstyle":"-", "color":"#9AA7B5", "lw":.7})
    axes.set_xscale("log")
    axes.set_ylim(0, 1.04)
    axes.yaxis.set_major_formatter(ticker.PercentFormatter(1))
    axes.set_xlabel("按得分降序排列的前 k 个节点", fontproperties=cn)
    axes.set_ylabel("累计 PageRank 分值占比", fontproperties=cn)
    style(axes)
    save(figure, "pagerank_concentration")

    projection = np.loadtxt(RESULTS / "node2vec_projection.csv", delimiter=",", skiprows=1)
    ids = projection[:, 0].astype(int)
    query = summary["node2vec"]["sample_seed_page_id"]
    neighbors = [r["page_id"] for r in summary["node2vec"]["nearest_pages"]]
    is_neighbor = np.isin(ids, neighbors)
    query_index = int(np.flatnonzero(ids == query)[0])
    figure, axes = plt.subplots(figsize=(6.9, 4.6))
    points = axes.scatter(projection[:, 1], projection[:, 2], c=projection[:, 3], cmap="YlGnBu",
                          s=13, alpha=.72, linewidths=0, rasterized=True)
    axes.scatter(projection[is_neighbor, 1], projection[is_neighbor, 2], facecolor="none",
                 edgecolor=ORANGE, linewidth=1.25, s=78, label="前十名近邻", zorder=3)
    axes.scatter(projection[query_index, 1], projection[query_index, 2], marker="*",
                 color="#D96B2B", edgecolor="white", linewidth=.8, s=190, label="查询节点", zorder=5)
    axes.annotate(str(query), projection[query_index, 1:3], xytext=(18, 18),
                  textcoords="offset points", fontsize=9, color="#B74D16",
                  arrowprops={"arrowstyle":"-", "color":"#B74D16", "lw":.8})
    axes.set_xlabel(f"PC1 ({analysis['embedding']['pc1_explained_variance_ratio']:.2%})")
    axes.set_ylabel(f"PC2 ({analysis['embedding']['pc2_explained_variance_ratio']:.2%})")
    axes.legend(prop=small, frameon=False, loc="lower left")
    colorbar = figure.colorbar(points, ax=axes, fraction=.045, pad=.03)
    colorbar.set_label("与查询节点的余弦相似度", fontproperties=small)
    colorbar.outline.set_visible(False)
    style(axes, "both")
    save(figure, "node2vec_projection")

    with (RESULTS / "node2vec_neighbor_heatmap.csv").open(newline="", encoding="utf-8") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        rows = list(reader)
    matrix = np.asarray([[float(value) for value in row[1:]] for row in rows])
    labels = [str(query)+"*"] + [str(v) for v in neighbors]
    figure, axes = plt.subplots(figsize=(6.8, 5.5))
    image = axes.imshow(matrix, cmap="YlGnBu", vmin=.85, vmax=1, interpolation="nearest")
    axes.set_xticks(range(11), labels, rotation=45, ha="right", fontsize=8)
    axes.set_yticks(range(11), labels, fontsize=8)
    axes.tick_params(length=0)
    axes.set_xticks(np.arange(-.5, 11, 1), minor=True)
    axes.set_yticks(np.arange(-.5, 11, 1), minor=True)
    axes.grid(which="minor", color="white", linewidth=.8)
    axes.tick_params(which="minor", length=0)
    for i in range(11):
        for j in range(11):
            axes.text(j, i, f"{matrix[i,j]:.2f}", ha="center", va="center", fontsize=6.7,
                      color="white" if matrix[i,j]>.94 else INK)
    colorbar = figure.colorbar(image, ax=axes, fraction=.045, pad=.03)
    colorbar.set_label("余弦相似度", fontproperties=small)
    colorbar.outline.set_visible(False)
    save(figure, "node2vec_similarity_heatmap")


if __name__ == "__main__":
    main()

