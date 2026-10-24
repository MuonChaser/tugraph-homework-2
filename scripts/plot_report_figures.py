"""Generate monochrome report figures from the TuGraph export and result JSON."""
from pathlib import Path
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.font_manager import FontProperties

ROOT = Path(__file__).resolve().parents[1]


def chinese_font():
    for family in ("SimSun", "Noto Serif CJK SC", "Source Han Serif SC", "Songti SC"):
        try:
            filename = font_manager.findfont(FontProperties(family=family), fallback_to_default=False)
            return FontProperties(fname=filename, size=10)
        except ValueError:
            continue
    return None


def save_figure(figure, name):
    (ROOT / "figures").mkdir(exist_ok=True)
    (ROOT / "latex_report" / "figures").mkdir(parents=True, exist_ok=True)
    figure.tight_layout(pad=0.7)
    figure.savefig(ROOT / "figures" / f"{name}.png", dpi=200, bbox_inches="tight")
    figure.savefig(ROOT / "latex_report" / "figures" / f"{name}.pdf", bbox_inches="tight")
    plt.close(figure)


def format_axes(axes):
    axes.spines[["top", "right"]].set_visible(False)
    axes.tick_params(width=0.65, labelsize=9)


def main():
    summary = json.loads((ROOT / "results" / "experiment_summary.json").read_text(encoding="utf-8"))
    plt.rcParams.update({
        "font.family": ["Times New Roman", "DejaVu Serif"],
        "font.size": 10,
        "mathtext.fontset": "stix",
        "axes.linewidth": 0.65,
        "pdf.fonttype": 42,
    })
    chinese = chinese_font()
    top = summary["pagerank"]["top20"][:10]
    figure, axes = plt.subplots(figsize=(6.0, 3.25))
    axes.bar(range(10), [row["pagerank"] for row in top], color="0.55",
             edgecolor="black", linewidth=0.45)
    axes.set_xticks(range(10), [str(row["page_id"]) for row in top], rotation=40, ha="right")
    axes.set_ylabel("PageRank 得分" if chinese else "PageRank score", fontproperties=chinese)
    axes.set_xlabel("节点编号" if chinese else "Node ID", fontproperties=chinese)
    format_axes(axes)
    save_figure(figure, "pagerank_top10")

    edges = np.loadtxt(ROOT / "tugraph_export" / "hyperlink_webpage_webpage.csv",
                       delimiter=",", dtype=np.int32, usecols=(0, 1))
    base = int(edges.min())
    source, target = edges[:, 0] - base, edges[:, 1] - base
    node_count = int(edges.max()) - base + 1
    outdegree = np.bincount(source, minlength=node_count).astype(float)
    rank = np.full(node_count, 1.0 / node_count)
    alpha = summary["pagerank"]["alpha"]
    for _ in range(summary["pagerank"]["max_iter"]):
        incoming = np.bincount(target, weights=rank[source] / outdegree[source],
                               minlength=node_count)
        updated = alpha * (incoming + rank[outdegree == 0].sum() / node_count)
        updated += (1.0 - alpha) / node_count
        residual = np.abs(updated - rank).sum()
        rank = updated
        if residual < summary["pagerank"]["tolerance_l1"]:
            break
    if int(np.argmax(rank) + base) != top[0]["page_id"]:
        raise ValueError("The TuGraph export and stored PageRank result do not match.")

    figure, axes = plt.subplots(figsize=(6.0, 3.25))
    axes.hist(np.log10(rank[rank > 0]), bins=45, color="0.55",
              edgecolor="white", linewidth=0.45)
    axes.set_xlabel(r"$\log_{10}(\mathrm{PageRank})$")
    axes.set_ylabel("节点数" if chinese else "Number of nodes", fontproperties=chinese)
    format_axes(axes)
    save_figure(figure, "pagerank_distribution")


if __name__ == "__main__":
    main()

