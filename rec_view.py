import argparse
import os
import sqlite3
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib import font_manager

def configure_font(font_path):
    if font_path and os.path.exists(font_path):
        try:
            font_manager.fontManager.addfont(font_path)
            font_name = font_manager.FontProperties(fname=font_path).get_name()
            matplotlib.rcParams["font.family"] = "sans-serif"
            matplotlib.rcParams["font.sans-serif"] = [font_name]
            matplotlib.rcParams["axes.unicode_minus"] = False
        except Exception:
            pass

def load_topk_subgraph(conn, k):
    cur = conn.cursor()
    cur.execute("SELECT src, dst FROM edges")
    degree = {}
    edges = []
    for src, dst in cur.fetchall():
        src = int(src); dst = int(dst)
        edges.append((src, dst))
        degree[src] = degree.get(src, 0) + 1
        degree[dst] = degree.get(dst, 0) + 1
    selected = set(x for x, _ in sorted(degree.items(), key=lambda x: x[1], reverse=True)[:k])
    sub_edges = [(s, d) for s, d in edges if s in selected and d in selected]
    if not selected:
        return [], {}
    cur.execute("SELECT id, name FROM nodes WHERE id IN (%s)" % ",".join(str(i) for i in selected))
    id2name = {int(i): n for i, n in cur.fetchall()}
    return sub_edges, id2name

def communities_for_graph(G):
    try:
        import community as community_louvain
        part = community_louvain.best_partition(G)
        return part
    except Exception:
        comms = list(nx.community.greedy_modularity_communities(G))
        label = {}
        for cid, c in enumerate(comms):
            for node in c:
                label[node] = cid
        return label

def draw_communities(edges, id2name, out_path, with_labels, seed, layout, dpi):
    G = nx.Graph()
    G.add_nodes_from(id2name.keys())
    if edges:
        G.add_edges_from(edges)
    part = communities_for_graph(G)
    deg = dict(G.degree())
    sizes = [max(50, deg.get(n, 0) * 2) for n in G.nodes()]
    if layout == "spring":
        pos = nx.spring_layout(G, seed=seed) if seed is not None else nx.spring_layout(G)
    elif layout == "random":
        pos = nx.random_layout(G)
    elif layout == "spectral":
        pos = nx.spectral_layout(G)
    else:
        pos = nx.spring_layout(G, seed=seed) if seed is not None else nx.spring_layout(G)
    labels = {n: id2name.get(n, str(n)) for n in G.nodes()}
    cmap = plt.get_cmap("tab20")
    groups = {}
    for n, c in part.items():
        groups.setdefault(c, []).append(n)
    plt.figure(figsize=(12, 9), dpi=dpi)
    for cid, nodes in groups.items():
        color = cmap(cid % 20)
        nsizes = [sizes[list(G.nodes()).index(n)] for n in nodes]
        nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_size=nsizes, node_color=[color], alpha=0.85)
    nx.draw_networkx_edges(G, pos, edge_color="#9C9C9C", alpha=0.25, width=0.5)
    if with_labels:
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=6)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--sizes", type=str, default="100,500,1000,10000")
    ap.add_argument("--out_dir", type=str, default=".")
    ap.add_argument("--with-labels", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--layout", type=str, choices=["spring", "random", "spectral"], default="spring")
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--font", type=str, default="HarmonyOS_Sans_SC.ttf")
    args = ap.parse_args()
    configure_font(os.path.abspath(args.font))
    sizes = [int(x) for x in args.sizes.split(",") if x.strip().isdigit()]
    db_path = os.path.abspath(args.db)
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        for k in sizes:
            edges, id2name = load_topk_subgraph(conn, k)
            out_path = os.path.join(out_dir, f"rec_comm_{k}.png")
            draw_communities(edges, id2name, out_path, args.with_labels, args.seed, args.layout, args.dpi)
            print("saved", out_path)
    finally:
        conn.close()

if __name__ == "__main__":
    main()