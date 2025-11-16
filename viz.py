import argparse
import os
import sqlite3
from typing import Dict, List, Set, Tuple, Optional

import matplotlib

# Use non-interactive backend for headless environments
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib import font_manager


def load_topk_subgraph(conn: sqlite3.Connection, k: int) -> Tuple[List[Tuple[int, int]], Dict[int, str]]:
    cur = conn.cursor()
    cur.execute("SELECT src, dst FROM edges")
    degree: Dict[int, int] = {}
    edges: List[Tuple[int, int]] = []
    for src, dst in cur.fetchall():
        edges.append((src, dst))
        degree[src] = degree.get(src, 0) + 1
        degree[dst] = degree.get(dst, 0) + 1
    # Pick top-k by degree
    selected: Set[int] = set(x for x, _ in sorted(degree.items(), key=lambda x: x[1], reverse=True)[:k])
    sub_edges = [(s, d) for s, d in edges if s in selected and d in selected]
    # Load names
    cur.execute("SELECT id, name FROM nodes WHERE id IN (%s)" % ",".join(str(i) for i in selected))
    id2name: Dict[int, str] = {int(i): n for i, n in cur.fetchall()}
    return sub_edges, id2name


def load_ego_subgraph(conn: sqlite3.Connection, center_name: str, k: int) -> Tuple[List[Tuple[int, int]], Dict[int, str]]:
    cur = conn.cursor()
    cur.execute("SELECT id FROM nodes WHERE name = ?", (center_name,))
    row = cur.fetchone()
    if not row:
        raise SystemExit(f"Center name not found: {center_name}")
    center_id = int(row[0])
    # Neighbors of center
    cur.execute("SELECT dst FROM edges WHERE src = ? UNION SELECT src FROM edges WHERE dst = ?", (center_id, center_id))
    neighbors = [int(r[0]) for r in cur.fetchall()]
    neighbors = neighbors[:k]
    selected: Set[int] = set(neighbors + [center_id])
    # Edges among selected set
    # Build parameter list for IN clause
    in_clause = ",".join(str(i) for i in selected)
    cur.execute(f"SELECT src, dst FROM edges WHERE src IN ({in_clause}) AND dst IN ({in_clause})")
    sub_edges = [(int(s), int(d)) for s, d in cur.fetchall()]
    # Names
    cur.execute(f"SELECT id, name FROM nodes WHERE id IN ({in_clause})")
    id2name: Dict[int, str] = {int(i): n for i, n in cur.fetchall()}
    return sub_edges, id2name


def load_all_nodes_and_degrees(conn: sqlite3.Connection) -> Tuple[Dict[int, str], Dict[int, int]]:
    cur = conn.cursor()
    cur.execute("SELECT id, name FROM nodes")
    id2name: Dict[int, str] = {int(i): n for i, n in cur.fetchall()}
    # Aggregate degrees from both src and dst
    degrees: Dict[int, int] = {}
    cur.execute("SELECT src AS id, COUNT(*) AS d FROM edges GROUP BY src")
    for i, d in cur.fetchall():
        degrees[int(i)] = degrees.get(int(i), 0) + int(d)
    cur.execute("SELECT dst AS id, COUNT(*) AS d FROM edges GROUP BY dst")
    for i, d in cur.fetchall():
        degrees[int(i)] = degrees.get(int(i), 0) + int(d)
    return id2name, degrees


def sample_edges(conn: sqlite3.Connection, sample_count: int) -> List[Tuple[int, int]]:
    if sample_count <= 0:
        return []
    cur = conn.cursor()
    cur.execute("SELECT src, dst FROM edges ORDER BY RANDOM() LIMIT ?", (sample_count,))
    return [(int(s), int(d)) for s, d in cur.fetchall()]


def configure_font(font_path: Optional[str]) -> None:
    if font_path and os.path.exists(font_path):
        try:
            font_manager.fontManager.addfont(font_path)
            font_name = font_manager.FontProperties(fname=font_path).get_name()
            matplotlib.rcParams["font.family"] = "sans-serif"
            matplotlib.rcParams["font.sans-serif"] = [font_name]
            matplotlib.rcParams["axes.unicode_minus"] = False
            print(f"Using font: {font_name} from {font_path}")
        except Exception as e:
            print(f"Warning: failed to configure font '{font_path}': {e}")


def draw_graph(edges: List[Tuple[int, int]], id2name: Dict[int, str], out_path: str, with_labels: bool, seed: Optional[int], node_degrees: Optional[Dict[int, int]] = None, layout: str = "spring", dpi: int = 150) -> None:
    G = nx.Graph()
    G.add_nodes_from(id2name.keys())
    if edges:
        G.add_edges_from(edges)
    # Node sizes scaled by degree (prefer provided degrees if available)
    degrees = node_degrees if node_degrees is not None else dict(G.degree())
    sizes = [max(50, degrees.get(n, 0) * 2) for n in G.nodes()]
    # Layout selection
    if layout == "spring":
        pos = nx.spring_layout(G, seed=seed) if seed is not None else nx.spring_layout(G)
    elif layout == "random":
        pos = nx.random_layout(G)
    elif layout == "spectral":
        pos = nx.spectral_layout(G)
    else:
        pos = nx.spring_layout(G, seed=seed) if seed is not None else nx.spring_layout(G)
    plt.figure(figsize=(12, 9), dpi=dpi)
    nx.draw_networkx_nodes(G, pos, node_size=sizes, node_color="#4C78A8", alpha=0.8)
    nx.draw_networkx_edges(G, pos, edge_color="#9C9C9C", alpha=0.3, width=0.5)
    if with_labels:
        labels = {n: id2name.get(n, str(n)) for n in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=6)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize actor co-occurrence graph from SQLite database")
    parser.add_argument("--db", required=True, help="Path to SQLite database (graph.sqlite)")
    parser.add_argument("--mode", choices=["topk", "ego", "full"], default="topk", help="Subgraph selection mode")
    parser.add_argument("--k", type=int, default=200, help="Number of nodes for subgraph")
    parser.add_argument("--center-name", type=str, default=None, help="Center actor name for ego mode")
    parser.add_argument("--out", type=str, default="viz.png", help="Output image path")
    parser.add_argument("--seed", type=int, default=42, help="Layout seed for reproducibility")
    parser.add_argument("--with-labels", action="store_true", help="Draw node labels (slower/cluttered)")
    parser.add_argument("--edge-sample-count", type=int, default=0, help="For full mode: randomly sample N edges to aid layout")
    parser.add_argument("--layout", type=str, choices=["spring", "random", "spectral"], default="spring", help="Layout algorithm for drawing")
    parser.add_argument("--dpi", type=int, default=150, help="Output image DPI")
    parser.add_argument("--font", type=str, default="/home/rtx2080ti/code/data-sci-work/HarmonyOS_Sans_SC.ttf", help="Path to TTF font for labels")
    args = parser.parse_args()

    db_path = os.path.abspath(args.db)
    configure_font(args.font)
    conn = sqlite3.connect(db_path)
    try:
        if args.mode == "topk":
            edges, id2name = load_topk_subgraph(conn, args.k)
            node_degrees = None
        elif args.mode == "ego":
            if not args.center_name:
                raise SystemExit("--center-name is required for ego mode")
            edges, id2name = load_ego_subgraph(conn, args.center_name, args.k)
            node_degrees = None
        else:  # full
            id2name, degrees = load_all_nodes_and_degrees(conn)
            edges = sample_edges(conn, args.edge_sample_count)
            node_degrees = degrees
        print(f"Subgraph edges: {len(edges)}, nodes: {len(id2name)}")
        out_path = os.path.abspath(args.out)
        draw_graph(edges, id2name, out_path, args.with_labels, args.seed, node_degrees=node_degrees, layout=args.layout, dpi=args.dpi)
        print(f"Saved visualization to: {out_path}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()