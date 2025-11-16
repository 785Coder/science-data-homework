import argparse
import os
from graph_builder import GraphBuilder


def main():
    parser = argparse.ArgumentParser(description="Build actor co-occurrence graph from media dataset")
    parser.add_argument("--input", required=True, help="Path to media_dataset.json (JSONL)")
    parser.add_argument("--db", default="graph.sqlite", help="Path to SQLite database to store the graph")
    parser.add_argument("--max-lines", type=int, default=None, help="Optionally limit number of lines for quick runs")
    args = parser.parse_args()

    input_path = os.path.abspath(args.input)
    db_path = os.path.abspath(args.db)

    print(f"Building graph from: {input_path}")
    print(f"Database will be stored at: {db_path}")

    gb = GraphBuilder(db_path)
    try:
        gb.build_from_file(input_path, max_lines=args.max_lines)
        nodes, edges = gb.get_counts()
        print(f"Graph build complete. Nodes: {nodes}, Edges: {edges}")
    finally:
        gb.close()


if __name__ == "__main__":
    main()
