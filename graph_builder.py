import json
import sqlite3
from itertools import combinations
from typing import Iterable, List, Optional


class GraphBuilder:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA synchronous=NORMAL;")
        self._init_schema()
        self._name_cache: dict[str, int] = {}

    def _init_schema(self) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS nodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS edges (
                src INTEGER NOT NULL,
                dst INTEGER NOT NULL,
                UNIQUE(src, dst)
            );
            """
        )
        # Indexes to speed up lookups
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_nodes_name ON nodes(name);
            """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_edges_src ON edges(src);
            """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_edges_dst ON edges(dst);
            """
        )
        self.conn.commit()

    def close(self) -> None:
        self.conn.commit()
        self.conn.close()

    def _normalize_name(self, name: Optional[str]) -> Optional[str]:
        if not name:
            return None
        n = name.strip()
        return n if n else None

    def _normalize_title(self, title: Optional[str]) -> Optional[str]:
        if not title:
            return None
        t = title.strip()
        return t if t else None

    def _parse_main_actors(self, main_actors: Optional[str]) -> List[str]:
        if not main_actors:
            return []
        # Normalize possible fullwidth separator
        s = str(main_actors).replace("／", "/")
        parts = [self._normalize_name(p) for p in s.split("/")]
        return [p for p in parts if p]

    def _get_or_create_node_id(self, name: str) -> int:
        cached = self._name_cache.get(name)
        if cached is not None:
            return cached
        cur = self.conn.cursor()
        cur.execute("INSERT OR IGNORE INTO nodes(name) VALUES (?)", (name,))
        # SELECT id regardless of insert/ignore outcome
        cur.execute("SELECT id FROM nodes WHERE name = ?", (name,))
        row = cur.fetchone()
        if row is None:
            raise RuntimeError(f"Failed to resolve node id for name: {name}")
        node_id = int(row[0])
        self._name_cache[name] = node_id
        return node_id

    def add_movie(self, participant_names: Iterable[str]) -> None:
        # Deduplicate within the same movie
        unique_names = sorted({n for n in (self._normalize_name(x) for x in participant_names) if n})
        if len(unique_names) < 2:
            return
        # Map names to node ids
        ids = [self._get_or_create_node_id(n) for n in unique_names]
        # Insert undirected edges as ordered pairs (min, max)
        cur = self.conn.cursor()
        for a, b in combinations(ids, 2):
            src, dst = (a, b) if a < b else (b, a)
            cur.execute("INSERT OR IGNORE INTO edges(src, dst) VALUES (?, ?)", (src, dst))

    def build_from_file(self, jsonl_path: str, max_lines: Optional[int] = None) -> None:
        # First pass: aggregate all actor names by video_original_name
        title_to_names: dict[str, set[str]] = {}
        processed = 0
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if max_lines is not None and processed >= max_lines:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    # Skip malformed lines
                    continue
                title = self._normalize_title(obj.get("video_original_name"))
                if not title:
                    processed += 1
                    continue
                main_actors = obj.get("main_actors")
                names_list = self._parse_main_actors(main_actors)
                bucket = title_to_names.setdefault(title, set())
                for name in names_list:
                    bucket.add(name)
                processed += 1

        # Second pass: for each title, add edges among all aggregated actors
        titles_processed = 0
        for names in title_to_names.values():
            self.add_movie(list(names))
            titles_processed += 1
            if titles_processed % 50 == 0:
                self.conn.commit()
        self.conn.commit()

    def get_counts(self) -> tuple[int, int]:
        cur = self.conn.cursor()
        cur.execute("SELECT COUNT(*) FROM nodes")
        nodes = int(cur.fetchone()[0])
        cur.execute("SELECT COUNT(*) FROM edges")
        edges = int(cur.fetchone()[0])
        return nodes, edges