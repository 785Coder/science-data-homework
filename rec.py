import argparse
import json
import os
import sys
import time
import random
import math
import hashlib
from collections import defaultdict, Counter

try:
    import psutil  # type: ignore
except Exception:
    psutil = None

try:
    import numpy as np  # type: ignore
except Exception:
    np = None

try:
    import networkx as nx  # type: ignore
except Exception:
    nx = None

try:
    from sklearn.feature_extraction.text import HashingVectorizer  # type: ignore
    from sklearn.preprocessing import normalize as sk_normalize  # type: ignore
except Exception:
    HashingVectorizer = None
    sk_normalize = None

try:
    import faiss  # type: ignore
except Exception:
    faiss = None

def load_records(path, limit=None):
    records = []
    if not os.path.exists(path):
        return records
    with open(path, "r", encoding="utf-8") as f:
        head = f.read(1)
        f.seek(0)
        if head == "[":
            data = json.load(f)
            for i, item in enumerate(data):
                if limit and i >= limit:
                    break
                records.append(item)
        else:
            for i, line in enumerate(f):
                if not line.strip():
                    continue
                if limit and i >= limit:
                    break
                try:
                    item = json.loads(line)
                    records.append(item)
                except Exception:
                    continue
    return records

def load_graph_db(db_path, degree_topk=None):
    import sqlite3
    id_to_name = {}
    name_to_id = {}
    adj = {}
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    for row in cur.execute("SELECT id, name FROM nodes"):
        i, n = int(row[0]), str(row[1])
        id_to_name[i] = n
        name_to_id[n] = i
        adj[i] = set()
    for row in cur.execute("SELECT src, dst FROM edges"):
        a, b = int(row[0]), int(row[1])
        if a in adj and b in adj:
            adj[a].add(b)
            adj[b].add(a)
    conn.close()
    if degree_topk is not None and degree_topk > 0:
        deg = sorted(((i, len(adj[i])) for i in adj.keys()), key=lambda x: x[1], reverse=True)
        keep = set(i for i, _ in deg[:degree_topk])
        id_to_name = {i: id_to_name[i] for i in keep}
        name_to_id = {id_to_name[i]: i for i in keep}
        adj = {i: set(j for j in nbrs if j in keep) for i, nbrs in adj.items() if i in keep}
    names = [id_to_name[i] for i in sorted(id_to_name.keys())]
    order = {i: idx for idx, i in enumerate(sorted(id_to_name.keys()))}
    neighbors_by_idx = {order[i]: [id_to_name[j] for j in sorted(adj[i])] for i in adj.keys()}
    G = None
    if nx is not None:
        G = nx.Graph()
        for i in id_to_name.keys():
            G.add_node(order[i], name=id_to_name[i])
        for i, nbrs in adj.items():
            for j in nbrs:
                if i in id_to_name and j in id_to_name:
                    G.add_edge(order[i], order[j])
    return names, neighbors_by_idx, G

def pick_name(item):
    return item.get("video_original_name") or item.get("name") or item.get("title") or str(item.get("id", ""))

def pick_actors(item):
    a = item.get("main_actors") or item.get("actors") or []
    if isinstance(a, str):
        return [x for x in a.replace("/", " ").replace(",", " ").split() if x]
    return [str(x) for x in a if x]

def pick_truth(item):
    t = item.get("rec_same_videos") or item.get("similar") or []
    if isinstance(t, str):
        return [x for x in t.replace("/", " ").replace(",", " ").split() if x]
    return [pick_name(x) if isinstance(x, dict) else str(x) for x in t]

def tokenize_name(name):
    s = name.lower()
    buf = []
    cur = []
    for ch in s:
        if ch.isalnum() or ord(ch) > 127:
            cur.append(ch)
        else:
            if cur:
                buf.append("".join(cur))
                cur = []
    if cur:
        buf.append("".join(cur))
    return buf

def hashed_vector(tokens, dim):
    vec = [0.0] * dim
    for tok in tokens:
        h = hashlib.md5(tok.encode("utf-8")).hexdigest()
        idx = int(h[:8], 16) % dim
        vec[idx] += 1.0
    return vec

def normalize(v):
    if np is not None:
        arr = np.asarray(v, dtype=np.float32)
        n = np.linalg.norm(arr)
        if n > 0:
            arr = arr / n
        return arr
    s = math.sqrt(sum(x * x for x in v))
    if s > 0:
        return [x / s for x in v]
    return v

def build_embeddings(items, dim):
    names = [pick_name(x) for x in items]
    actors = [pick_actors(x) for x in items]
    if HashingVectorizer is not None and np is not None:
        docs = []
        for i in range(len(items)):
            toks = tokenize_name(names[i]) + actors[i]
            docs.append(" ".join(toks))
        hv = HashingVectorizer(n_features=dim, analyzer=lambda s: s.split(), alternate_sign=False, norm=None)
        Xs = hv.transform(docs)
        X = Xs.toarray().astype(np.float32)
        X = sk_normalize(X, norm="l2", axis=1)
        return names, X
    vecs = []
    for i in range(len(items)):
        toks = tokenize_name(names[i]) + actors[i]
        v = hashed_vector(toks, dim)
        vecs.append(normalize(v))
    return names, vecs

def l2(a, b):
    if np is not None:
        d = a - b
        return float(np.dot(d, d))
    return sum((x - y) * (x - y) for x, y in zip(a, b))

def cosine(a, b):
    if np is not None:
        return 1.0 - float(np.dot(a, b))
    s = sum(x * y for x, y in zip(a, b))
    return 1.0 - s

def dist_fn(metric):
    return l2 if metric == "l2" else cosine

def kmeans_pp_init(X, k, metric, rng):
    n = len(X)
    centers = []
    idx0 = rng.randrange(n)
    centers.append(X[idx0])
    for _ in range(1, k):
        dists = []
        df = dist_fn(metric)
        for i in range(n):
            m = min(df(X[i], c) for c in centers)
            dists.append(m)
        total = sum(dists)
        if total == 0:
            centers.append(X[rng.randrange(n)])
            continue
        r = rng.random() * total
        s = 0.0
        pick = 0
        for i, d in enumerate(dists):
            s += d
            if s >= r:
                pick = i
                break
        centers.append(X[pick])
    return centers

def kmeans_fit(X, k, metric, max_iter, seed):
    rng = random.Random(seed)
    centers = kmeans_pp_init(X, k, metric, rng)
    df = dist_fn(metric)
    for _ in range(max_iter):
        assigns = [[] for _ in range(k)]
        for i, x in enumerate(X):
            j = min(range(k), key=lambda c: df(x, centers[c]))
            assigns[j].append(i)
        new_centers = []
        for c in range(k):
            if not assigns[c]:
                new_centers.append(centers[c])
                continue
            if np is not None:
                m = np.mean(np.asarray([X[i] for i in assigns[c]], dtype=np.float32), axis=0)
                new_centers.append(m)
            else:
                acc = [0.0] * len(X[0])
                for i in assigns[c]:
                    for j in range(len(acc)):
                        acc[j] += X[i][j]
                for j in range(len(acc)):
                    acc[j] /= len(assigns[c])
                new_centers.append(acc)
        shift = 0.0
        for a, b in zip(centers, new_centers):
            shift += df(a, b)
        centers = new_centers
        if shift < 1e-6:
            break
    assigns = [[] for _ in range(k)]
    for i, x in enumerate(X):
        j = min(range(k), key=lambda c: df(x, centers[c]))
        assigns[j].append(i)
    return centers, assigns

class IVFFlatIndex:
    def __init__(self, nlist=16, metric="cosine", nprobe=1, seed=42):
        self.nlist = nlist
        self.metric = metric
        self.nprobe = nprobe
        self.seed = seed
        self.centers = None
        self.lists = None
        self.X = None
        self.faiss_index = None

    def build(self, X):
        self.X = X
        if faiss is not None and np is not None:
            d = X.shape[1] if hasattr(X, "shape") else len(X[0])
            if self.metric == "l2":
                quant = faiss.IndexFlatL2(d)
            else:
                quant = faiss.IndexFlatIP(d)
            index = faiss.IndexIVFFlat(quant, d, max(1, self.nlist))
            if self.metric != "l2":
                pass
            Xa = np.asarray(X, dtype=np.float32)
            index.train(Xa)
            index.add(Xa)
            index.nprobe = max(1, self.nprobe)
            self.faiss_index = index
            return
        centers, lists = kmeans_fit(X, self.nlist, self.metric, 20, self.seed)
        self.centers = centers
        self.lists = lists

    def search(self, q, topk):
        if self.faiss_index is not None and np is not None:
            qa = np.asarray([q], dtype=np.float32)
            D, I = self.faiss_index.search(qa, topk)
            return list(zip(I[0].tolist(), D[0].tolist()))
        df = dist_fn(self.metric)
        if self.centers is None or self.lists is None:
            return []
        dcs = [(i, df(q, c)) for i, c in enumerate(self.centers)]
        dcs.sort(key=lambda x: x[1])
        probes = [i for i, _ in dcs[:max(1, self.nprobe)]]
        cand = []
        for p in probes:
            for idx in self.lists[p]:
                cand.append((idx, df(q, self.X[idx])))
        cand.sort(key=lambda x: x[1])
        return cand[:topk]

class HNSWIndex:
    def __init__(self, M=16, ef=64, metric="cosine"):
        self.M = M
        self.ef = ef
        self.metric = metric
        self.backend = None
        self.X = None
        self.graph = None

    def _try_hnswlib(self):
        try:
            import hnswlib  # type: ignore
            return hnswlib
        except Exception:
            return None

    def build(self, X):
        self.X = X
        hnswlib = self._try_hnswlib()
        if hnswlib is not None and np is not None:
            dim = len(X[0])
            space = "cosine" if self.metric != "l2" else "l2"
            p = hnswlib.Index(space=space, dim=dim)
            p.init_index(max_elements=len(X), ef_construction=self.ef, M=self.M)
            p.add_items(np.asarray(X, dtype=np.float32))
            p.set_ef(self.ef)
            self.backend = p
        else:
            df = dist_fn(self.metric)
            adj = [[] for _ in range(len(X))]
            for i, xi in enumerate(X):
                d = [(j, df(xi, X[j])) for j in range(len(X)) if j != i]
                d.sort(key=lambda x: x[1])
                adj[i] = [j for j, _ in d[:self.M]]
            self.graph = adj

    def search(self, q, topk):
        if self.backend is not None:
            labels, dists = self.backend.knn_query(np.asarray([q], dtype=np.float32), k=topk)
            return list(zip(labels[0].tolist(), dists[0].tolist()))
        df = dist_fn(self.metric)
        cand = [(i, df(q, x)) for i, x in enumerate(self.X)]
        cand.sort(key=lambda x: x[1])
        return cand[:topk]

    def communities(self):
        edges = []
        if self.backend is not None and np is not None:
            for i, xi in enumerate(self.X):
                labels, _ = self.backend.knn_query(np.asarray([xi], dtype=np.float32), k=min(self.M, len(self.X)))
                for j in labels[0].tolist():
                    if i != j:
                        edges.append((i, j))
        else:
            for i, nbrs in enumerate(self.graph or []):
                for j in nbrs:
                    edges.append((i, j))
        if nx is None:
            return {i: i for i in range(len(self.X))}
        G = nx.Graph()
        G.add_nodes_from(range(len(self.X)))
        G.add_edges_from(edges)
        try:
            import community as community_louvain  # type: ignore
            part = community_louvain.best_partition(G)
            return part
        except Exception:
            comms = list(nx.community.greedy_modularity_communities(G))
            label = {}
            for cid, c in enumerate(comms):
                for node in c:
                    label[node] = cid
            return label

def precision_recall_f1(pred, truth):
    tp = len(set(pred) & set(truth))
    p = tp / max(1, len(pred))
    r = tp / max(1, len(truth))
    if p + r == 0:
        f1 = 0.0
    else:
        f1 = 2 * p * r / (p + r)
    return p, r, f1

def measure_memory():
    if psutil is not None:
        proc = psutil.Process(os.getpid())
        return proc.memory_info().rss
    import tracemalloc
    if not tracemalloc.is_tracing():
        tracemalloc.start()
    cur, peak = tracemalloc.get_traced_memory()
    return peak

def run_algo(algo, names, X, truth_map, topk, metric):
    t0 = time.perf_counter()
    mem0 = measure_memory()
    if algo == "ivfflat":
        index = IVFFlatIndex(nlist=max(1, int(math.sqrt(len(X)))), metric=metric, nprobe=2)
    else:
        index = HNSWIndex(M=16, ef=64, metric=metric)
    index.build(X)
    mem1 = measure_memory()
    build_time = time.perf_counter() - t0
    build_mem = max(0, mem1 - mem0)
    ps, rs, fs = [], [], []
    for i, q in enumerate(X):
        res = index.search(q, topk)
        rec = [names[j] for j, _ in res if j != i]
        truth = truth_map.get(names[i], [])
        p, r, f1 = precision_recall_f1(rec, truth)
        ps.append(p)
        rs.append(r)
        fs.append(f1)
    avg_p = sum(ps) / max(1, len(ps))
    avg_r = sum(rs) / max(1, len(rs))
    avg_f = sum(fs) / max(1, len(fs))
    comm = {}
    if algo == "hnsw":
        comm = index.communities()
    return {
        "algo": algo,
        "build_time": build_time,
        "build_mem": build_mem,
        "precision": avg_p,
        "recall": avg_r,
        "f1": avg_f,
        "communities": comm,
    }

def analyze_pros_consivfflat():
    return "IVFFlat构建较慢但查询快，需良好聚类；内存相对可控，nprobe影响准确率与速度。"

def analyze_pros_conshnsw():
    return "HNSW构建较快、查询稳定低延迟，内存开销较大；近邻图可用于社区发现。"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, default=None)
    ap.add_argument("--db", type=str, default=None)
    ap.add_argument("--algo", type=str, default="both", choices=["ivfflat", "hnsw", "both"])
    ap.add_argument("--metric", type=str, default="cosine", choices=["cosine", "l2"])
    ap.add_argument("--dim", type=int, default=512)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--graph_topk", type=int, default=None)
    args = ap.parse_args()

    names = []
    X = None
    truth_map = {}
    G = None
    if args.db:
        names, neighbors_by_idx, G = load_graph_db(args.db, args.graph_topk)
        docs = []
        for i in range(len(names)):
            toks = [names[i]] + neighbors_by_idx.get(i, [])
            docs.append(" ".join(toks))
        if HashingVectorizer is not None and np is not None:
            hv = HashingVectorizer(n_features=args.dim, analyzer=lambda s: s.split(), alternate_sign=False, norm=None)
            Xs = hv.transform(docs)
            Xa = Xs.toarray().astype(np.float32)
            X = sk_normalize(Xa, norm="l2", axis=1)
        else:
            vecs = []
            for i in range(len(names)):
                toks = docs[i].split()
                v = hashed_vector(toks, args.dim)
                vecs.append(normalize(v))
            X = np.asarray(vecs, dtype=np.float32) if np is not None else vecs
        for i in range(len(names)):
            truth_map[names[i]] = neighbors_by_idx.get(i, [])
    else:
        items = load_records(args.input or "demo-data.json", args.limit)
        if not items:
            print("no data")
            sys.exit(1)
        names, X = build_embeddings(items, args.dim)
        for it in items:
            truth_map[pick_name(it)] = pick_truth(it)
    algos = [args.algo] if args.algo != "both" else ["ivfflat", "hnsw"]
    results = []
    for a in algos:
        r = run_algo(a, names, X, truth_map, args.topk, args.metric)
        results.append(r)
    for r in results:
        print("algo", r["algo"]) 
        print("build_time_s", round(r["build_time"], 6))
        print("build_mem_bytes", r["build_mem"]) 
        print("acc(precision)", round(r["precision"], 6))
        print("recall", round(r["recall"], 6))
        print("f1", round(r["f1"], 6))
        if r["algo"] == "hnsw":
            print("communities", len(set(r["communities"].values())) if r["communities"] else 0)
        print()
    if args.db and nx is not None and G is not None:
        try:
            import community as community_louvain  # type: ignore
            part = community_louvain.best_partition(G)
            print("graph_louvain_communities", len(set(part.values())))
        except Exception:
            comms = list(nx.community.greedy_modularity_communities(G))
            print("graph_greedy_communities", len(comms))
    print("IVFFlat", analyze_pros_consivfflat())
    print("HNSW", analyze_pros_conshnsw())

if __name__ == "__main__":
    main()