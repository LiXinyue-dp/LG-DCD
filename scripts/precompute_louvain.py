"""
Pre-compute Louvain community detection for all datasets.
Saves results to files for quick loading in LGDCD.py.
"""

import numpy as np
import json
import os
from collections import defaultdict
from sklearn.metrics import normalized_mutual_info_score
import time


def louvain_communities_dense(mat, node_list=None, seed=42):
    """Fast Louvain community detection for dense matrix (from utils.py get_mat)."""
    rng = np.random.RandomState(seed)
    if node_list is None:
        node_list = list(range(len(mat)))

    n = len(node_list)

    # Build adjacency dict
    adj = {}
    for i in range(n):
        neighbors = {}
        for j in range(n):
            if mat[node_list[i], node_list[j]] != 0:
                neighbors[j] = mat[node_list[i], node_list[j]]
        adj[i] = neighbors

    k = np.array([sum(adj[i].values()) for i in range(n)])
    m = np.sum(k) / 2.0
    if m == 0:
        return {node_list[i]: 0 for i in range(n)}

    communities = {i: i for i in range(n)}
    comm_weight = defaultdict(float)
    comm_nodes = defaultdict(set)
    for i in range(n):
        c = communities[i]
        comm_nodes[c].add(i)
        for j, w in adj[i].items():
            if i < j:
                comm_weight[c] += w

    def modularity():
        Q = 0.0
        for c in comm_nodes:
            members = list(comm_nodes[c])
            for i in members:
                for j, w in adj[i].items():
                    if i < j and communities[j] == c:
                        ki = k[i]
                        kj = k[j]
                        Q += w - (ki * kj) / (2 * m)
        return Q / (2 * m)

    best_Q = modularity()
    improved = True
    max_iter = 5
    iteration = 0

    while improved and iteration < max_iter:
        improved = False
        iteration += 1
        for i in range(n):
            current_c = communities[i]
            ki = k[i]
            neighbor_comms = {}
            for j, w in adj[i].items():
                cj = communities[j]
                neighbor_comms[cj] = neighbor_comms.get(cj, 0.0) + w

            for target_c, sum_w in neighbor_comms.items():
                if target_c == current_c:
                    continue
                delta_Q = sum_w - (ki * comm_weight[target_c]) / (2 * m)
                delta_Q -= (ki * comm_weight[current_c]) / (2 * m) if current_c != target_c else 0

                if delta_Q > 1e-10:
                    comm_nodes[current_c].discard(i)
                    comm_nodes[target_c].add(i)
                    communities[i] = target_c
                    best_Q += delta_Q
                    improved = True
                    break

    return {node_list[i]: communities[i] for i in range(n)}


def load_mat_from_file(data_path, node_num, time_index):
    """Load snapshot from edge list file into dense matrix."""
    mat0_index = np.zeros((node_num, node_num))
    snapshot_file = f'{data_path}_{time_index}.txt'

    if not os.path.exists(snapshot_file):
        print(f"  WARNING: File not found: {snapshot_file}")
        return None

    with open(snapshot_file, 'r', encoding='utf-8') as f:
        for line in f:
            s = line.strip().split('\n')
            temp = s[0].split()
            temp = [int(t) for t in temp]
            src = temp[0]
            tar = temp[1]
            if mat0_index[src - 1, tar - 1] != 0:
                mat0_index[src - 1, tar - 1] += 1
                mat0_index[tar - 1, src - 1] += 1
            else:
                mat0_index[src - 1, tar - 1] = 1
                mat0_index[tar - 1, src - 1] = 1

    return mat0_index


def save_louvain_result(result, filepath):
    """Save Louvain result to JSON file."""
    # Convert sets to lists for JSON serialization
    serializable = {}
    for node_id, comm_id in result['assignment'].items():
        serializable[str(node_id)] = comm_id

    data = {
        'assignment': serializable,
        'num_communities': result['num_communities'],
        'num_nodes': result['num_nodes'],
        'modularity': result['modularity'],
        'dataset': result.get('dataset', 'unknown'),
        'node_num': result.get('node_num', 0),
        'time_index': result.get('time_index', 0)
    }

    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def compute_modularity(assignment, mat):
    """Compute modularity Q for a community assignment."""
    n = len(mat)
    comm_members = defaultdict(set)
    for node, comm in assignment.items():
        if node < n:
            comm_members[comm].add(node)

    adj = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            w = mat[i, j]
            if w > 0:
                adj[i, j] = w
                adj[j, i] = w

    m = np.sum(np.triu(adj, 1))
    if m == 0:
        return 0.0

    Q = 0.0
    for c in comm_members:
        members = list(comm_members[c])
        for i in members:
            for j in members:
                if i < j:
                    ki = np.sum(adj[i])
                    kj = np.sum(adj[j])
                    Q += adj[i, j] - (ki * kj) / (2 * m)
    return Q / (2 * m)


def main():
    datasets = [
        {
            'name': 'EmailDept1',
            'data_path': "./data/EmailDept1_LDP/EmailDept1",
            'node_num': 319,
            'time_index': 0,
        },
        {
            'name': 'Forum',
            'data_path': "./data/Forum_LDP/FbForum",
            'node_num': 899,
            'time_index': 0,
        },
        {
            'name': 'Tech_AS',
            'data_path': "./data/Tech_LDP/tech",
            'node_num': 5000,
            'time_index': 0,
        },
        {
            'name': 'MathOverflow_a2q',
            'data_path': "./data/MathOverflow_a2q_LDP/MathOverflow_a2q",
            'node_num': 21688,
            'time_index': 0,
        },
    ]

    output_dir = "./precomputed_louvain"
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("Pre-computing Louvain community detection for all datasets")
    print("=" * 70)

    results = {}

    for ds in datasets:
        name = ds['name']
        data_path = ds['data_path']
        node_num = ds['node_num']
        time_index = ds['time_index']

        output_file = os.path.join(output_dir, f"{name}_snapshot0.json")

        print(f"\n[{name}]")
        print(f"  Node count: {node_num}")
        print(f"  Data path: {data_path}")
        print(f"  Output file: {output_file}")

        # Check if already computed
        if os.path.exists(output_file):
            print(f"  [SKIP] Already exists, delete to recompute")
            continue

        # Load snapshot
        t_load = time.time()
        print(f"  Loading snapshot...", end=" ", flush=True)
        mat = load_mat_from_file(data_path, node_num, time_index)
        if mat is None:
            print(f"FAILED - file not found")
            continue
        print(f"done ({time.time() - t_load:.2f}s)")

        # Run Louvain
        t_louvain = time.time()
        print(f"  Running Louvain...", end=" ", flush=True)
        all_nodes = list(range(node_num))
        assignment = louvain_communities_dense(mat, all_nodes, seed=42)
        print(f"done ({time.time() - t_louvain:.2f}s)")

        # Compute modularity
        Q = compute_modularity(assignment, mat)

        # Count communities
        num_communities = len(set(assignment.values()))

        # Save result
        result = {
            'assignment': assignment,
            'num_communities': num_communities,
            'num_nodes': node_num,
            'modularity': Q,
            'dataset': name,
            'node_num': node_num,
            'time_index': time_index
        }

        save_louvain_result(result, output_file)

        results[name] = {
            'communities': num_communities,
            'modularity': Q,
            'time': time.time() - t_load
        }

        print(f"  Communities: {num_communities}")
        print(f"  Modularity: {Q:.4f}")
        print(f"  Saved to: {output_file}")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Output directory: {output_dir}")
    print("\nFiles created:")
    for name in results:
        r = results[name]
        print(f"  - {name}_snapshot0.json: {r['communities']} communities, Q={r['modularity']:.4f}")
    print("=" * 70)


if __name__ == '__main__':
    main()