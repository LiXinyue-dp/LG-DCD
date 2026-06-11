"""
Pre-compute Louvain community detection for ALL snapshots of ALL datasets.
Saves results to files for quick loading in LGDCD.py.

Usage: python precompute_louvain_all.py
"""

import numpy as np
import json
import os
import time
import pickle
from collections import defaultdict


def louvain_communities_dense(mat, node_list=None, seed=42):
    """Fast Louvain community detection for dense matrix."""
    rng = np.random.RandomState(seed)
    if node_list is None:
        node_list = list(range(len(mat)))

    n = len(node_list)

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
        return None

    with open(snapshot_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            src, tar = int(parts[0]), int(parts[1])
            if 0 < src <= node_num and 0 < tar <= node_num:
                if mat0_index[src - 1, tar - 1] != 0:
                    mat0_index[src - 1, tar - 1] += 1
                    mat0_index[tar - 1, src - 1] += 1
                else:
                    mat0_index[src - 1, tar - 1] = 1
                    mat0_index[tar - 1, src - 1] = 1

    return mat0_index


def save_louvain_batch(results_dict, filepath):
    """Save batch of Louvain results using pickle for efficiency."""
    with open(filepath, 'wb') as f:
        pickle.dump(results_dict, f)


def load_louvain_batch(filepath):
    """Load batch of Louvain results."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def precompute_dataset(name, data_path, node_num, snapshot_num, output_dir, limit_snapshots=None):
    """Precompute Louvain for all snapshots of a dataset."""
    if limit_snapshots is not None:
        snapshot_num = min(snapshot_num, limit_snapshots)

    output_file = os.path.join(output_dir, f"{name}_all_snapshots.pkl")

    # Check if already computed
    if os.path.exists(output_file):
        file_size = os.path.getsize(output_file) / (1024 * 1024)
        print(f"  [SKIP] Already exists ({file_size:.1f} MB), delete to recompute")
        return output_file

    print(f"  Precomputing {snapshot_num} snapshots...")
    t_start = time.time()

    results = {}
    for t in range(snapshot_num):
        mat = load_mat_from_file(data_path, node_num, t)
        if mat is None:
            print(f"    WARNING: snapshot {t} not found, stopping")
            break

        assignment = louvain_communities_dense(mat, list(range(node_num)), seed=42)
        num_communities = len(set(assignment.values()))
        results[t] = {
            'assignment': assignment,
            'num_communities': num_communities
        }

        if (t + 1) % 100 == 0:
            elapsed = time.time() - t_start
            avg_time = elapsed / (t + 1)
            remaining = avg_time * (snapshot_num - t - 1)
            print(f"    {t+1}/{snapshot_num} done, avg {avg_time:.1f}s/snap, ~{remaining/60:.1f}min remaining")

    t_total = time.time() - t_start
    print(f"  Computed {len(results)} snapshots in {t_total/60:.1f} minutes")

    # Save
    save_louvain_batch(results, output_file)
    file_size = os.path.getsize(output_file) / (1024 * 1024)
    print(f"  Saved to {output_file} ({file_size:.1f} MB)")

    return output_file


def main():
    datasets = [
        {
            'name': 'EmailDept1',
            'data_path': "./data/EmailDept1_LDP/EmailDept1",
            'node_num': 319,
            'snapshot_num': 173,
        },
        {
            'name': 'Forum',
            'data_path': "./data/Forum_LDP/FbForum",
            'node_num': 899,
            'snapshot_num': 24,
        },
        {
            'name': 'Tech_AS',
            'data_path': "./data/Tech_LDP/tech",
            'node_num': 5000,
            'snapshot_num': 24,
        },
        {
            'name': 'MathOverflow_a2q',
            'data_path': "./data/MathOverflow_a2q_LDP/MathOverflow_a2q",
            'node_num': 21688,
            'snapshot_num': 2350,
        },
    ]

    output_dir = "./precomputed_louvain"
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("Pre-computing Louvain for ALL snapshots of ALL datasets")
    print("=" * 70)

    # Limit MathOverflow for testing (set None to compute all)
    MATH_OVERFLOW_LIMIT = None  # Set to 100 for testing, None for full

    results = {}
    for ds in datasets:
        name = ds['name']
        data_path = ds['data_path']
        node_num = ds['node_num']
        snapshot_num = ds['snapshot_num']

        limit = MATH_OVERFLOW_LIMIT if name == 'MathOverflow_a2q' else None

        print(f"\n[{name}] node_num={node_num}, snapshots={snapshot_num}")

        filepath = precompute_dataset(
            name, data_path, node_num, snapshot_num,
            output_dir, limit_snapshots=limit
        )
        results[name] = filepath

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for name, filepath in results.items():
        size = os.path.getsize(filepath) / (1024 * 1024)
        print(f"  {name}: {size:.1f} MB")
    print("=" * 70)


if __name__ == '__main__':
    main()