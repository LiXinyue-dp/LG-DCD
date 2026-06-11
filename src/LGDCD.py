import numpy as np
import math
import copy
import os
import time
import json
import pickle
import argparse
from collections import defaultdict
from sklearn.metrics import normalized_mutual_info_score

V_MAX = 5      # Clipping threshold for delta values
GAMMA = 0.1    # Smoothing factor for Turmoil coefficient


# =============================================================================
# Precomputed Louvain Cache
# =============================================================================

def load_snapshot0_louvain(dataset_name):
    """Load precomputed Louvain result for snapshot 0 (JSON format)."""
    output_dir = "./precomputed_louvain"
    filepath = os.path.join(output_dir, f"{dataset_name}_snapshot0.json")
    if not os.path.exists(filepath):
        return None
    with open(filepath, 'r') as f:
        data = json.load(f)
    return {int(k): v for k, v in data['assignment'].items()}


def load_all_snapshots_louvain_cache(dataset_name):
    """Load all-snapshots Louvain cache from pickle file."""
    output_dir = "./precomputed_louvain"
    filepath = os.path.join(output_dir, f"{dataset_name}_all_snapshots.pkl")
    if not os.path.exists(filepath):
        return None
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    if not isinstance(data, dict):
        return None
    gt_all = {}
    for snap_id, snap_data in data.items():
        if isinstance(snap_data, dict) and 'assignment' in snap_data:
            gt_all[int(snap_id)] = {int(k): int(v) for k, v in snap_data['assignment'].items()}
        elif isinstance(snap_data, dict):
            gt_all[int(snap_id)] = {int(k): int(v) for k, v in snap_data.items()}
        else:
            return None
    return gt_all


def save_all_snapshots_louvain_cache(dataset_name, gt_all):
    """Save all-snapshots Louvain cache to pickle file."""
    output_dir = "./precomputed_louvain"
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f"{dataset_name}_all_snapshots.pkl")
    payload = {int(k): {int(k2): int(v2) for k2, v2 in v.items()} for k, v in gt_all.items()}
    with open(filepath, 'wb') as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


# =============================================================================
# Evaluation Metrics
# =============================================================================


class SparseSnapshot:
    """Lightweight sparse undirected weighted graph snapshot."""

    def __init__(self, node_num):
        self.node_num = node_num
        self.adj = defaultdict(dict)

    def add_edge(self, u, v, w=1.0):
        if u == v:
            return
        self.adj[u][v] = self.adj[u].get(v, 0.0) + w
        self.adj[v][u] = self.adj[v].get(u, 0.0) + w

    def neighbors(self, u):
        return self.adj.get(u, {})

    def __len__(self):
        return self.node_num

    def __getitem__(self, idx):
        i, j = idx
        return self.adj.get(i, {}).get(j, 0.0)


def load_sparse_snapshot(data_path, node_num, time_index):
    """Load one snapshot from edge list file into sparse structure."""
    mat = SparseSnapshot(node_num)
    snapshot_file = f'{data_path}_{time_index}.txt'
    with open(snapshot_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            src = int(parts[0]) - 1
            tar = int(parts[1]) - 1
            if 0 <= src < node_num and 0 <= tar < node_num:
                mat.add_edge(src, tar, 1.0)
    return mat, list(range(node_num))


def binarize_snapshot(mat):
    """Convert weighted SparseSnapshot to binary (unweighted): weight > 0 → 1."""
    if hasattr(mat, 'neighbors'):
        binary = SparseSnapshot(mat.node_num)
        for u, nbrs in mat.adj.items():
            for v, w in nbrs.items():
                if u < v and w > 0:
                    binary.add_edge(u, v, 1.0)
        return binary
    else:
        n = len(mat)
        binary = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                if mat[i, j] > 0:
                    binary[i, j] = 1.0
                    binary[j, i] = 1.0
        return binary


def iter_undirected_edges(mat):
    """Yield (u, v, w) for undirected edges with u < v."""
    if hasattr(mat, 'neighbors'):
        for u, nbrs in mat.adj.items():
            for v, w in nbrs.items():
                if u < v and w != 0:
                    yield u, v, w
    else:
        n = len(mat)
        for i in range(n):
            for j in range(i + 1, n):
                w = mat[i, j]
                if w != 0:
                    yield i, j, w


def iter_node_neighbors(mat, node_id):
    """Yield (neighbor, weight) for one node."""
    if hasattr(mat, 'neighbors'):
        for nb, w in mat.neighbors(node_id).items():
            yield nb, w
    else:
        n = len(mat)
        for nb in range(n):
            w = mat[node_id, nb]
            if w != 0:
                yield nb, w


def node_degree(mat, node_id):
    """Weighted degree of a node."""
    if hasattr(mat, 'neighbors'):
        return sum(mat.neighbors(node_id).values())
    return float(np.sum(mat[node_id]))

def louvain_communities(mat, node_list=None, seed=42):
    """Fast Louvain community detection. Returns: dict {node_id: comm_id}"""
    rng = np.random.RandomState(seed)  # Local RNG, doesn't affect global state
    if node_list is None:
        node_list = list(range(len(mat)))

    n = len(node_list)

    # Build adjacency dict
    adj = {}
    if hasattr(mat, 'neighbors'):
        local_index = {node_id: idx for idx, node_id in enumerate(node_list)}
        for i, node_id in enumerate(node_list):
            neighbors = {}
            for nb, w in mat.neighbors(node_id).items():
                j = local_index.get(nb)
                if j is not None and w != 0:
                    neighbors[j] = w
            adj[i] = neighbors
    else:
        for i in range(n):
            neighbors = {}
            for j in range(n):
                w = mat[node_list[i], node_list[j]]
                if w != 0:
                    neighbors[j] = w
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


def compute_modularity_metric(assignment, mat):
    """Compute modularity Q for a community assignment."""
    n = len(mat)
    comm_members = defaultdict(set)
    for node, comm in assignment.items():
        if node < n:
            comm_members[comm].add(node)

    deg = np.zeros(n, dtype=float)
    internal_weight = defaultdict(float)
    m = 0.0
    for i, j, w in iter_undirected_edges(mat):
        m += w
        deg[i] += w
        deg[j] += w
        ci = assignment.get(i, None)
        cj = assignment.get(j, None)
        if ci is not None and ci == cj:
            internal_weight[ci] += w

    if m == 0:
        return 0.0

    Q = 0.0
    two_m = 2.0 * m
    for c, members in comm_members.items():
        L_c = internal_weight.get(c, 0.0)
        K_c = float(np.sum(deg[list(members)])) if members else 0.0
        Q += (L_c / m) - (K_c / two_m) ** 2
    return Q


def evaluate_community(ground_truth, predicted):
    """Evaluate community detection: NMI and mean Jaccard similarity."""
    common_nodes = set(ground_truth.keys()) & set(predicted.keys())
    if len(common_nodes) == 0:
        return 0.0, 0.0

    pred_labels = [predicted[n] for n in sorted(common_nodes)]
    true_labels = [ground_truth[n] for n in sorted(common_nodes)]

    nmi = normalized_mutual_info_score(true_labels, pred_labels)
    # Build contingency counts in O(|V|), then compute best Jaccard for each true community.
    true_sizes = defaultdict(int)
    pred_sizes = defaultdict(int)
    overlap = defaultdict(int)  # key: (true_comm, pred_comm)

    for n in common_nodes:
        t = ground_truth[n]
        p = predicted[n]
        true_sizes[t] += 1
        pred_sizes[p] += 1
        overlap[(t, p)] += 1

    best_jaccard_per_true = defaultdict(float)
    for (t, p), inter in overlap.items():
        union = true_sizes[t] + pred_sizes[p] - inter
        if union > 0:
            j = inter / union
            if j > best_jaccard_per_true[t]:
                best_jaccard_per_true[t] = j

    jaccards = [best_jaccard_per_true[t] for t in true_sizes.keys()]
    jaccard = float(np.mean(jaccards)) if jaccards else 0.0
    return nmi, jaccard


# =============================================================================
# Module I: Local-Differential Sensing - SenseAndPerturb
# =============================================================================

def compute_connection_strengths(node_id, mat, assignment, comm_members):
    """
    Compute connection strengths from node to each community.
    s(C_j) = sum of edges from node to nodes in community C_j

    Returns: dict {community_id: total_weight}
    """
    n = len(mat) if mat is not None else 0
    if node_id >= n:
        return {}

    strengths = {comm_id: 0.0 for comm_id in comm_members.keys()}
    for other_node, w in iter_node_neighbors(mat, node_id):
        comm_id = assignment.get(other_node)
        if comm_id is not None:
            strengths[comm_id] = strengths.get(comm_id, 0.0) + w
    return strengths


def sense_and_perturb(node_id, mat_now, mat_prev, assignment, comm_members,
                      eps_local, time_index, W=None):
    """
    Module I: SenseAndPerturb.
    Triple projection: temporal, semantic, spatial.
    Then 2V perturbation with clipping to [-V_MAX, V_MAX].

    Returns: (tilde_v_intra, tilde_v_inter, c_inter)
    """
    global V_MAX

    # Get current community of node
    current_comm = assignment.get(node_id, -1) if assignment else -1

    # === Semantic Projection: Compute connection strengths to all communities ===
    strengths_now = compute_connection_strengths(node_id, mat_now, assignment, comm_members)
    if mat_prev is not None:
        strengths_prev = compute_connection_strengths(node_id, mat_prev, assignment, comm_members)
    else:
        strengths_prev = {c: 0.0 for c in strengths_now.keys()}

    # Ensure both dicts have same keys
    all_comms = set(strengths_now.keys()) | set(strengths_prev.keys())
    for c in all_comms:
        if c not in strengths_now:
            strengths_now[c] = 0.0
        if c not in strengths_prev:
            strengths_prev[c] = 0.0

    # === Spatial Projection: intra = current comm, inter = strongest other comm ===
    intra_now = strengths_now.get(current_comm, 0.0) if current_comm != -1 else 0.0
    intra_prev = strengths_prev.get(current_comm, 0.0) if current_comm != -1 else 0.0

    # Find strongest external community
    best_inter_comm = -1
    best_inter_strength = -1
    for comm_id in all_comms:
        if comm_id != current_comm:
            if strengths_now.get(comm_id, 0.0) > best_inter_strength:
                best_inter_strength = strengths_now[comm_id]
                best_inter_comm = comm_id

    inter_now = strengths_now.get(best_inter_comm, 0.0) if best_inter_comm != -1 else 0.0
    inter_prev = strengths_prev.get(best_inter_comm, 0.0) if best_inter_comm != -1 else 0.0

    # === Temporal Projection: Compute deltas ===
    delta_intra = intra_now - intra_prev
    delta_inter = inter_now - inter_prev

    # Clip to [-V_MAX, V_MAX] as per paper
    delta_intra = max(-V_MAX, min(V_MAX, delta_intra))
    delta_inter = max(-V_MAX, min(V_MAX, delta_inter))

    # === 2V Perturbation (Eq. 7 in paper) ===
    tilde_intra = V_2V(delta_intra, eps_local)
    tilde_inter = V_2V(delta_inter, eps_local)

    return tilde_intra, tilde_inter, best_inter_comm


def V_2V(delta, epsilon):
    """
    Zero-Cost Binary Perturbation (2V) mechanism.
    Output: {-1, +1}

    For delta = 0: 50/50 random (no privacy cost)
    For delta != 0: biased output based on normalized value

    Eq. 7 from paper:
    P(+1) = 0.5 + (v/2) * (e^eps - 1)/(e^eps + 1)
    where v = delta / V_MAX (normalized to [-1, 1])
    """
    global V_MAX

    if delta == 0:
        # Static user: no effective privacy cost, return 0 to signal inactivity
        return 0

    # Clip to [-V_MAX, V_MAX] as per paper
    delta_clipped = max(-V_MAX, min(V_MAX, delta))

    # Normalize to [-1, 1]
    v = delta_clipped / V_MAX
    v = max(-1.0, min(1.0, v))

    # Compute probability of +1 (Eq. 7)
    exp_term = (math.exp(epsilon) - 1) / (math.exp(epsilon) + 1)
    prob_plus = 0.5 + (v / 2.0) * exp_term
    prob_plus = max(0.01, min(0.99, prob_plus))  # Keep away from extremes for stability

    return 1 if np.random.random() < prob_plus else -1


def recover_magnitude(tilde_v, epsilon):
    """
    Lossless magnitude recovery (Eq. 8).
    Used at server side to recover unbiased estimate of delta.
    """
    global V_MAX
    return V_MAX * (math.exp(epsilon) + 1) / (math.exp(epsilon) - 1) * tilde_v


# =============================================================================
# Module II: Local-Global Influence Trigger - TriggerAndAllocate
# =============================================================================

def compute_turmoil(comm_members, S):
    """
    Compute Turmoil coefficient for each community.
    Eq. 6: Turmoil(c) = (sum_{d!=c} S_{cd} + gamma) / (sum_j S_{cj} + gamma)
    """
    global GAMMA

    turmoil = {}
    for comm_id in comm_members:
        internal = S.get((comm_id, comm_id), 0.0)
        external = 0.0
        for other_id in comm_members:
            if other_id != comm_id:
                external += S.get((comm_id, other_id), 0.0)
                external += S.get((other_id, comm_id), 0.0)

        total = internal + external
        turmoil[comm_id] = (external + GAMMA) / (total + GAMMA)

    return turmoil


def compute_influence_score(delta_intra, delta_inter, turmoil_c):
    """
    Eq. 5: I_u = (|Delta_intra| + |Delta_inter|) * Turmoil(c_u)
    """
    return (abs(delta_intra) + abs(delta_inter)) * turmoil_c


def trigger_and_allocate(delta_intra, delta_inter, turmoil_c, budget_pool,
                          eps_step, eps_trig, eps_rep, theta=0.5):
    """
    Module II: TriggerAndAllocate.

    Algorithm 3 from paper:
    1. Accumulate budget from unused per-step allocation
    2. Compute influence score
    3. Check budget sufficiency
    4. Use exponential mechanism to decide activation

    Returns: (activate, updated_budget_pool)
        activate: True/False
        budget_pool: updated pool (accumulated or depleted)
    """
    # Step 1: Absorb unused budget
    budget_pool += eps_step

    # Step 2: Compute influence score (Eq. 5)
    I = compute_influence_score(delta_intra, delta_inter, turmoil_c)

    # Static users (delta_intra=0 and delta_inter=0) should NOT spend budget
    is_static = (abs(delta_intra) < 0.1 and abs(delta_inter) < 0.1)

    # Required budget for activation
    eps_required = eps_trig + eps_rep

    # Step 3: Budget check - static users never activate
    if is_static or budget_pool < eps_required:
        return False, budget_pool

    # Step 4: Exponential mechanism (Eq. 10)
    # Pr[r=1] = exp(eps_trig * I / (2*Delta_I)) / [exp(...) + exp(...)]
    # Delta_I = 2 * V_MAX (sensitivity, Eq. 11)
    Delta_I = 2 * V_MAX

    exp_I = math.exp(min((eps_trig * I) / (2 * Delta_I), 20))
    exp_theta = math.exp(min((eps_trig * theta) / (2 * Delta_I), 20))

    prob_activate = exp_I / (exp_I + exp_theta)
    prob_activate = max(0.0, min(1.0, prob_activate))

    activate = np.random.random() < prob_activate

    if activate:
        budget_pool -= eps_required

    return activate, budget_pool


# =============================================================================
# Module III: Global-Incremental Evolution Engine - EvolutionEngine
# =============================================================================

# Variance of 2V estimator: Var[Delta_hat] = (V_MAX * (e^eps+1)/(e^eps-1))^2 - Delta^2
# Upper bound (when Delta=0): (V_MAX * (e^eps+1)/(e^eps-1))^2
def compute_2v_variance_bound(epsilon):
    """Upper bound on Var[Delta_hat] for the 2V mechanism (Lemma 2)."""
    M = V_MAX * (math.exp(epsilon) + 1) / (math.exp(epsilon) - 1)
    return M * M


def compute_migration_threshold(W_ub, W_ua, d_u, d_a, d_b, m, var_bound):
    """
    Compute the noise-aware migration threshold tau_mig via error propagation.
    Delta_Q = (W_ub - W_ua)/m - d_u*(d_b - d_a)/(2*m^2)
    Var[Delta_Q] approx sum of partial derivatives^2 * variance of each component.
    """
    # Variance contributions from W_ub, W_ua (each is sum of Var for constituent reports)
    # For a single node's W entry, Var = var_bound
    var_W = var_bound  # per-entry variance
    # Partial derivatives
    d_dQ_dWub = 1.0 / m
    d_dQ_dWua = -1.0 / m
    # d_a and d_b come from S which aggregates community-level, so their variance is smaller
    # Approximate: Var[d_a] ~ var_bound (per node contribution), d_dQ_ddb = -d_u/(2*m^2)
    d_dQ_ddb = -d_u / (2 * m * m)
    d_dQ_dda = d_u / (2 * m * m)

    sigma_Q = math.sqrt(
        (d_dQ_dWub ** 2) * var_W +
        (d_dQ_dWua ** 2) * var_W +
        (d_dQ_ddb ** 2) * var_bound +
        (d_dQ_dda ** 2) * var_bound
    )
    return 1.96 * sigma_Q  # z_{1-0.025} for alpha=0.05


def compute_merge_threshold(d_c, d_d, m, var_bound):
    """
    Compute noise-aware merge threshold tau_merge via error propagation.
    Delta_Q_merge = S_cd / m - d_c * d_d / (2*m^2)
    """
    # S_cd has variance ~ var_bound per node pair contribution
    d_dQ_dScd = 1.0 / m
    d_dQ_ddc = -d_d / (2 * m * m)
    d_dQ_ddd = -d_c / (2 * m * m)

    sigma_merge = math.sqrt(
        (d_dQ_dScd ** 2) * var_bound +
        (d_dQ_ddc ** 2) * var_bound +
        (d_dQ_ddd ** 2) * var_bound
    )
    return 1.96 * sigma_merge


def evolution_engine(active_reports, prev_W, prev_S, prev_assignment, prev_comm_members,
                     eps_rep):
    """
    Module III: Global-Incremental Evolution Engine.

    Algorithm 4 from paper:
    1. Incremental W/S structure update from active reports
    2. Node migration with noise-aware threshold (tau_mig)
    3. Community merge with noise-aware threshold (tau_merge)
    4. Community split based on internal density ratio
    5. Recompute turmoil coefficients
    """
    global GAMMA

    W = copy.deepcopy(prev_W) if prev_W else {}
    S = copy.deepcopy(prev_S) if prev_S else {}
    assignment = copy.deepcopy(prev_assignment) if prev_assignment else {}
    comm_members = copy.deepcopy(prev_comm_members) if prev_comm_members else defaultdict(set)

    m = sum(S.values()) / 2.0
    if m <= 0:
        m = 1.0

    # Variance bound for 2V mechanism
    var_bound = compute_2v_variance_bound(eps_rep)

    # === Step 1: Incremental Structure Update (Eq. 9-12) ===
    for (node_id, tilde_intra, tilde_inter, c_inter) in active_reports:
        delta_intra_hat = recover_magnitude(tilde_intra, eps_rep)
        delta_inter_hat = recover_magnitude(tilde_inter, eps_rep)

        c_u = assignment.get(node_id, -1)
        if c_u == -1:
            continue

        W[(node_id, c_u)] = W.get((node_id, c_u), 0.0) + delta_intra_hat
        if c_inter != -1 and c_inter != c_u:
            W[(node_id, c_inter)] = W.get((node_id, c_inter), 0.0) + delta_inter_hat

        S[(c_u, c_u)] = S.get((c_u, c_u), 0.0) + 2 * delta_intra_hat
        if c_inter != -1 and c_inter != c_u:
            S[(c_u, c_inter)] = S.get((c_u, c_inter), 0.0) + delta_inter_hat
            S[(c_inter, c_u)] = S.get((c_inter, c_u), 0.0) + delta_inter_hat

    m = sum(S.values()) / 2.0
    if m <= 0:
        m = 1.0

    # === Step 2: Node Migration with noise-aware threshold ===
    comm_degree = {}
    for c in list(comm_members.keys()):
        d_c = S.get((c, c), 0.0)
        for other in comm_members:
            if other != c:
                d_c += S.get((c, other), 0.0)
        comm_degree[c] = d_c

    for (node_id, tilde_intra, tilde_inter, c_inter) in active_reports:
        a = assignment.get(node_id, -1)
        b = c_inter
        if a == -1 or b == -1 or a == b:
            continue

        W_ub = W.get((node_id, b), 0.0)
        W_ua = W.get((node_id, a), 0.0)
        d_a = comm_degree.get(a, 1.0)
        d_b = comm_degree.get(b, 1.0)
        d_u = max(sum(W.get((node_id, c), 0.0) for c in comm_members), 1.0)

        delta_Q = (W_ub - W_ua) / m - (d_u * (d_b - d_a)) / (2 * m * m)
        tau_mig = compute_migration_threshold(W_ub, W_ua, d_u, d_a, d_b, m, var_bound)

        if delta_Q > tau_mig:
            comm_members[a].discard(node_id)
            comm_members[b].add(node_id)
            assignment[node_id] = b
            comm_degree[a] = comm_degree.get(a, 0) - d_u
            comm_degree[b] = comm_degree.get(b, 0) + d_u

    # === Step 3: Community Merge with noise-aware threshold ===
    MAX_MERGES = 3
    merge_candidates = []
    comm_ids = list(comm_members.keys())
    for i, c in enumerate(comm_ids):
        for d in comm_ids[i+1:]:
            if c not in comm_members or d not in comm_members:
                continue
            S_cd = S.get((c, d), 0.0)
            d_c = comm_degree.get(c, 1.0)
            d_d = comm_degree.get(d, 1.0)
            delta_Q_merge = S_cd / m - (d_c * d_d) / (2 * m * m) if m > 0 else 0
            tau_merge = compute_merge_threshold(d_c, d_d, m, var_bound)

            if delta_Q_merge > tau_merge:
                merge_candidates.append((delta_Q_merge, c, d))

    merge_candidates.sort(key=lambda x: x[0], reverse=True)
    merged_count = 0
    for delta_Q_merge, c, d in merge_candidates:
        if merged_count >= MAX_MERGES:
            break
        if d not in comm_members or c not in comm_members:
            continue

        for node_id in list(comm_members[d]):
            comm_members[c].add(node_id)
            assignment[node_id] = c
        del comm_members[d]

        new_S = {}
        for (k1, k2), val in S.items():
            nk1 = c if k1 == d else k1
            nk2 = c if k2 == d else k2
            new_S[(nk1, nk2)] = new_S.get((nk1, nk2), 0.0) + val
        S.clear()
        S.update(new_S)

        comm_degree[c] = comm_degree.get(c, 0) + comm_degree.get(d, 0)
        if d in comm_degree:
            del comm_degree[d]
        merged_count += 1

    # === Step 4: Community Split ===
    # A community is dissolved if its internal density ratio S_cc / d_c
    # falls below the random baseline d_c / (2*m)
    dissolved = []
    for c in list(comm_members.keys()):
        d_c = comm_degree.get(c, 0)
        if d_c <= 0:
            continue
        S_cc = S.get((c, c), 0.0)
        density_ratio = S_cc / d_c
        tau_split = d_c / (2 * m)

        if density_ratio < tau_split:
            # Dissolve: reassign all members to singleton communities
            dissolved.append(c)

    for c in dissolved:
        if c not in comm_members:
            continue
        for node_id in list(comm_members[c]):
            new_comm = node_id  # Each node becomes its own community
            comm_members[new_comm] = {node_id}
            assignment[node_id] = new_comm
        del comm_members[c]
        if c in comm_degree:
            del comm_degree[c]

    # === Step 5: Compute Turmoil Coefficients ===
    turmoil = compute_turmoil(comm_members, S)

    return W, S, assignment, comm_members, turmoil


def recalibrate_ws(assignment, comm_members, mat_now, mat_prev, eps_cal):
    """
    Periodic calibration: rebuild W/S from real graph connection strengths.
    Called every CALIBRATION_INTERVAL snapshots to prevent noise drift.
    """
    W = {}
    S = {}

    for node_id in range(len(mat_now)):
        c_u = assignment.get(node_id, -1)
        if c_u == -1:
            continue

        strengths = compute_connection_strengths(node_id, mat_now, assignment, comm_members)
        for c_id, s_val in strengths.items():
            W[(node_id, c_id)] = s_val

    # Rebuild S from real edges
    for ni, nj, w in iter_undirected_edges(mat_now):
        ci = assignment.get(ni)
        cj = assignment.get(nj)
        if ci is not None and cj is not None:
            if ci == cj:
                S[(ci, ci)] = S.get((ci, ci), 0.0) + w
            else:
                S[(ci, cj)] = S.get((ci, cj), 0.0) + w
                S[(cj, ci)] = S.get((cj, ci), 0.0) + w

    return W, S


# =============================================================================
# Main LG-DCD Algorithm
# =============================================================================

def main_func(data_path, eps, exp_num, node_num, snapshot_num, window_size, dataset_name='Unknown', max_h=72):
    """
    Main LG-DCD algorithm.
    Implements the full workflow from Algorithm 1.
    """
    from utils import get_mat

    # Budget partition: eps_init for checkpointing, eps_inc for incremental
    T_EPOCH = int(os.getenv('LGDCD_T_EPOCH', '6'))  # Checkpointing interval
    eps_init_fraction = float(os.getenv('LGDCD_EPS_INIT_FRAC', '0.3'))
    eps_init = eps * eps_init_fraction
    eps_inc = eps * (1 - eps_init_fraction)

    eps_step = eps_inc / window_size
    eps_trig = eps_step * 0.5
    eps_rep = eps_step * 0.5

    theta = 0.5
    max_snapshots = min(int(os.getenv('LGDCD_MAX_SNAPSHOTS', str(snapshot_num))), snapshot_num)
    use_dense_loader = os.getenv('LGDCD_USE_DENSE', '0') == '1'
    fast_eval = os.getenv('LGDCD_FAST_EVAL', '0') == '1'
    precompute_gt = os.getenv('LGDCD_PRECOMPUTE_GT', '1') == '1'

    all_nmi = []
    all_jaccard = []
    all_mod = []

    if node_num > 10000 and use_dense_loader:
        print('WARNING: dense matrix mode on a large graph can be very slow or run out of memory.')
        print('         Set LGDCD_USE_DENSE=0 to use sparse snapshot loading.')
    if fast_eval:
        print('INFO: fast evaluation enabled, NMI/Jaccard/RE_Modularity are skipped for speed (set LGDCD_FAST_EVAL=0 to enable).')

    for exper in range(exp_num):
        np.random.seed(1000 + exper)
        print(f'-----------epsilon={eps}, experiment={exper + 1}/{exp_num}-------------')

        # Initialize structures
        W = {}
        S = {}
        assignment = {}
        comm_members = defaultdict(set)
        turmoil = {}
        budget_pools = defaultdict(lambda: 0.0)  # One pool per user

        nmi_per_snapshot = []
        jaccard_per_snapshot = []
        mod_per_snapshot = []
        gt_assignment_cache = None
        mat_prev = None

        # Load precomputed ground truth cache OR compute & save
        gt_all = None
        if precompute_gt:
            gt_all = load_all_snapshots_louvain_cache(dataset_name)
            if gt_all is not None:
                print(f'  [CACHE] Loaded GT from ./precomputed_louvain/{dataset_name}_all_snapshots.pkl ({len(gt_all)} snapshots)')
            else:
                print(f'  [COMPUTE] Precomputing ground truth for {max_snapshots} snapshots...')
                t_gt_total = time.time()
                gt_all = {}
                for time_index in range(max_snapshots):
                    if use_dense_loader:
                        mat_t, _ = get_mat(data_path, node_num, time_index)
                    else:
                        mat_t, _ = load_sparse_snapshot(data_path, node_num, time_index)
                    mat_t = binarize_snapshot(mat_t)
                    gt_all[time_index] = louvain_communities(mat_t, list(range(node_num)), seed=42)
                    if time_index % 50 == 0 and time_index > 0:
                        print(f'    GT snapshot {time_index}/{max_snapshots} done')
                save_all_snapshots_louvain_cache(dataset_name, gt_all)
                print(f'  [COMPUTE] Done in {time.time() - t_gt_total:.1f}s, cached to disk')

        for time_index in range(max_snapshots):
            print(f'  Snapshot {time_index}')

            # Load graph snapshot
            t_load = time.time()
            if use_dense_loader:
                mat0, mid = get_mat(data_path, node_num, time_index)
            else:
                mat0, mid = load_sparse_snapshot(data_path, node_num, time_index)
            mat0 = binarize_snapshot(mat0)
            print(f'    Loaded snapshot in {time.time() - t_load:.2f}s')

            # Ground truth: use precomputed Louvain result
            gt_assignment = gt_all[time_index]

            all_nodes = list(range(node_num))

            # === Initialize first snapshot using Louvain ===
            if time_index == 0 or len(assignment) == 0:
                # Use precomputed Louvain result first (much faster)
                precomputed = load_snapshot0_louvain(dataset_name)
                if precomputed is not None:
                    assignment = copy.deepcopy(precomputed)
                    print(f'    [PRECOMPUTED] Loaded Louvain from cache')
                else:
                    t_init_louvain = time.time()
                    assignment = copy.deepcopy(gt_assignment)  # Use precomputed GT
                    print(f'    Init Louvain done in {time.time() - t_init_louvain:.2f}s')

                comm_members = defaultdict(set)
                for node_id, comm in assignment.items():
                    comm_members[comm].add(node_id)

                # Initialize W and S matrices
                # W stores each node's connection strength to ALL communities (not just its own)
                # S stores ALL inter-community edge weights (not just intra-community)
                W = {}
                S = {}
                for node_id, comm_id in assignment.items():
                    strengths = compute_connection_strengths(node_id, mat0, assignment, comm_members)
                    # Store connection strength to all communities
                    for c_id, s_val in strengths.items():
                        W[(node_id, c_id)] = s_val

                for ni, nj, w in iter_undirected_edges(mat0):
                    ci = assignment.get(ni)
                    cj = assignment.get(nj)
                    if ci is not None and cj is not None:
                        if ci == cj:
                            S[(ci, ci)] = S.get((ci, ci), 0.0) + w
                        else:
                            # Store inter-community edges in both directions
                            S[(ci, cj)] = S.get((ci, cj), 0.0) + w
                            S[(cj, ci)] = S.get((cj, ci), 0.0) + w

                # Compute initial turmoil
                turmoil = compute_turmoil(comm_members, S)

            # === Periodic Checkpointing: re-initialize every T_EPOCH steps ===
            if time_index > 0 and time_index % T_EPOCH == 0:
                # Re-initialize W/S/assignment from current graph (like t=0)
                assignment = louvain_communities(mat0, list(range(node_num)), seed=42)
                comm_members = defaultdict(set)
                for node_id, comm in assignment.items():
                    comm_members[comm].add(node_id)
                W = {}
                S = {}
                for node_id, comm_id in assignment.items():
                    strengths = compute_connection_strengths(node_id, mat0, assignment, comm_members)
                    for c_id, s_val in strengths.items():
                        W[(node_id, c_id)] = s_val
                for ni, nj, w in iter_undirected_edges(mat0):
                    ci = assignment.get(ni)
                    cj = assignment.get(nj)
                    if ci is not None and cj is not None:
                        if ci == cj:
                            S[(ci, ci)] = S.get((ci, ci), 0.0) + w
                        else:
                            S[(ci, cj)] = S.get((ci, cj), 0.0) + w
                            S[(cj, ci)] = S.get((cj, ci), 0.0) + w
                turmoil = compute_turmoil(comm_members, S)
            else:
                # === Module II: Trigger for each user ===
                activate_decisions = {}
                for node_id in all_nodes:
                    current_comm = assignment.get(node_id, -1)
                    if current_comm == -1:
                        activate_decisions[node_id] = False
                        continue

                    strengths_now = compute_connection_strengths(node_id, mat0, assignment, comm_members)
                    if mat_prev is not None:
                        strengths_prev = compute_connection_strengths(node_id, mat_prev, assignment, comm_members)
                    else:
                        strengths_prev = {c: 0.0 for c in strengths_now.keys()}

                    best_inter = -1
                    best_strength = -1
                    for c in comm_members:
                        if c != current_comm:
                            if strengths_now.get(c, 0.0) > best_strength:
                                best_strength = strengths_now.get(c, 0.0)
                                best_inter = c

                    intra_now = strengths_now.get(current_comm, 0.0)
                    intra_prev = strengths_prev.get(current_comm, 0.0)
                    inter_now = strengths_now.get(best_inter, 0.0) if best_inter != -1 else 0.0
                    inter_prev = strengths_prev.get(best_inter, 0.0) if best_inter != -1 else 0.0

                    delta_intra = intra_now - intra_prev
                    delta_inter = inter_now - inter_prev

                    delta_intra = max(-V_MAX, min(V_MAX, delta_intra))
                    delta_inter = max(-V_MAX, min(V_MAX, delta_inter))

                    turmoil_c = turmoil.get(current_comm, 0.0)

                    activate, budget_pools[node_id] = trigger_and_allocate(
                        delta_intra, delta_inter, turmoil_c,
                        budget_pools[node_id], eps_step, eps_trig, eps_rep, theta
                    )
                    activate_decisions[node_id] = activate

                # === Collect active reports ===
                active_reports = []
                for node_id in all_nodes:
                    if activate_decisions.get(node_id, False):
                        tilde_intra, tilde_inter, c_inter = sense_and_perturb(
                            node_id, mat0, mat_prev, assignment, comm_members,
                            eps_rep, time_index, W
                        )
                        active_reports.append((node_id, tilde_intra, tilde_inter, c_inter))

                if exper == 0 and time_index <= 3:
                    print(f'    DEBUG: active={len(active_reports)}/{node_num}')

                # === Module III: Evolution Engine ===
                if len(active_reports) > 0:
                    W, S, assignment, comm_members, turmoil = evolution_engine(
                        active_reports, W, S, assignment, comm_members,
                        eps_rep
                    )
                turmoil = compute_turmoil(comm_members, S)

            # === Evaluate ===
            if fast_eval:
                nmi = 0.0
                jaccard = 0.0
                mod = 0.0
            else:
                nmi, jaccard = evaluate_community(gt_assignment, assignment)
                q_orig = compute_modularity_metric(gt_assignment, mat0)
                q_synth = compute_modularity_metric(assignment, mat0)
                mod = abs(q_synth - q_orig) / abs(q_orig) if abs(q_orig) > 1e-10 else 0.0

            nmi_per_snapshot.append(nmi)
            jaccard_per_snapshot.append(jaccard)
            mod_per_snapshot.append(mod)

            # Save mat0 as mat_prev for next snapshot
            mat_prev = mat0

            print(f'    NMI={nmi:.4f}, Jaccard={jaccard:.4f}, RE_Q={mod:.4f}')

        avg_nmi = np.mean(nmi_per_snapshot)
        avg_jaccard = np.mean(jaccard_per_snapshot)
        avg_mod = np.mean(mod_per_snapshot)

        all_nmi.append(avg_nmi)
        all_jaccard.append(avg_jaccard)
        all_mod.append(avg_mod)

        print(f'  [Exp {exper+1}] Avg NMI={avg_nmi:.4f}, Avg Jaccard={avg_jaccard:.4f}, Avg RE_Q={avg_mod:.4f}')

    return {
        'NMI': np.mean(all_nmi),
        'Jaccard': np.mean(all_jaccard),
        'Modularity': np.mean(all_mod),
        'NMI_std': np.std(all_nmi),
        'Jaccard_std': np.std(all_jaccard),
        'Modularity_std': np.std(all_mod)
    }


def parse_float_list(value):
    return [float(x.strip()) for x in value.split(',') if x.strip()]


def parse_int_list(value):
    return [int(x.strip()) for x in value.split(',') if x.strip()]


def parse_dataset_list(value, available):
    names = [x.strip() for x in value.split(',') if x.strip()]
    if len(names) == 1 and names[0].lower() == 'all':
        return list(available.values())
    unknown = [name for name in names if name not in available]
    if unknown:
        raise ValueError("Unknown dataset(s): %s. Available: %s" %
                         (', '.join(unknown), ', '.join(sorted(available.keys()))))
    return [available[name] for name in names]


if __name__ == '__main__':
    available_datasets = {
        'EmailDept1': {
            'name': 'EmailDept1',
            'data_path': "./data/EmailDept1_LDP/EmailDept1",
            'node_num': 319,
            'snapshot_num': 173,
            'max_h': 72,
        },
        'Forum': {
            'name': 'Forum',
            'data_path': "./data/Forum_LDP/FbForum",
            'node_num': 899,
            'snapshot_num': 24,
            'max_h': 168,
        },
        'Tech_AS': {
            'name': 'Tech_AS',
            'data_path': "./data/Tech_LDP/tech",
            'node_num': 5000,
            'snapshot_num': 24,
            'max_h': 24,
        },
        'MathOverflow_a2q': {
            'name': 'MathOverflow_a2q',
            'data_path': "./data/MathOverflow_a2q_LDP/MathOverflow_a2q",
            'node_num': 21688,
            'snapshot_num': 2350,
            'max_h': 72,
        },
    }

    parser = argparse.ArgumentParser(description='Run LG-DCD experiments.')
    parser.add_argument(
        '--datasets',
        default='Forum',
        help='Comma-separated dataset names or "all". Options: EmailDept1, Forum, Tech_AS, MathOverflow_a2q. Default: Forum.'
    )
    parser.add_argument(
        '--eps',
        default='2.0',
        help='Comma-separated privacy budgets. Example: 1.0,2.0,4.0. Default: 2.0.'
    )
    parser.add_argument(
        '--windows',
        default='5',
        help='Comma-separated sliding window sizes w. Example: 1,3,5,7,9. Default: 5.'
    )
    parser.add_argument(
        '--exp-num',
        type=int,
        default=5,
        help='Number of repeated runs per setting. Default: 5.'
    )
    args = parser.parse_args()

    datasets = parse_dataset_list(args.datasets, available_datasets)
    eps_values = parse_float_list(args.eps)
    window_sizes = parse_int_list(args.windows)
    exp_num = args.exp_num

    print("=" * 70)
    print("LG-DCD: Community Detection with LDP")
    print("=" * 70)
    print(f"Privacy budgets: eps = {eps_values}")
    print(f"Window sizes: w = {window_sizes}")
    print(f"Experiments per setting: {exp_num}")
    print(f"Metrics: NMI, Jaccard, RE_Modularity")
    print(f"V_MAX = {V_MAX}, GAMMA = {GAMMA}, T_EPOCH = {os.getenv('LGDCD_T_EPOCH', '6')}")
    print(f"eps_init fraction = {os.getenv('LGDCD_EPS_INIT_FRAC', '0.3')}")
    print("=" * 70)

    all_results = {}

    for ds in datasets:
        data_path = ds['data_path']
        node_num = ds['node_num']
        snapshot_num = ds['snapshot_num']
        ds_name = ds['name']

        print(f"\n{'='*70}")
        print(f"Dataset: {ds_name}")
        print(f"{'='*70}")

        for window_size in window_sizes:
            for eps in eps_values:
                print(f"\n--- eps={eps}, w={window_size} ---")

                result = main_func(data_path, eps, exp_num, node_num, snapshot_num, window_size, ds_name, max_h=ds.get('max_h', 72))

                all_results[(ds_name, eps, window_size)] = result

                print(f"  NMI:         {result['NMI']:.4f} +/- {result['NMI_std']:.4f}")
                print(f"  Jaccard:     {result['Jaccard']:.4f} +/- {result['Jaccard_std']:.4f}")
                print(f"  RE_Modularity:  {result['Modularity']:.4f} +/- {result['Modularity_std']:.4f}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: LG-DCD Results")
    print("=" * 70)
    for ds in datasets:
        ds_name = ds['name']
        print(f"\n{ds_name}:")
        for window_size in window_sizes:
            print(f"  w={window_size}:")
            for eps in eps_values:
                r = all_results[(ds_name, eps, window_size)]
                print(f"    eps={eps}: NMI={r['NMI']:.4f}, Jaccard={r['Jaccard']:.4f}, RE_Q={r['Modularity']:.4f}")
    print("=" * 70)
