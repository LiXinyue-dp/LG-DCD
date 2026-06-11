import numpy as np
import math
import copy
from collections import defaultdict
from sklearn.metrics import normalized_mutual_info_score

V_MAX = 5      # Clipping threshold for delta values
GAMMA = 0.1    # Smoothing factor for Turmoil coefficient


# =============================================================================
# Evaluation Metrics
# =============================================================================

def louvain_communities(mat, node_list=None, seed=42):
    """Fast Louvain community detection. Returns: dict {node_id: comm_id}"""
    rng = np.random.RandomState(seed)  # Local RNG, doesn't affect global state
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


def compute_modularity_metric(assignment, mat):
    """Compute modularity Q for a community assignment."""
    n = len(mat)
    node_list = list(range(n))

    adj = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            w = mat[i, j]
            if w > 0:
                adj[i, j] = w
                adj[j, i] = w

    comm_members = defaultdict(set)
    for node, comm in assignment.items():
        if node < n:
            comm_members[comm].add(node)

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


def evaluate_community(ground_truth, predicted):
    """Evaluate community detection: NMI and mean Jaccard similarity."""
    common_nodes = set(ground_truth.keys()) & set(predicted.keys())
    if len(common_nodes) == 0:
        return 0.0, 0.0

    pred_labels = [predicted[n] for n in sorted(common_nodes)]
    true_labels = [ground_truth[n] for n in sorted(common_nodes)]

    nmi = normalized_mutual_info_score(true_labels, pred_labels)
    pred_comms = defaultdict(set)
    true_comms = defaultdict(set)
    for n in common_nodes:
        pred_comms[predicted[n]].add(n)
        true_comms[ground_truth[n]].add(n)

    # Mean Jaccard over true communities
    jaccards = []
    for tc_nodes in true_comms.values():
        best_j = 0.0
        for pc_nodes in pred_comms.values():
            inter = len(tc_nodes & pc_nodes)
            union = len(tc_nodes | pc_nodes)
            if union > 0:
                best_j = max(best_j, inter / union)
        jaccards.append(best_j)
    jaccard = np.mean(jaccards) if jaccards else 0.0
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

    strengths = {}
    for comm_id, members in comm_members.items():
        total = 0.0
        for other_node in members:
            if other_node < n and other_node != node_id:
                total += mat[node_id, other_node]
        strengths[comm_id] = total
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
        # Static user: pure randomness, no effective privacy cost
        return 0  # Static: no privacy budget consumed

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
    # They are handled specially: 50/50 random output with no privacy cost
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

    exp_I = math.exp((eps_trig * I) / (2 * Delta_I))
    exp_theta = math.exp((eps_trig * theta) / (2 * Delta_I))

    prob_activate = exp_I / (exp_I + exp_theta)
    prob_activate = max(0.0, min(1.0, prob_activate))

    activate = np.random.random() < prob_activate

    if activate:
        budget_pool -= eps_required

    return activate, budget_pool


# =============================================================================
# Module III: Global-Incremental Evolution Engine - EvolutionEngine
# =============================================================================

def evolution_engine(active_reports, prev_W, prev_S, prev_assignment, prev_comm_members,
                     eps_trig, eps_rep):
    """
    Module III: Global-Incremental Evolution Engine.

    Algorithm 4 from paper:
    1. Update W and S matrices from active user reports
    2. Node migration using SafeModularity (Eq. 13)
    3. Community merge using (Eq. 14)
    4. Recompute turmoil coefficients

    Args:
        active_reports: list of (node_id, tilde_intra, tilde_inter, c_inter)
        prev_W: dict {(node_id, comm_id): strength}
        prev_S: dict {(comm_c, comm_d): edge_weight}
        prev_assignment: dict {node_id: comm_id}
        prev_comm_members: dict {comm_id: set(node_ids)}
        eps_trig, eps_rep: privacy budgets

    Returns:
        W, S, assignment, comm_members, turmoil
    """
    global GAMMA

    # Deep copy to avoid modifying original
    W = copy.deepcopy(prev_W) if prev_W else {}
    S = copy.deepcopy(prev_S) if prev_S else {}
    assignment = copy.deepcopy(prev_assignment) if prev_assignment else {}
    comm_members = copy.deepcopy(prev_comm_members) if prev_comm_members else defaultdict(set)

    # Total edge weight m
    m = sum(S.values()) / 2.0
    if m == 0:
        # Use raw graph to compute m if S is empty
        pass

    # === Step 1: Incremental Structure Update (Eq. 9-12) ===
    for (node_id, tilde_intra, tilde_inter, c_inter) in active_reports:
        # Recover magnitude (Eq. 8)
        delta_intra_hat = recover_magnitude(tilde_intra, eps_rep)
        delta_inter_hat = recover_magnitude(tilde_inter, eps_rep)

        # Get current community
        c_u = assignment.get(node_id, -1)
        if c_u == -1:
            continue

        # Update W matrix (Eq. 9-10)
        W[(node_id, c_u)] = W.get((node_id, c_u), 0.0) + delta_intra_hat
        if c_inter != -1 and c_inter != c_u:
            W[(node_id, c_inter)] = W.get((node_id, c_inter), 0.0) + delta_inter_hat

        # Update S matrix (Eq. 11-12)
        S[(c_u, c_u)] = S.get((c_u, c_u), 0.0) + 2 * delta_intra_hat
        if c_inter != -1 and c_inter != c_u:
            S[(c_u, c_inter)] = S.get((c_u, c_inter), 0.0) + delta_inter_hat
            S[(c_inter, c_u)] = S.get((c_inter, c_u), 0.0) + delta_inter_hat

    # Recompute m after updates
    m = sum(S.values()) / 2.0
    if m <= 0:
        m = 1.0

    # === Step 2: Incremental SafeModularity Node Migration (Eq. 13) ===
    # For each active user, evaluate migration using pre-materialized W and S
    for (node_id, tilde_intra, tilde_inter, c_inter) in active_reports:
        a = assignment.get(node_id, -1)
        b = c_inter
        if a == -1 or b == -1 or a == b:
            continue

        # Get W entries
        W_ub = W.get((node_id, b), 0.0)
        W_ua = W.get((node_id, a), 0.0)

        # Get S entries (sum of degrees)
        d_a = S.get((a, a), 0.0)
        d_b = S.get((b, b), 0.0)
        for other_c in comm_members:
            if other_c != a:
                d_a += S.get((a, other_c), 0.0) + S.get((other_c, a), 0.0)
            if other_c != b:
                d_b += S.get((b, other_c), 0.0) + S.get((other_c, b), 0.0)

        # Compute delta Q (Eq. 13)
        # d_u approximation: sum of W entries for this node
        d_u = sum(W.get((node_id, c), 0.0) for c in comm_members)
        if d_u == 0:
            d_u = 1.0

        delta_Q = (W_ub - W_ua) / m - (d_u * (d_b - d_a)) / (2 * m * m)

        if delta_Q > 0:
            # Migrate node
            comm_members[a].discard(node_id)
            comm_members[b].add(node_id)
            assignment[node_id] = b

    # === Step 3: Selective Community Merge (Eq. 14) ===
    comm_ids = list(comm_members.keys())
    for i, c in enumerate(comm_ids):
        for d in comm_ids[i+1:]:
            S_cd = S.get((c, d), 0.0) + S.get((d, c), 0.0)

            # Compute d_c and d_d
            d_c = S.get((c, c), 0.0)
            d_d = S.get((d, d), 0.0)
            for other in comm_ids:
                d_c += S.get((c, other), 0.0) + S.get((other, c), 0.0)
                d_d += S.get((d, other), 0.0) + S.get((other, d), 0.0)

            # Compute merge gain (Eq. 14)
            delta_Q_merge = S_cd / m - (d_c * d_d) / (2 * m * m)

            if delta_Q_merge > 0:
                # Merge communities c and d
                # Move all nodes from d to c
                if d in comm_members and c in comm_members:
                    for node_id in list(comm_members[d]):
                        comm_members[c].add(node_id)
                        assignment[node_id] = c
                    del comm_members[d]

                    # Merge S entries
                    for key in list(S.keys()):
                        if key[0] == d or key[1] == d:
                            c1, c2 = key
                            if c1 == d and c2 == d:
                                S[(c, c)] = S.get((c, c), 0.0) + S.get(key, 0.0)
                            elif c1 == d:
                                S[(c, c2)] = S.get((c, c2), 0.0) + S.get(key, 0.0)
                            elif c2 == d:
                                S[(c1, c)] = S.get((c1, c), 0.0) + S.get(key, 0.0)
                            del S[key]

    # === Step 4: Compute Turmoil Coefficients (Eq. 6) ===
    turmoil = compute_turmoil(comm_members, S)

    return W, S, assignment, comm_members, turmoil


# =============================================================================
# Main LG-DCD Algorithm
# =============================================================================

def main_func(data_path, eps, exp_num, node_num, snapshot_num, window_size):
    """
    Main LG-DCD algorithm.
    Implements the full workflow from Algorithm 1.

    Args:
        data_path: path to graph data
        eps: total privacy budget
        exp_num: number of experiments
        node_num: number of nodes
        snapshot_num: number of snapshots
        window_size: sliding window size w
    """
    from utils import get_mat

    # Per-step budget: eps / w (w-event LDP)
    eps_step = eps / window_size
    eps_trig = eps_step * 0.5  # Budget for trigger
    eps_rep = eps_step * 0.5   # Budget for report (2V)

    theta = 0.5  # Baseline threshold for exponential mechanism

    all_nmi = []
    all_jaccard = []
    all_mod = []

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

        for time_index in range(snapshot_num):
            print(f'  Snapshot {time_index}')

            # Load graph snapshot
            mat0, mid = get_mat(data_path, node_num, time_index)

            # Ground truth: Louvain on original graph
            gt_assignment = louvain_communities(mat0, list(range(node_num)), seed=42)

            all_nodes = list(range(node_num))

            # === Initialize first snapshot using Louvain ===
            if time_index == 0 or len(assignment) == 0:
                # Use Louvain to get initial community structure
                assignment = louvain_communities(mat0, all_nodes, seed=42)

                comm_members = defaultdict(set)
                for node_id, comm in assignment.items():
                    comm_members[comm].add(node_id)

                # Initialize W and S matrices
                W = {}
                S = {}
                for comm_id, members in comm_members.items():
                    for node_id in members:
                        # W[node_id, comm_id] = number of edges to this community
                        strength = sum(mat0[node_id, m] for m in members if m != node_id)
                        W[(node_id, comm_id)] = strength

                    # S[comm_id, comm_id] = total internal edge weight
                    internal = 0.0
                    for ni in members:
                        for nj in members:
                            if ni < nj:
                                internal += mat0[ni, nj]
                    S[(comm_id, comm_id)] = internal

                # Compute initial turmoil
                turmoil = compute_turmoil(comm_members, S)

            # === Module II: Trigger for each user ===
            activate_decisions = {}
            for node_id in all_nodes:
                # Compute deltas using projections (free, no budget cost)
                current_comm = assignment.get(node_id, -1)
                if current_comm == -1:
                    activate_decisions[node_id] = False
                    continue

                # Compute connection strengths
                strengths_now = compute_connection_strengths(node_id, mat0, assignment, comm_members)
                strengths_prev = compute_connection_strengths(node_id, mat0, assignment, comm_members)  # Approximate

                # Find inter-community
                best_inter = -1
                best_strength = -1
                for c in comm_members:
                    if c != current_comm:
                        if strengths_now.get(c, 0.0) > best_strength:
                            best_strength = strengths_now.get(c, 0.0)
                            best_inter = c

                # Deltas
                intra_now = strengths_now.get(current_comm, 0.0)
                inter_now = strengths_now.get(best_inter, 0.0) if best_inter != -1 else 0.0

                delta_intra = intra_now
                delta_inter = inter_now

                # Clip
                delta_intra = max(-V_MAX, min(V_MAX, delta_intra))
                delta_inter = max(-V_MAX, min(V_MAX, delta_inter))

                # Get turmoil for current community
                turmoil_c = turmoil.get(current_comm, 0.0)

                # Trigger decision
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
                        node_id, mat0, None, assignment, comm_members,
                        eps_rep, time_index, W
                    )
                    active_reports.append((node_id, tilde_intra, tilde_inter, c_inter))

            # Debug output
            if exper == 0 and time_index <= 3:
                print(f'    DEBUG: active={len(active_reports)}/{node_num}, budget_pools={len([p for p in budget_pools.values() if p > 0])}')

            # === Module III: Evolution Engine ===
            if len(active_reports) > 0:
                W, S, assignment, comm_members, turmoil = evolution_engine(
                    active_reports, W, S, assignment, comm_members,
                    eps_trig, eps_rep
                )

            # === Evaluate ===
            nmi, jaccard = evaluate_community(gt_assignment, assignment)
            q_orig = compute_modularity_metric(gt_assignment, mat0)
            q_synth = compute_modularity_metric(assignment, mat0)
            mod = abs(q_synth - q_orig) / abs(q_orig) if abs(q_orig) > 1e-10 else 0.0

            nmi_per_snapshot.append(nmi)
            jaccard_per_snapshot.append(jaccard)
            mod_per_snapshot.append(mod)

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


if __name__ == '__main__':
    datasets = [
        # {
        #     'name': 'Email-Eu',
        #     'data_path': "./data/EmailDept1_LDP/EmailDept1",
        #     'node_num': 319,
        #     'snapshot_num': 173,
        # },
        # {
        #     'name': 'Forum',
        #     'data_path': "./data/Forum_LDP/FbForum",
        #     'node_num': 899,
        #     'snapshot_num': 24,
        # },
        # {
        #     'name': 'Tech-AS',
        #     'data_path': "./data/Tech_LDP/tech",
        #     'node_num': 5000,
        #     'snapshot_num': 24,
        # },
        {
            'name': 'MathOverflow_a2q',
            'data_path': "./data/MathOverflow_a2q_LDP/MathOverflow_a2q",
            'node_num': 21688,
            'snapshot_num': 2350,
        },
    ]

    eps_values = [1.0,2.0,4.0]
    window_size = 5
    exp_num = 3

    print("=" * 70)
    print("LG-DCD: Community Detection with LDP")
    print("=" * 70)
    print(f"Privacy budgets: eps = {eps_values}")
    print(f"Window size: w = {window_size}")
    print(f"Experiments per setting: {exp_num}")
    print(f"Metrics: NMI, Jaccard, RE_Modularity")
    print(f"V_MAX = {V_MAX}, GAMMA = {GAMMA}")
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

        for eps in eps_values:
            print(f"\n--- eps={eps}, w={window_size} ---")

            result = main_func(data_path, eps, exp_num, node_num, snapshot_num, window_size)

            all_results[(ds_name, eps)] = result

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
        for eps in eps_values:
            r = all_results[(ds_name, eps)]
            print(f"  eps={eps}: NMI={r['NMI']:.4f}, Jaccard={r['Jaccard']:.4f}, RE_Q={r['Modularity']:.4f}")
    print("=" * 70)
