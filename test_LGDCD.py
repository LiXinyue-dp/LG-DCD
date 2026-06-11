"""
Simple test script for LG-DCD algorithm.
Run this to test basic functionality before full comparison.
"""

import numpy as np
import networkx as nx
from collections import defaultdict
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import LGDCD
import utils


def test_lgdcd_basic():
    """Test basic LG-DCD functionality."""
    print("Testing LG-DCD basic functionality...")

    # Create a simple test graph
    n = 20
    mat = np.zeros((n, n))

    # Create 2 communities: {0-9} and {10-19}
    # Community 1: nodes 0-9 fully connected
    for i in range(10):
        for j in range(i + 1, 10):
            mat[i, j] = 1
            mat[j, i] = 1

    # Community 2: nodes 10-19 fully connected
    for i in range(10, 20):
        for j in range(i + 1, 20):
            mat[i, j] = 1
            mat[j, i] = 1

    # Some cross-community edges (sparse)
    mat[0, 10] = mat[10, 0] = 0.5
    mat[1, 11] = mat[11, 1] = 0.5

    print("Test graph created with 2 communities")
    print(f"Total edges: {np.sum(np.triu(mat, 1))}")

    # Test Louvain-style clustering
    assignment = LGDCD.louvain_style_clustering(mat, list(range(n)), max_iter=10)

    comm_count = len(set(assignment.values()))
    print(f"Detected communities: {comm_count}")

    # Test sense_and_perturb
    prev_assignment = assignment
    comm_members = defaultdict(set)
    for node_id, comm in assignment.items():
        comm_members[comm].add(node_id)

    # Test for node 0
    perturb_val = LGDCD.sense_and_perturb(0, mat, None, prev_assignment, comm_members, eps_local=1.0)
    print(f"Perturb value for node 0 (no prev): {perturb_val}")

    # Test with prev matrix (simulate small change)
    mat_prev = mat.copy()
    mat_prev[0, 1] = 0  # Remove one edge to simulate change
    mat_prev[1, 0] = 0

    perturb_val2 = LGDCD.sense_and_perturb(0, mat, mat_prev, prev_assignment, comm_members, eps_local=1.0)
    print(f"Perturb value for node 0 (with prev): {perturb_val2}")

    # Test evolution engine
    assignment, A, E, comm_members, Q = LGDCD.evolution_engine(
        mat, list(range(n)),
        prev_assignment=None, prev_comm_members=None,
        prev_A=None, prev_E=None,
        perturbed_views=None, activated_users=None, eps_local=1.0
    )

    print(f"Evolution Engine result: communities={len(set(assignment.values()))}, Q={Q:.4f}")

    print("\nBasic test PASSED!")


def test_on_real_data():
    """Test on actual Forum data."""
    print("\n" + "=" * 50)
    print("Testing on real Forum data...")
    print("=" * 50)

    data_path = "./data/Forum_LDP/FbForum"
    node_num = 899
    snapshot_num = 3  # Test on first 3 snapshots

    results = []

    for time_index in range(snapshot_num):
        print(f"\nSnapshot {time_index}:")
        mat0, mid = utils.get_mat(data_path, node_num, time_index)
        print(f"  Graph loaded: {node_num} nodes")

        # Get ground truth
        G = nx.from_numpy_array(mat0, create_using=nx.Graph)
        ground_truth = LGDCD.louvain_style_clustering(mat0, list(range(node_num)), max_iter=20)
        print(f"  Ground truth communities: {len(set(ground_truth.values()))}")

        # Run LG-DCD
        assignment, A, E, comm_members, Q = LGDCD.evolution_engine(
            mat0, list(range(node_num)),
            prev_assignment=None, prev_comm_members=None,
            prev_A=None, prev_E=None,
            perturbed_views=None, activated_users=None, eps_local=1.0
        )

        # Evaluate
        pred_labels = [assignment.get(n, 0) for n in range(node_num)]
        true_labels = [ground_truth.get(n, 0) for n in range(node_num)]

        try:
            from sklearn.metrics import normalized_mutual_info_score
            nmi = normalized_mutual_info_score(true_labels, pred_labels)
            pred_comms = {}
            true_comms = {}
            for n in range(node_num):
                pred_comms.setdefault(assignment.get(n, 0), set()).add(n)
                true_comms.setdefault(ground_truth.get(n, 0), set()).add(n)
            jaccard_scores = []
            for pred_nodes in pred_comms.values():
                best_jaccard = 0.0
                for true_nodes in true_comms.values():
                    union_size = len(pred_nodes | true_nodes)
                    if union_size == 0:
                        continue
                    jaccard = len(pred_nodes & true_nodes) / union_size
                    if jaccard > best_jaccard:
                        best_jaccard = jaccard
                jaccard_scores.append(best_jaccard)
            mean_jaccard = float(np.mean(jaccard_scores)) if jaccard_scores else 0.0
            print(f"  LG-DCD: NMI={nmi:.4f}, Jaccard={mean_jaccard:.4f}, Q={Q:.4f}")
            results.append({'nmi': nmi, 'jaccard': mean_jaccard, 'q': Q})
        except Exception as e:
            print(f"  Evaluation error: {e}")

    print("\nReal data test completed!")


if __name__ == '__main__':
    test_lgdcd_basic()
    test_on_real_data()