from collections import defaultdict
from pathlib import Path
import os
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import LGDCD


def build_two_block_graph(n=20):
    mat = np.zeros((n, n))
    for i in range(10):
        for j in range(i + 1, 10):
            mat[i, j] = mat[j, i] = 1
    for i in range(10, 20):
        for j in range(i + 1, 20):
            mat[i, j] = mat[j, i] = 1
    mat[0, 10] = mat[10, 0] = 0.5
    mat[1, 11] = mat[11, 1] = 0.5
    return mat


def test_core_functions():
    print("Testing core LG-DCD functions...")
    mat = build_two_block_graph()
    n = len(mat)

    assignment = LGDCD.louvain_communities(mat, list(range(n)), seed=42)
    comm_members = defaultdict(set)
    for node_id, comm_id in assignment.items():
        comm_members[comm_id].add(node_id)
    print(f"  communities: {len(set(assignment.values()))}")

    W, S = LGDCD.recalibrate_ws(assignment, comm_members, mat, None, eps_cal=1.0)
    turmoil = LGDCD.compute_turmoil(comm_members, S)
    assert W and S and turmoil

    mat_prev = mat.copy()
    mat_prev[0, 1] = mat_prev[1, 0] = 0
    report = LGDCD.sense_and_perturb(
        0, mat, mat_prev, assignment, comm_members, eps_local=1.0, time_index=1, W=W
    )
    assert len(report) == 3

    W2, S2, assignment2, comm_members2, turmoil2 = LGDCD.evolution_engine(
        [(0, report[0], report[1], report[2])],
        W,
        S,
        assignment,
        comm_members,
        eps_rep=1.0,
    )
    assert W2 and S2 and assignment2 and comm_members2 and turmoil2
    print("  core functions ok")


def test_short_pipeline():
    print("Testing short LG-DCD pipeline on Forum...")
    os.environ["LGDCD_MAX_SNAPSHOTS"] = "2"
    os.environ["LGDCD_FAST_EVAL"] = "1"
    result = LGDCD.main_func(
        data_path="./data/Forum_LDP/FbForum",
        eps=1.0,
        exp_num=1,
        node_num=899,
        snapshot_num=24,
        window_size=5,
        dataset_name="Forum",
        max_h=168,
    )
    assert isinstance(result, dict)
    print("  short pipeline ok")


if __name__ == "__main__":
    test_core_functions()
    test_short_pipeline()
    print("LG-DCD smoke tests passed.")
