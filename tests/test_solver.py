"""求解器单元测试：基本约束、同只/异只判断、不可行情形，
以及与暴力枚举（小规模随机实例）的全局最优性对照。"""

import itertools
import random
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.example import build_example_detections
from app.solver import solve_tracking

W_DIST, W_APP, PENALTY = 1.0, 10000.0, 15000.0


def mk(bid, frame, cx, cy, vec, size=10):
    return {
        "id": bid,
        "frame": frame,
        "rect": [cx - size / 2, cy - size / 2, size, size],
        "vector": list(vec),
    }


def edge_cost(d1, d2, w_dist=W_DIST, w_app=W_APP):
    c1 = np.array([d1["rect"][0] + d1["rect"][2] / 2, d1["rect"][1] + d1["rect"][3] / 2])
    c2 = np.array([d2["rect"][0] + d2["rect"][2] / 2, d2["rect"][1] + d2["rect"][3] / 2])
    v1, v2 = np.array(d1["vector"], float), np.array(d2["vector"], float)
    cos = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    return w_dist * float(np.sum((c1 - c2) ** 2)) + w_app * (1 - cos)


# ---------------------------------------------------------------- 基本用例

def test_empty():
    r = solve_tracking([], [], W_DIST, W_APP, PENALTY)
    assert r["feasible"] and r["animal_count"] == 0


def test_simple_chain_one_trajectory():
    dets = [mk("a", 0, 10, 10, [1, 0]), mk("b", 1, 20, 10, [1, 0]),
            mk("c", 2, 30, 10, [1, 0])]
    r = solve_tracking(dets, [], W_DIST, W_APP, PENALTY)
    assert r["feasible"]
    assert r["n_trajectories"] == 1 and r["animal_count"] == 1
    expect = edge_cost(dets[0], dets[1]) + edge_cost(dets[1], dets[2]) + PENALTY
    assert r["total_cost"] == pytest.approx(expect)
    assert r["trajectories"][0]["boxes"] == [
        {"id": "a", "frame": 0}, {"id": "b", "frame": 1}, {"id": "c", "frame": 2}]


def test_gap2_link():
    dets = [mk("a", 0, 10, 10, [1, 0]), mk("b", 2, 30, 10, [1, 0])]
    r = solve_tracking(dets, [], W_DIST, W_APP, PENALTY)
    assert r["feasible"] and r["n_trajectories"] == 1


def test_gap3_cannot_link():
    dets = [mk("a", 0, 10, 10, [1, 0]), mk("b", 3, 30, 10, [1, 0])]
    r = solve_tracking(dets, [], W_DIST, W_APP, PENALTY)
    assert r["feasible"] and r["n_trajectories"] == 2 and r["animal_count"] == 2


def test_same_forces_expensive_merge():
    # 两框相距很远，无判断时各自成轨更便宜；标记同只后必须合并
    dets = [mk("a", 0, 0, 0, [1, 0]), mk("b", 1, 500, 500, [0, 1])]
    r_free = solve_tracking(dets, [], W_DIST, W_APP, 10.0)
    assert r_free["n_trajectories"] == 2
    r_same = solve_tracking(dets, [{"a": 0, "b": 1, "type": "same"}], W_DIST, W_APP, 10.0)
    assert r_same["feasible"] and r_same["n_trajectories"] == 1


def test_same_same_frame_infeasible():
    dets = [mk("a", 0, 0, 0, [1, 0]), mk("b", 0, 50, 0, [1, 0])]
    r = solve_tracking(dets, [{"a": 0, "b": 1, "type": "same"}], W_DIST, W_APP, PENALTY)
    assert not r["feasible"]
    assert "animal_count" not in r  # 无解时不给计数


def test_different_forces_split():
    # 相邻帧、外观相同、罚分极高 → 本应合并；标记异只后必须拆开
    dets = [mk("a", 0, 10, 10, [1, 0]), mk("b", 1, 12, 10, [1, 0])]
    r_free = solve_tracking(dets, [], W_DIST, W_APP, 1e6)
    assert r_free["n_trajectories"] == 1
    r_diff = solve_tracking(dets, [{"a": 0, "b": 1, "type": "different"}], W_DIST, W_APP, 1e6)
    assert r_diff["feasible"] and r_diff["n_trajectories"] == 2


def test_transitive_conflict_infeasible():
    dets = [mk("a", 0, 0, 0, [1, 0]), mk("b", 1, 10, 0, [1, 0]), mk("c", 2, 20, 0, [1, 0])]
    judgments = [
        {"a": 0, "b": 1, "type": "same"},
        {"a": 1, "b": 2, "type": "same"},
        {"a": 0, "b": 2, "type": "different"},
    ]
    r = solve_tracking(dets, judgments, W_DIST, W_APP, PENALTY)
    assert not r["feasible"]


def test_example_two_animals():
    """内置交叉遮挡示例：全局最优应为 2 条轨迹（2 只动物），且不得串号。"""
    dets = build_example_detections()
    assert len(dets) == 11  # 6 + 5（B 第 3 帧被遮挡）
    r = solve_tracking(dets, [], W_DIST, W_APP, PENALTY)
    assert r["feasible"]
    assert r["animal_count"] == 2 and r["n_trajectories"] == 2
    got = sorted(tuple(b["id"] for b in t["boxes"]) for t in r["trajectories"])
    expect = sorted([
        ("A0", "A1", "A2", "A3", "A4", "A5"),
        ("B0", "B1", "B2", "B4", "B5"),  # 跨 2 帧连接（帧 2 -> 帧 4）
    ])
    assert got == expect, f"示例轨迹不正确（发生串号或断裂）: {got}"


# ---------------------------------------------------------------- 暴力枚举对照

def brute_force(detections, judgments, w_dist, w_app, penalty):
    """枚举所有路径覆盖，返回 (feasible, best_cost, best_ntraj)。"""
    n = len(detections)
    if n == 0:
        return True, 0.0, 0
    frames = [d["frame"] for d in detections]

    # 同只等价类
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    diff_pairs = []
    for j in judgments:
        a, b = j["a"], j["b"]
        if j["type"] == "same":
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb
        else:
            diff_pairs.append((a, b))
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    classes = [sorted(g, key=lambda i: frames[i]) for g in groups.values() if len(g) > 1]
    for g in classes:
        fs = [frames[i] for i in g]
        if len(set(fs)) != len(fs):
            return False, None, None
    checked_diff = []
    for a, b in diff_pairs:
        if find(a) == find(b):
            return False, None, None
        if frames[a] == frames[b]:
            continue
        checked_diff.append((a, b) if frames[a] < frames[b] else (b, a))

    # 每个节点的后继候选
    succ_options = []
    for i in range(n):
        opts = [None]
        for j in range(n):
            if 1 <= frames[j] - frames[i] <= 2:
                opts.append(j)
        succ_options.append(opts)

    best = (False, None, None)
    for choice in itertools.product(*succ_options):
        indeg = [0] * n
        ok = True
        cost = 0.0
        for i, j in enumerate(choice):
            if j is None:
                continue
            indeg[j] += 1
            if indeg[j] > 1:
                ok = False
                break
            cost += edge_cost(detections[i], detections[j], w_dist, w_app)
        if not ok:
            continue
        succ = {i: j for i, j in enumerate(choice) if j is not None}

        def reachable(a, b):
            cur, seen = a, {a}
            while cur in succ:
                cur = succ[cur]
                if cur == b:
                    return True
                if cur in seen:
                    return False
                seen.add(cur)
            return False

        if any(not reachable(g[0], m) for g in classes for m in g[1:]):
            continue
        if any(reachable(a, b) for a, b in checked_diff):
            continue
        n_traj = n - len(succ)
        total = cost + penalty * n_traj
        if best[1] is None or total < best[1]:
            best = (True, total, n_traj)
    return best


@pytest.mark.parametrize("seed", range(60))
def test_against_brute_force(seed):
    rng = random.Random(seed)
    n_frames = rng.randint(2, 4)
    n = rng.randint(2, 7)
    dets = [
        mk(f"b{k}", rng.randrange(n_frames), rng.uniform(0, 100), rng.uniform(0, 100),
           [rng.uniform(0.1, 1) for _ in range(3)])
        for k in range(n)
    ]
    judgments = []
    for _ in range(rng.randint(0, 2)):
        a, b = rng.sample(range(n), 2)
        judgments.append({"a": a, "b": b, "type": rng.choice(["same", "different"])})
    w_dist = rng.uniform(0, 2)
    w_app = rng.uniform(0, 50)
    penalty = rng.uniform(0, 500)

    r = solve_tracking(dets, judgments, w_dist, w_app, penalty)
    feasible, best_cost, best_ntraj = brute_force(dets, judgments, w_dist, w_app, penalty)
    assert r["feasible"] == feasible, f"seed={seed} 可行性不一致"
    if feasible:
        assert r["total_cost"] == pytest.approx(best_cost, abs=1e-6), f"seed={seed} 非全局最优"
        assert r["n_trajectories"] == best_ntraj
