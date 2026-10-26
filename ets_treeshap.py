"""
ets_treeshap.py  -  TreeSHAP explanations for the NumPy Random Forest (ets_rf.py)
FYP: An Explainable Prediction Model Using Random Forest for KTMB ETS Ridership Surges

Two independent ways of computing the same Shapley values:

1. tree_shap / forest_shap  -  TreeSHAP (Lundberg, Erion & Lee, 2018, "Consistent Individualized
   Feature Attribution for Tree Ensembles", Algorithm 2, path-dependent). Walks each tree once,
   tracking for every root-to-leaf path how much of the "credit" each feature on the path gets.
   Polynomial time. Written here to explain MANY rows at once: every per-row quantity is a NumPy
   vector, so one walk of the tree serves all rows.

2. exact_shapley  -  brute force from the Shapley definition. For every subset S of features
   (2^13 = 8,192 subsets) it computes v(S) = expected prediction when only the features in S are
   known (the same tree-based conditional expectation TreeSHAP uses: an unknown feature sends the
   prediction down BOTH branches, weighted by how many training samples went each way), then
       phi_i = sum over S without i of  |S|! (M-|S|-1)! / M!  *  ( v(S + {i}) - v(S) )
   Exponential time, used only to VERIFY TreeSHAP on a few rows.

Properties that must hold (checked in 05_explain.py):
   additivity:  base_value + sum(phi) = model prediction, for every row
   agreement:   TreeSHAP == exact enumeration (to rounding error)
"""
from math import factorial
from pathlib import Path

import numpy as np


# ==========================================================================
# 1. TreeSHAP (path-dependent), vectorised over rows
# ==========================================================================
def _extend(feat, zero, one, pw, ud, pz, po, pf):
    """Add a new element (feature pf, zero fraction pz, one fraction po) to the path and
    update the path weights. Lengths before the call: ud elements; after: ud + 1."""
    feat.append(pf)
    zero.append(pz)
    one = np.vstack([one[:ud], po[None, :]])
    pw = np.vstack([pw[:ud], np.full((1, one.shape[1]), 1.0 if ud == 0 else 0.0)])
    for i in range(ud - 1, -1, -1):
        pw[i + 1] += po * pw[i] * (i + 1) / (ud + 1)
        pw[i] = pz * pw[i] * (ud - i) / (ud + 1)
    return feat, zero, one, pw


def _unwind(feat, zero, one, pw, ud, k):
    """Remove path element k (undo its _extend) - used when a feature appears twice on a path."""
    o, z = one[k], zero[k]
    nz = o != 0
    o_safe = np.where(nz, o, 1.0)
    next_one = pw[ud].copy()
    pw = pw.copy()
    for i in range(ud - 1, -1, -1):
        old = pw[i].copy()
        w_one = next_one * (ud + 1) / ((i + 1) * o_safe)
        w_zero = pw[i] * (ud + 1) / (z * (ud - i))
        pw[i] = np.where(nz, w_one, w_zero)
        next_one = np.where(nz, old - pw[i] * z * (ud - i) / (ud + 1), next_one)
    feat = feat[:k] + feat[k + 1:ud + 1]
    zero = zero[:k] + zero[k + 1:ud + 1]
    one = np.vstack([one[:k], one[k + 1:ud + 1]])
    return feat, zero, one, pw[:ud]


def _unwound_sums(zero, one, pw, ud):
    """For every path element 1..ud at once: total weight of the path with that element removed."""
    O = one[1:ud + 1]                                     # (ud, rows)
    Z = np.asarray(zero[1:ud + 1])[:, None]               # (ud, 1)
    nz = O != 0
    O_safe = np.where(nz, O, 1.0)
    next_one = np.repeat(pw[ud][None, :], ud, axis=0)
    total = np.zeros_like(O)
    for j in range(ud - 1, -1, -1):
        tmp = next_one * (ud + 1) / ((j + 1) * O_safe)
        total += np.where(nz, tmp, pw[j] / (Z * (ud - j) / (ud + 1)))
        next_one = pw[j] - tmp * Z * (ud - j) / (ud + 1)
    return total, O, Z


def tree_shap(tree, X):
    """SHAP values of one RegressionTree for every row of X -> array (rows, features)."""
    X = np.asarray(X, dtype=np.float64)
    n, p = X.shape
    phi = np.zeros((n, p))
    ones = np.ones(n)

    def recurse(node, feat, zero, one, pw, ud, pz, po, pf):
        feat, zero, one, pw = _extend(list(feat[:ud]), list(zero[:ud]), one, pw, ud, pz, po, pf)
        f = tree.feature[node]
        if f < 0:                                         # leaf: hand out credit to path features
            if ud == 0:
                return
            w, O, Z = _unwound_sums(zero, one, pw, ud)
            contrib = w * (O - Z) * tree.value[node]
            for k, fk in enumerate(feat[1:ud + 1]):
                phi[:, fk] += contrib[k]
            return
        go_left = X[:, f] <= tree.threshold[node]
        l, r = tree.left[node], tree.right[node]
        iz, io = 1.0, ones
        if f in feat[1:]:                                 # feature already on the path: merge
            k = feat.index(f, 1)
            iz, io = zero[k], one[k]
            feat, zero, one, pw = _unwind(feat, zero, one, pw, ud, k)
            ud -= 1
        c = tree.cover[node]
        recurse(l, feat, zero, one, pw, ud + 1, iz * tree.cover[l] / c, io * go_left, f)
        recurse(r, feat, zero, one, pw, ud + 1, iz * tree.cover[r] / c, io * ~go_left, f)

    recurse(0, [], [], np.empty((0, n)), np.empty((0, n)), 0, 1.0, ones, -1)
    return phi


def forest_shap(forest, X, verbose=False):
    """SHAP values of the whole forest = average of the trees' SHAP values.
    Returns (phi, base_value) where base_value = average prediction over the training data
    (each tree's root value), so that base_value + phi.sum(axis=1) == forest.predict(X)."""
    X = np.asarray(X, dtype=np.float64)
    phi = np.zeros(X.shape)
    for t, tree in enumerate(forest.trees):
        phi += tree_shap(tree, X)
        if verbose and (t + 1) % max(1, len(forest.trees) // 10) == 0:
            print(f"    TreeSHAP: tree {t + 1}/{len(forest.trees)}")
    phi /= len(forest.trees)
    base = float(np.mean([tree.value[0] for tree in forest.trees]))
    return phi, base


def model_tag(path):
    """Identity of a saved model file (size + modification time): changes whenever it is retrained."""
    st = Path(path).stat()
    return f"{st.st_size}-{int(st.st_mtime)}"


def forest_shap_cached(forest, X, cache_path, tag, progress=None, max_rows=60000):
    """forest_shap with a cache, so a daily update only explains rows that changed.
    SHAP values depend only on the model and the row's feature values, so a row whose exact feature
    vector was already explained by the same model (same tag) reuses its stored values unchanged.
    Returns (phi, base_value, n_computed)."""
    X = np.asarray(X, dtype=np.float64)
    cache_path = Path(cache_path)
    known = {}
    if cache_path.exists():
        z = np.load(cache_path, allow_pickle=False)
        if "X" in z.files and "tag" in z.files and str(z["tag"]) == tag:
            known = {x.tobytes(): p for x, p in zip(z["X"], z["phi"])}
    keys = [x.tobytes() for x in X]
    todo = np.array([i for i, k in enumerate(keys) if k not in known], dtype=int)
    phi = np.zeros(X.shape)
    for i, k in enumerate(keys):
        if k in known:
            phi[i] = known[k]
    if len(todo):
        sub = X[todo]
        acc = np.zeros(sub.shape)
        for t, tree in enumerate(forest.trees, 1):
            acc += tree_shap(tree, sub)
            if progress:
                progress(t, len(forest.trees), len(todo))
        phi[todo] = acc / len(forest.trees)
    # save: this run's rows first, then older rows still worth keeping (up to max_rows)
    merged = dict(zip(keys, phi))
    for k, p in known.items():
        if len(merged) >= max_rows:
            break
        merged.setdefault(k, p)
    keep = list(merged.items())
    np.savez_compressed(cache_path, tag=np.array(tag),
                        X=np.stack([np.frombuffer(k, dtype=np.float64) for k, _ in keep]),
                        phi=np.stack([p for _, p in keep]))
    base = float(np.mean([tree.value[0] for tree in forest.trees]))
    return phi, base, int(len(todo))


# ==========================================================================
# 2. Exact Shapley values by enumerating every feature subset (verification only)
# ==========================================================================
def _subset_values(tree, x, masks):
    """v(S) for every subset S (one row of masks per subset): expected tree output when
    only the features in S are known. Unknown feature -> both branches, cover-weighted."""
    def val(node):
        f = tree.feature[node]
        if f < 0:
            return np.full(masks.shape[0], tree.value[node])
        l, r = tree.left[node], tree.right[node]
        vl, vr = val(l), val(r)
        known = masks[:, f]
        follow = vl if x[f] <= tree.threshold[node] else vr
        average = (tree.cover[l] * vl + tree.cover[r] * vr) / tree.cover[node]
        return np.where(known, follow, average)
    return val(0)


def exact_shapley(forest, x):
    """Shapley values of one row by full enumeration of all 2^M subsets."""
    x = np.asarray(x, dtype=np.float64)
    m = x.size
    ids = np.arange(2 ** m)
    masks = ((ids[:, None] >> np.arange(m)) & 1).astype(bool)        # subset -> which features in it
    v = np.mean([_subset_values(t, x, masks) for t in forest.trees], axis=0)
    size = masks.sum(axis=1)
    weight = np.array([factorial(s) * factorial(m - s - 1) / factorial(m) for s in range(m)])
    phi = np.zeros(m)
    for i in range(m):
        without = ~masks[:, i]
        S = ids[without]
        phi[i] = np.sum(weight[size[S]] * (v[S | (1 << i)] - v[S]))
    return phi, float(v[0])                                            # v(empty set) = base value
