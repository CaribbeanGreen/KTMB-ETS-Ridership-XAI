"""
ets_rf.py  -  Random Forest regression implemented with NumPy only
FYP: An Explainable Prediction Model Using Random Forest for KTMB ETS Ridership Surges

Contents
  RegressionTree   CART regression tree (variance-reduction splits)
  RandomForest     bootstrap aggregation (bagging) of RegressionTrees + random feature subsets
  mae / rmse / r2 / mape   evaluation metrics

Every tree is stored as flat arrays (one entry per node), which makes prediction fast and
gives TreeSHAP direct access to the structure it needs (split feature, threshold, children,
node value and node cover = number of training samples that reached the node).

    node arrays:  feature[i]   index of split feature, -1 for a leaf
                  threshold[i] go left if x[feature] <= threshold
                  left[i], right[i]  child node ids (-1 for a leaf)
                  value[i]     mean target of training samples in the node (the prediction)
                  cover[i]     number of (bootstrap) training samples in the node
                  impurity[i]  sum of squared errors in the node
"""
import json

import numpy as np


# ==========================================================================
# Helper
# ==========================================================================
def encode_features(X):
    """Give every distinct value of every feature its own 'bin' number.
    Feature f's values occupy bins block_start..block_end-1 in ascending order, so
    codes[i, f] = bin of row i's value for feature f. Built once, reused by every node."""
    n, p = X.shape
    codes = np.empty((n, p), dtype=np.int64)
    bin_value, bin_feature, starts = [], [], []
    offset = 0
    for f in range(p):
        u, inv = np.unique(X[:, f], return_inverse=True)
        codes[:, f] = inv + offset
        starts.append(offset)
        bin_value.append(u)
        bin_feature.append(np.full(u.size, f))
        offset += u.size
    starts = np.array(starts)
    bin_feature = np.concatenate(bin_feature)
    return {"codes": codes, "n_features": p, "n_bins": offset,
            "bin_value": np.concatenate(bin_value), "bin_feature": bin_feature,
            "block_start": starts[bin_feature], "block_end": np.append(starts[1:], offset)}


# ==========================================================================
# Regression tree
# ==========================================================================
class RegressionTree:
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features=None):
        self.max_depth = max_depth if max_depth is not None else np.inf
        self.min_samples_split = max(2, min_samples_split)
        self.min_samples_leaf = max(1, min_samples_leaf)
        self.max_features = max_features          # number of features tried per split (None = all)

    # ---------------------------------------------------------------- split search
    def _best_split(self, enc, y, idx, rng):
        """Return (feature, threshold, gain) of the split that most reduces the
        sum of squared errors (SSE), or None if the node cannot be split.

        Candidate thresholds: every gap between two different values of a feature that occur
        in this node. Each feature's values were coded once as their rank among the distinct
        values, and each feature owns its own block of 'bins'. One bincount over the node's
        codes therefore gives, for EVERY feature and EVERY distinct value at once, the number
        of samples and the sums of y and y^2. Running sums within each feature's block then
        give the SSE of both children for all thresholds of all features:
            SSE = sum(y^2) - (sum(y))^2 / n        (for the left child and the right child)
        This finds exactly the same splits as sorting the samples feature by feature.
        """
        n = idx.size
        y_node = y[idx]
        y_sq = y_node ** 2
        total, total_sq = y_node.sum(), y_sq.sum()
        parent_sse = total_sq - total * total / n
        if parent_sse <= 1e-12:
            return None                                   # all targets identical: nothing to gain

        n_feat, n_bins = enc["n_features"], enc["n_bins"]
        flat = enc["codes"][idx].ravel()                  # bin id of every (sample, feature)
        count = np.bincount(flat, minlength=n_bins).astype(np.float64)
        sum_y = np.bincount(flat, weights=np.repeat(y_node, n_feat), minlength=n_bins)
        sum_sq = np.bincount(flat, weights=np.repeat(y_sq, n_feat), minlength=n_bins)

        # running sums restarted at the start of each feature's block
        start = enc["block_start"]                         # first bin of each bin's feature
        def block_cumsum(v):
            c = np.cumsum(v)
            return c - (c[start] - v[start])
        n_left = block_cumsum(count)                       # samples with value <= this bin's value
        csum, csq = block_cumsum(sum_y), block_cumsum(sum_sq)
        n_right = n - n_left

        # a threshold sits after a value that occurs (count > 0) and before at least one more
        valid = (count > 0) & (n_right > 0) \
            & (n_left >= self.min_samples_leaf) & (n_right >= self.min_samples_leaf)
        if not valid.any():
            return None
        feat_ok = np.zeros(n_feat, dtype=bool)
        feat_ok[enc["bin_feature"][valid]] = True

        # random feature subset: visit features in random order and keep the first k that can
        # split (like standard CART, look further only if a chosen feature cannot split)
        k = n_feat if self.max_features is None else self.max_features
        order = rng.permutation(n_feat)
        chosen = order[feat_ok[order]][:k]
        use = valid & np.isin(enc["bin_feature"], chosen)

        with np.errstate(divide="ignore", invalid="ignore"):
            sse = (csq - csum ** 2 / n_left) + ((total_sq - csq) - (total - csum) ** 2 / n_right)
        sse = np.where(use, sse, np.inf)
        b = int(np.argmin(sse))
        gain = parent_sse - sse[b]
        if not np.isfinite(sse[b]) or gain <= 1e-12:
            return None

        f = int(enc["bin_feature"][b])
        end = enc["block_end"][f]
        nxt = b + 1 + int(np.flatnonzero(count[b + 1:end])[0])   # next value present in this node
        vals = enc["bin_value"]
        return f, (vals[b] + vals[nxt]) / 2.0, gain

    # ---------------------------------------------------------------- training
    def fit(self, X, y, rng=None, enc=None):
        rng = rng if rng is not None else np.random.default_rng()
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if enc is None:
            enc = encode_features(X)

        feature, threshold, left, right, value, cover, impurity = [], [], [], [], [], [], []
        self.gain_by_feature = np.zeros(X.shape[1])

        def new_node(idx):
            yi = y[idx]
            feature.append(-1); threshold.append(0.0); left.append(-1); right.append(-1)
            value.append(yi.mean()); cover.append(idx.size)
            impurity.append(float(((yi - yi.mean()) ** 2).sum()))
            return len(value) - 1

        root = new_node(np.arange(X.shape[0]))
        stack = [(root, np.arange(X.shape[0]), 0)]          # (node id, sample indices, depth)
        while stack:
            node, idx, depth = stack.pop()
            if depth >= self.max_depth or idx.size < self.min_samples_split:
                continue
            split = self._best_split(enc, y, idx, rng)
            if split is None:
                continue
            f, thr, gain = split
            go_left = X[idx, f] <= thr
            li, ri = idx[go_left], idx[~go_left]
            l_node, r_node = new_node(li), new_node(ri)
            feature[node], threshold[node], left[node], right[node] = f, thr, l_node, r_node
            self.gain_by_feature[f] += gain
            stack.append((r_node, ri, depth + 1))
            stack.append((l_node, li, depth + 1))

        self.feature = np.array(feature, dtype=np.int64)
        self.threshold = np.array(threshold, dtype=np.float64)
        self.left = np.array(left, dtype=np.int64)
        self.right = np.array(right, dtype=np.int64)
        self.value = np.array(value, dtype=np.float64)
        self.cover = np.array(cover, dtype=np.float64)
        self.impurity = np.array(impurity, dtype=np.float64)
        return self

    # ---------------------------------------------------------------- prediction
    def apply(self, X):
        """Leaf node id reached by each row (all rows walk down the tree together)."""
        X = np.asarray(X, dtype=np.float64)
        node = np.zeros(X.shape[0], dtype=np.int64)
        rows = np.arange(X.shape[0])
        active = self.feature[node] >= 0
        while active.any():
            r, n = rows[active], node[active]
            go_left = X[r, self.feature[n]] <= self.threshold[n]
            node[r] = np.where(go_left, self.left[n], self.right[n])
            active = self.feature[node] >= 0
        return node

    def predict(self, X):
        return self.value[self.apply(X)]

    @property
    def n_nodes(self):
        return self.value.size

    @property
    def depth(self):
        d = np.zeros(self.n_nodes, dtype=np.int64)
        for i in range(self.n_nodes):                 # children always have larger ids than parents
            if self.feature[i] >= 0:
                d[self.left[i]] = d[self.right[i]] = d[i] + 1
        return int(d.max())


# ==========================================================================
# Random forest
# ==========================================================================
class RandomForest:
    """Bagging of regression trees.

    n_trees           number of trees in the forest
    max_depth         maximum depth of each tree (None = grow until leaves are pure / too small)
    min_samples_leaf  minimum training samples in a leaf
    max_features      features tried at each split: fraction (0-1], int, or None for all
    bootstrap         sample the training set with replacement for each tree
    """

    def __init__(self, n_trees=100, max_depth=None, min_samples_split=2, min_samples_leaf=1,
                 max_features=1 / 3, bootstrap=True, random_state=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state

    def _n_features_per_split(self, n_features):
        mf = self.max_features
        if mf is None:
            return n_features
        if isinstance(mf, float) and mf <= 1.0:
            return max(1, int(round(mf * n_features)))
        return max(1, min(n_features, int(mf)))

    def fit(self, X, y, feature_names=None, verbose=False):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n, p = X.shape
        self.n_features = p
        self.feature_names = list(feature_names) if feature_names is not None else [f"x{i}" for i in range(p)]
        k = self._n_features_per_split(p)
        rng = np.random.default_rng(self.random_state)
        enc = encode_features(X)                     # done once, shared by every tree

        self.trees = []
        oob_sum, oob_count = np.zeros(n), np.zeros(n)
        for t in range(self.n_trees):
            # 1) bootstrap sample: n draws with replacement (about 63% unique rows)
            sample = rng.integers(0, n, n) if self.bootstrap else np.arange(n)
            # 2) grow one tree on that sample, trying k random features at each split
            tree = RegressionTree(self.max_depth, self.min_samples_split, self.min_samples_leaf, k)
            tree.fit(X[sample], y[sample], rng, dict(enc, codes=enc["codes"][sample]))
            self.trees.append(tree)
            # 3) out-of-bag rows (never drawn) give a free validation estimate
            if self.bootstrap:
                oob = np.ones(n, dtype=bool)
                oob[sample] = False
                oob_sum[oob] += tree.predict(X[oob])
                oob_count[oob] += 1
            if verbose and (t + 1) % max(1, self.n_trees // 10) == 0:
                print(f"    tree {t + 1}/{self.n_trees}  ({tree.n_nodes} nodes, depth {tree.depth})")

        if self.bootstrap:
            seen = oob_count > 0
            self.oob_prediction_ = np.where(seen, oob_sum / np.maximum(oob_count, 1), np.nan)
            self.oob_r2_ = r2(y[seen], self.oob_prediction_[seen])
        return self

    def predict(self, X):
        """Forest prediction = average of all tree predictions."""
        X = np.asarray(X, dtype=np.float64)
        return np.mean([t.predict(X) for t in self.trees], axis=0)

    def feature_importance(self):
        """Mean decrease in impurity: each tree's SSE reduction per feature, normalised
        to sum to 1 within the tree, then averaged over trees."""
        imp = np.zeros(self.n_features)
        for t in self.trees:
            s = t.gain_by_feature.sum()
            if s > 0:
                imp += t.gain_by_feature / s
        imp /= len(self.trees)
        return dict(zip(self.feature_names, imp))

    # ---------------------------------------------------------------- save / load (no pickle)
    def save(self, path):
        sizes = np.array([t.n_nodes for t in self.trees])
        cat = lambda attr: np.concatenate([getattr(t, attr) for t in self.trees])
        params = {k: getattr(self, k) for k in
                  ["n_trees", "max_depth", "min_samples_split", "min_samples_leaf", "max_features",
                   "bootstrap", "random_state"]}
        np.savez_compressed(
            path, sizes=sizes, feature=cat("feature"), threshold=cat("threshold"), left=cat("left"),
            right=cat("right"), value=cat("value"), cover=cat("cover"), impurity=cat("impurity"),
            gain=np.stack([t.gain_by_feature for t in self.trees]),
            meta=json.dumps({"params": params, "feature_names": self.feature_names,
                             "oob_r2": getattr(self, "oob_r2_", None)}))

    @classmethod
    def load(cls, path):
        z = np.load(path, allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        forest = cls(**meta["params"])
        forest.feature_names = meta["feature_names"]
        forest.n_features = len(forest.feature_names)
        forest.oob_r2_ = meta.get("oob_r2")
        forest.trees = []
        # read each array from the file ONCE (every z[...] access decompresses the whole array again)
        attrs = ["feature", "threshold", "left", "right", "value", "cover", "impurity"]
        arrays = {attr: z[attr] for attr in attrs}
        gain = z["gain"]
        start = 0
        for i, size in enumerate(z["sizes"]):
            t = RegressionTree()
            sl = slice(start, start + size)
            for attr in attrs:
                setattr(t, attr, arrays[attr][sl].copy())
            t.gain_by_feature = gain[i].copy()
            forest.trees.append(t)
            start += size
        return forest


# ==========================================================================
# Metrics
# ==========================================================================
def mae(y, p):
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(p))))


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def r2(y, p):
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    ss_res = np.sum((y - p) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return float(1 - ss_res / ss_tot)


def mape(y, p):
    """Mean absolute percentage error (rows with y = 0 are skipped)."""
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    m = y != 0
    return float(np.mean(np.abs((y[m] - p[m]) / y[m])) * 100)
