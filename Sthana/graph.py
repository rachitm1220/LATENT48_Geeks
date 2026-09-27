"""
Spatial reasoning over the seat graph (a small graph neural network).

Every tracked seat is a NODE. Each seat is connected by an EDGE to its k nearest
seats within a radius, measured in "desk sizes" so the camera's perspective (far
desks look smaller) doesn't distort who counts as a neighbour.
Edge weight  w_ij = exp(-(d_ij / sigma)^2).

Node state is a probability distribution over three classes:
    [occupied, held (belongings only), free]

Input features per node
    x_i = c_i * onehot(detector observation) + (1 - c_i) * h_i(previous frame)
where c_i is the detector's evidence confidence (0 for a desk hidden in this frame).

Message passing (L rounds, a GCN layer with fixed, interpretable weights):
    M = A_hat · H · W                      (A_hat: row-normalised adjacency)
    H <- softmax( log H + beta * (1 - c) ⊙ M )
W is a 3x3 class-compatibility matrix: row = neighbour's class, column = the effect
on my class. Busy areas cluster, so an occupied neighbour raises my "occupied"
logit and lowers my "free" logit, and so on.

The gate (1 - c) is the key idea: a seat with strong detector evidence barely
moves, while an uncertain or hidden seat leans on its neighbours and its own
history. The model never overrides a confident detection.

Outputs per seat: refined probabilities, a flag (consistent / uncertain /
disagree / inferred), neighbours, and neighbourhood occupancy, which also
drives the "best seat now" recommendation (the quietest free seat).

The weights are hand-set (no labelled training data yet). With labelled seat
sequences, W, beta and sigma can be fitted by minimising cross-entropy on the
refined probabilities, keeping the same forward pass.
"""
import math

import numpy as np

CLASSES = ("occupied", "held", "free")

# rows: neighbour class, cols: effect on my class logits
DEFAULT_W = np.array([
    [0.8, 0.3, -0.5],    # neighbour occupied
    [0.3, 0.6, -0.3],    # neighbour held (belongings)
    [-0.5, -0.3, 0.8],   # neighbour free
])

EPS = 1e-6


def _softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class SpatialGNN:
    def __init__(self, k=4, radius=2.6, sigma=1.6, beta=1.4, layers=2, temporal=0.6,
                 uncertain=0.5, W=None):
        self.k = k
        self.radius = radius
        self.sigma = sigma
        self.beta = beta
        self.layers = layers
        self.temporal = temporal
        self.uncertain = uncertain
        self.W = DEFAULT_W if W is None else np.asarray(W, float)
        self.edges = []

    # --------------------------------------------------------
    def build_graph(self, nodes):
        """Adjacency matrix from seat centres/sizes (pixels). Symmetric kNN within radius."""
        n = len(nodes)
        A = np.zeros((n, n))
        if n < 2:
            self.edges = []
            return A
        C = np.array([nd["center"] for nd in nodes], float)
        S = np.array([max(1.0, math.sqrt(nd["size"][0] * nd["size"][1])) for nd in nodes])
        D = np.sqrt(((C[:, None, :] - C[None, :, :]) ** 2).sum(-1))
        D = D / ((S[:, None] + S[None, :]) / 2)              # in desk sizes
        np.fill_diagonal(D, np.inf)
        for i in range(n):
            for j in np.argsort(D[i])[: self.k]:
                if D[i, j] <= self.radius:
                    w = math.exp(-(D[i, j] / self.sigma) ** 2)
                    A[i, j] = A[j, i] = max(A[i, j], w)
        self.edges = [(nodes[i]["id"], nodes[j]["id"], round(float(A[i, j]), 3))
                      for i in range(n) for j in range(i + 1, n) if A[i, j] > 0]
        return A

    # --------------------------------------------------------
    def run(self, nodes):
        """
        nodes: [{"id", "center": [x, y] px, "size": [w, h] px,
                 "obs": 0|1|2 or None, "evidence": 0..1, "visible": bool,
                 "prev": [p_occ, p_held, p_free] or None}]
        Returns {id: result}.
        """
        n = len(nodes)
        if n == 0:
            self.edges = []
            return {}
        A = self.build_graph(nodes)
        deg = A.sum(1, keepdims=True)
        A_hat = np.divide(A, deg, out=np.zeros_like(A), where=deg > 0)

        c = np.array([nd["evidence"] if nd["visible"] and nd["obs"] is not None else 0.0
                      for nd in nodes])
        prior = np.array([nd["prev"] if nd.get("prev") is not None else [1 / 3] * 3
                          for nd in nodes], float)
        onehot = np.zeros((n, 3))
        for i, nd in enumerate(nodes):
            if nd["obs"] is not None:
                onehot[i, nd["obs"]] = 1.0
            else:
                onehot[i] = prior[i]
        # temporal blend of history, then evidence-weighted observation
        base = self.temporal * prior + (1 - self.temporal) * onehot
        H = c[:, None] * onehot + (1 - c[:, None]) * base
        H = np.clip(H, EPS, 1)
        H /= H.sum(1, keepdims=True)

        gate = (1 - c)[:, None]
        for _ in range(self.layers):
            M = A_hat @ H @ self.W
            H = _softmax(np.log(H) + self.beta * gate * M)

        # neighbourhood occupancy: how busy the seats around me are (held counts half)
        busy = H[:, 0] + 0.5 * H[:, 1]
        nbr_busy = np.divide(A @ busy, deg[:, 0], out=np.full(n, np.nan), where=deg[:, 0] > 0)

        out = {}
        for i, nd in enumerate(nodes):
            p = H[i]
            k = int(p.argmax())
            if not nd["visible"]:
                flag = "inferred"
            elif nd["obs"] is not None and k != nd["obs"] and p[k] >= 0.55:
                flag = "disagree"
            elif c[i] < self.uncertain:
                flag = "uncertain"
            else:
                flag = "consistent"
            nbrs = [nodes[j]["id"] for j in np.argsort(-A[i]) if A[i, j] > 0]
            out[nd["id"]] = {
                "p": [round(float(x), 3) for x in p],
                "state": CLASSES[k],
                "confidence": round(float(c[i]), 3),
                "flag": flag,
                "neighbours": nbrs,
                "nbr_busy": None if math.isnan(nbr_busy[i]) else round(float(nbr_busy[i]), 3),
            }
        return out
