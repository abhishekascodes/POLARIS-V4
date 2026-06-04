#!/usr/bin/env python3
"""
POLARIS v5 — Emergent Language Analysis
========================================
Analyses what AI ministers *learned to communicate* through 16-dimensional
latent vectors produced by the LatentDiplomacy module.

Modules:
  • LatentLanguageAnalyzer  — PCA/t-SNE embedding, k-means clustering,
                               protocol description
  • InformationMetrics      — MI, channel capacity, redundancy,
                               coordination entropy

Self-contained: imports only torch, numpy, math, random, collections, typing.
"""

import torch
import numpy as np
import math
import random
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict, Counter


# ═══════════════════════════════════════════════════════════════════
#  PURE-NUMPY UTILITIES  (no sklearn)
# ═══════════════════════════════════════════════════════════════════

def _pca(X: np.ndarray, n_components: int = 2) -> np.ndarray:
    """PCA via SVD.  X: (N, D) -> (N, n_components)."""
    X_centered = X - X.mean(axis=0)
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    return X_centered @ Vt[:n_components].T


def _kmeans(X: np.ndarray, k: int, max_iter: int = 100, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Minimal k-means.  X: (N, D).
    Returns (labels, centroids).
    """
    rng = np.random.RandomState(seed)
    N, D = X.shape
    # k-means++ initialisation (simplified: random distinct points)
    idx = rng.choice(N, size=min(k, N), replace=False)
    centroids = X[idx].copy()

    labels = np.zeros(N, dtype=np.int64)
    for _ in range(max_iter):
        # Assign
        dists = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)  # (N, k)
        new_labels = np.argmin(dists, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        # Update
        for j in range(k):
            members = X[labels == j]
            if len(members) > 0:
                centroids[j] = members.mean(axis=0)
    return labels, centroids


def _simple_tsne(X: np.ndarray, n_components: int = 2, perplexity: float = 15.0,
                 lr: float = 100.0, n_iter: int = 300, seed: int = 42) -> np.ndarray:
    """
    Simplified t-SNE-like neighbour embedding (Barnes-Hut-free).

    Good enough for < 5 000 points; produces interpretable 2-D layouts.
    Falls back to PCA if the input has <= 3 dimensions already.
    """
    N, D = X.shape
    if D <= n_components or N < 4:
        return _pca(X, n_components)

    rng = np.random.RandomState(seed)

    # --- High-dim affinities (Gaussian) ---
    sq_dists = np.sum((X[:, None] - X[None, :]) ** 2, axis=2)  # (N, N)

    # Binary-search for per-point sigma to match target perplexity
    target_entropy = np.log(perplexity)
    P = np.zeros((N, N))
    for i in range(N):
        lo, hi = 1e-10, 1e4
        for _ in range(50):
            sigma = (lo + hi) / 2.0
            pij = np.exp(-sq_dists[i] / (2.0 * sigma))
            pij[i] = 0.0
            s = pij.sum()
            if s < 1e-12:
                lo = sigma
                continue
            pij /= s
            ent = -np.sum(pij[pij > 0] * np.log(pij[pij > 0]))
            if ent > target_entropy:
                hi = sigma
            else:
                lo = sigma
        P[i] = pij

    # Symmetrise
    P = (P + P.T) / (2.0 * N)
    P = np.maximum(P, 1e-12)

    # --- Low-dim embedding (Student-t kernel) ---
    Y = rng.randn(N, n_components).astype(np.float64) * 1e-4
    velocity = np.zeros_like(Y)
    momentum = 0.5

    for it in range(n_iter):
        sq_d_low = np.sum((Y[:, None] - Y[None, :]) ** 2, axis=2)
        Q_num = 1.0 / (1.0 + sq_d_low)
        np.fill_diagonal(Q_num, 0.0)
        Q = Q_num / Q_num.sum()
        Q = np.maximum(Q, 1e-12)

        # Gradient
        PQ_diff = P - Q
        grad = np.zeros_like(Y)
        for i in range(N):
            diff = Y[i] - Y  # (N, n_components)
            grad[i] = 4.0 * np.sum((PQ_diff[i] * Q_num[i])[:, None] * diff, axis=0)

        if it > 100:
            momentum = 0.8
        velocity = momentum * velocity - lr * grad
        Y += velocity

    return Y


# ═══════════════════════════════════════════════════════════════════
#  1.  LATENT LANGUAGE ANALYZER
# ═══════════════════════════════════════════════════════════════════

class LatentLanguageAnalyzer:
    """
    Records (latent_message, context_state, action_taken, outcome) tuples
    from agent communication and analyses the emergent protocol.

    Key outputs:
      • cluster_labels     – what each message cluster correlates with
      • mutual_information  – MI(messages, outcomes) in bits
      • channel_capacity    – effective bits used in the 16-dim channel
      • vocabulary_size     – number of distinct message clusters

    Parameters
    ----------
    latent_dim : int   Dimensionality of the latent channel (default 16).
    max_clusters : int Upper bound on k for the elbow heuristic.
    """

    def __init__(self, latent_dim: int = 16, max_clusters: int = 12):
        self.latent_dim = latent_dim
        self.max_clusters = max_clusters

        # Collected data
        self._messages: List[np.ndarray] = []
        self._states: List[np.ndarray] = []
        self._actions: List[int] = []
        self._rewards: List[float] = []

        # Analysis cache (lazily computed)
        self._analysis_cache: Optional[Dict[str, Any]] = None
        self._embedding_cache: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    #  Data collection
    # ------------------------------------------------------------------

    def record(
        self,
        message_vec: Any,
        state: Any,
        action: int,
        reward: float,
    ) -> None:
        """
        Store one communication data point.

        Parameters
        ----------
        message_vec : array-like (latent_dim,)
        state       : array-like or None
        action      : int — discrete action taken after receiving message
        reward      : float — outcome
        """
        if isinstance(message_vec, torch.Tensor):
            message_vec = message_vec.detach().cpu().numpy()
        msg = np.asarray(message_vec, dtype=np.float64).ravel()
        assert msg.shape[0] == self.latent_dim, (
            f"Message dim mismatch: expected {self.latent_dim}, got {msg.shape[0]}"
        )
        self._messages.append(msg)

        if state is not None:
            if isinstance(state, torch.Tensor):
                state = state.detach().cpu().numpy()
            self._states.append(np.asarray(state, dtype=np.float64).ravel())
        else:
            self._states.append(np.zeros(1))

        self._actions.append(int(action))
        self._rewards.append(float(reward))
        # Invalidate cache
        self._analysis_cache = None
        self._embedding_cache = None

    # ------------------------------------------------------------------
    #  Helpers
    # ------------------------------------------------------------------

    def _best_k(self, X: np.ndarray) -> int:
        """Pick k via the elbow method (max 2nd-derivative of inertia)."""
        N = X.shape[0]
        k_range = range(2, min(self.max_clusters + 1, N))
        inertias = []
        for k in k_range:
            labels, centroids = _kmeans(X, k)
            inertia = sum(np.sum((X[labels == j] - centroids[j]) ** 2)
                          for j in range(k))
            inertias.append(inertia)
        if len(inertias) < 3:
            return 2
        # Second derivative
        diffs = np.diff(inertias)
        diffs2 = np.diff(diffs)
        best_idx = int(np.argmax(diffs2)) + 2  # offset by 2 because of two diffs
        return list(k_range)[best_idx]

    @staticmethod
    def _mi_bits(x_labels: np.ndarray, y_labels: np.ndarray) -> float:
        """Mutual information in bits between two integer label arrays."""
        N = len(x_labels)
        if N == 0:
            return 0.0
        joint = Counter(zip(x_labels, y_labels))
        cx = Counter(x_labels)
        cy = Counter(y_labels)
        mi = 0.0
        for (xi, yi), nxy in joint.items():
            pxy = nxy / N
            px = cx[xi] / N
            py = cy[yi] / N
            if pxy > 0 and px > 0 and py > 0:
                mi += pxy * math.log2(pxy / (px * py))
        return mi

    # ------------------------------------------------------------------
    #  Core analysis
    # ------------------------------------------------------------------

    def analyze(self) -> Dict[str, Any]:
        """
        Run the full emergent-language analysis pipeline.

        Returns
        -------
        dict with:
            cluster_labels     : dict  cluster_id -> {dominant_action, avg_reward, count}
            mutual_information : float MI(messages, outcomes) in bits
            channel_capacity   : float effective bits used in channel
            vocabulary_size    : int   number of message clusters
            num_samples        : int
        """
        if self._analysis_cache is not None:
            return self._analysis_cache

        N = len(self._messages)
        assert N >= 4, f"Need >= 4 recorded samples to analyse, have {N}."

        X = np.stack(self._messages)  # (N, latent_dim)
        actions = np.array(self._actions)
        rewards = np.array(self._rewards)

        # --- Clustering ---
        k = self._best_k(X)
        labels, centroids = _kmeans(X, k)
        vocab_size = k

        # Characterise each cluster
        cluster_info: Dict[int, Dict[str, Any]] = {}
        for c in range(k):
            mask = labels == c
            if mask.sum() == 0:
                continue
            c_actions = actions[mask]
            c_rewards = rewards[mask]
            action_counts = Counter(c_actions.tolist())
            dominant_action = action_counts.most_common(1)[0][0]
            cluster_info[int(c)] = {
                "dominant_action": dominant_action,
                "action_distribution": dict(action_counts),
                "avg_reward": float(c_rewards.mean()),
                "std_reward": float(c_rewards.std()) if len(c_rewards) > 1 else 0.0,
                "count": int(mask.sum()),
            }

        # --- Mutual information (messages -> outcomes) ---
        # Discretise rewards into quartile bins
        if rewards.std() > 1e-9:
            reward_bins = np.digitize(
                rewards,
                np.percentile(rewards, [25, 50, 75]),
            )
        else:
            reward_bins = np.zeros(N, dtype=int)

        mi_action = self._mi_bits(labels, actions)
        mi_reward = self._mi_bits(labels, reward_bins)
        mi_combined = max(mi_action, mi_reward)

        # --- Channel capacity estimate ---
        # Upper bound: log2(vocab_size).  Effective: MI(msg clusters, actions).
        capacity_upper = math.log2(max(vocab_size, 1))
        effective_capacity = min(mi_action, capacity_upper)

        self._analysis_cache = {
            "cluster_labels": cluster_info,
            "mutual_information": mi_combined,
            "mi_message_action": mi_action,
            "mi_message_reward": mi_reward,
            "channel_capacity": effective_capacity,
            "channel_capacity_upper_bound": capacity_upper,
            "vocabulary_size": vocab_size,
            "num_samples": N,
        }
        return self._analysis_cache

    # ------------------------------------------------------------------
    #  Embedding for visualisation
    # ------------------------------------------------------------------

    def get_embedding_data(self) -> Dict[str, Any]:
        """
        Return 2-D coordinates suitable for scatter plotting.

        Returns
        -------
        dict with:
            coords_2d : np.ndarray (N, 2)
            pca_2d    : np.ndarray (N, 2)
            actions   : np.ndarray (N,)
            rewards   : np.ndarray (N,)
            cluster_labels : np.ndarray (N,)
        """
        if self._embedding_cache is not None:
            return self._embedding_cache

        X = np.stack(self._messages)
        pca_2d = _pca(X, 2)

        # Attempt t-SNE-like for richer layout (cap at 2000 for speed)
        if X.shape[0] <= 2000:
            tsne_2d = _simple_tsne(X, n_components=2, perplexity=min(15, X.shape[0] // 2))
        else:
            tsne_2d = pca_2d  # fall back

        # Cluster labels (reuse analysis if available)
        analysis = self.analyze()
        k = analysis["vocabulary_size"]
        labels, _ = _kmeans(X, k)

        self._embedding_cache = {
            "coords_2d": tsne_2d,
            "pca_2d": pca_2d,
            "actions": np.array(self._actions),
            "rewards": np.array(self._rewards),
            "cluster_labels": labels,
        }
        return self._embedding_cache

    # ------------------------------------------------------------------
    #  Human-readable protocol description
    # ------------------------------------------------------------------

    def protocol_description(self) -> str:
        """
        Generate a natural-language summary of the emergent protocol
        the agents have developed.
        """
        analysis = self.analyze()
        lines: List[str] = []
        lines.append("=" * 60)
        lines.append("  EMERGENT COMMUNICATION PROTOCOL SUMMARY")
        lines.append("=" * 60)
        lines.append(f"  Vocabulary size : {analysis['vocabulary_size']} distinct message types")
        lines.append(f"  Channel usage   : {analysis['channel_capacity']:.2f} / "
                     f"{analysis['channel_capacity_upper_bound']:.2f} bits")
        lines.append(f"  MI(msg->action)  : {analysis['mi_message_action']:.3f} bits")
        lines.append(f"  MI(msg->reward)  : {analysis['mi_message_reward']:.3f} bits")
        lines.append("")

        for cid, info in sorted(analysis["cluster_labels"].items()):
            pct = 100 * info["count"] / analysis["num_samples"]
            lines.append(f"  Signal #{cid}  ({info['count']} msgs, {pct:.0f}%)")
            lines.append(f"    -> triggers action {info['dominant_action']} "
                         f"(avg reward {info['avg_reward']:.3f} +/- {info['std_reward']:.3f})")
            top_actions = sorted(info["action_distribution"].items(),
                                 key=lambda kv: -kv[1])[:3]
            lines.append(f"    -> action dist (top-3): "
                         + ", ".join(f"a{a}:{n}" for a, n in top_actions))

        lines.append("")

        # Qualitative assessment
        mi = analysis["mutual_information"]
        if mi > 1.5:
            verdict = "RICH -- messages carry substantial information about outcomes."
        elif mi > 0.5:
            verdict = "MODERATE -- messages partially predict outcomes."
        else:
            verdict = "WEAK -- messages carry little actionable information."
        lines.append(f"  Protocol quality: {verdict}")
        lines.append("=" * 60)
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════
#  2.  INFORMATION METRICS
# ═══════════════════════════════════════════════════════════════════

class InformationMetrics:
    """
    Information-theoretic measures of agent communication channels.

    All methods are stateless; pass data directly.
    """

    # ------------------------------------------------------------------

    @staticmethod
    def mutual_information(messages: np.ndarray, outcomes: np.ndarray) -> float:
        """
        MI(messages, outcomes) in bits.

        Both inputs are 1-D integer label arrays of equal length
        (pre-discretised).
        """
        N = len(messages)
        if N == 0:
            return 0.0
        joint = Counter(zip(messages, outcomes))
        cx = Counter(messages)
        cy = Counter(outcomes)
        mi = 0.0
        for (xi, yi), nxy in joint.items():
            pxy = nxy / N
            px = cx[xi] / N
            py = cy[yi] / N
            if pxy > 0 and px > 0 and py > 0:
                mi += pxy * math.log2(pxy / (px * py))
        return float(mi)

    # ------------------------------------------------------------------

    @staticmethod
    def channel_capacity(messages: np.ndarray) -> float:
        """
        Effective channel capacity (bits) -- entropy of the message
        distribution, H(M).  This is the *used* capacity;
        the theoretical max is log2(|M|).

        Parameters
        ----------
        messages : 1-D integer label array (cluster IDs of messages).
        """
        N = len(messages)
        if N == 0:
            return 0.0
        counts = Counter(messages)
        ent = 0.0
        for c in counts.values():
            p = c / N
            if p > 0:
                ent -= p * math.log2(p)
        return float(ent)

    # ------------------------------------------------------------------

    @staticmethod
    def redundancy(messages: np.ndarray) -> float:
        """
        Fraction of channel capacity that is *wasted* (redundant).

        redundancy = 1 - H(M) / log2(|M|)

        0 = every message type equally likely (no waste)
        1 = only one message type ever used (total waste)
        """
        N = len(messages)
        if N == 0:
            return 1.0
        unique = len(set(messages))
        if unique <= 1:
            return 1.0
        H = InformationMetrics.channel_capacity(messages)
        H_max = math.log2(unique)
        return float(1.0 - H / H_max) if H_max > 0 else 1.0

    # ------------------------------------------------------------------

    @staticmethod
    def coordination_entropy(joint_actions: np.ndarray) -> float:
        """
        Entropy of the *joint* action distribution H(A_1, A_2, ..., A_N)
        in bits.

        Parameters
        ----------
        joint_actions : (T, N) integer array where T = timesteps, N = agents.

        Low entropy -> agents are locked into stereotyped coordination.
        High entropy -> varied / exploratory coordination.
        """
        T = joint_actions.shape[0]
        if T == 0:
            return 0.0
        # Tuple-ify each row for counting
        tuples = [tuple(row) for row in joint_actions]
        counts = Counter(tuples)
        ent = 0.0
        for c in counts.values():
            p = c / T
            if p > 0:
                ent -= p * math.log2(p)
        return float(ent)


# ═══════════════════════════════════════════════════════════════════
#  VALIDATION
# ═══════════════════════════════════════════════════════════════════

def validate_emergent() -> None:
    """
    Creates synthetic communication data and runs the full
    LatentLanguageAnalyzer + InformationMetrics pipeline.
    """

    LATENT_DIM = 16
    NUM_ACTIONS = 16
    NUM_AGENTS = 5
    NUM_SAMPLES = 300

    print("=" * 64)
    print("  POLARIS v5 -- Emergent Language Analysis Validation")
    print("=" * 64)

    rng = np.random.RandomState(42)

    # ------------------------------------------------------------------
    #  Generate synthetic messages with structure:
    #    3 "ground-truth" message types mapped loosely to 3 action groups
    # ------------------------------------------------------------------
    analyzer = LatentLanguageAnalyzer(latent_dim=LATENT_DIM, max_clusters=8)

    centres = rng.randn(3, LATENT_DIM) * 2.0          # 3 clusters
    action_groups = [[0, 1, 2, 3, 4],                  # cluster 0 -> low actions
                     [5, 6, 7, 8, 9, 10],              # cluster 1 -> mid actions
                     [11, 12, 13, 14, 15]]              # cluster 2 -> high actions

    for _ in range(NUM_SAMPLES):
        cluster_id = rng.randint(0, 3)
        msg = centres[cluster_id] + rng.randn(LATENT_DIM) * 0.3
        action = rng.choice(action_groups[cluster_id])
        # Reward = higher for higher actions (artificial trend)
        reward = 0.5 + 0.03 * action + rng.randn() * 0.1
        state = torch.randn(21)
        analyzer.record(torch.tensor(msg, dtype=torch.float32), state, action, reward)

    print(f"\n  Recorded {NUM_SAMPLES} message-action-reward tuples.\n")

    # --- Full analysis ---
    analysis = analyzer.analyze()
    print(f"  [analyze]")
    print(f"    vocabulary_size     = {analysis['vocabulary_size']}")
    print(f"    MI(msg->action)      = {analysis['mi_message_action']:.4f} bits")
    print(f"    MI(msg->reward)      = {analysis['mi_message_reward']:.4f} bits")
    print(f"    channel_capacity    = {analysis['channel_capacity']:.4f} bits "
          f"(upper {analysis['channel_capacity_upper_bound']:.4f})")
    for cid, info in sorted(analysis["cluster_labels"].items()):
        print(f"    cluster {cid}: dom_action={info['dominant_action']}, "
              f"avg_reward={info['avg_reward']:.3f}, n={info['count']}")

    # --- Embedding ---
    emb = analyzer.get_embedding_data()
    print(f"\n  [get_embedding_data]")
    print(f"    PCA  shape   = {emb['pca_2d'].shape}")
    print(f"    t-SNE shape  = {emb['coords_2d'].shape}")
    print(f"    cluster ids  = {sorted(set(emb['cluster_labels'].tolist()))}")

    # --- Protocol description ---
    desc = analyzer.protocol_description()
    print(f"\n{desc}")

    # --- Information Metrics ---
    print("\n  [InformationMetrics]")
    im = InformationMetrics()

    msg_labels = emb["cluster_labels"]
    action_labels = np.array(analyzer._actions)
    mi = im.mutual_information(msg_labels, action_labels)
    print(f"    MI(clusters, actions)   = {mi:.4f} bits")

    cap = im.channel_capacity(msg_labels)
    print(f"    channel_capacity(H(M))  = {cap:.4f} bits")

    red = im.redundancy(msg_labels)
    print(f"    redundancy              = {red:.4f}")

    # Joint actions: simulate 50 timesteps x 5 agents
    joint_acts = rng.randint(0, NUM_ACTIONS, size=(50, NUM_AGENTS))
    ce = im.coordination_entropy(joint_acts)
    print(f"    coordination_entropy    = {ce:.4f} bits")

    # Low-entropy scenario: agents always do the same thing
    uniform_acts = np.tile([0, 1, 2, 3, 4], (50, 1))
    ce_low = im.coordination_entropy(uniform_acts)
    print(f"    coord_entropy(uniform)  = {ce_low:.4f} bits  (expect 0)")

    print("\n" + "=" * 64)
    print("  EMERGENT ANALYSIS VALIDATION PASSED")
    print("=" * 64)


if __name__ == "__main__":
    validate_emergent()
