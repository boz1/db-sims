#!/usr/bin/env python3
"""
Decentralized Building Simulator (db-sims) - Core simulation engine

Inspired by GeoDec research, generalized for studying location choice in various
distributed and decentralized block building regimes.

Implements distance-ranked sharing with two interchangeable learning policies:
  (A) EMA + softmax
  (B) Individual UCB bandit
"""

import numpy as np
from typing import List, Tuple, Dict
from dataclasses import dataclass, field
from abc import ABC, abstractmethod


@dataclass
class Region:
    """A geographical region."""
    id: int
    name: str


@dataclass
class Source:
    """A signal source with constant value."""
    id: int
    name: str
    value: float  # V_I: constant base value
    home_region: int = 0  # optional: which region it's "anchored" to


@dataclass
class Proposer:
    """A proposer/agent with learning state."""
    id: int
    policy: 'LearningPolicy'
    current_region: int = 0  # Persistent region assignment

    def choose_region_and_source(self, regions: List[Region], sources: List[Source]) -> Tuple[int, int]:
        """Choose region and source based on policy."""
        return self.policy.choose(regions, sources)

    def update(self, region_id: int, source_id: int, reward: float):
        """Update learning state based on observed reward."""
        self.policy.update(region_id, source_id, reward)

    def set_region(self, region_id: int):
        """Set the proposer's current region."""
        self.current_region = region_id


class LearningPolicy(ABC):
    """Abstract base class for learning policies."""

    @abstractmethod
    def choose(self, regions: List[Region], sources: List[Source]) -> Tuple[int, int]:
        """Choose (region_id, source_id)."""
        pass

    @abstractmethod
    def update(self, region_id: int, source_id: int, reward: float):
        """Update policy state after observing reward."""
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Return policy name for logging."""
        pass


class EMASoftmaxPolicy(LearningPolicy):
    """
    Policy A: EMA + softmax with two-stage selection.
    Individual scoreboard per proposer.
    """

    def __init__(self, n_regions: int, n_sources: int,
                 eta: float = 0.1,
                 beta_reg: float = 2.0,
                 beta_src: float = 2.0,
                 cost_c: float = 0.0):
        """
        Args:
            n_regions: Number of regions
            n_sources: Number of sources
            eta: EMA learning rate
            beta_reg: Temperature for region selection
            beta_src: Temperature for source selection
            cost_c: Migration cost (subtracted from region score)
        """
        self.n_regions = n_regions
        self.n_sources = n_sources
        self.eta = eta
        self.beta_reg = beta_reg
        self.beta_src = beta_src
        self.cost_c = cost_c

        # Scoreboard: u_hat[r, I] for each (region, source) pair
        self.u_hat = np.zeros((n_regions, n_sources))

    def choose(self, regions: List[Region], sources: List[Source]) -> Tuple[int, int]:
        """Two-stage softmax selection."""
        # Stage 1: Choose region
        # U^reg(r) = max_I u_hat(r, I) - c
        U_reg = np.max(self.u_hat, axis=1) - self.cost_c

        # Softmax over regions
        exp_scores = np.exp(self.beta_reg * U_reg)
        probs_reg = exp_scores / np.sum(exp_scores)
        region_id = np.random.choice(self.n_regions, p=probs_reg)

        # Stage 2: Choose source given region
        # Softmax over sources for the chosen region
        exp_scores_src = np.exp(self.beta_src * self.u_hat[region_id, :])
        probs_src = exp_scores_src / np.sum(exp_scores_src)
        source_id = np.random.choice(self.n_sources, p=probs_src)

        return region_id, source_id

    def update(self, region_id: int, source_id: int, reward: float):
        """EMA update: u_hat(r, I) <- (1-eta)*u_hat(r, I) + eta*R."""
        self.u_hat[region_id, source_id] = (
            (1 - self.eta) * self.u_hat[region_id, source_id] +
            self.eta * reward
        )

    def get_name(self) -> str:
        return "EMA-Softmax"


class UCBPolicy(LearningPolicy):
    """
    Policy B: Individual UCB bandit.
    Arms are (region, source) pairs.
    """

    def __init__(self, n_regions: int, n_sources: int, alpha: float = 1.0):
        """
        Args:
            n_regions: Number of regions
            n_sources: Number of sources
            alpha: Exploration parameter
        """
        self.n_regions = n_regions
        self.n_sources = n_sources
        self.alpha = alpha

        # Counts and empirical means for each (region, source) arm
        self.N = np.zeros((n_regions, n_sources))
        self.mu_hat = np.zeros((n_regions, n_sources))
        self.t = 0  # Local clock

    def choose(self, regions: List[Region], sources: List[Source]) -> Tuple[int, int]:
        """UCB selection: argmax [mu_hat(r,I) + alpha*sqrt(log(1+t)/(1+N(r,I)))]."""
        # Compute UCB scores
        exploration_bonus = self.alpha * np.sqrt(
            np.log(1 + self.t) / (1 + self.N)
        )
        ucb_scores = self.mu_hat + exploration_bonus

        # Flatten and find argmax
        flat_idx = np.argmax(ucb_scores)
        region_id = flat_idx // self.n_sources
        source_id = flat_idx % self.n_sources

        return region_id, source_id

    def update(self, region_id: int, source_id: int, reward: float):
        """Standard UCB update."""
        self.N[region_id, source_id] += 1
        n = self.N[region_id, source_id]
        self.mu_hat[region_id, source_id] += (reward - self.mu_hat[region_id, source_id]) / n
        self.t += 1

    def get_name(self) -> str:
        return "UCB"


class MCPSimulator:
    """
    Core simulator for studying location choice in decentralized block building.

    Implements distance-ranked sharing mechanism to model geographic incentives
    in various distributed/decentralized building regimes (e.g., multiple concurrent
    proposers, distributed builder networks, decentralized ordering protocols).

    Note: Class retains 'MCP' name for backward compatibility.
    """

    def __init__(self,
                 regions: List[Region],
                 sources: List[Source],
                 proposers: List[Proposer],
                 distance_matrix: np.ndarray,
                 rank_weights: List[float] = None,
                 K: int = 5,
                 seed: int = 42):
        """
        Args:
            regions: List of regions
            sources: List of sources
            proposers: List of proposers (agents)
            distance_matrix: D[r, I] = distance from region r to source I
            rank_weights: w_j for rank j (default: 2^(-j))
            K: Number of concurrent proposers per slot
            seed: Random seed
        """
        self.regions = regions
        self.sources = sources
        self.proposers = proposers
        self.distance_matrix = distance_matrix
        self.K = K

        self.n_regions = len(regions)
        self.n_sources = len(sources)
        self.n_proposers = len(proposers)

        # Default rank weights: w_j = 2^(-j), j=0,1,2,...
        if rank_weights is None:
            # Make enough weights for all possible ranks
            self.rank_weights = [2**(-j) for j in range(self.n_regions)]
        else:
            self.rank_weights = rank_weights

        np.random.seed(seed)

        # Initialize proposers evenly across regions
        self._initialize_proposer_distribution()

        # Tracking
        self.slot = 0
        self.region_counts_history = []
        self.source_counts_history = []
        self.reward_history = []
        self.proposer_distribution_history = []  # Track distribution of all proposers over time

    def _initialize_proposer_distribution(self):
        """Initialize proposers evenly across regions."""
        for i, proposer in enumerate(self.proposers):
            # Distribute evenly: proposer i goes to region (i mod n_regions)
            initial_region = i % self.n_regions
            proposer.set_region(initial_region)

    def _get_proposer_distribution(self) -> np.ndarray:
        """Get current distribution of all proposers across regions."""
        distribution = np.zeros(self.n_regions)
        for proposer in self.proposers:
            distribution[proposer.current_region] += 1
        return distribution

    def run_slot(self):
        """Execute one slot: sample proposers, they choose, compute rewards, update."""
        # Sample K proposers uniformly
        selected_indices = np.random.choice(self.n_proposers, size=self.K, replace=False)
        selected_proposers = [self.proposers[i] for i in selected_indices]

        # Each proposer chooses (region, source)
        choices = []
        for prop in selected_proposers:
            r, I = prop.choose_region_and_source(self.regions, self.sources)
            choices.append((prop, r, I))

        # Compute Ω and rewards for each source
        rewards = {}

        for source_idx in range(self.n_sources):
            # Find participating regions for this source
            participating = {}  # region_id -> count
            for prop, r, I in choices:
                if I == source_idx:
                    participating[r] = participating.get(r, 0) + 1

            if not participating:
                continue

            # Rank regions by distance to source
            region_ids = list(participating.keys())
            distances = [self.distance_matrix[r, source_idx] for r in region_ids]
            sorted_indices = np.argsort(distances)
            ranked_regions = [region_ids[i] for i in sorted_indices]

            # Compute Ω for each participating region
            # Ω_r(j) = w_j / sum_{ℓ} w_ℓ * x_r(ℓ)
            occupancies = [participating[r] for r in ranked_regions]
            denominator = sum(self.rank_weights[j] * occupancies[j]
                            for j in range(len(ranked_regions)))

            omega = {}
            for j, r in enumerate(ranked_regions):
                omega[r] = self.rank_weights[j] / denominator

            # Compute per-proposer contribution for this source
            source_value = self.sources[source_idx].value
            for prop, r, I in choices:
                if I == source_idx:
                    contrib = omega[r] * source_value
                    if prop.id not in rewards:
                        rewards[prop.id] = 0.0
                    rewards[prop.id] += contrib

        # Update proposers and their regions
        for prop, r, I in choices:
            reward = rewards.get(prop.id, 0.0)
            prop.update(r, I, reward)
            # Update proposer's persistent region
            prop.set_region(r)

        # Track statistics
        region_counts = np.zeros(self.n_regions)
        source_counts = np.zeros(self.n_sources)
        slot_rewards = []

        for prop, r, I in choices:
            region_counts[r] += 1
            source_counts[I] += 1
            slot_rewards.append(rewards.get(prop.id, 0.0))

        self.region_counts_history.append(region_counts)
        self.source_counts_history.append(source_counts)
        self.reward_history.append(slot_rewards)

        # Track overall proposer distribution (all proposers, not just selected K)
        proposer_distribution = self._get_proposer_distribution()
        self.proposer_distribution_history.append(proposer_distribution)

        self.slot += 1

    def run(self, n_slots: int):
        """Run simulation for n_slots."""
        for _ in range(n_slots):
            self.run_slot()

    def get_statistics(self) -> Dict:
        """Compute summary statistics."""
        region_counts = np.array(self.region_counts_history)
        source_counts = np.array(self.source_counts_history)
        proposer_distribution = np.array(self.proposer_distribution_history)

        # Average counts over time
        avg_region_counts = np.mean(region_counts, axis=0)
        avg_source_counts = np.mean(source_counts, axis=0)
        avg_proposer_distribution = np.mean(proposer_distribution, axis=0)

        # Average reward per slot
        all_rewards = [r for slot_rewards in self.reward_history for r in slot_rewards]
        avg_reward = np.mean(all_rewards) if all_rewards else 0.0

        # Concentration metrics (Gini coefficient for regions)
        def gini(x):
            """Compute Gini coefficient."""
            sorted_x = np.sort(x)
            n = len(x)
            cumsum = np.cumsum(sorted_x)
            return (2 * np.sum((np.arange(1, n+1)) * sorted_x)) / (n * cumsum[-1]) - (n + 1) / n if cumsum[-1] > 0 else 0.0

        region_gini = gini(avg_region_counts)
        source_gini = gini(avg_source_counts)
        proposer_dist_gini = gini(avg_proposer_distribution)

        # Entropy
        def entropy(counts):
            """Compute normalized entropy."""
            probs = counts / np.sum(counts) if np.sum(counts) > 0 else counts
            probs = probs[probs > 0]
            return -np.sum(probs * np.log(probs)) / np.log(len(counts)) if len(probs) > 0 else 0.0

        region_entropy = entropy(avg_region_counts)
        source_entropy = entropy(avg_source_counts)
        proposer_dist_entropy = entropy(avg_proposer_distribution)

        return {
            'avg_region_counts': avg_region_counts,
            'avg_source_counts': avg_source_counts,
            'avg_proposer_distribution': avg_proposer_distribution,
            'avg_reward': avg_reward,
            'region_gini': region_gini,
            'source_gini': source_gini,
            'proposer_dist_gini': proposer_dist_gini,
            'region_entropy': region_entropy,
            'source_entropy': source_entropy,
            'proposer_dist_entropy': proposer_dist_entropy,
            'total_slots': len(self.region_counts_history)
        }


# Example usage: See my_experiments.py for how to use this simulator
