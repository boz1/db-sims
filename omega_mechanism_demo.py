#!/usr/bin/env python3
"""
Demonstration of the Omega (Omega) mechanism for distance-ranked sharing.

This script shows concrete examples of how the reward-sharing mechanism works
and demonstrates its key properties:
1. Proximity advantage (closer regions get better rank weights)
2. Concentration penalty (crowding dilutes per-proposer rewards)
3. Trade-offs between being close vs. avoiding crowds
"""

import numpy as np


def compute_omega(distances, occupancies, rank_weights=None):
    """
    Compute Omega (per-proposer share) for each region.

    Args:
        distances: List of distances for participating regions
        occupancies: List of occupancies (number of proposers) for each region
        rank_weights: Rank weights (default: 2^(-j))

    Returns:
        List of Omega values (per-proposer shares) for each region
    """
    # Sort by distance
    sorted_indices = np.argsort(distances)
    sorted_occupancies = [occupancies[i] for i in sorted_indices]

    # Default rank weights: 2^(-j) for rank j
    if rank_weights is None:
        rank_weights = [2**(-j) for j in range(len(distances))]

    # Compute denominator: sum w_j x x_j
    denominator = sum(rank_weights[j] * sorted_occupancies[j]
                     for j in range(len(distances)))

    # Compute Omega for each region
    omega_sorted = [rank_weights[j] / denominator for j in range(len(distances))]

    # Unsort to match original order
    omega = [0] * len(distances)
    for j, orig_idx in enumerate(sorted_indices):
        omega[orig_idx] = omega_sorted[j]

    return omega


def print_scenario(scenario_name, regions, distances, occupancies, source_value):
    """Print a scenario and compute Omega values."""
    print(f"\n{'='*70}")
    print(f"Scenario: {scenario_name}")
    print(f"{'='*70}")

    print(f"\nSource Value: {source_value}")
    print(f"\n{'Region':<15} {'Distance':<12} {'Occupancy':<12} {'Rank':<8}")
    print("-" * 70)

    # Compute ranks
    sorted_indices = np.argsort(distances)
    ranks = [0] * len(distances)
    for rank, idx in enumerate(sorted_indices):
        ranks[idx] = rank + 1

    for i, region in enumerate(regions):
        print(f"{region:<15} {distances[i]:<12.1f} {occupancies[i]:<12d} {ranks[i]:<8d}")

    # Compute Omega
    omega = compute_omega(distances, occupancies)

    print(f"\n{'Region':<15} {'Omega (per-prop)':<15} {'Reward/prop':<15} {'Total Reward':<15}")
    print("-" * 70)

    total_reward = 0
    for i, region in enumerate(regions):
        reward_per_prop = omega[i] * source_value
        total_reward_region = reward_per_prop * occupancies[i]
        total_reward += total_reward_region
        print(f"{region:<15} {omega[i]:<15.4f} {reward_per_prop:<15.4f} {total_reward_region:<15.4f}")

    print(f"\n{'Total distributed:':<15} {'':<15} {'':<15} {total_reward:<15.4f}")
    print(f"(Should equal source value: {source_value})")


def demo_proximity_advantage():
    """Demo: Closer regions get higher per-proposer rewards."""
    regions = ["At Source", "Medium", "Far"]
    distances = [0.0, 2.0, 4.0]  # Source positioned at first region
    occupancies = [1, 1, 1]  # Equal occupancy
    source_value = 10.0

    print_scenario(
        "Proximity Advantage (equal occupancy)",
        regions, distances, occupancies, source_value
    )

    print("\n** Key insight: With equal occupancy, closer regions get higher per-proposer rewards.")
    print("   Rank 1 (at source) gets 4x, Rank 2 gets 2x, Rank 3 gets 1x the base rate.")


def demo_concentration_penalty():
    """Demo: Crowding in the closest region dilutes rewards."""
    regions = ["At Source (crowded)", "Far (alone)"]
    distances = [0.0, 4.0]  # Source at first region
    occupancies_scenarios = [
        ([1, 1], "Equal occupancy"),
        ([2, 1], "2:1 crowding"),
        ([4, 1], "4:1 crowding"),
        ([8, 1], "8:1 crowding"),
    ]
    source_value = 10.0

    for occupancies, description in occupancies_scenarios:
        print_scenario(
            f"Concentration Penalty - {description}",
            regions, distances, occupancies, source_value
        )

    print("\n** Key insight: As the region at the source gets crowded, per-proposer reward decreases.")
    print("   Eventually, being alone in a far region can yield higher per-proposer reward!")


def demo_multi_region_equilibrium():
    """Demo: Three regions with different distance-occupancy trade-offs."""
    regions = ["North (at source)", "Central (medium)", "South (far)"]
    distances = [0.0, 2.0, 4.0]  # Source at North region
    occupancies = [3, 2, 1]
    source_value = 15.0

    print_scenario(
        "Multi-Region Equilibrium",
        regions, distances, occupancies, source_value
    )

    omega = compute_omega(distances, occupancies)
    rewards = [omega[i] * source_value for i in range(len(regions))]

    print("\n** Analysis:")
    print(f"   Per-proposer rewards: North={rewards[0]:.4f}, Central={rewards[1]:.4f}, South={rewards[2]:.4f}")

    if max(rewards) == min(rewards):
        print("   [OK] Perfect equilibrium: all regions offer equal per-proposer rewards!")
    else:
        max_idx = np.argmax(rewards)
        print(f"   [X] Not equilibrium: {regions[max_idx]} offers highest reward")
        print(f"     Proposers should migrate there, reducing its Omega value")


def demo_value_distance_tradeoff():
    """Demo: High-value far source vs. low-value source at region."""
    print(f"\n{'='*70}")
    print("Scenario: Value-Distance Trade-off")
    print(f"{'='*70}")

    regions = ["North"]  # Single region choosing between sources
    occupancy = 1

    sources = [
        ("SourceA (at region, low value)", 0.0, 8.0),
        ("SourceB (far, high value)", 4.0, 15.0)
    ]

    print("\nComparing two sources from the same region:")
    print(f"\n{'Source':<35} {'Distance':<12} {'Value':<12} {'Omega':<12} {'Reward':<12}")
    print("-" * 75)

    for source_name, distance, value in sources:
        # When alone, Omega = w_rank / (w_rank x occupancy) = 1/occupancy = 1
        omega = 1.0  # Rank 1, alone
        reward = omega * value
        print(f"{source_name:<35} {distance:<12.1f} {value:<12.1f} {omega:<12.4f} {reward:<12.4f}")

    print("\n** Key insight: Value and proximity both matter.")
    print("   Proposers must balance high-value sources with sources at their location.")
    print("   As others choose the high-value source, its Omega drops (concentration penalty).")


def demo_rank_weights():
    """Demo: Show how different rank weight schemes affect distribution."""
    print(f"\n{'='*70}")
    print("Scenario: Effect of Rank Weight Schemes")
    print(f"{'='*70}")

    regions = ["Rank 1 (at source)", "Rank 2", "Rank 3"]
    distances = [0.0, 2.0, 4.0]  # Source at Rank 1 region
    occupancies = [1, 1, 1]
    source_value = 10.0

    weight_schemes = [
        ([1.0, 0.5, 0.25], "Exponential (2^-j) - DEFAULT"),
        ([1.0, 1.0, 1.0], "Uniform (no rank advantage)"),
        ([4.0, 2.0, 1.0], "Steep exponential"),
        ([1.0, 0.8, 0.6], "Linear decay"),
    ]

    for weights, name in weight_schemes:
        print(f"\n{name}:")
        print(f"Weights: {weights}")

        omega = compute_omega(distances, occupancies, rank_weights=weights)
        rewards = [omega[i] * source_value for i in range(len(regions))]

        print(f"Per-proposer rewards: {', '.join(f'{r:.4f}' for r in rewards)}")
        print(f"Rank 1 advantage: {rewards[0]/rewards[2]:.2f}x over Rank 3")

    print("\n** Key insight: Rank weights control how much proximity matters.")
    print("   Steeper weights = stronger incentive to be at source location.")
    print("   Uniform weights = location doesn't matter (only occupancy).")


def demo_equilibrium_analysis():
    """Demo: Find equilibrium distribution for a simple scenario."""
    print(f"\n{'='*70}")
    print("Scenario: Equilibrium Analysis")
    print(f"{'='*70}")

    print("\nGiven: 2 regions, 6 proposers total, distances [0.0, 2.0] (source at first region), source value = 12.0")
    print("\nTesting different distributions:")
    print(f"\n{'Distribution':<20} {'Omega_at_src':<12} {'Omega_far':<12} {'R_at_src':<12} {'R_far':<12} {'Balanced?':<12}")
    print("-" * 80)

    regions = ["At Source", "Far"]
    distances = [0.0, 2.0]  # Source at first region
    source_value = 12.0
    total_proposers = 6

    for n_close in range(1, total_proposers):
        n_far = total_proposers - n_close
        occupancies = [n_close, n_far]

        omega = compute_omega(distances, occupancies)
        rewards = [omega[i] * source_value for i in range(len(regions))]

        diff = abs(rewards[0] - rewards[1])
        balanced = "[OK] EQUILIBRIUM" if diff < 0.01 else ""

        print(f"[{n_close}, {n_far}]              {omega[0]:<12.4f} {omega[1]:<12.4f} "
              f"{rewards[0]:<12.4f} {rewards[1]:<12.4f} {balanced:<12}")

    print("\n** Key insight: Equilibrium occurs when per-proposer rewards are equal.")
    print("   At equilibrium, no proposer has incentive to switch regions.")
    print("   Learning algorithms should converge toward these distributions.")


def main():
    """Run all demonstrations."""
    print("="*70)
    print("Omega Mechanism Demonstration")
    print("Distance-Ranked Sharing with Occupancy Penalty")
    print("="*70)

    demo_proximity_advantage()
    demo_concentration_penalty()
    demo_multi_region_equilibrium()
    demo_value_distance_tradeoff()
    demo_rank_weights()
    demo_equilibrium_analysis()

    print("\n" + "="*70)
    print("Summary")
    print("="*70)
    print("""
The Omega mechanism implements distance-ranked sharing with two key properties:

1. PROXIMITY ADVANTAGE: Closer regions get higher rank weights (w_j = 2^(-j))
   -> Incentivizes choosing regions near high-value sources

2. CONCENTRATION PENALTY: Crowded regions split rewards across many proposers
   -> Incentivizes diversity and avoiding over-concentration

Together, these create an equilibrium where:
- Multiple regions participate (diversity)
- Closer regions attract more proposers (efficiency)
- No single proposer wants to switch (stability)

The learning algorithms (EMA-Softmax, UCB) navigate this landscape:
- Exploration helps discover good (region, source) combinations
- Exploitation drives toward high-reward configurations
- Stochasticity prevents collapse to a single region
""")
    print("="*70)


if __name__ == "__main__":
    main()
