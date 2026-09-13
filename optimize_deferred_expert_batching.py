from __future__ import annotations

import argparse
import csv
import math
from dataclasses import asdict, dataclass
from pathlib import Path

from run_mac_studio_m5_ultra_scaling import calculate_capacity_agent_limit
from src.llm_architecture_simulator.expert_routing import (
    generate_normalized_gaussian_expert_popularity,
)
from src.llm_architecture_simulator.published_model_profiles import (
    BYTES_PER_GB,
    PUBLISHED_MODEL_PROFILES,
    PublishedMoeModelProfile,
)


@dataclass(frozen=True)
class DeferredBatchingCandidate:
    model: str
    node_count: int
    agent_count: int
    coldest_expert_fraction: float
    cold_horizon_rounds: int
    expected_cold_expert_selections_per_token: float
    probability_one_layer_waits: float
    expected_added_layer_round_waits_per_token: float
    expert_weight_gb_per_token: float
    total_memory_gb_per_token: float
    memory_traffic_reduction_fraction: float
    memory_bound_throughput_gain_fraction: float
    pareto_efficient: bool = False


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Enumerate cold-expert batching horizons and report the memory/latency "
            "Pareto frontier before running event simulations"
        )
    )
    parser.add_argument("--node-count", type=int, default=7)
    parser.add_argument("--context-tokens", type=int, default=100_000)
    parser.add_argument("--kv-resident-fraction", type=float, default=0.80)
    parser.add_argument("--output", type=Path, default=Path("results/deferred_expert_batching.csv"))
    return parser.parse_args()


def approximate_inclusion_probabilities(
    popularity_weights: tuple[float, ...],
    selected_expert_count: int,
) -> tuple[float, ...]:
    """Approximate weighted-without-replacement inclusion probabilities.

    The scale is chosen so probabilities sum to Top-K while no expert exceeds one.
    This is deterministic and sufficient for screening scheduling policies; finalists
    still need the discrete-event simulation.
    """
    lower, upper = 0.0, selected_expert_count / min(popularity_weights)
    for _ in range(100):
        scale = (lower + upper) / 2
        probability_sum = sum(min(1.0, scale * weight) for weight in popularity_weights)
        if probability_sum < selected_expert_count:
            lower = scale
        else:
            upper = scale
    return tuple(min(1.0, upper * weight) for weight in popularity_weights)


def approximate_expected_capped_batch_loads(
    expected_request_count: float,
    maximum_batch_size: int,
) -> float:
    """Approximate E[ceil(Poisson(lambda) / batch_size)] without SciPy."""
    if expected_request_count <= 0:
        return 0.0
    expected_loads = 1.0 - math.exp(-expected_request_count)
    threshold = maximum_batch_size
    standard_deviation = math.sqrt(expected_request_count)
    while threshold < expected_request_count + 8 * standard_deviation:
        z = (threshold + 0.5 - expected_request_count) / standard_deviation
        expected_loads += 0.5 * math.erfc(z / math.sqrt(2.0))
        threshold += maximum_batch_size
    return expected_loads


def model_memory_terms(
    profile: PublishedMoeModelProfile,
    agent_count: int,
    node_count: int,
    context_tokens: int,
    kv_resident_fraction: float,
) -> tuple[float, float, float, tuple[float, ...]]:
    configuration = profile.create_simulation_configuration(
        node_count=node_count,
        continuously_active_agent_count=agent_count,
        compute_peak_tflops_per_node=100.0,
        memory_bandwidth_gb_per_second_per_node=1_200.0,
        network_link_bandwidth_gb_per_second_each_direction=7.5,
        network_link_fixed_latency_ns=50_000.0,
        random_seed=7,
        collapse_repeated_layers=True,
        context_token_count=context_tokens,
        kv_resident_fraction=kv_resident_fraction,
    )
    popularity = generate_normalized_gaussian_expert_popularity(
        profile.routed_expert_count,
        configuration.expert_popularity_sigma,
        random_seed=7,
    )
    inclusion_probabilities = approximate_inclusion_probabilities(
        popularity,
        profile.selected_routed_expert_count_per_token,
    )
    shared_batch_size = min(
        configuration.maximum_shared_layer_batch_size,
        max(1.0, agent_count / node_count),
    )
    shared_weight_bytes_per_token = (
        configuration.shared_layer_weight_bytes_read_per_batch / shared_batch_size
    )
    kv_and_activation_bytes_per_token = (
        configuration.shared_layer_activation_bytes_per_token
    )
    return (
        shared_weight_bytes_per_token,
        kv_and_activation_bytes_per_token,
        configuration.expert_weight_bytes_read_per_batch,
        inclusion_probabilities,
    )


def evaluate_candidate(
    profile: PublishedMoeModelProfile,
    node_count: int,
    agent_count: int,
    context_tokens: int,
    kv_resident_fraction: float,
    coldest_expert_fraction: float,
    cold_horizon_rounds: int,
) -> DeferredBatchingCandidate:
    shared_bytes, kv_bytes, expert_weight_bytes, inclusion_probabilities = model_memory_terms(
        profile,
        agent_count,
        node_count,
        context_tokens,
        kv_resident_fraction,
    )
    ranked_indexes = sorted(
        range(len(inclusion_probabilities)),
        key=inclusion_probabilities.__getitem__,
    )
    cold_count = round(len(ranked_indexes) * coldest_expert_fraction)
    cold_indexes = set(ranked_indexes[:cold_count])

    expert_loads_per_round = 0.0
    baseline_expert_loads_per_round = 0.0
    expected_cold_selections = 0.0
    for expert_index, probability in enumerate(inclusion_probabilities):
        baseline_expert_loads_per_round += approximate_expected_capped_batch_loads(
            agent_count * probability,
            128,
        )
        if expert_index in cold_indexes:
            expected_cold_selections += probability
            expert_loads_per_round += (
                approximate_expected_capped_batch_loads(
                    agent_count * probability * cold_horizon_rounds,
                    128,
                )
                / cold_horizon_rounds
            )
        else:
            expert_loads_per_round += approximate_expected_capped_batch_loads(
                agent_count * probability,
                128,
            )

    baseline_expert_bytes = (
        expert_weight_bytes * baseline_expert_loads_per_round / agent_count
    )
    candidate_expert_bytes = expert_weight_bytes * expert_loads_per_round / agent_count
    baseline_total_bytes = kv_bytes + shared_bytes + baseline_expert_bytes
    candidate_total_bytes = kv_bytes + shared_bytes + candidate_expert_bytes
    probability_layer_waits = 1.0 - math.exp(-expected_cold_selections)
    expected_added_waits = (
        profile.moe_layer_count
        * probability_layer_waits
        * (cold_horizon_rounds - 1)
        / 2
    )
    return DeferredBatchingCandidate(
        model=profile.name,
        node_count=node_count,
        agent_count=agent_count,
        coldest_expert_fraction=coldest_expert_fraction,
        cold_horizon_rounds=cold_horizon_rounds,
        expected_cold_expert_selections_per_token=expected_cold_selections,
        probability_one_layer_waits=probability_layer_waits,
        expected_added_layer_round_waits_per_token=expected_added_waits,
        expert_weight_gb_per_token=candidate_expert_bytes / BYTES_PER_GB,
        total_memory_gb_per_token=candidate_total_bytes / BYTES_PER_GB,
        memory_traffic_reduction_fraction=(
            1.0 - candidate_total_bytes / baseline_total_bytes
        ),
        memory_bound_throughput_gain_fraction=(
            baseline_total_bytes / candidate_total_bytes - 1.0
        ),
    )


def mark_pareto_frontier(
    candidates: list[DeferredBatchingCandidate],
) -> list[DeferredBatchingCandidate]:
    marked = []
    for candidate in candidates:
        dominated = any(
            other.memory_bound_throughput_gain_fraction
            >= candidate.memory_bound_throughput_gain_fraction
            and other.expected_added_layer_round_waits_per_token
            <= candidate.expected_added_layer_round_waits_per_token
            and (
                other.memory_bound_throughput_gain_fraction
                > candidate.memory_bound_throughput_gain_fraction
                or other.expected_added_layer_round_waits_per_token
                < candidate.expected_added_layer_round_waits_per_token
            )
            for other in candidates
        )
        marked.append(
            DeferredBatchingCandidate(
                **{
                    **asdict(candidate),
                    "pareto_efficient": not dominated,
                }
            )
        )
    return marked


def run_optimization(arguments: argparse.Namespace) -> list[DeferredBatchingCandidate]:
    usable_memory_bytes_per_node = int(512 * BYTES_PER_GB * 0.90)
    rows = []
    for profile in PUBLISHED_MODEL_PROFILES:
        agent_count = calculate_capacity_agent_limit(
            profile,
            arguments.node_count,
            arguments.context_tokens,
            arguments.kv_resident_fraction,
            usable_memory_bytes_per_node,
        )
        model_rows = [
            evaluate_candidate(
                profile,
                arguments.node_count,
                agent_count,
                arguments.context_tokens,
                arguments.kv_resident_fraction,
                cold_fraction,
                horizon,
            )
            for cold_fraction in (0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 1.00)
            for horizon in (2, 3, 4)
        ]
        rows.extend(mark_pareto_frontier(model_rows))
    return rows


def main() -> None:
    arguments = parse_arguments()
    rows = run_optimization(arguments)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    with arguments.output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=asdict(rows[0]).keys())
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)
    for row in rows:
        if row.pareto_efficient:
            print(
                f"{row.model}: coldest {row.coldest_expert_fraction:.0%}, "
                f"{row.cold_horizon_rounds} rounds -> "
                f"+{row.memory_bound_throughput_gain_fraction:.2%} throughput ceiling, "
                f"+{row.expected_added_layer_round_waits_per_token:.1f} layer-round waits"
            )


if __name__ == "__main__":
    main()
