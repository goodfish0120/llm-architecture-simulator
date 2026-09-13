from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

from src.llm_architecture_simulator.published_model_profiles import (
    BYTES_PER_GB,
    PUBLISHED_MODEL_PROFILES,
    PublishedMoeModelProfile,
)
from src.llm_architecture_simulator.stochastic_moe_simulation import (
    StochasticMoeArchitectureSimulator,
)
from plot_pressure_curves import write_pressure_curve_charts


@dataclass(frozen=True)
class ConcurrencyObservation:
    model: str
    node_count: int
    agent_count: int
    tokens_per_second: float
    simulated_duration_ms: float
    completed_token_count: int
    mean_token_latency_ms: float
    p95_token_latency_ms: float
    mean_expert_batch_size: float
    remote_expert_route_fraction: float
    mean_compute_busy_fraction: float
    mean_memory_bandwidth_busy_fraction: float
    maximum_network_link_busy_fraction: float


@dataclass(frozen=True)
class ScalingResult:
    model: str
    node_count: int
    fits_in_memory: bool
    checkpoint_gb: float
    aggregate_usable_memory_gb: float
    memory_capacity_agent_limit_at_assumed_context: int | None
    saturation_agent_count_at_95_percent_of_peak: int | None
    observed_best_tokens_per_second: float | None
    stopping_reason: str
    dominant_busy_resource: str | None


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find saturated decode throughput on 1-7 M5 Ultra Mac Studios"
    )
    parser.add_argument("--output-directory", type=Path, default=Path("results"))
    parser.add_argument("--context-tokens", type=int, default=100_000)
    parser.add_argument("--kv-resident-fraction", type=float, default=0.80)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--plateau-gain", type=float, default=0.05)
    parser.add_argument("--maximum-agents", type=int, default=4096)
    parser.add_argument("--samples-per-decade", type=int, default=6)
    parser.add_argument("--tokens-per-agent", type=int, default=4)
    return parser.parse_args()


def calculate_throughput_after_one_token_per_agent(
    completion_times_ns: list[float], agent_count: int
) -> tuple[float, float]:
    if len(completion_times_ns) <= agent_count:
        return 0.0, 0.0
    ordered_times = sorted(completion_times_ns)
    measurement_start_ns = ordered_times[agent_count - 1]
    measured_times = [
        time_ns
        for time_ns in ordered_times[agent_count:]
        if time_ns > measurement_start_ns
    ]
    if not measured_times or measured_times[-1] <= measurement_start_ns:
        return 0.0, measurement_start_ns
    throughput = len(measured_times) / (
        (measured_times[-1] - measurement_start_ns) / 1e9
    )
    return throughput, measurement_start_ns


def calculate_capacity_agent_limit(
    profile: PublishedMoeModelProfile,
    node_count: int,
    context_tokens: int,
    kv_resident_fraction: float,
    usable_memory_bytes_per_node: int,
) -> int:
    available_runtime_bytes = (
        usable_memory_bytes_per_node * node_count - profile.checkpoint_bytes
    )
    if available_runtime_bytes < 0:
        return 0
    kv_bytes_per_agent = (
        profile.approximate_kv_cache_bytes_per_context_token
        * context_tokens
        * kv_resident_fraction
    )
    return int(available_runtime_bytes // kv_bytes_per_agent)


def simulate_one_concurrency(
    profile: PublishedMoeModelProfile,
    node_count: int,
    agent_count: int,
    seed: int,
    context_tokens: int,
    kv_resident_fraction: float,
    tokens_per_agent: int = 4,
) -> ConcurrencyObservation:
    configuration = profile.create_simulation_configuration(
        node_count=node_count,
        continuously_active_agent_count=agent_count,
        compute_peak_tflops_per_node=100.0,
        memory_bandwidth_gb_per_second_per_node=1_200.0,
        network_link_bandwidth_gb_per_second_each_direction=7.5,
        network_link_fixed_latency_ns=50_000.0,
        random_seed=seed,
        collapse_repeated_layers=True,
        context_token_count=context_tokens,
        kv_resident_fraction=kv_resident_fraction,
    )
    simulator = StochasticMoeArchitectureSimulator(configuration)
    target_completed_tokens = max(128, agent_count * tokens_per_agent)
    while (
        len(simulator.observer.completed_token_times_ns) < target_completed_tokens
        and simulator.event_engine.current_time_ns < 10e9
    ):
        simulator.run_for_simulated_nanoseconds(50e6)

    completed_times = simulator.observer.completed_token_times_ns
    throughput, measurement_start_ns = calculate_throughput_after_one_token_per_agent(
        completed_times, agent_count
    )
    measurement_end_ns = simulator.event_engine.current_time_ns
    occupancy = simulator.calculate_resource_occupancy_between_times(
        measurement_start_time_ns=measurement_start_ns,
        measurement_end_time_ns=measurement_end_ns,
    )
    compute_busy = [
        value for name, value in occupancy.items() if name.startswith("compute[")
    ]
    memory_busy = [
        value for name, value in occupancy.items() if name.startswith("memory[")
    ]
    network_busy = [
        value
        for name, value in occupancy.items()
        if not name.startswith(("compute[", "memory["))
    ]
    summary = simulator.observer.summarize_steady_state_behavior(
        simulation_end_time_ns=measurement_end_ns,
        measurement_window_ns=50e6,
        warmup_time_ns=measurement_start_ns,
    )
    measured_latencies_ns = sorted(
        latency_ns
        for completion_time_ns, latency_ns
        in simulator.observer.completed_token_latency_records
        if completion_time_ns > measurement_start_ns
    )
    p95_latency_index = max(0, math.ceil(len(measured_latencies_ns) * 0.95) - 1)
    return ConcurrencyObservation(
        model=profile.name,
        node_count=node_count,
        agent_count=agent_count,
        tokens_per_second=throughput,
        simulated_duration_ms=measurement_end_ns / 1e6,
        completed_token_count=len(completed_times),
        mean_token_latency_ms=(
            sum(measured_latencies_ns) / len(measured_latencies_ns) / 1e6
            if measured_latencies_ns
            else 0.0
        ),
        p95_token_latency_ms=(
            measured_latencies_ns[p95_latency_index] / 1e6
            if measured_latencies_ns
            else 0.0
        ),
        mean_expert_batch_size=summary["mean_expert_batch_size"],
        remote_expert_route_fraction=summary["remote_expert_route_fraction"],
        mean_compute_busy_fraction=sum(compute_busy) / len(compute_busy),
        mean_memory_bandwidth_busy_fraction=sum(memory_busy) / len(memory_busy),
        maximum_network_link_busy_fraction=max(network_busy, default=0.0),
    )


def scan_until_plateau(
    profile: PublishedMoeModelProfile,
    node_count: int,
    capacity_agent_limit: int,
    arguments: argparse.Namespace,
) -> tuple[list[ConcurrencyObservation], str]:
    observations: list[ConcurrencyObservation] = []
    practical_agent_limit = min(capacity_agent_limit, arguments.maximum_agents)
    agent_counts = create_logarithmic_agent_count_grid(
        practical_agent_limit,
        arguments.samples_per_decade,
    )

    for agent_count in agent_counts:
        observation = simulate_one_concurrency(
            profile,
            node_count,
            agent_count,
            arguments.seed,
            arguments.context_tokens,
            arguments.kv_resident_fraction,
            arguments.tokens_per_agent,
        )
        observations.append(observation)
        print(
            f"{profile.name}: {node_count} nodes, {agent_count} agents -> "
            f"{observation.tokens_per_second:.1f} tok/s",
            flush=True,
        )
    prior_peak = max(
        item.tokens_per_second for item in observations[:-1]
    ) if len(observations) > 1 else 0.0
    final_point_reveals_more_than_five_percent_unrealized_throughput = (
        observations[-1].tokens_per_second
        > prior_peak * (1.0 + arguments.plateau_gain)
    )
    if (
        practical_agent_limit == capacity_agent_limit
        and final_point_reveals_more_than_five_percent_unrealized_throughput
    ):
        stopping_reason = "memory_capacity_before_throughput_plateau"
    elif practical_agent_limit < capacity_agent_limit:
        stopping_reason = "runaway_guard_before_plateau"
    else:
        stopping_reason = "throughput_plateau"
    return observations, stopping_reason


def create_logarithmic_agent_count_grid(
    maximum_agent_count: int,
    samples_per_decade: int,
) -> list[int]:
    if maximum_agent_count < 1:
        return []
    if samples_per_decade < 1:
        raise ValueError("samples_per_decade must be positive")

    counts = set(range(1, min(maximum_agent_count, 8) + 1))
    exponent_count = math.ceil(
        math.log10(maximum_agent_count) * samples_per_decade
    )
    for exponent_index in range(exponent_count + 1):
        count = round(10 ** (exponent_index / samples_per_decade))
        if 1 <= count <= maximum_agent_count:
            counts.add(count)
    counts.add(maximum_agent_count)
    return sorted(counts)


def dominant_busy_resource(observation: ConcurrencyObservation) -> str:
    resource_by_busy_fraction = {
        "gpu_compute": observation.mean_compute_busy_fraction,
        "memory_bandwidth": observation.mean_memory_bandwidth_busy_fraction,
        "thunderbolt_link": observation.maximum_network_link_busy_fraction,
    }
    return max(resource_by_busy_fraction, key=resource_by_busy_fraction.get)


def run_scaling_sweep(
    arguments: argparse.Namespace,
) -> tuple[list[ScalingResult], list[ConcurrencyObservation]]:
    usable_memory_bytes_per_node = int(512 * BYTES_PER_GB * 0.90)
    scaling_results: list[ScalingResult] = []
    concurrency_observations: list[ConcurrencyObservation] = []
    for profile in PUBLISHED_MODEL_PROFILES:
        for node_count in range(1, 8):
            capacity_agent_limit = calculate_capacity_agent_limit(
                profile,
                node_count,
                arguments.context_tokens,
                arguments.kv_resident_fraction,
                usable_memory_bytes_per_node,
            )
            fits = capacity_agent_limit >= 1
            observations, stopping_reason = (
                scan_until_plateau(
                    profile, node_count, capacity_agent_limit, arguments
                )
                if fits
                else ([], "weights_do_not_fit")
            )
            concurrency_observations.extend(observations)
            peak = (
                max(observations, key=lambda item: item.tokens_per_second)
                if observations
                else None
            )
            saturated = None
            if peak is not None:
                saturated = next(
                    observation
                    for observation in observations
                    if observation.tokens_per_second
                    >= peak.tokens_per_second * (1.0 - arguments.plateau_gain)
                )
            scaling_results.append(
                ScalingResult(
                    model=profile.name,
                    node_count=node_count,
                    fits_in_memory=fits,
                    checkpoint_gb=profile.checkpoint_bytes / BYTES_PER_GB,
                    aggregate_usable_memory_gb=(
                        usable_memory_bytes_per_node * node_count / BYTES_PER_GB
                    ),
                    memory_capacity_agent_limit_at_assumed_context=(
                        capacity_agent_limit if fits else None
                    ),
                    saturation_agent_count_at_95_percent_of_peak=(
                        saturated.agent_count if saturated else None
                    ),
                    observed_best_tokens_per_second=(
                        peak.tokens_per_second if peak else None
                    ),
                    stopping_reason=stopping_reason,
                    dominant_busy_resource=(
                        dominant_busy_resource(peak) if peak else None
                    ),
                )
            )
    return scaling_results, concurrency_observations


def write_csv(path: Path, rows: list[object]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=asdict(rows[0]).keys())
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def write_results(
    output_directory: Path,
    scaling_results: list[ScalingResult],
    concurrency_observations: list[ConcurrencyObservation],
    arguments: argparse.Namespace,
) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    write_csv(
        output_directory / "mac_studio_m5_ultra_scaling.csv", scaling_results
    )
    write_csv(
        output_directory / "mac_studio_m5_ultra_concurrency_scan.csv",
        concurrency_observations,
    )
    write_pressure_curve_charts(output_directory, concurrency_observations)
    metadata = {
        "hardware": {
            "node": "Mac Studio M5 Ultra, 80-core GPU, 512 GB unified memory",
            "memory_bandwidth_gb_per_second_per_node": 1200.0,
            "usable_memory_fraction": 0.90,
            "assumed_effective_compute_tflops_per_node": 100.0,
        },
        "network": {
            "transport": "RDMA over Thunderbolt 5",
            "topology": (
                "direct full mesh; six ports per node support at most seven "
                "fully connected nodes"
            ),
            "physical_bandwidth_gbps_each_direction": 80.0,
            "modeled_effective_bandwidth_gbps_each_direction": 60.0,
            "modeled_fixed_latency_microseconds_per_original_layer": 50.0,
        },
        "capacity": {
            "context_tokens_per_agent": arguments.context_tokens,
            "kv_resident_fraction": arguments.kv_resident_fraction,
            "kv_cache_bytes_per_context_token_are_architecture_estimates": True,
        },
        "saturation_search": {
            "agent_counts": (
                "logarithmic pressure sweep plus exact low-concurrency and memory-ceiling "
                "points; saturation is the first point reaching 95 percent of the observed peak"
            ),
            "samples_per_decade": arguments.samples_per_decade,
            "completed_tokens_per_agent_target": arguments.tokens_per_agent,
            "plateau_gain": arguments.plateau_gain,
            "maximum_agents_is_runaway_guard_not_claimed_limit": (
                arguments.maximum_agents
            ),
        },
        "model_profiles": [asdict(profile) for profile in PUBLISHED_MODEL_PROFILES],
        "limitations": [
            "This is a mechanism simulation, not an M5 Ultra benchmark.",
            "Repeated transformer layers are collapsed into one equivalent-work macro layer for the saturation sweep.",
            "Apple does not publish the effective model-kernel TFLOPS used here.",
            "Checkpoint bytes are measured from the official Hugging Face shards.",
            "Ten percent of unified memory is reserved for runtime and non-KV headroom.",
            "KV cache estimates omit architecture-specific fixed recurrent state.",
            "The non-resident KV fraction is treated as compressed or omitted; miss recovery is not modeled.",
            "Sparse auxiliary or n-gram stores count for capacity but not dense FLOPs.",
        ],
    }
    (output_directory / "mac_studio_m5_ultra_scaling_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


def main() -> None:
    arguments = parse_arguments()
    scaling_results, concurrency_observations = run_scaling_sweep(arguments)
    write_results(
        arguments.output_directory,
        scaling_results,
        concurrency_observations,
        arguments,
    )


if __name__ == "__main__":
    main()
