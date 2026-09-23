"""Run a reproducible, single-scenario hardware counterfactual sweep.

This is a simulator mechanism check, not a benchmark of an M5 Ultra.  Each
variant changes exactly one modeled hardware capability from the same Kimi K3,
seven-node, 681-active-agent scenario.  ``control_x1`` is intentionally
identical to ``baseline`` and provides a deterministic negative control.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

from src.llm_architecture_simulator.published_model_profiles import (
    PUBLISHED_MODEL_PROFILES,
    PublishedMoeModelProfile,
)
from src.llm_architecture_simulator.stochastic_moe_simulation import (
    StochasticMoeArchitectureSimulator,
)
from src.llm_architecture_simulator.simulation_runtime_setup import (
    create_expert_routers_by_layer_index,
)


PROFILE_NAME = "Kimi K3"
NODE_COUNT = 7
AGENT_COUNT = 681
CONTEXT_TOKENS = 100_000
KV_RESIDENT_FRACTION = 0.80
SEED = 7
MAX_SIMULATED_DURATION_NS = 10e9
KEYED_ROUTING = "keyed_logical_work"


@dataclass(frozen=True)
class HardwareVariant:
    name: str
    compute_tflops_per_node: float = 100.0
    memory_gbps_per_node: float = 1_200.0
    network_gbps_each_direction: float = 7.5
    network_latency_ns: float = 50_000.0


VARIANTS = (
    HardwareVariant("baseline"),
    HardwareVariant("control_x1"),
    HardwareVariant("compute_x2", compute_tflops_per_node=200.0),
    HardwareVariant("hbm_bandwidth_x2", memory_gbps_per_node=2_400.0),
    HardwareVariant("network_bandwidth_x2", network_gbps_each_direction=15.0),
    HardwareVariant("network_latency_div2", network_latency_ns=25_000.0),
)


@dataclass(frozen=True)
class CounterfactualResult:
    source_commit: str
    profile: str
    node_count: int
    agent_count: int
    context_tokens: int
    kv_resident_fraction: float
    seed: int
    tokens_per_agent: int
    variant: str
    compute_tflops_per_node: float
    memory_gbps_per_node: float
    network_gbps_each_direction: float
    network_latency_ns: float
    completed_token_count: int
    simulated_duration_ms: float
    tokens_per_second: float
    p95_token_latency_ms: float
    compute_busy_fraction: float
    memory_busy_fraction: float
    max_network_busy_fraction: float
    compute_idle_reasons: str
    memory_idle_reasons: str
    delta_throughput_vs_baseline: float
    delta_p95_vs_baseline: float
    run_status: str


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep one Kimi K3 / 7-node / 681-agent simulator scenario"
    )
    parser.add_argument("--output-directory", type=Path, default=Path("results"))
    parser.add_argument(
        "--tokens-per-agent",
        type=int,
        default=10,
        help="Completed tokens targeted per active agent (default: 10).",
    )
    return parser.parse_args()


def source_commit() -> str:
    """Return the checked-out source commit without depending on Git at import time."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, encoding="utf-8"
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def kimi_k3_profile() -> PublishedMoeModelProfile:
    return next(profile for profile in PUBLISHED_MODEL_PROFILES if profile.name == PROFILE_NAME)


def calculate_throughput_after_one_token_per_agent(
    completion_times_ns: list[float], agent_count: int
) -> tuple[float, float]:
    """Use the existing scaling sweep's warmup boundary and throughput definition."""
    if len(completion_times_ns) <= agent_count:
        return 0.0, 0.0
    ordered_times = sorted(completion_times_ns)
    measurement_start_ns = ordered_times[agent_count - 1]
    measured_times = [time for time in ordered_times[agent_count:] if time > measurement_start_ns]
    if not measured_times or measured_times[-1] <= measurement_start_ns:
        return 0.0, measurement_start_ns
    return len(measured_times) / ((measured_times[-1] - measurement_start_ns) / 1e9), measurement_start_ns


def average_idle_reasons(
    by_node: dict[str, dict[str, float]], prefix: str
) -> str:
    """Serialize a node-average reason breakdown in stable CSV-safe JSON."""
    reason_names = sorted(
        reason for reasons in by_node.values() for reason in reasons if reason.startswith(prefix)
    )
    averages = {
        reason: sum(reasons.get(reason, 0.0) for reasons in by_node.values()) / len(by_node)
        for reason in reason_names
    } if by_node else {}
    return json.dumps(averages, sort_keys=True, separators=(",", ":"))


def run_variant(
    profile: PublishedMoeModelProfile,
    variant: HardwareVariant,
    tokens_per_agent: int,
    commit: str,
    routing_randomness: str = KEYED_ROUTING,
) -> CounterfactualResult:
    if tokens_per_agent < 1:
        raise ValueError("tokens_per_agent must be positive")
    configuration = profile.create_simulation_configuration(
        node_count=NODE_COUNT,
        continuously_active_agent_count=AGENT_COUNT,
        compute_peak_tflops_per_node=variant.compute_tflops_per_node,
        memory_bandwidth_gb_per_second_per_node=variant.memory_gbps_per_node,
        network_link_bandwidth_gb_per_second_each_direction=variant.network_gbps_each_direction,
        network_link_fixed_latency_ns=variant.network_latency_ns,
        random_seed=SEED,
        collapse_repeated_layers=True,
        context_token_count=CONTEXT_TOKENS,
        kv_resident_fraction=KV_RESIDENT_FRACTION,
    )
    configuration.routing_randomness = routing_randomness
    simulator = StochasticMoeArchitectureSimulator(configuration)
    target_completed_tokens = max(128, AGENT_COUNT * tokens_per_agent)
    while (
        len(simulator.observer.completed_token_times_ns) < target_completed_tokens
        and simulator.event_engine.current_time_ns < MAX_SIMULATED_DURATION_NS
    ):
        simulator.run_for_simulated_nanoseconds(50e6)

    completion_times = simulator.observer.completed_token_times_ns
    throughput, measurement_start_ns = calculate_throughput_after_one_token_per_agent(
        completion_times, AGENT_COUNT
    )
    measurement_end_ns = simulator.event_engine.current_time_ns
    occupancy = simulator.calculate_resource_occupancy_between_times(
        measurement_start_ns, measurement_end_ns
    )
    compute_busy = [value for name, value in occupancy.items() if name.startswith("compute[")]
    memory_busy = [value for name, value in occupancy.items() if name.startswith("memory[")]
    network_busy = [
        value for name, value in occupancy.items()
        if not name.startswith(("compute[", "memory["))
    ]
    latencies_ns = sorted(
        latency for completion_time, latency in simulator.observer.completed_token_latency_records
        if completion_time > measurement_start_ns
    )
    p95_index = max(0, math.ceil(len(latencies_ns) * 0.95) - 1)
    idle_reasons = simulator.calculate_compute_and_memory_idle_reasons_between_times(
        measurement_start_ns, measurement_end_ns
    )
    return CounterfactualResult(
        source_commit=commit,
        profile=profile.name,
        node_count=NODE_COUNT,
        agent_count=AGENT_COUNT,
        context_tokens=CONTEXT_TOKENS,
        kv_resident_fraction=KV_RESIDENT_FRACTION,
        seed=SEED,
        tokens_per_agent=tokens_per_agent,
        variant=variant.name,
        compute_tflops_per_node=variant.compute_tflops_per_node,
        memory_gbps_per_node=variant.memory_gbps_per_node,
        network_gbps_each_direction=variant.network_gbps_each_direction,
        network_latency_ns=variant.network_latency_ns,
        completed_token_count=len(completion_times),
        simulated_duration_ms=measurement_end_ns / 1e6,
        tokens_per_second=throughput,
        p95_token_latency_ms=latencies_ns[p95_index] / 1e6 if latencies_ns else 0.0,
        compute_busy_fraction=sum(compute_busy) / len(compute_busy),
        memory_busy_fraction=sum(memory_busy) / len(memory_busy),
        max_network_busy_fraction=max(network_busy, default=0.0),
        compute_idle_reasons=average_idle_reasons(idle_reasons, "compute_"),
        memory_idle_reasons=average_idle_reasons(idle_reasons, "memory_"),
        delta_throughput_vs_baseline=0.0,
        delta_p95_vs_baseline=0.0,
        run_status=(
            "completed_target"
            if len(completion_times) >= target_completed_tokens
            else "simulated_time_limit_reached"
        ),
    )


def run_sweep(
    tokens_per_agent: int,
    commit: str | None = None,
    routing_randomness: str = KEYED_ROUTING,
) -> list[CounterfactualResult]:
    profile = kimi_k3_profile()
    resolved_commit = source_commit() if commit is None else commit
    raw_results = [
        run_variant(
            profile, variant, tokens_per_agent, resolved_commit, routing_randomness
        )
        for variant in VARIANTS
    ]
    baseline = raw_results[0]
    return [
        CounterfactualResult(
            **{
                **asdict(result),
                "delta_throughput_vs_baseline": result.tokens_per_second - baseline.tokens_per_second,
                "delta_p95_vs_baseline": result.p95_token_latency_ms - baseline.p95_token_latency_ms,
            }
        )
        for result in raw_results
    ]


def calculate_route_prefix_digest(
    profile: PublishedMoeModelProfile,
    variant: HardwareVariant,
    routing_randomness: str,
    agent_prefix: int = 8,
    token_prefix: int = 4,
) -> str:
    """Digest a canonical logical route prefix without consuming a mutable stream."""
    configuration = profile.create_simulation_configuration(
        node_count=NODE_COUNT,
        continuously_active_agent_count=AGENT_COUNT,
        compute_peak_tflops_per_node=variant.compute_tflops_per_node,
        memory_bandwidth_gb_per_second_per_node=variant.memory_gbps_per_node,
        network_link_bandwidth_gb_per_second_each_direction=variant.network_gbps_each_direction,
        network_link_fixed_latency_ns=variant.network_latency_ns,
        random_seed=SEED,
        collapse_repeated_layers=True,
        context_token_count=CONTEXT_TOKENS,
        kv_resident_fraction=KV_RESIDENT_FRACTION,
    )
    configuration.routing_randomness = routing_randomness
    if routing_randomness != KEYED_ROUTING:
        raise ValueError("route digest requires keyed logical routing")
    routers = create_expert_routers_by_layer_index(configuration)
    digest = hashlib.sha256()
    for agent_id in range(agent_prefix):
        for token_ordinal in range(token_prefix):
            for layer_index in sorted(routers):
                route = routers[layer_index].sample_distinct_expert_indexes_for_logical_step(
                    agent_id, token_ordinal, layer_index, SEED
                )
                digest.update(f"{agent_id}:{token_ordinal}:{layer_index}:{route}\n".encode())
    return digest.hexdigest()


def write_results(
    output_directory: Path,
    results: list[CounterfactualResult],
    routing_randomness: str = KEYED_ROUTING,
) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    csv_path = output_directory / "mac_studio_m5_ultra_counterfactual.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=asdict(results[0]).keys())
        writer.writeheader()
        writer.writerows(asdict(result) for result in results)
    profile = kimi_k3_profile()
    route_digests = {
        variant.name: calculate_route_prefix_digest(profile, variant, routing_randomness)
        for variant in VARIANTS
    }
    if len(set(route_digests.values())) != 1:
        raise RuntimeError("counterfactual route-prefix identity check failed")
    metadata = {
        "purpose": "single-scenario simulator counterfactual bottleneck sweep",
        "scenario": {
            "profile": PROFILE_NAME,
            "node_count": NODE_COUNT,
            "agent_count": AGENT_COUNT,
            "context_tokens": CONTEXT_TOKENS,
            "kv_resident_fraction": KV_RESIDENT_FRACTION,
            "seed": SEED,
            "tokens_per_agent": results[0].tokens_per_agent,
            "warmup": "exclude the first completed token per agent",
            "stop_rule": "target completed tokens or 10 simulated seconds",
        },
        "variants": [asdict(variant) for variant in VARIANTS],
        "negative_control": "control_x1 is identical to baseline and should reproduce it exactly.",
        "implementation_commit": results[0].source_commit,
        "workload_control": {
            "routing_randomness": routing_randomness,
            "logical_identity": "seed, workload_agent_id, workload_token_ordinal, model_layer_index",
            "route_prefix_digest": next(iter(route_digests.values())),
            "route_identity_check": "passed",
        },
        "historical_reference_only": {
            "681_agent_scan_throughput_tok_s": 1430.9811369323036,
            "681_agent_scan_p95_ms": 539.6001001859472,
            "global_scan_best_throughput_tok_s_not_this_baseline": 1488.6109678717396,
        },
        "limitations": [
            "Synthetic mechanism simulation, not a hardware benchmark.",
            "A busy fraction is an occupancy signal, not a causal bottleneck finding.",
            "This output does not validate simulator accuracy against real hardware.",
            "Route-workload comparability across hardware variants is unverified and exploratory until a design verdict and auditable check establish an identical route trace; hardware changes can alter event ordering and the seeded router's sampled workload.",
        ],
    }
    (output_directory / "mac_studio_m5_ultra_counterfactual_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


def main() -> None:
    arguments = parse_arguments()
    results = run_sweep(arguments.tokens_per_agent, routing_randomness=KEYED_ROUTING)
    write_results(arguments.output_directory, results, routing_randomness=KEYED_ROUTING)
    for result in results:
        print(f"{result.variant}: {result.tokens_per_second:.3f} tok/s ({result.run_status})")


if __name__ == "__main__":
    main()
