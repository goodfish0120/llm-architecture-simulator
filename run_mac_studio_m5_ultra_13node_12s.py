"""13-node, six-port symmetric topology: 12 s model time, 2 s warmup."""
from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path

from src.llm_architecture_simulator.network_topology import (
    BidirectionalNetworkLink,
    RoutedNetworkTopology,
)
from src.llm_architecture_simulator.published_model_profiles import (
    BYTES_PER_GB,
    PUBLISHED_MODEL_PROFILES,
)
from src.llm_architecture_simulator.stochastic_moe_simulation import StochasticMoeArchitectureSimulator

NODE_COUNT = 13
MEMORY_PER_NODE_GB = 96
USABLE_FRACTION = 0.90
CONTEXT_TOKENS = 100000
KV_RESIDENT_FRACTION = 0.80
NEIGHBOR_OFFSETS = frozenset((1, 3, 4, 9, 10, 12))
VARIANTS = {
    "baseline": (80.0, 1200.0, 7.5),
    "memory_x2": (80.0, 2400.0, 7.5),
    "memory_half": (80.0, 600.0, 7.5),
    "network_x2": (80.0, 1200.0, 15.0),
    "compute_x2": (160.0, 1200.0, 7.5),
}

def build_topology(link_gbps: float, latency_ns: float) -> RoutedNetworkTopology:
    links = []
    for i in range(NODE_COUNT):
        for j in range(i + 1, NODE_COUNT):
            if (j - i) in NEIGHBOR_OFFSETS:
                links.append(BidirectionalNetworkLink(
                    link_name=f"m{i+1:02d}_m{j+1:02d}",
                    endpoint_a=RoutedNetworkTopology.compute_node_endpoint(i),
                    endpoint_b=RoutedNetworkTopology.compute_node_endpoint(j),
                    bytes_per_ns_each_direction=link_gbps,
                    fixed_latency_ns=latency_ns,
                ))
    topo = RoutedNetworkTopology(NODE_COUNT, links)
    topo.validate_compute_node_port_limits({i: 6 for i in range(NODE_COUNT)})
    degrees = [len(topo.links_by_endpoint[topo.compute_node_endpoint(i)]) for i in range(NODE_COUNT)]
    assert len(links) == 39 and degrees == [6] * NODE_COUNT, (len(links), degrees)
    hop_counts = Counter()
    for i in range(NODE_COUNT):
        for j in range(i + 1, NODE_COUNT):
            route = topo._find_and_cache_shortest_hop_route(
                topo.compute_node_endpoint(i), topo.compute_node_endpoint(j))
            assert 1 <= len(route) <= 2, (i, j, len(route))
            hop_counts[len(route)] += 1
    assert hop_counts == {1: 39, 2: 39}, hop_counts
    return topo

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("qwen", "deepseek"), required=True)
    parser.add_argument("--agents", type=int, required=True)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--seconds", type=int, default=12)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--output-dir", default="results/13_node_96gb")
    args = parser.parse_args()
    assert args.seconds - args.warmup >= 10, "Need >=10 measured simulated seconds"
    name = "Qwen3.8 Flash Next" if args.model == "qwen" else "DeepSeek V4.1 Flash"
    profile = next(p for p in PUBLISHED_MODEL_PROFILES if p.name == name)
    available = NODE_COUNT * MEMORY_PER_NODE_GB * BYTES_PER_GB * USABLE_FRACTION
    remaining = available - profile.checkpoint_bytes
    kv_per_agent = profile.approximate_kv_cache_bytes_per_context_token * CONTEXT_TOKENS * KV_RESIDENT_FRACTION
    agent_limit = math.floor(remaining / kv_per_agent) if remaining > 0 else 0
    assert 1 <= args.agents <= agent_limit, (name, args.agents, agent_limit)
    compute, memory, network = VARIANTS[args.variant]

    cfg = profile.create_simulation_configuration(
        node_count=NODE_COUNT,
        continuously_active_agent_count=args.agents,
        compute_peak_tflops_per_node=compute,
        memory_bandwidth_gb_per_second_per_node=memory,
        network_link_bandwidth_gb_per_second_each_direction=network,
        network_link_fixed_latency_ns=50000,
        random_seed=7,
        collapse_repeated_layers=True,
        context_token_count=CONTEXT_TOKENS,
        kv_resident_fraction=KV_RESIDENT_FRACTION,
    )
    cfg.routing_randomness = "keyed_logical_work"
    simulator = StochasticMoeArchitectureSimulator(cfg)
    simulator.network_topology = build_topology(network, cfg.network_link_fixed_latency_ns)
    wall_start = time.perf_counter()
    print("CONFIG", json.dumps({
        "model": name, "agents": args.agents, "variant": args.variant,
        "model_seconds": args.seconds, "warmup_seconds": args.warmup,
        "max_agents_100k": agent_limit, "checkpoint_gb": profile.checkpoint_bytes / BYTES_PER_GB,
        "usable_aggregate_gb": available / BYTES_PER_GB,
        "topology_edges": 39, "hop_1_pairs": 39, "hop_2_pairs": 39,
        "compute_tflops_assumption": compute, "memory_gb_s": memory,
        "network_gb_s_each_direction": network,
        "collapse_repeated_layers": True,
        "routing_randomness": cfg.routing_randomness,
    }), flush=True)
    per_second = []
    for end_second in range(1, args.seconds + 1):
        simulator.run_for_simulated_nanoseconds(1e9)
        completions = simulator.observer.completed_token_times_ns
        count = sum((end_second - 1) * 1e9 < t <= end_second * 1e9 for t in completions)
        per_second.append(count)
        print(f"SECOND model={args.model} variant={args.variant} t={end_second}s completed={count} wall={time.perf_counter()-wall_start:.1f}s", flush=True)

    measurement_begin_ns = args.warmup * 1e9
    measurement_end_ns = args.seconds * 1e9
    completions = simulator.observer.completed_token_times_ns
    measured_count = sum(measurement_begin_ns < t <= measurement_end_ns for t in completions)
    throughput = measured_count / (args.seconds - args.warmup)
    latencies_ms = sorted(latency / 1e6 for at, latency in simulator.observer.completed_token_latency_records
                          if measurement_begin_ns < at <= measurement_end_ns)
    p95 = latencies_ms[min(len(latencies_ms)-1, math.ceil(.95*len(latencies_ms))-1)] if latencies_ms else None
    occupancy = simulator.calculate_resource_occupancy_between_times(
        measurement_begin_ns, measurement_end_ns)
    compute_busy = [v for k, v in occupancy.items() if k.startswith("compute[")]
    memory_busy = [v for k, v in occupancy.items() if k.startswith("memory[")]
    net_busy = [v for k, v in occupancy.items() if not k.startswith(("compute[", "memory["))]
    productive_compute = [n.compute_resource.calculate_productive_fraction_between_times(
        measurement_begin_ns, measurement_end_ns)
        for n in simulator.compute_and_memory_node_by_id]
    productive_memory = [n.memory_resource.calculate_productive_fraction_between_times(
        measurement_begin_ns, measurement_end_ns)
        for n in simulator.compute_and_memory_node_by_id]
    def mean(values):
        return sum(values) / len(values) if values else 0
    result = {
        "model": name, "agents": args.agents, "variant": args.variant,
        "simulation_seconds": args.seconds,
        "measurement_seconds": args.seconds - args.warmup,
        "completed_tokens_after_warmup": measured_count,
        "aggregate_tokens_per_second": round(throughput, 3),
        "mean_per_agent_tokens_per_second": round(throughput / args.agents, 3),
        "p95_token_latency_ms": round(p95, 3) if p95 is not None else None,
        "per_second_completed_tokens": per_second,
        "last_5s_vs_first_5s_throughput_ratio": round(
            sum(per_second[-5:]) / max(1, sum(per_second[args.warmup:args.warmup+5])), 4),
        "avg_compute_reserved_fraction": round(mean(compute_busy), 4),
        "avg_memory_reserved_fraction": round(mean(memory_busy), 4),
        "avg_compute_productive_fraction": round(mean(productive_compute), 4),
        "avg_memory_productive_fraction": round(mean(productive_memory), 4),
        "avg_network_direction_busy_fraction": round(mean(net_busy), 4),
        "peak_network_direction_busy_fraction": round(max(net_busy, default=0), 4),
        "wall_clock_seconds": round(time.perf_counter()-wall_start, 2),
        "memory_capacity_agent_limit": agent_limit,
        "topology": "13-node Paley degree-6, 39 links, diameter 2, deterministic BFS shortest hop",
        "limitations": ["Synthetic uncalibrated hardware timings", "Collapsed transformer layers", "No relay-node host/RDMA forwarding costs", "Single shortest route per endpoint pair", "KV miss recovery not modeled"],
    }
    print("FINAL_RESULT", json.dumps(result), flush=True)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{args.model}_{args.agents}_{args.variant}.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("RESULT_FILE", str(path), flush=True)

if __name__ == "__main__":
    main()
