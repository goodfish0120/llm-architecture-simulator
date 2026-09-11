from __future__ import annotations

import argparse

from .stochastic_moe_simulation import (
    StochasticMoeArchitectureSimulator,
    StochasticMoeSimulationConfiguration,
)


def parse_optional_comma_separated_layer_indexes(value: str | None) -> tuple[int, ...] | None:
    if value is None:
        return None
    if not value.strip():
        return tuple()
    return tuple(int(part.strip()) for part in value.split(","))


def main() -> None:
    argument_parser = argparse.ArgumentParser(
        description="Run a stochastic LLM/MoE architecture simulation"
    )
    argument_parser.add_argument("--ms", type=float, default=500.0)
    argument_parser.add_argument("--window-ms", type=float, default=10.0)
    argument_parser.add_argument("--warmup-ms", type=float, default=20.0)
    argument_parser.add_argument("--nodes", type=int, default=4)
    argument_parser.add_argument("--agents", type=int, default=128)
    argument_parser.add_argument("--layers", type=int, default=4)
    argument_parser.add_argument(
        "--moe-layers",
        type=str,
        default=None,
        help="comma-separated layer indexes; omitted means every layer is MoE",
    )
    argument_parser.add_argument("--experts", type=int, default=16)
    argument_parser.add_argument("--top-k", type=int, default=2)
    argument_parser.add_argument("--batch-window-us", type=float, default=1000.0)
    argument_parser.add_argument(
        "--network-topology",
        choices=("switch_star", "full_mesh", "daisy_chain", "ring"),
        default="switch_star",
    )
    argument_parser.add_argument("--seed", type=int, default=7)
    arguments = argument_parser.parse_args()

    configuration = StochasticMoeSimulationConfiguration(
        node_count=arguments.nodes,
        continuously_active_agent_count=arguments.agents,
        model_layer_count=arguments.layers,
        moe_layer_indexes=parse_optional_comma_separated_layer_indexes(arguments.moe_layers),
        routed_expert_count_per_moe_layer=arguments.experts,
        selected_expert_count_per_token=arguments.top_k,
        expert_batching_window_ns=arguments.batch_window_us * 1e3,
        network_topology_kind=arguments.network_topology,
        random_seed=arguments.seed,
    )
    simulator = StochasticMoeArchitectureSimulator(configuration)
    simulation_duration_ns = arguments.ms * 1e6
    observer = simulator.run_for_simulated_nanoseconds(simulation_duration_ns)

    print("window,tokens_per_second")
    for window_index, tokens_per_second in enumerate(
        observer.calculate_tokens_per_second_for_fixed_time_windows(
            simulation_duration_ns,
            arguments.window_ms * 1e6,
            arguments.warmup_ms * 1e6,
        )
    ):
        print(f"{window_index},{tokens_per_second:.0f}")

    print("\nsummary")
    for metric_name, metric_value in observer.summarize_steady_state_behavior(
        simulation_duration_ns,
        arguments.window_ms * 1e6,
        arguments.warmup_ms * 1e6,
    ).items():
        print(f"{metric_name}: {metric_value}")

    print("\nresource_occupancy")
    for resource_name, occupancy_fraction in (
        simulator.calculate_resource_occupancy_over_simulation_duration(
            simulation_duration_ns
        ).items()
    ):
        print(f"{resource_name}: {occupancy_fraction:.1%}")


if __name__ == "__main__":
    main()
