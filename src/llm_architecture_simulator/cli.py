from __future__ import annotations

import argparse

from .simulation_configuration import (
    StochasticMoeSimulationConfiguration,
    TransformerLayerConfiguration,
)
from .stochastic_moe_simulation import StochasticMoeArchitectureSimulator


def parse_optional_comma_separated_layer_indexes(
    value: str | None,
) -> tuple[int, ...] | None:
    if value is None:
        return None
    if not value.strip():
        return tuple()
    return tuple(int(part.strip()) for part in value.split(","))


def parse_command_line_arguments() -> argparse.Namespace:
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
    argument_parser.add_argument(
        "--rendezvous-policy",
        choices=("sequence_owner", "largest_local_expert_group"),
        default="sequence_owner",
    )
    argument_parser.add_argument("--seed", type=int, default=7)
    return argument_parser.parse_args()


def create_model_layers_from_command_line_arguments(
    arguments: argparse.Namespace,
) -> tuple[TransformerLayerConfiguration, ...]:
    requested_moe_layer_indexes = parse_optional_comma_separated_layer_indexes(
        arguments.moe_layers
    )
    moe_layer_indexes = (
        set(range(arguments.layers))
        if requested_moe_layer_indexes is None
        else set(requested_moe_layer_indexes)
    )
    if any(
        layer_index < 0 or layer_index >= arguments.layers
        for layer_index in moe_layer_indexes
    ):
        raise ValueError("--moe-layers contains a layer outside --layers")

    return tuple(
        TransformerLayerConfiguration(
            routed_expert_count=arguments.experts,
            selected_expert_count_per_token=arguments.top_k,
        )
        if layer_index in moe_layer_indexes
        else TransformerLayerConfiguration()
        for layer_index in range(arguments.layers)
    )


def create_simulation_configuration_from_command_line_arguments(
    arguments: argparse.Namespace,
) -> StochasticMoeSimulationConfiguration:
    return StochasticMoeSimulationConfiguration(
        node_count=arguments.nodes,
        continuously_active_agent_count=arguments.agents,
        model_layers=create_model_layers_from_command_line_arguments(arguments),
        expert_batching_window_ns=arguments.batch_window_us * 1e3,
        network_topology_kind=arguments.network_topology,
        expert_result_rendezvous_policy=arguments.rendezvous_policy,
        random_seed=arguments.seed,
    )


def print_throughput_windows(
    simulator: StochasticMoeArchitectureSimulator,
    simulation_duration_ns: float,
    measurement_window_ns: float,
    warmup_time_ns: float,
) -> None:
    print("window,tokens_per_second")
    tokens_per_second_by_window = (
        simulator.observer.calculate_tokens_per_second_for_fixed_time_windows(
            simulation_end_time_ns=simulation_duration_ns,
            measurement_window_ns=measurement_window_ns,
            warmup_time_ns=warmup_time_ns,
        )
    )
    for window_index, tokens_per_second in enumerate(tokens_per_second_by_window):
        print(f"{window_index},{tokens_per_second:.0f}")


def print_steady_state_summary(
    simulator: StochasticMoeArchitectureSimulator,
    simulation_duration_ns: float,
    measurement_window_ns: float,
    warmup_time_ns: float,
) -> None:
    print("\nsummary")
    summary = simulator.observer.summarize_steady_state_behavior(
        simulation_end_time_ns=simulation_duration_ns,
        measurement_window_ns=measurement_window_ns,
        warmup_time_ns=warmup_time_ns,
    )
    for metric_name, metric_value in summary.items():
        print(f"{metric_name}: {metric_value}")


def print_resource_occupancy(
    simulator: StochasticMoeArchitectureSimulator,
    simulation_duration_ns: float,
) -> None:
    print("\nresource_occupancy")
    occupancy_by_resource = (
        simulator.calculate_resource_occupancy_over_simulation_duration(
            simulation_duration_ns=simulation_duration_ns
        )
    )
    for resource_name, occupancy_fraction in occupancy_by_resource.items():
        print(f"{resource_name}: {occupancy_fraction:.1%}")


def print_compute_and_memory_idle_reasons(
    simulator: StochasticMoeArchitectureSimulator,
    simulation_duration_ns: float,
) -> None:
    print("\ncompute_and_memory_idle_reasons")
    idle_reasons_by_node = simulator.calculate_compute_and_memory_idle_reasons_between_times(
        measurement_start_time_ns=0.0,
        measurement_end_time_ns=simulation_duration_ns,
    )
    for node_name, idle_reasons in idle_reasons_by_node.items():
        print(node_name)
        for reason_name, fraction in idle_reasons.items():
            print(f"  {reason_name}: {fraction:.1%}")


def main() -> None:
    arguments = parse_command_line_arguments()
    configuration = create_simulation_configuration_from_command_line_arguments(arguments)
    simulator = StochasticMoeArchitectureSimulator(configuration)

    simulation_duration_ns = arguments.ms * 1e6
    measurement_window_ns = arguments.window_ms * 1e6
    warmup_time_ns = arguments.warmup_ms * 1e6

    simulator.run_for_simulated_nanoseconds(
        simulation_duration_ns=simulation_duration_ns
    )

    print_throughput_windows(
        simulator=simulator,
        simulation_duration_ns=simulation_duration_ns,
        measurement_window_ns=measurement_window_ns,
        warmup_time_ns=warmup_time_ns,
    )
    print_steady_state_summary(
        simulator=simulator,
        simulation_duration_ns=simulation_duration_ns,
        measurement_window_ns=measurement_window_ns,
        warmup_time_ns=warmup_time_ns,
    )
    print_resource_occupancy(
        simulator=simulator,
        simulation_duration_ns=simulation_duration_ns,
    )
    print_compute_and_memory_idle_reasons(
        simulator=simulator,
        simulation_duration_ns=simulation_duration_ns,
    )


if __name__ == "__main__":
    main()
