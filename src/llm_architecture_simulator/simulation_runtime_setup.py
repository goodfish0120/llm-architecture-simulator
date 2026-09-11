from __future__ import annotations

"""Build concrete runtime structures before the token event flow begins.

The simulation configuration describes the system declaratively. The simulator's
main event flow assumes compute resources, topology, routers, placements, and batch
gates already exist, so this module resolves that setup boundary and keeps
construction plumbing out of token-lifecycle code.
"""

from .discrete_event_engine import ParticleBatchingGate
from .expert_routing import (
    WeightedTopKExpertRouter,
    generate_normalized_gaussian_expert_popularity,
)
from .hardware_resources import ComputeAndMemoryNode, SerializedThroughputResource
from .network_topology import NetworkTopologyFactory, RoutedNetworkTopology
from .particle_flow import ParticleBatchingRule
from .simulation_configuration import StochasticMoeSimulationConfiguration


def create_compute_and_memory_nodes(
    configuration: StochasticMoeSimulationConfiguration,
) -> list[ComputeAndMemoryNode]:
    return [
        ComputeAndMemoryNode(
            compute_resource=SerializedThroughputResource(
                resource_name=f"compute[{node_id}]",
                work_units_per_ns=configuration.compute_peak_tflops_per_node * 1e3,
                startup_latency_ns=3_000.0,
            ),
            memory_resource=SerializedThroughputResource(
                resource_name=f"memory[{node_id}]",
                work_units_per_ns=configuration.memory_bandwidth_gb_per_second_per_node,
                startup_latency_ns=100.0,
            ),
        )
        for node_id in range(configuration.node_count)
    ]


def create_network_topology(
    configuration: StochasticMoeSimulationConfiguration,
) -> RoutedNetworkTopology:
    topology_factory_by_kind = {
        "full_mesh": NetworkTopologyFactory.create_direct_full_mesh_between_compute_nodes,
        "daisy_chain": NetworkTopologyFactory.create_bidirectional_daisy_chain_between_compute_nodes,
        "ring": NetworkTopologyFactory.create_bidirectional_ring_between_compute_nodes,
        "switch_star": NetworkTopologyFactory.create_nonblocking_central_switch_star,
    }
    topology_factory = topology_factory_by_kind[configuration.network_topology_kind]
    return topology_factory(
        configuration.node_count,
        configuration.network_link_bandwidth_gb_per_second_each_direction,
        configuration.network_link_fixed_latency_ns,
    )


def create_expert_routers_by_layer_index(
    configuration: StochasticMoeSimulationConfiguration,
) -> dict[int, WeightedTopKExpertRouter]:
    routers: dict[int, WeightedTopKExpertRouter] = {}
    for layer_index, layer_configuration in enumerate(configuration.model_layers):
        if not layer_configuration.uses_routed_experts:
            continue
        routers[layer_index] = WeightedTopKExpertRouter(
            relative_expert_popularity=generate_normalized_gaussian_expert_popularity(
                expert_count=layer_configuration.routed_expert_count,
                popularity_sigma=configuration.expert_popularity_sigma,
                random_seed=configuration.random_seed + layer_index * 2,
            ),
            selected_expert_count_per_token=(
                layer_configuration.selected_expert_count_per_token
            ),
            random_seed=configuration.random_seed + layer_index * 2 + 1,
        )
    return routers


def place_each_layer_experts_across_compute_nodes(
    configuration: StochasticMoeSimulationConfiguration,
) -> dict[int, tuple[int, ...]]:
    return {
        layer_index: tuple(
            expert_index % configuration.node_count
            for expert_index in range(layer_configuration.routed_expert_count)
        )
        for layer_index, layer_configuration in enumerate(configuration.model_layers)
        if layer_configuration.uses_routed_experts
    }


def create_shared_layer_batching_gates(
    configuration: StochasticMoeSimulationConfiguration,
) -> dict[tuple[int, int], ParticleBatchingGate]:
    return {
        (layer_index, node_id): ParticleBatchingGate(
            gate_name=f"layer_{layer_index}_node_{node_id}_shared_work",
            maximum_resource_width_per_service=(
                configuration.maximum_shared_layer_batch_size
            ),
            batching_rule=ParticleBatchingRule.BATCH_IF_READY_WITHIN_TIME_WINDOW,
            batching_window_ns=configuration.shared_layer_batching_window_ns,
        )
        for layer_index in range(configuration.model_layer_count)
        for node_id in range(configuration.node_count)
    }


def create_expert_batching_gates(
    configuration: StochasticMoeSimulationConfiguration,
) -> dict[tuple[int, int], ParticleBatchingGate]:
    gates: dict[tuple[int, int], ParticleBatchingGate] = {}
    for layer_index, layer_configuration in enumerate(configuration.model_layers):
        if not layer_configuration.uses_routed_experts:
            continue
        for expert_index in range(layer_configuration.routed_expert_count):
            gates[(layer_index, expert_index)] = ParticleBatchingGate(
                gate_name=f"layer_{layer_index}_expert_{expert_index}",
                maximum_resource_width_per_service=(
                    configuration.maximum_expert_batch_size
                ),
                batching_rule=ParticleBatchingRule.BATCH_IF_READY_WITHIN_TIME_WINDOW,
                batching_window_ns=configuration.expert_batching_window_ns,
            )
    return gates
