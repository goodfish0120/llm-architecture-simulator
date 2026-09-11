from __future__ import annotations

from dataclasses import dataclass
from math import log2

from .discrete_event_engine import (
    DiscreteEventSimulationEngine,
    ParticleBatchingGate,
    ScheduledSimulationEvent,
)
from .expert_routing import (
    WeightedTopKExpertRouter,
    generate_normalized_gaussian_expert_popularity,
)
from .hardware_resources import (
    ComputeAndMemoryNode,
    SerializedThroughputResource,
    SharedNetworkFabric,
)
from .observations import SimulationObserver
from .particle_flow import (
    DependencyJoinTracker,
    LogicalWorkUnit,
    ParticleBatchingRule,
    WorkParticle,
)


@dataclass
class StochasticMoeSimulationConfiguration:
    node_count: int = 4
    continuously_active_agent_count: int = 128
    moe_layer_count: int = 4
    routed_expert_count_per_layer: int = 16
    selected_expert_count_per_token: int = 2
    random_seed: int = 7

    expert_batching_window_ns: float = 1_000_000.0
    maximum_expert_batch_size: int = 128

    compute_peak_tflops_per_node: float = 100.0
    memory_bandwidth_gb_per_second_per_node: float = 500.0
    network_bandwidth_gb_per_second_per_node: float = 20.0
    shared_switch_bandwidth_gb_per_second: float = 70.0
    fixed_network_latency_ns: float = 4_000.0

    non_expert_flops_per_token_per_layer: float = 10e6
    non_expert_bytes_per_token_per_layer: float = 0.8e6
    expert_flops_per_token: float = 30e6
    expert_weight_bytes_read_per_batch: float = 4e6
    activation_bytes_transferred_per_expert_branch: float = 16_384.0
    expert_popularity_sigma: float = 0.8


class StochasticMoeArchitectureSimulator:
    def __init__(self, configuration: StochasticMoeSimulationConfiguration) -> None:
        self.configuration = configuration
        self.event_engine = DiscreteEventSimulationEngine()
        self.observer = SimulationObserver()
        self.dependency_join_tracker = DependencyJoinTracker()
        self.next_globally_unique_token_id = 0

        self.compute_and_memory_node_by_id = [
            ComputeAndMemoryNode(
                SerializedThroughputResource(
                    f"compute[{node_id}]",
                    configuration.compute_peak_tflops_per_node * 1e3,
                    3_000.0,
                ),
                SerializedThroughputResource(
                    f"memory[{node_id}]",
                    configuration.memory_bandwidth_gb_per_second_per_node,
                    100.0,
                ),
            )
            for node_id in range(configuration.node_count)
        ]

        self.network_fabric = SharedNetworkFabric(
            configuration.node_count,
            configuration.network_bandwidth_gb_per_second_per_node,
            configuration.shared_switch_bandwidth_gb_per_second,
            configuration.fixed_network_latency_ns,
        )

        self.expert_router = WeightedTopKExpertRouter(
            generate_normalized_gaussian_expert_popularity(
                configuration.routed_expert_count_per_layer,
                configuration.expert_popularity_sigma,
                configuration.random_seed,
            ),
            configuration.selected_expert_count_per_token,
            configuration.random_seed + 1,
        )

        self.node_id_by_expert_index = [
            expert_index % configuration.node_count
            for expert_index in range(configuration.routed_expert_count_per_layer)
        ]

        self.batching_gate_by_layer_and_expert = {
            (layer_index, expert_index): ParticleBatchingGate(
                f"layer_{layer_index}_expert_{expert_index}",
                configuration.maximum_expert_batch_size,
                ParticleBatchingRule.BATCH_IF_READY_WITHIN_TIME_WINDOW,
                configuration.expert_batching_window_ns,
            )
            for layer_index in range(configuration.moe_layer_count)
            for expert_index in range(configuration.routed_expert_count_per_layer)
        }

        self.next_valid_flush_time_by_layer_and_expert: dict[
            tuple[int, int],
            float,
        ] = {}

        self.event_engine.register_event_handler(
            "execute_non_expert_layer_work",
            self._execute_non_expert_layer_work_then_route_tokens_to_experts,
        )
        self.event_engine.register_event_handler(
            "route_tokens_to_experts",
            self._sample_experts_and_transfer_remote_branches_to_expert_nodes,
        )
        self.event_engine.register_event_handler(
            "expert_branch_arrived",
            self._enqueue_expert_branches_and_schedule_batch_flush,
        )
        self.event_engine.register_event_handler(
            "flush_expert_batch",
            self._execute_next_expert_batch_if_this_flush_event_is_still_valid,
        )
        self.event_engine.register_event_handler(
            "expert_batch_finished",
            self._transfer_finished_expert_branches_back_to_sequence_owner_nodes,
        )
        self.event_engine.register_event_handler(
            "expert_branch_returned_to_owner",
            self._release_tokens_whose_selected_expert_branches_have_all_finished,
        )
        self.event_engine.register_event_handler(
            "token_finished_all_layers",
            self._record_completed_tokens_and_immediately_start_next_token_for_each_agent,
        )

        for workload_agent_id in range(configuration.continuously_active_agent_count):
            node_that_owns_sequence_state = workload_agent_id % configuration.node_count
            self._schedule_new_token_for_agent(
                workload_agent_id,
                node_that_owns_sequence_state,
                0.0,
            )

    def _schedule_new_token_for_agent(
        self,
        workload_agent_id: int,
        node_that_owns_sequence_state: int,
        ready_time_ns: float,
    ) -> None:
        token_id = self.next_globally_unique_token_id
        self.next_globally_unique_token_id += 1
        particle = WorkParticle(
            (
                LogicalWorkUnit(
                    globally_unique_token_id=token_id,
                    workload_agent_id=workload_agent_id,
                    node_that_owns_sequence_state=node_that_owns_sequence_state,
                ),
            ),
            "execute_non_expert_layer_work",
            ready_time_ns,
            node_that_owns_sequence_state,
            model_layer_index=0,
        )
        self.event_engine.schedule_event(
            ready_time_ns,
            "execute_non_expert_layer_work",
            particle,
        )

    @staticmethod
    def _estimate_expert_kernel_efficiency_from_local_expert_batch_size(
        local_expert_batch_size: int,
    ) -> float:
        measured_or_estimated_efficiency_points = (
            (1, 0.06),
            (2, 0.10),
            (4, 0.18),
            (8, 0.30),
            (16, 0.45),
            (32, 0.60),
            (64, 0.72),
            (128, 0.80),
        )
        if local_expert_batch_size <= 1:
            return measured_or_estimated_efficiency_points[0][1]

        for (batch_size_0, efficiency_0), (batch_size_1, efficiency_1) in zip(
            measured_or_estimated_efficiency_points,
            measured_or_estimated_efficiency_points[1:],
        ):
            if local_expert_batch_size <= batch_size_1:
                interpolation_fraction = (
                    log2(local_expert_batch_size) - log2(batch_size_0)
                ) / (log2(batch_size_1) - log2(batch_size_0))
                return efficiency_0 + interpolation_fraction * (
                    efficiency_1 - efficiency_0
                )

        return 0.84

    def _execute_non_expert_layer_work_then_route_tokens_to_experts(
        self,
        event_engine: DiscreteEventSimulationEngine,
        event: ScheduledSimulationEvent,
    ) -> None:
        particle = event.work_particle
        if particle.current_node_id is None:
            raise ValueError("layer work requires a current node")

        _, completion_time_ns, _ = self.compute_and_memory_node_by_id[
            particle.current_node_id
        ].execute_kernel_with_compute_and_memory_overlap(
            event.scheduled_time_ns,
            self.configuration.non_expert_flops_per_token_per_layer
            * particle.logical_unit_count,
            self.configuration.non_expert_bytes_per_token_per_layer
            * particle.logical_unit_count,
        )

        event_engine.schedule_event(
            completion_time_ns,
            "route_tokens_to_experts",
            WorkParticle(
                particle.logical_work_units,
                "route_tokens_to_experts",
                completion_time_ns,
                particle.current_node_id,
                model_layer_index=particle.model_layer_index,
            ),
        )

    def _sample_experts_and_transfer_remote_branches_to_expert_nodes(
        self,
        event_engine: DiscreteEventSimulationEngine,
        event: ScheduledSimulationEvent,
    ) -> None:
        particle = event.work_particle
        expert_branches_grouped_by_destination: dict[
            tuple[int, int, int],
            list[LogicalWorkUnit],
        ] = {}

        for unit in particle.logical_work_units:
            selected_expert_indexes = (
                self.expert_router.sample_distinct_expert_indexes_for_one_token()
            )
            dependency_join_id = (
                f"layer_{particle.model_layer_index}_token_{unit.globally_unique_token_id}"
            )
            self.dependency_join_tracker.register_expected_branch_count(
                dependency_join_id,
                len(selected_expert_indexes),
            )

            if unit.node_that_owns_sequence_state is None:
                raise ValueError("expert route requires a sequence owner node")

            for branch_index, expert_index in enumerate(selected_expert_indexes):
                expert_node_id = self.node_id_by_expert_index[expert_index]
                grouping_key = (
                    expert_index,
                    expert_node_id,
                    unit.node_that_owns_sequence_state,
                )
                expert_branches_grouped_by_destination.setdefault(
                    grouping_key,
                    [],
                ).append(
                    LogicalWorkUnit(
                        globally_unique_token_id=unit.globally_unique_token_id,
                        workload_agent_id=unit.workload_agent_id,
                        dependency_join_id=dependency_join_id,
                        branch_index_inside_dependency_join=branch_index,
                        node_that_owns_sequence_state=unit.node_that_owns_sequence_state,
                    )
                )

        for (
            expert_index,
            expert_node_id,
            sequence_owner_node_id,
        ), logical_work_units in expert_branches_grouped_by_destination.items():
            is_remote_route = sequence_owner_node_id != expert_node_id
            self.observer.record_expert_route(
                is_remote_route,
                len(logical_work_units),
            )

            arrival_time_ns = event.scheduled_time_ns
            if is_remote_route:
                _, arrival_time_ns = self.network_fabric.transfer_bytes_between_nodes(
                    sequence_owner_node_id,
                    expert_node_id,
                    event.scheduled_time_ns,
                    self.configuration.activation_bytes_transferred_per_expert_branch
                    * len(logical_work_units),
                )

            event_engine.schedule_event(
                arrival_time_ns,
                "expert_branch_arrived",
                WorkParticle(
                    tuple(logical_work_units),
                    "expert_branch_arrived",
                    arrival_time_ns,
                    expert_node_id,
                    route_name=str(expert_index),
                    model_layer_index=particle.model_layer_index,
                ),
            )

    def _enqueue_expert_branches_and_schedule_batch_flush(
        self,
        event_engine: DiscreteEventSimulationEngine,
        event: ScheduledSimulationEvent,
    ) -> None:
        particle = event.work_particle
        if particle.route_name is None:
            raise ValueError("expert arrival requires an expert route")

        layer_and_expert = (
            particle.model_layer_index,
            int(particle.route_name),
        )
        gate = self.batching_gate_by_layer_and_expert[layer_and_expert]
        gate.enqueue_particle_and_batch_when_allowed(particle)

        if layer_and_expert not in self.next_valid_flush_time_by_layer_and_expert:
            scheduled_flush_time_ns = (
                event.scheduled_time_ns
                + self.configuration.expert_batching_window_ns
            )
            self._replace_valid_flush_time_and_schedule_flush_event(
                event_engine,
                layer_and_expert,
                scheduled_flush_time_ns,
                particle,
            )

        if gate.queued_logical_unit_count >= self.configuration.maximum_expert_batch_size:
            self._replace_valid_flush_time_and_schedule_flush_event(
                event_engine,
                layer_and_expert,
                event.scheduled_time_ns,
                particle,
            )

    def _replace_valid_flush_time_and_schedule_flush_event(
        self,
        event_engine: DiscreteEventSimulationEngine,
        layer_and_expert: tuple[int, int],
        scheduled_flush_time_ns: float,
        representative_particle: WorkParticle,
    ) -> None:
        self.next_valid_flush_time_by_layer_and_expert[
            layer_and_expert
        ] = scheduled_flush_time_ns
        event_engine.schedule_event(
            scheduled_flush_time_ns,
            "flush_expert_batch",
            WorkParticle(
                (),
                "flush_expert_batch",
                scheduled_flush_time_ns,
                representative_particle.current_node_id,
                representative_particle.route_name,
                representative_particle.model_layer_index,
            ),
            layer_and_expert=layer_and_expert,
        )

    def _execute_next_expert_batch_if_this_flush_event_is_still_valid(
        self,
        event_engine: DiscreteEventSimulationEngine,
        event: ScheduledSimulationEvent,
    ) -> None:
        layer_and_expert = event.event_data["layer_and_expert"]
        if (
            self.next_valid_flush_time_by_layer_and_expert.get(layer_and_expert)
            != event.scheduled_time_ns
        ):
            return

        del self.next_valid_flush_time_by_layer_and_expert[layer_and_expert]
        gate = self.batching_gate_by_layer_and_expert[layer_and_expert]
        queue_depth_before_execution = gate.queued_logical_unit_count
        batch_particle = gate.take_next_ready_particle_that_fits_gate_capacity(
            event.scheduled_time_ns
        )
        if batch_particle is None:
            return
        if batch_particle.current_node_id is None:
            raise ValueError("expert batch requires a current node")

        self.observer.record_expert_batch_execution(
            batch_particle.logical_unit_count,
            queue_depth_before_execution,
        )

        expert_kernel_efficiency = (
            self._estimate_expert_kernel_efficiency_from_local_expert_batch_size(
                batch_particle.logical_unit_count
            )
        )
        _, completion_time_ns, _ = self.compute_and_memory_node_by_id[
            batch_particle.current_node_id
        ].execute_kernel_with_compute_and_memory_overlap(
            event.scheduled_time_ns,
            self.configuration.expert_flops_per_token
            * batch_particle.logical_unit_count,
            self.configuration.expert_weight_bytes_read_per_batch,
            expert_kernel_efficiency,
        )

        event_engine.schedule_event(
            completion_time_ns,
            "expert_batch_finished",
            WorkParticle(
                batch_particle.logical_work_units,
                "expert_batch_finished",
                completion_time_ns,
                batch_particle.current_node_id,
                batch_particle.route_name,
                batch_particle.model_layer_index,
                batch_particle.resource_width_per_unit,
            ),
        )

        if gate.waiting_particles:
            self._replace_valid_flush_time_and_schedule_flush_event(
                event_engine,
                layer_and_expert,
                event.scheduled_time_ns,
                batch_particle,
            )

    def _transfer_finished_expert_branches_back_to_sequence_owner_nodes(
        self,
        event_engine: DiscreteEventSimulationEngine,
        event: ScheduledSimulationEvent,
    ) -> None:
        particle = event.work_particle
        if particle.current_node_id is None:
            raise ValueError("finished expert batch requires a current node")

        work_units_by_sequence_owner: dict[int, list[LogicalWorkUnit]] = {}
        for unit in particle.logical_work_units:
            if unit.node_that_owns_sequence_state is None:
                raise ValueError("finished expert branch requires a sequence owner")
            work_units_by_sequence_owner.setdefault(
                unit.node_that_owns_sequence_state,
                [],
            ).append(unit)

        for sequence_owner_node_id, logical_work_units in work_units_by_sequence_owner.items():
            arrival_time_ns = event.scheduled_time_ns
            if sequence_owner_node_id != particle.current_node_id:
                _, arrival_time_ns = self.network_fabric.transfer_bytes_between_nodes(
                    particle.current_node_id,
                    sequence_owner_node_id,
                    event.scheduled_time_ns,
                    self.configuration.activation_bytes_transferred_per_expert_branch
                    * len(logical_work_units),
                )

            event_engine.schedule_event(
                arrival_time_ns,
                "expert_branch_returned_to_owner",
                WorkParticle(
                    tuple(logical_work_units),
                    "expert_branch_returned_to_owner",
                    arrival_time_ns,
                    sequence_owner_node_id,
                    model_layer_index=particle.model_layer_index,
                ),
            )

    def _release_tokens_whose_selected_expert_branches_have_all_finished(
        self,
        event_engine: DiscreteEventSimulationEngine,
        event: ScheduledSimulationEvent,
    ) -> None:
        particle = event.work_particle
        released_particles = (
            self.dependency_join_tracker.accept_completed_branches_and_release_finished_tokens(
                particle,
                "execute_non_expert_layer_work",
                event.scheduled_time_ns,
            )
        )

        for released_particle in released_particles:
            next_layer_index = particle.model_layer_index + 1
            if next_layer_index < self.configuration.moe_layer_count:
                event_engine.schedule_event(
                    event.scheduled_time_ns,
                    "execute_non_expert_layer_work",
                    WorkParticle(
                        released_particle.logical_work_units,
                        "execute_non_expert_layer_work",
                        event.scheduled_time_ns,
                        released_particle.current_node_id,
                        model_layer_index=next_layer_index,
                    ),
                )
            else:
                event_engine.schedule_event(
                    event.scheduled_time_ns,
                    "token_finished_all_layers",
                    WorkParticle(
                        released_particle.logical_work_units,
                        "token_finished_all_layers",
                        event.scheduled_time_ns,
                        released_particle.current_node_id,
                        model_layer_index=particle.model_layer_index,
                    ),
                )

    def _record_completed_tokens_and_immediately_start_next_token_for_each_agent(
        self,
        event_engine: DiscreteEventSimulationEngine,
        event: ScheduledSimulationEvent,
    ) -> None:
        for unit in event.work_particle.logical_work_units:
            if unit.node_that_owns_sequence_state is None:
                raise ValueError("completed token requires a sequence owner")
            self.observer.record_completed_token(event.scheduled_time_ns)
            self._schedule_new_token_for_agent(
                unit.workload_agent_id,
                unit.node_that_owns_sequence_state,
                event.scheduled_time_ns,
            )

    def run_for_simulated_nanoseconds(
        self,
        simulation_duration_ns: float,
        maximum_event_count: int | None = None,
    ) -> SimulationObserver:
        self.event_engine.run_until_simulated_time(
            simulation_duration_ns,
            maximum_event_count,
        )
        return self.observer

    def calculate_resource_occupancy_over_simulation_duration(
        self,
        simulation_duration_ns: float,
    ) -> dict[str, float]:
        if simulation_duration_ns <= 0:
            return {}

        occupancy_by_resource: dict[str, float] = {}
        for node_id, node in enumerate(self.compute_and_memory_node_by_id):
            occupancy_by_resource[f"compute[{node_id}]"] = min(
                1.0,
                node.compute_resource.accumulated_busy_time_ns
                / simulation_duration_ns,
            )
            occupancy_by_resource[f"memory[{node_id}]"] = min(
                1.0,
                node.memory_resource.accumulated_busy_time_ns
                / simulation_duration_ns,
            )
            occupancy_by_resource[f"nic_tx[{node_id}]"] = min(
                1.0,
                self.network_fabric.transmit_resource_by_node[
                    node_id
                ].accumulated_busy_time_ns
                / simulation_duration_ns,
            )
            occupancy_by_resource[f"nic_rx[{node_id}]"] = min(
                1.0,
                self.network_fabric.receive_resource_by_node[
                    node_id
                ].accumulated_busy_time_ns
                / simulation_duration_ns,
            )

        occupancy_by_resource["switch"] = min(
            1.0,
            self.network_fabric.shared_switch_resource.accumulated_busy_time_ns
            / simulation_duration_ns,
        )
        return occupancy_by_resource
