from __future__ import annotations

from math import log2

from .discrete_event_engine import DiscreteEventSimulationEngine, ScheduledSimulationEvent
from .observations import SimulationObserver
from .particle_flow import DependencyJoinTracker, LogicalWorkUnit, WorkParticle
from .resource_utilization import (
    calculate_compute_and_memory_idle_reason_fractions_between_times,
)
from .runtime_policies import choose_rendezvous_node_for_selected_expert_nodes
from .simulation_configuration import StochasticMoeSimulationConfiguration
from .simulation_runtime_setup import (
    create_compute_and_memory_nodes,
    create_expert_batching_gates,
    create_expert_routers_by_layer_index,
    create_network_topology,
    create_shared_layer_batching_gates,
    place_each_layer_experts_across_compute_nodes,
)


class StochasticMoeArchitectureSimulator:
    def __init__(self, configuration: StochasticMoeSimulationConfiguration) -> None:
        configuration.validate()
        self.configuration = configuration

        self.event_engine = DiscreteEventSimulationEngine()
        self.observer = SimulationObserver()
        self.dependency_join_tracker = DependencyJoinTracker()
        self.next_globally_unique_token_id = 0
        self.next_workload_token_ordinal_by_agent = [
            0 for _ in range(configuration.continuously_active_agent_count)
        ]

        self.compute_and_memory_node_by_id = create_compute_and_memory_nodes(configuration)
        self.network_topology = create_network_topology(configuration)
        self.expert_router_by_layer_index = create_expert_routers_by_layer_index(configuration)
        self.node_id_by_expert_index_by_layer_index = (
            place_each_layer_experts_across_compute_nodes(configuration)
        )
        self.shared_layer_batching_gate_by_layer_and_node = (
            create_shared_layer_batching_gates(configuration)
        )
        self.expert_batching_gate_by_layer_and_expert = (
            create_expert_batching_gates(configuration)
        )

        self.next_valid_shared_layer_flush_time_by_layer_and_node: dict[
            tuple[int, int], float
        ] = {}
        self.next_valid_expert_flush_time_by_layer_and_expert: dict[
            tuple[int, int], float
        ] = {}

        self._register_simulation_event_handlers()
        self._schedule_initial_token_for_each_continuously_active_agent()

    def _register_simulation_event_handlers(self) -> None:
        event_handler_by_type = {
            "shared_layer_work_arrived": self._enqueue_shared_layer_work_and_schedule_batch_flush,
            "flush_shared_layer_batch": self._execute_next_shared_layer_batch_if_flush_is_valid,
            "shared_layer_batch_finished": self._continue_after_shared_layer_batch_finishes,
            "route_tokens_to_experts": self._sample_experts_and_transfer_branches_to_expert_nodes,
            "expert_branch_arrived": self._enqueue_expert_branches_and_schedule_batch_flush,
            "flush_expert_batch": self._execute_next_expert_batch_if_flush_is_valid,
            "expert_batch_finished": self._transfer_finished_expert_branches_to_rendezvous_nodes,
            "expert_branch_arrived_at_rendezvous": self._join_completed_expert_branches_and_release_finished_tokens,
            "joined_tokens_arrived_at_sequence_owner": self._continue_joined_tokens_from_sequence_owner,
            "token_finished_all_layers": self._record_completed_tokens_and_start_next_token_for_each_agent,
        }
        for event_type, handler in event_handler_by_type.items():
            self.event_engine.register_event_handler(event_type, handler)

    def _schedule_initial_token_for_each_continuously_active_agent(self) -> None:
        for workload_agent_id in range(self.configuration.continuously_active_agent_count):
            self._schedule_new_token_for_agent(
                workload_agent_id=workload_agent_id,
                node_that_owns_sequence_state=workload_agent_id % self.configuration.node_count,
                ready_time_ns=0.0,
            )

    def run_for_simulated_nanoseconds(
        self,
        simulation_duration_ns: float,
        maximum_event_count: int | None = None,
    ) -> SimulationObserver:
        if simulation_duration_ns < 0:
            raise ValueError("simulation_duration_ns must be non-negative")
        self.event_engine.run_until_simulated_time(
            end_time_ns=self.event_engine.current_time_ns + simulation_duration_ns,
            maximum_event_count=maximum_event_count,
        )
        return self.observer

    def calculate_resource_occupancy_between_times(
        self,
        measurement_start_time_ns: float,
        measurement_end_time_ns: float,
    ) -> dict[str, float]:
        occupancy_by_resource: dict[str, float] = {}
        for node_id, node in enumerate(self.compute_and_memory_node_by_id):
            occupancy_by_resource[f"compute[{node_id}]"] = (
                node.compute_resource.calculate_reserved_fraction_between_times(
                    measurement_start_time_ns, measurement_end_time_ns
                )
            )
            occupancy_by_resource[f"memory[{node_id}]"] = (
                node.memory_resource.calculate_reserved_fraction_between_times(
                    measurement_start_time_ns, measurement_end_time_ns
                )
            )
        occupancy_by_resource.update(
            self.network_topology.calculate_directional_link_busy_fractions_between_times(
                measurement_start_time_ns, measurement_end_time_ns
            )
        )
        return occupancy_by_resource

    def calculate_resource_occupancy_over_simulation_duration(
        self,
        simulation_duration_ns: float,
    ) -> dict[str, float]:
        return self.calculate_resource_occupancy_between_times(
            measurement_start_time_ns=0.0,
            measurement_end_time_ns=simulation_duration_ns,
        )

    def calculate_compute_and_memory_idle_reasons_between_times(
        self,
        measurement_start_time_ns: float,
        measurement_end_time_ns: float,
    ) -> dict[str, dict[str, float]]:
        return {
            f"node[{node_id}]": calculate_compute_and_memory_idle_reason_fractions_between_times(
                node=node,
                measurement_start_time_ns=measurement_start_time_ns,
                measurement_end_time_ns=measurement_end_time_ns,
            )
            for node_id, node in enumerate(self.compute_and_memory_node_by_id)
        }

    def _schedule_new_token_for_agent(
        self,
        workload_agent_id: int,
        node_that_owns_sequence_state: int,
        ready_time_ns: float,
    ) -> None:
        token_id = self.next_globally_unique_token_id
        self.next_globally_unique_token_id += 1
        token_ordinal = self.next_workload_token_ordinal_by_agent[workload_agent_id]
        self.next_workload_token_ordinal_by_agent[workload_agent_id] += 1
        self.observer.record_token_started(token_id, ready_time_ns)
        self.event_engine.schedule_event(
            scheduled_time_ns=ready_time_ns,
            event_type="shared_layer_work_arrived",
            work_particle=WorkParticle(
                logical_work_units=(
                LogicalWorkUnit(
                    globally_unique_token_id=token_id,
                    workload_agent_id=workload_agent_id,
                    workload_token_ordinal=token_ordinal,
                        node_that_owns_sequence_state=node_that_owns_sequence_state,
                    ),
                ),
                operation_name="shared_layer_work_arrived",
                earliest_ready_time_ns=ready_time_ns,
                current_node_id=node_that_owns_sequence_state,
                model_layer_index=0,
            ),
        )

    def _enqueue_shared_layer_work_and_schedule_batch_flush(
        self,
        event_engine: DiscreteEventSimulationEngine,
        event: ScheduledSimulationEvent,
    ) -> None:
        particle = event.work_particle
        if particle.current_node_id is None:
            raise ValueError("shared layer work requires a current node")
        for unit in particle.logical_work_units:
            if unit.node_that_owns_sequence_state != particle.current_node_id:
                raise ValueError("shared layer work must execute where sequence state is owned")

        gate_key = (particle.model_layer_index, particle.current_node_id)
        gate = self.shared_layer_batching_gate_by_layer_and_node[gate_key]
        gate.enqueue_particle_and_batch_when_allowed(particle)

        if gate_key not in self.next_valid_shared_layer_flush_time_by_layer_and_node:
            self._schedule_earliest_shared_layer_flush(
                event_engine=event_engine,
                gate_key=gate_key,
                desired_flush_time_ns=(
                    event.scheduled_time_ns
                    + self.configuration.shared_layer_batching_window_ns
                ),
            )

        if gate.queued_logical_unit_count >= self.configuration.maximum_shared_layer_batch_size:
            self._schedule_earliest_shared_layer_flush(
                event_engine=event_engine,
                gate_key=gate_key,
                desired_flush_time_ns=event.scheduled_time_ns,
            )

    def _schedule_earliest_shared_layer_flush(
        self,
        event_engine: DiscreteEventSimulationEngine,
        gate_key: tuple[int, int],
        desired_flush_time_ns: float,
    ) -> None:
        layer_index, node_id = gate_key
        actual_flush_time_ns = max(
            desired_flush_time_ns,
            self._next_time_compute_and_memory_node_can_start_kernel(node_id),
        )
        current_flush_time_ns = self.next_valid_shared_layer_flush_time_by_layer_and_node.get(
            gate_key
        )
        if current_flush_time_ns is not None and current_flush_time_ns <= actual_flush_time_ns:
            return

        self.next_valid_shared_layer_flush_time_by_layer_and_node[gate_key] = (
            actual_flush_time_ns
        )
        event_engine.schedule_event(
            scheduled_time_ns=actual_flush_time_ns,
            event_type="flush_shared_layer_batch",
            work_particle=WorkParticle(
                logical_work_units=(),
                operation_name="flush_shared_layer_batch",
                earliest_ready_time_ns=actual_flush_time_ns,
                current_node_id=node_id,
                model_layer_index=layer_index,
            ),
            key=gate_key,
        )

    def _execute_next_shared_layer_batch_if_flush_is_valid(
        self,
        event_engine: DiscreteEventSimulationEngine,
        event: ScheduledSimulationEvent,
    ) -> None:
        gate_key = event.event_data["key"]
        if (
            self.next_valid_shared_layer_flush_time_by_layer_and_node.get(gate_key)
            != event.scheduled_time_ns
        ):
            return

        node_id = gate_key[1]
        node_available_time_ns = self._next_time_compute_and_memory_node_can_start_kernel(
            node_id
        )
        if node_available_time_ns > event.scheduled_time_ns:
            del self.next_valid_shared_layer_flush_time_by_layer_and_node[gate_key]
            self._schedule_earliest_shared_layer_flush(
                event_engine=event_engine,
                gate_key=gate_key,
                desired_flush_time_ns=node_available_time_ns,
            )
            return

        del self.next_valid_shared_layer_flush_time_by_layer_and_node[gate_key]
        gate = self.shared_layer_batching_gate_by_layer_and_node[gate_key]
        batch_particle = gate.take_next_ready_particle_that_fits_gate_capacity(
            current_time_ns=event.scheduled_time_ns
        )
        if batch_particle is None or batch_particle.current_node_id is None:
            return

        bytes_accessed = (
            self.configuration.shared_layer_weight_bytes_read_per_batch
            + self.configuration.shared_layer_activation_bytes_per_token
            * batch_particle.logical_unit_count
        )
        _, completion_time_ns, _ = self.compute_and_memory_node_by_id[
            batch_particle.current_node_id
        ].execute_kernel_with_compute_and_memory_overlap(
            ready_time_ns=event.scheduled_time_ns,
            floating_point_operations=(
                self.configuration.shared_layer_flops_per_token
                * batch_particle.logical_unit_count
            ),
            bytes_accessed=bytes_accessed,
        )

        event_engine.schedule_event(
            scheduled_time_ns=completion_time_ns,
            event_type="shared_layer_batch_finished",
            work_particle=WorkParticle(
                logical_work_units=batch_particle.logical_work_units,
                operation_name="shared_layer_batch_finished",
                earliest_ready_time_ns=completion_time_ns,
                current_node_id=batch_particle.current_node_id,
                model_layer_index=batch_particle.model_layer_index,
            ),
        )

        if gate.waiting_particles:
            self._schedule_earliest_shared_layer_flush(
                event_engine=event_engine,
                gate_key=gate_key,
                desired_flush_time_ns=completion_time_ns,
            )

    def _continue_after_shared_layer_batch_finishes(
        self,
        event_engine: DiscreteEventSimulationEngine,
        event: ScheduledSimulationEvent,
    ) -> None:
        particle = event.work_particle
        layer_configuration = self.configuration.model_layers[particle.model_layer_index]

        if layer_configuration.uses_routed_experts:
            event_engine.schedule_event(
                scheduled_time_ns=event.scheduled_time_ns,
                event_type="route_tokens_to_experts",
                work_particle=WorkParticle(
                    logical_work_units=particle.logical_work_units,
                    operation_name="route_tokens_to_experts",
                    earliest_ready_time_ns=event.scheduled_time_ns,
                    current_node_id=particle.current_node_id,
                    model_layer_index=particle.model_layer_index,
                ),
            )
            return

        self._schedule_next_layer_or_token_completion(
            event_engine=event_engine,
            particle=particle,
            ready_time_ns=event.scheduled_time_ns,
        )

    def _sample_experts_and_transfer_branches_to_expert_nodes(
        self,
        event_engine: DiscreteEventSimulationEngine,
        event: ScheduledSimulationEvent,
    ) -> None:
        particle = event.work_particle
        if particle.current_node_id is None:
            raise ValueError("expert routing requires a current node")

        layer_index = particle.model_layer_index
        expert_router = self.expert_router_by_layer_index[layer_index]
        expert_node_ids = self.node_id_by_expert_index_by_layer_index[layer_index]
        expert_branches_grouped_by_destination: dict[
            tuple[int, int], list[LogicalWorkUnit]
        ] = {}

        for unit in particle.logical_work_units:
            if unit.node_that_owns_sequence_state != particle.current_node_id:
                raise ValueError("expert routing must start where sequence state is owned")

            selected_expert_indexes = (
                expert_router.sample_distinct_expert_indexes_for_logical_step(
                    workload_agent_id=unit.workload_agent_id,
                    workload_token_ordinal=unit.workload_token_ordinal,
                    model_layer_index=layer_index,
                    base_random_seed=self.configuration.random_seed,
                )
                if self.configuration.routing_randomness == "keyed_logical_work"
                else expert_router.sample_distinct_expert_indexes_for_one_token()
            )
            if self.configuration.routing_randomness == "keyed_logical_work":
                self.observer.record_selected_expert_indexes(
                    workload_agent_id=unit.workload_agent_id,
                    workload_token_ordinal=unit.workload_token_ordinal,
                    model_layer_index=layer_index,
                    selected_expert_indexes=selected_expert_indexes,
                )
            selected_expert_node_ids = tuple(
                expert_node_ids[expert_index] for expert_index in selected_expert_indexes
            )
            rendezvous_node_id = choose_rendezvous_node_for_selected_expert_nodes(
                selected_expert_node_ids=selected_expert_node_ids,
                sequence_owner_node_id=unit.node_that_owns_sequence_state,
                rendezvous_policy=self.configuration.expert_result_rendezvous_policy,
            )
            dependency_join_id = (
                f"layer_{layer_index}_token_{unit.globally_unique_token_id}"
            )
            self.dependency_join_tracker.register_expected_branch_count(
                dependency_join_id=dependency_join_id,
                branch_count=len(selected_expert_indexes),
            )

            for branch_index, expert_index in enumerate(selected_expert_indexes):
                expert_node_id = expert_node_ids[expert_index]
                expert_branches_grouped_by_destination.setdefault(
                    (expert_index, expert_node_id), []
                ).append(
                    LogicalWorkUnit(
                        globally_unique_token_id=unit.globally_unique_token_id,
                        workload_agent_id=unit.workload_agent_id,
                        workload_token_ordinal=unit.workload_token_ordinal,
                        dependency_join_id=dependency_join_id,
                        branch_index_inside_dependency_join=branch_index,
                        node_that_owns_sequence_state=unit.node_that_owns_sequence_state,
                        expert_result_rendezvous_node_id=rendezvous_node_id,
                    )
                )

        for (expert_index, expert_node_id), units in (
            expert_branches_grouped_by_destination.items()
        ):
            is_remote_route = particle.current_node_id != expert_node_id
            self.observer.record_expert_route(
                routing_time_ns=event.scheduled_time_ns,
                is_remote_route=is_remote_route,
                branch_count=len(units),
            )

            arrival_time_ns = event.scheduled_time_ns
            if is_remote_route:
                _, arrival_time_ns = self.network_topology.transfer_bytes_between_compute_nodes(
                    particle.current_node_id,
                    expert_node_id,
                    event.scheduled_time_ns,
                    self.configuration.activation_bytes_transferred_per_expert_branch
                    * len(units),
                )

            event_engine.schedule_event(
                scheduled_time_ns=arrival_time_ns,
                event_type="expert_branch_arrived",
                work_particle=WorkParticle(
                    logical_work_units=tuple(units),
                    operation_name="expert_branch_arrived",
                    earliest_ready_time_ns=arrival_time_ns,
                    current_node_id=expert_node_id,
                    route_name=str(expert_index),
                    model_layer_index=layer_index,
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
        if particle.current_node_id is None:
            raise ValueError("expert arrival requires a current node")

        expert_index = int(particle.route_name)
        gate_key = (particle.model_layer_index, expert_index)
        expected_expert_node_id = self.node_id_by_expert_index_by_layer_index[
            particle.model_layer_index
        ][expert_index]
        if particle.current_node_id != expected_expert_node_id:
            raise ValueError("expert branch arrived at the wrong expert node")

        gate = self.expert_batching_gate_by_layer_and_expert[gate_key]
        gate.enqueue_particle_and_batch_when_allowed(particle)

        if gate_key not in self.next_valid_expert_flush_time_by_layer_and_expert:
            self._schedule_earliest_expert_flush(
                event_engine=event_engine,
                gate_key=gate_key,
                desired_flush_time_ns=(
                    event.scheduled_time_ns + self.configuration.expert_batching_window_ns
                ),
            )

        if gate.queued_logical_unit_count >= self.configuration.maximum_expert_batch_size:
            self._schedule_earliest_expert_flush(
                event_engine=event_engine,
                gate_key=gate_key,
                desired_flush_time_ns=event.scheduled_time_ns,
            )

    def _schedule_earliest_expert_flush(
        self,
        event_engine: DiscreteEventSimulationEngine,
        gate_key: tuple[int, int],
        desired_flush_time_ns: float,
    ) -> None:
        layer_index, expert_index = gate_key
        node_id = self.node_id_by_expert_index_by_layer_index[layer_index][expert_index]
        actual_flush_time_ns = max(
            desired_flush_time_ns,
            self._next_time_compute_and_memory_node_can_start_kernel(node_id),
        )
        current_flush_time_ns = self.next_valid_expert_flush_time_by_layer_and_expert.get(
            gate_key
        )
        if current_flush_time_ns is not None and current_flush_time_ns <= actual_flush_time_ns:
            return

        self.next_valid_expert_flush_time_by_layer_and_expert[gate_key] = (
            actual_flush_time_ns
        )
        event_engine.schedule_event(
            scheduled_time_ns=actual_flush_time_ns,
            event_type="flush_expert_batch",
            work_particle=WorkParticle(
                logical_work_units=(),
                operation_name="flush_expert_batch",
                earliest_ready_time_ns=actual_flush_time_ns,
                current_node_id=node_id,
                route_name=str(expert_index),
                model_layer_index=layer_index,
            ),
            key=gate_key,
        )

    def _execute_next_expert_batch_if_flush_is_valid(
        self,
        event_engine: DiscreteEventSimulationEngine,
        event: ScheduledSimulationEvent,
    ) -> None:
        gate_key = event.event_data["key"]
        if (
            self.next_valid_expert_flush_time_by_layer_and_expert.get(gate_key)
            != event.scheduled_time_ns
        ):
            return

        layer_index, expert_index = gate_key
        node_id = self.node_id_by_expert_index_by_layer_index[layer_index][expert_index]
        node_available_time_ns = self._next_time_compute_and_memory_node_can_start_kernel(
            node_id
        )
        if node_available_time_ns > event.scheduled_time_ns:
            del self.next_valid_expert_flush_time_by_layer_and_expert[gate_key]
            self._schedule_earliest_expert_flush(
                event_engine=event_engine,
                gate_key=gate_key,
                desired_flush_time_ns=node_available_time_ns,
            )
            return

        del self.next_valid_expert_flush_time_by_layer_and_expert[gate_key]
        gate = self.expert_batching_gate_by_layer_and_expert[gate_key]
        queue_depth_before_execution = gate.queued_logical_unit_count
        batch_particle = gate.take_next_ready_particle_that_fits_gate_capacity(
            current_time_ns=event.scheduled_time_ns
        )
        if batch_particle is None or batch_particle.current_node_id is None:
            return

        self.observer.record_expert_batch_execution(
            execution_time_ns=event.scheduled_time_ns,
            executed_batch_size=batch_particle.logical_unit_count,
            queue_depth_before_execution=queue_depth_before_execution,
        )
        _, completion_time_ns, _ = self.compute_and_memory_node_by_id[
            batch_particle.current_node_id
        ].execute_kernel_with_compute_and_memory_overlap(
            ready_time_ns=event.scheduled_time_ns,
            floating_point_operations=(
                self.configuration.expert_flops_per_token
                * batch_particle.logical_unit_count
            ),
            bytes_accessed=self.configuration.expert_weight_bytes_read_per_batch,
            compute_efficiency_multiplier=(
                self._estimate_expert_kernel_efficiency_from_local_expert_batch_size(
                    batch_particle.logical_unit_count
                )
            ),
        )

        event_engine.schedule_event(
            scheduled_time_ns=completion_time_ns,
            event_type="expert_batch_finished",
            work_particle=WorkParticle(
                logical_work_units=batch_particle.logical_work_units,
                operation_name="expert_batch_finished",
                earliest_ready_time_ns=completion_time_ns,
                current_node_id=batch_particle.current_node_id,
                route_name=batch_particle.route_name,
                model_layer_index=batch_particle.model_layer_index,
                resource_width_per_unit=batch_particle.resource_width_per_unit,
            ),
        )

        if gate.waiting_particles:
            self._schedule_earliest_expert_flush(
                event_engine=event_engine,
                gate_key=gate_key,
                desired_flush_time_ns=completion_time_ns,
            )

    def _transfer_finished_expert_branches_to_rendezvous_nodes(
        self,
        event_engine: DiscreteEventSimulationEngine,
        event: ScheduledSimulationEvent,
    ) -> None:
        particle = event.work_particle
        if particle.current_node_id is None:
            raise ValueError("finished expert batch requires a current node")

        work_units_by_rendezvous_node: dict[int, list[LogicalWorkUnit]] = {}
        for unit in particle.logical_work_units:
            if unit.expert_result_rendezvous_node_id is None:
                raise ValueError(
                    "finished expert branch requires an expert result rendezvous node"
                )
            work_units_by_rendezvous_node.setdefault(
                unit.expert_result_rendezvous_node_id, []
            ).append(unit)

        for rendezvous_node_id, units in work_units_by_rendezvous_node.items():
            arrival_time_ns = event.scheduled_time_ns
            if rendezvous_node_id != particle.current_node_id:
                _, arrival_time_ns = self.network_topology.transfer_bytes_between_compute_nodes(
                    particle.current_node_id,
                    rendezvous_node_id,
                    event.scheduled_time_ns,
                    self.configuration.activation_bytes_transferred_per_expert_branch
                    * len(units),
                )

            event_engine.schedule_event(
                scheduled_time_ns=arrival_time_ns,
                event_type="expert_branch_arrived_at_rendezvous",
                work_particle=WorkParticle(
                    logical_work_units=tuple(units),
                    operation_name="expert_branch_arrived_at_rendezvous",
                    earliest_ready_time_ns=arrival_time_ns,
                    current_node_id=rendezvous_node_id,
                    model_layer_index=particle.model_layer_index,
                ),
            )

    def _join_completed_expert_branches_and_release_finished_tokens(
        self,
        event_engine: DiscreteEventSimulationEngine,
        event: ScheduledSimulationEvent,
    ) -> None:
        released_particles = (
            self.dependency_join_tracker
            .accept_completed_branches_and_release_tokens_after_all_selected_experts_finish(
                particle=event.work_particle,
                next_operation_name="joined_expert_result_ready",
                ready_time_ns=event.scheduled_time_ns,
            )
        )
        for released_particle in released_particles:
            self._transfer_joined_tokens_to_sequence_owner_nodes(
                event_engine=event_engine,
                particle=released_particle,
                ready_time_ns=event.scheduled_time_ns,
            )

    def _transfer_joined_tokens_to_sequence_owner_nodes(
        self,
        event_engine: DiscreteEventSimulationEngine,
        particle: WorkParticle,
        ready_time_ns: float,
    ) -> None:
        if particle.current_node_id is None:
            raise ValueError("joined expert result requires a rendezvous node")
        if not particle.logical_work_units:
            return

        owner_node_ids = {
            unit.node_that_owns_sequence_state for unit in particle.logical_work_units
        }
        if None in owner_node_ids or len(owner_node_ids) != 1:
            raise ValueError("joined token particle must contain exactly one sequence owner")
        sequence_owner_node_id = next(iter(owner_node_ids))
        if sequence_owner_node_id is None:
            raise ValueError("sequence owner node is missing")

        arrival_time_ns = ready_time_ns
        if sequence_owner_node_id != particle.current_node_id:
            _, arrival_time_ns = self.network_topology.transfer_bytes_between_compute_nodes(
                particle.current_node_id,
                sequence_owner_node_id,
                ready_time_ns,
                self.configuration.activation_bytes_transferred_per_expert_branch
                * particle.logical_unit_count,
            )

        event_engine.schedule_event(
            scheduled_time_ns=arrival_time_ns,
            event_type="joined_tokens_arrived_at_sequence_owner",
            work_particle=WorkParticle(
                logical_work_units=particle.logical_work_units,
                operation_name="joined_tokens_arrived_at_sequence_owner",
                earliest_ready_time_ns=arrival_time_ns,
                current_node_id=sequence_owner_node_id,
                model_layer_index=particle.model_layer_index,
            ),
        )

    def _continue_joined_tokens_from_sequence_owner(
        self,
        event_engine: DiscreteEventSimulationEngine,
        event: ScheduledSimulationEvent,
    ) -> None:
        self._schedule_next_layer_or_token_completion(
            event_engine=event_engine,
            particle=event.work_particle,
            ready_time_ns=event.scheduled_time_ns,
        )

    def _schedule_next_layer_or_token_completion(
        self,
        event_engine: DiscreteEventSimulationEngine,
        particle: WorkParticle,
        ready_time_ns: float,
    ) -> None:
        next_layer_index = particle.model_layer_index + 1
        if next_layer_index < self.configuration.model_layer_count:
            event_engine.schedule_event(
                scheduled_time_ns=ready_time_ns,
                event_type="shared_layer_work_arrived",
                work_particle=WorkParticle(
                    logical_work_units=particle.logical_work_units,
                    operation_name="shared_layer_work_arrived",
                    earliest_ready_time_ns=ready_time_ns,
                    current_node_id=particle.current_node_id,
                    model_layer_index=next_layer_index,
                ),
            )
            return

        event_engine.schedule_event(
            scheduled_time_ns=ready_time_ns,
            event_type="token_finished_all_layers",
            work_particle=WorkParticle(
                logical_work_units=particle.logical_work_units,
                operation_name="token_finished_all_layers",
                earliest_ready_time_ns=ready_time_ns,
                current_node_id=particle.current_node_id,
                model_layer_index=particle.model_layer_index,
            ),
        )

    def _record_completed_tokens_and_start_next_token_for_each_agent(
        self,
        event_engine: DiscreteEventSimulationEngine,
        event: ScheduledSimulationEvent,
    ) -> None:
        for unit in event.work_particle.logical_work_units:
            if unit.node_that_owns_sequence_state is None:
                raise ValueError("completed token requires a sequence owner")
            self.observer.record_completed_token(
                unit.globally_unique_token_id,
                event.scheduled_time_ns,
            )
            if (
                self.configuration.maximum_completed_tokens_per_agent is not None
                and unit.workload_token_ordinal + 1
                >= self.configuration.maximum_completed_tokens_per_agent
            ):
                continue
            self._schedule_new_token_for_agent(
                workload_agent_id=unit.workload_agent_id,
                node_that_owns_sequence_state=unit.node_that_owns_sequence_state,
                ready_time_ns=event.scheduled_time_ns,
            )

    def _next_time_compute_and_memory_node_can_start_kernel(self, node_id: int) -> float:
        node = self.compute_and_memory_node_by_id[node_id]
        return max(
            node.compute_resource.next_free_time_ns,
            node.memory_resource.next_free_time_ns,
        )

    @staticmethod
    def _estimate_expert_kernel_efficiency_from_local_expert_batch_size(
        local_expert_batch_size: int,
    ) -> float:
        efficiency_points = (
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
            return efficiency_points[0][1]

        for (batch_size_0, efficiency_0), (batch_size_1, efficiency_1) in zip(
            efficiency_points, efficiency_points[1:]
        ):
            if local_expert_batch_size <= batch_size_1:
                interpolation_fraction = (
                    log2(local_expert_batch_size) - log2(batch_size_0)
                ) / (log2(batch_size_1) - log2(batch_size_0))
                return efficiency_0 + interpolation_fraction * (
                    efficiency_1 - efficiency_0
                )

        return 0.84
