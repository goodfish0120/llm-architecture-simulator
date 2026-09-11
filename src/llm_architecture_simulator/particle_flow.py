from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ParticleBatchingRule(str, Enum):
    NEVER_BATCH = "never"
    BATCH_ONLY_IF_READY_AT_EXACTLY_THE_SAME_TIME = "exact"
    BATCH_IF_READY_WITHIN_TIME_WINDOW = "window"


@dataclass(frozen=True)
class LogicalWorkUnit:
    globally_unique_token_id: int
    workload_agent_id: int
    dependency_join_id: str | None = None
    branch_index_inside_dependency_join: int | None = None
    node_that_owns_sequence_state: int | None = None
    expert_result_rendezvous_node_id: int | None = None


@dataclass(frozen=True)
class WorkParticle:
    logical_work_units: tuple[LogicalWorkUnit, ...]
    operation_name: str
    earliest_ready_time_ns: float = 0.0
    current_node_id: int | None = None
    route_name: str | None = None
    model_layer_index: int = 0
    resource_width_per_unit: float = 1.0

    @property
    def logical_unit_count(self) -> int:
        return len(self.logical_work_units)

    @property
    def total_resource_width(self) -> float:
        return self.logical_unit_count * self.resource_width_per_unit

    @property
    def batching_identity(self) -> tuple:
        return (
            self.operation_name,
            self.current_node_id,
            self.route_name,
            self.model_layer_index,
            self.resource_width_per_unit,
        )

    def can_batch_with(
        self,
        other: "WorkParticle",
        batching_rule: ParticleBatchingRule,
        batching_window_ns: float,
    ) -> bool:
        if batching_rule == ParticleBatchingRule.NEVER_BATCH:
            return False
        if self.batching_identity != other.batching_identity:
            return False
        ready_time_difference = abs(self.earliest_ready_time_ns - other.earliest_ready_time_ns)
        if batching_rule == ParticleBatchingRule.BATCH_ONLY_IF_READY_AT_EXACTLY_THE_SAME_TIME:
            return ready_time_difference == 0
        return ready_time_difference <= batching_window_ns

    def batch_with(
        self,
        other: "WorkParticle",
        batching_rule: ParticleBatchingRule,
        batching_window_ns: float,
    ) -> "WorkParticle":
        if not self.can_batch_with(other, batching_rule, batching_window_ns):
            raise ValueError("work particles are not batch-compatible")
        return WorkParticle(
            logical_work_units=self.logical_work_units + other.logical_work_units,
            operation_name=self.operation_name,
            earliest_ready_time_ns=max(
                self.earliest_ready_time_ns,
                other.earliest_ready_time_ns,
            ),
            current_node_id=self.current_node_id,
            route_name=self.route_name,
            model_layer_index=self.model_layer_index,
            resource_width_per_unit=self.resource_width_per_unit,
        )

    def split_to_fit_maximum_logical_units(
        self,
        maximum_logical_units: int,
    ) -> tuple["WorkParticle", "WorkParticle | None"]:
        if maximum_logical_units <= 0:
            raise ValueError("maximum_logical_units must be positive")
        if self.logical_unit_count <= maximum_logical_units:
            return self, None

        selected_particle = WorkParticle(
            logical_work_units=self.logical_work_units[:maximum_logical_units],
            operation_name=self.operation_name,
            earliest_ready_time_ns=self.earliest_ready_time_ns,
            current_node_id=self.current_node_id,
            route_name=self.route_name,
            model_layer_index=self.model_layer_index,
            resource_width_per_unit=self.resource_width_per_unit,
        )
        remainder_particle = WorkParticle(
            logical_work_units=self.logical_work_units[maximum_logical_units:],
            operation_name=self.operation_name,
            earliest_ready_time_ns=self.earliest_ready_time_ns,
            current_node_id=self.current_node_id,
            route_name=self.route_name,
            model_layer_index=self.model_layer_index,
            resource_width_per_unit=self.resource_width_per_unit,
        )
        return selected_particle, remainder_particle


class DependencyJoinTracker:
    def __init__(self) -> None:
        self.expected_branch_count_by_join_id: dict[str, int] = {}
        self.arrived_branch_indexes_by_join_id: dict[str, set[int]] = {}

    def register_expected_branch_count(
        self,
        dependency_join_id: str,
        branch_count: int,
    ) -> None:
        if branch_count < 1:
            raise ValueError("branch_count must be positive")
        if dependency_join_id in self.expected_branch_count_by_join_id:
            raise ValueError(f"dependency join already exists: {dependency_join_id}")
        self.expected_branch_count_by_join_id[dependency_join_id] = branch_count

    def accept_completed_branches_and_release_tokens_after_all_selected_experts_finish(
        self,
        particle: WorkParticle,
        next_operation_name: str,
        ready_time_ns: float,
    ) -> list[WorkParticle]:
        if particle.current_node_id is None:
            raise ValueError("completed expert branches require a rendezvous node")

        released_units_by_sequence_owner: dict[int, list[LogicalWorkUnit]] = {}

        for unit in particle.logical_work_units:
            if unit.dependency_join_id is None:
                raise ValueError("dependency_join_id is missing")
            if unit.branch_index_inside_dependency_join is None:
                raise ValueError("branch_index_inside_dependency_join is missing")
            if unit.node_that_owns_sequence_state is None:
                raise ValueError("node_that_owns_sequence_state is missing")
            if unit.expert_result_rendezvous_node_id != particle.current_node_id:
                raise ValueError("expert branch arrived at the wrong rendezvous node")

            arrived_branch_indexes = self.arrived_branch_indexes_by_join_id.setdefault(
                unit.dependency_join_id,
                set(),
            )
            arrived_branch_indexes.add(unit.branch_index_inside_dependency_join)

            expected_branch_count = self.expected_branch_count_by_join_id[
                unit.dependency_join_id
            ]
            if len(arrived_branch_indexes) < expected_branch_count:
                continue

            released_units_by_sequence_owner.setdefault(
                unit.node_that_owns_sequence_state,
                [],
            ).append(
                LogicalWorkUnit(
                    globally_unique_token_id=unit.globally_unique_token_id,
                    workload_agent_id=unit.workload_agent_id,
                    node_that_owns_sequence_state=unit.node_that_owns_sequence_state,
                )
            )
            del self.arrived_branch_indexes_by_join_id[unit.dependency_join_id]
            del self.expected_branch_count_by_join_id[unit.dependency_join_id]

        return [
            WorkParticle(
                logical_work_units=tuple(units),
                operation_name=next_operation_name,
                earliest_ready_time_ns=ready_time_ns,
                current_node_id=particle.current_node_id,
                model_layer_index=particle.model_layer_index,
            )
            for units in released_units_by_sequence_owner.values()
        ]
