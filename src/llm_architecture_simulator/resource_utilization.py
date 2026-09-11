from __future__ import annotations

from dataclasses import dataclass

from .hardware_resources import (
    ComputeAndMemoryNode,
    SerializedThroughputResource,
    _calculate_interval_overlap_ns,
)


@dataclass(frozen=True)
class ResourceUtilizationBreakdown:
    productive_fraction: float
    reserved_but_not_productive_fraction: float
    completely_idle_fraction: float

    @property
    def reserved_fraction(self) -> float:
        return self.productive_fraction + self.reserved_but_not_productive_fraction

    def as_dict(self) -> dict[str, float]:
        return {
            "productive": self.productive_fraction,
            "reserved_but_not_productive": self.reserved_but_not_productive_fraction,
            "completely_idle": self.completely_idle_fraction,
        }


def calculate_serialized_resource_utilization_between_times(
    resource: SerializedThroughputResource,
    measurement_start_time_ns: float,
    measurement_end_time_ns: float,
) -> ResourceUtilizationBreakdown:
    reserved_fraction = resource.calculate_reserved_fraction_between_times(
        measurement_start_time_ns,
        measurement_end_time_ns,
    )
    productive_fraction = resource.calculate_productive_fraction_between_times(
        measurement_start_time_ns,
        measurement_end_time_ns,
    )
    return ResourceUtilizationBreakdown(
        productive_fraction=productive_fraction,
        reserved_but_not_productive_fraction=max(
            0.0,
            reserved_fraction - productive_fraction,
        ),
        completely_idle_fraction=max(0.0, 1.0 - reserved_fraction),
    )


def calculate_compute_and_memory_idle_reason_fractions_between_times(
    node: ComputeAndMemoryNode,
    measurement_start_time_ns: float,
    measurement_end_time_ns: float,
) -> dict[str, float]:
    measurement_duration_ns = measurement_end_time_ns - measurement_start_time_ns
    if measurement_duration_ns <= 0:
        return {
            "compute_productive": 0.0,
            "compute_waiting_for_memory": 0.0,
            "compute_idle_without_reserved_kernel": 0.0,
            "memory_productive": 0.0,
            "memory_waiting_for_compute": 0.0,
            "memory_idle_without_reserved_kernel": 0.0,
        }

    compute_utilization = calculate_serialized_resource_utilization_between_times(
        node.compute_resource,
        measurement_start_time_ns,
        measurement_end_time_ns,
    )
    memory_utilization = calculate_serialized_resource_utilization_between_times(
        node.memory_resource,
        measurement_start_time_ns,
        measurement_end_time_ns,
    )

    compute_waiting_for_memory_ns = 0.0
    memory_waiting_for_compute_ns = 0.0
    for kernel in node.executed_kernels:
        if kernel.memory_productive_end_time_ns > kernel.compute_productive_end_time_ns:
            compute_waiting_for_memory_ns += _calculate_interval_overlap_ns(
                kernel.compute_productive_end_time_ns,
                kernel.service_end_time_ns,
                measurement_start_time_ns,
                measurement_end_time_ns,
            )
        elif kernel.compute_productive_end_time_ns > kernel.memory_productive_end_time_ns:
            memory_waiting_for_compute_ns += _calculate_interval_overlap_ns(
                kernel.memory_productive_end_time_ns,
                kernel.service_end_time_ns,
                measurement_start_time_ns,
                measurement_end_time_ns,
            )

    return {
        "compute_productive": compute_utilization.productive_fraction,
        "compute_waiting_for_memory": min(
            1.0,
            compute_waiting_for_memory_ns / measurement_duration_ns,
        ),
        "compute_idle_without_reserved_kernel": compute_utilization.completely_idle_fraction,
        "memory_productive": memory_utilization.productive_fraction,
        "memory_waiting_for_compute": min(
            1.0,
            memory_waiting_for_compute_ns / measurement_duration_ns,
        ),
        "memory_idle_without_reserved_kernel": memory_utilization.completely_idle_fraction,
    }
