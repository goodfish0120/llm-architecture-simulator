from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SerializedThroughputResource:
    resource_name: str
    work_units_per_ns: float
    startup_latency_ns: float = 0.0
    next_free_time_ns: float = 0.0
    accumulated_queue_wait_time_ns: float = 0.0
    reserved_busy_intervals_ns: list[tuple[float, float]] = field(default_factory=list)

    def reserve_resource_until_work_finishes(
        self,
        ready_time_ns: float,
        work_amount: float,
        efficiency_multiplier: float = 1.0,
    ) -> tuple[float, float]:
        if work_amount < 0 or self.work_units_per_ns <= 0 or efficiency_multiplier <= 0:
            raise ValueError("invalid resource request")

        service_start_time_ns = max(ready_time_ns, self.next_free_time_ns)
        service_duration_ns = (
            self.startup_latency_ns
            + work_amount / (self.work_units_per_ns * efficiency_multiplier)
        )
        service_end_time_ns = service_start_time_ns + service_duration_ns

        self.accumulated_queue_wait_time_ns += service_start_time_ns - ready_time_ns
        self.reserved_busy_intervals_ns.append(
            (service_start_time_ns, service_end_time_ns)
        )
        self.next_free_time_ns = service_end_time_ns
        return service_start_time_ns, service_end_time_ns

    def reserve_exact_interval(
        self,
        ready_time_ns: float,
        service_start_time_ns: float,
        service_end_time_ns: float,
    ) -> None:
        if service_start_time_ns < ready_time_ns or service_end_time_ns < service_start_time_ns:
            raise ValueError("invalid service interval")
        self.accumulated_queue_wait_time_ns += service_start_time_ns - ready_time_ns
        self.reserved_busy_intervals_ns.append(
            (service_start_time_ns, service_end_time_ns)
        )
        self.next_free_time_ns = max(self.next_free_time_ns, service_end_time_ns)

    def calculate_busy_fraction_between_times(
        self,
        measurement_start_time_ns: float,
        measurement_end_time_ns: float,
    ) -> float:
        measurement_duration_ns = measurement_end_time_ns - measurement_start_time_ns
        if measurement_duration_ns <= 0:
            return 0.0

        busy_time_ns = 0.0
        for service_start_time_ns, service_end_time_ns in self.reserved_busy_intervals_ns:
            overlap_start_time_ns = max(measurement_start_time_ns, service_start_time_ns)
            overlap_end_time_ns = min(measurement_end_time_ns, service_end_time_ns)
            if overlap_end_time_ns > overlap_start_time_ns:
                busy_time_ns += overlap_end_time_ns - overlap_start_time_ns

        return min(1.0, busy_time_ns / measurement_duration_ns)


@dataclass
class ComputeAndMemoryNode:
    compute_resource: SerializedThroughputResource
    memory_resource: SerializedThroughputResource

    def execute_kernel_with_compute_and_memory_overlap(
        self,
        ready_time_ns: float,
        floating_point_operations: float,
        bytes_accessed: float,
        compute_efficiency_multiplier: float = 1.0,
    ) -> tuple[float, float, float]:
        service_start_time_ns = max(
            ready_time_ns,
            self.compute_resource.next_free_time_ns,
            self.memory_resource.next_free_time_ns,
        )
        compute_duration_ns = (
            self.compute_resource.startup_latency_ns
            + floating_point_operations
            / (
                self.compute_resource.work_units_per_ns
                * compute_efficiency_multiplier
            )
        )
        memory_duration_ns = (
            self.memory_resource.startup_latency_ns
            + bytes_accessed / self.memory_resource.work_units_per_ns
        )
        service_duration_ns = max(compute_duration_ns, memory_duration_ns)
        service_end_time_ns = service_start_time_ns + service_duration_ns

        self.compute_resource.reserve_exact_interval(
            ready_time_ns,
            service_start_time_ns,
            service_end_time_ns,
        )
        self.memory_resource.reserve_exact_interval(
            ready_time_ns,
            service_start_time_ns,
            service_end_time_ns,
        )

        return service_start_time_ns, service_end_time_ns, service_duration_ns
