from __future__ import annotations

from dataclasses import dataclass, field


def _calculate_interval_overlap_ns(
    interval_start_ns: float,
    interval_end_ns: float,
    measurement_start_ns: float,
    measurement_end_ns: float,
) -> float:
    overlap_start_ns = max(interval_start_ns, measurement_start_ns)
    overlap_end_ns = min(interval_end_ns, measurement_end_ns)
    return max(0.0, overlap_end_ns - overlap_start_ns)


@dataclass
class SerializedThroughputResource:
    resource_name: str
    work_units_per_ns: float
    startup_latency_ns: float = 0.0
    next_free_time_ns: float = 0.0
    accumulated_queue_wait_time_ns: float = 0.0
    reserved_busy_intervals_ns: list[tuple[float, float]] = field(default_factory=list)
    productive_work_intervals_ns: list[tuple[float, float]] = field(default_factory=list)

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

        self.reserve_interval_and_record_productive_subinterval(
            ready_time_ns=ready_time_ns,
            reserved_start_time_ns=service_start_time_ns,
            reserved_end_time_ns=service_end_time_ns,
            productive_start_time_ns=service_start_time_ns,
            productive_end_time_ns=service_end_time_ns,
        )
        return service_start_time_ns, service_end_time_ns

    def reserve_exact_interval_as_productive_work(
        self,
        ready_time_ns: float,
        service_start_time_ns: float,
        service_end_time_ns: float,
    ) -> None:
        self.reserve_interval_and_record_productive_subinterval(
            ready_time_ns=ready_time_ns,
            reserved_start_time_ns=service_start_time_ns,
            reserved_end_time_ns=service_end_time_ns,
            productive_start_time_ns=service_start_time_ns,
            productive_end_time_ns=service_end_time_ns,
        )

    def reserve_interval_and_record_productive_subinterval(
        self,
        ready_time_ns: float,
        reserved_start_time_ns: float,
        reserved_end_time_ns: float,
        productive_start_time_ns: float,
        productive_end_time_ns: float,
    ) -> None:
        if reserved_start_time_ns < ready_time_ns:
            raise ValueError("reserved interval cannot start before work is ready")
        if reserved_end_time_ns < reserved_start_time_ns:
            raise ValueError("reserved interval end cannot precede its start")
        if productive_start_time_ns < reserved_start_time_ns:
            raise ValueError("productive interval cannot start before reservation")
        if productive_end_time_ns < productive_start_time_ns:
            raise ValueError("productive interval end cannot precede its start")
        if productive_end_time_ns > reserved_end_time_ns:
            raise ValueError("productive interval cannot exceed reservation")

        self.accumulated_queue_wait_time_ns += reserved_start_time_ns - ready_time_ns
        self.reserved_busy_intervals_ns.append(
            (reserved_start_time_ns, reserved_end_time_ns)
        )
        if productive_end_time_ns > productive_start_time_ns:
            self.productive_work_intervals_ns.append(
                (productive_start_time_ns, productive_end_time_ns)
            )
        self.next_free_time_ns = max(self.next_free_time_ns, reserved_end_time_ns)

    @staticmethod
    def _calculate_fraction_covered_by_intervals_between_times(
        intervals_ns: list[tuple[float, float]],
        measurement_start_time_ns: float,
        measurement_end_time_ns: float,
    ) -> float:
        measurement_duration_ns = measurement_end_time_ns - measurement_start_time_ns
        if measurement_duration_ns <= 0:
            return 0.0

        covered_time_ns = sum(
            _calculate_interval_overlap_ns(
                interval_start_ns,
                interval_end_ns,
                measurement_start_time_ns,
                measurement_end_time_ns,
            )
            for interval_start_ns, interval_end_ns in intervals_ns
        )
        return min(1.0, covered_time_ns / measurement_duration_ns)

    def calculate_reserved_fraction_between_times(
        self,
        measurement_start_time_ns: float,
        measurement_end_time_ns: float,
    ) -> float:
        return self._calculate_fraction_covered_by_intervals_between_times(
            self.reserved_busy_intervals_ns,
            measurement_start_time_ns,
            measurement_end_time_ns,
        )

    def calculate_productive_fraction_between_times(
        self,
        measurement_start_time_ns: float,
        measurement_end_time_ns: float,
    ) -> float:
        return self._calculate_fraction_covered_by_intervals_between_times(
            self.productive_work_intervals_ns,
            measurement_start_time_ns,
            measurement_end_time_ns,
        )

    def calculate_busy_fraction_between_times(
        self,
        measurement_start_time_ns: float,
        measurement_end_time_ns: float,
    ) -> float:
        return self.calculate_reserved_fraction_between_times(
            measurement_start_time_ns,
            measurement_end_time_ns,
        )


@dataclass
class OverlappedComputeAndMemoryKernelExecution:
    service_start_time_ns: float
    service_end_time_ns: float
    compute_productive_end_time_ns: float
    memory_productive_end_time_ns: float


@dataclass
class ComputeAndMemoryNode:
    compute_resource: SerializedThroughputResource
    memory_resource: SerializedThroughputResource
    executed_kernels: list[OverlappedComputeAndMemoryKernelExecution] = field(
        default_factory=list
    )

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
        compute_productive_end_time_ns = service_start_time_ns + compute_duration_ns
        memory_productive_end_time_ns = service_start_time_ns + memory_duration_ns

        self.compute_resource.reserve_interval_and_record_productive_subinterval(
            ready_time_ns=ready_time_ns,
            reserved_start_time_ns=service_start_time_ns,
            reserved_end_time_ns=service_end_time_ns,
            productive_start_time_ns=service_start_time_ns,
            productive_end_time_ns=compute_productive_end_time_ns,
        )
        self.memory_resource.reserve_interval_and_record_productive_subinterval(
            ready_time_ns=ready_time_ns,
            reserved_start_time_ns=service_start_time_ns,
            reserved_end_time_ns=service_end_time_ns,
            productive_start_time_ns=service_start_time_ns,
            productive_end_time_ns=memory_productive_end_time_ns,
        )
        self.executed_kernels.append(
            OverlappedComputeAndMemoryKernelExecution(
                service_start_time_ns=service_start_time_ns,
                service_end_time_ns=service_end_time_ns,
                compute_productive_end_time_ns=compute_productive_end_time_ns,
                memory_productive_end_time_ns=memory_productive_end_time_ns,
            )
        )

        return service_start_time_ns, service_end_time_ns, service_duration_ns
