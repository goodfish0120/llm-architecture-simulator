from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SerializedThroughputResource:
    resource_name: str
    work_units_per_ns: float
    startup_latency_ns: float = 0.0
    next_free_time_ns: float = 0.0
    accumulated_busy_time_ns: float = 0.0
    accumulated_queue_wait_time_ns: float = 0.0

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
        self.accumulated_busy_time_ns += service_duration_ns
        self.next_free_time_ns = service_end_time_ns
        return service_start_time_ns, service_end_time_ns


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

        for resource in (self.compute_resource, self.memory_resource):
            resource.accumulated_queue_wait_time_ns += service_start_time_ns - ready_time_ns
            resource.accumulated_busy_time_ns += service_duration_ns
            resource.next_free_time_ns = service_end_time_ns

        return service_start_time_ns, service_end_time_ns, service_duration_ns


class SharedNetworkFabric:
    def __init__(
        self,
        node_count: int,
        endpoint_bytes_per_ns: float,
        switch_bytes_per_ns: float,
        fixed_network_latency_ns: float,
    ) -> None:
        self.fixed_network_latency_ns = fixed_network_latency_ns
        self.transmit_resource_by_node = [
            SerializedThroughputResource(f"nic_tx[{node_id}]", endpoint_bytes_per_ns)
            for node_id in range(node_count)
        ]
        self.receive_resource_by_node = [
            SerializedThroughputResource(f"nic_rx[{node_id}]", endpoint_bytes_per_ns)
            for node_id in range(node_count)
        ]
        self.shared_switch_resource = SerializedThroughputResource(
            "switch",
            switch_bytes_per_ns,
        )

    def transfer_bytes_between_nodes(
        self,
        source_node_id: int,
        destination_node_id: int,
        ready_time_ns: float,
        byte_count: float,
    ) -> tuple[float, float]:
        if source_node_id == destination_node_id:
            return ready_time_ns, ready_time_ns

        transmit_resource = self.transmit_resource_by_node[source_node_id]
        receive_resource = self.receive_resource_by_node[destination_node_id]
        switch_resource = self.shared_switch_resource

        serialization_rate_bytes_per_ns = min(
            transmit_resource.work_units_per_ns,
            receive_resource.work_units_per_ns,
            switch_resource.work_units_per_ns,
        )
        serialization_duration_ns = byte_count / serialization_rate_bytes_per_ns

        serialization_start_time_ns = max(
            ready_time_ns,
            transmit_resource.next_free_time_ns,
            switch_resource.next_free_time_ns,
            receive_resource.next_free_time_ns - self.fixed_network_latency_ns,
        )
        receive_start_time_ns = serialization_start_time_ns + self.fixed_network_latency_ns
        arrival_time_ns = receive_start_time_ns + serialization_duration_ns

        transmit_resource.accumulated_queue_wait_time_ns += (
            serialization_start_time_ns - ready_time_ns
        )
        switch_resource.accumulated_queue_wait_time_ns += (
            serialization_start_time_ns - ready_time_ns
        )
        receive_resource.accumulated_queue_wait_time_ns += max(
            0.0,
            receive_start_time_ns - (ready_time_ns + self.fixed_network_latency_ns),
        )

        for resource in (
            transmit_resource,
            receive_resource,
            switch_resource,
        ):
            resource.accumulated_busy_time_ns += serialization_duration_ns

        transmit_resource.next_free_time_ns = (
            serialization_start_time_ns + serialization_duration_ns
        )
        switch_resource.next_free_time_ns = (
            serialization_start_time_ns + serialization_duration_ns
        )
        receive_resource.next_free_time_ns = arrival_time_ns

        return serialization_start_time_ns, arrival_time_ns
