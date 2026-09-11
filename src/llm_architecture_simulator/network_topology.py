from __future__ import annotations

import heapq
from dataclasses import dataclass

from .hardware_resources import SerializedThroughputResource


@dataclass
class BidirectionalNetworkLink:
    link_name: str
    endpoint_a: str
    endpoint_b: str
    bytes_per_ns_each_direction: float
    fixed_latency_ns_per_hop: float

    def __post_init__(self) -> None:
        if self.bytes_per_ns_each_direction <= 0:
            raise ValueError("bytes_per_ns_each_direction must be positive")
        if self.fixed_latency_ns_per_hop < 0:
            raise ValueError("fixed_latency_ns_per_hop must be non-negative")
        self.a_to_b_serialization = SerializedThroughputResource(
            f"{self.link_name}:{self.endpoint_a}->{self.endpoint_b}",
            self.bytes_per_ns_each_direction,
        )
        self.b_to_a_serialization = SerializedThroughputResource(
            f"{self.link_name}:{self.endpoint_b}->{self.endpoint_a}",
            self.bytes_per_ns_each_direction,
        )

    def other_endpoint(self, endpoint: str) -> str:
        if endpoint == self.endpoint_a:
            return self.endpoint_b
        if endpoint == self.endpoint_b:
            return self.endpoint_a
        raise ValueError(f"{endpoint} is not attached to {self.link_name}")

    def serialization_resource_for_direction(
        self,
        source_endpoint: str,
        destination_endpoint: str,
    ) -> SerializedThroughputResource:
        if source_endpoint == self.endpoint_a and destination_endpoint == self.endpoint_b:
            return self.a_to_b_serialization
        if source_endpoint == self.endpoint_b and destination_endpoint == self.endpoint_a:
            return self.b_to_a_serialization
        raise ValueError("requested direction does not belong to this link")


class RoutedNetworkTopology:
    def __init__(self, compute_node_count: int, links: list[BidirectionalNetworkLink]) -> None:
        if compute_node_count < 1:
            raise ValueError("compute_node_count must be positive")
        self.compute_node_count = compute_node_count
        self.links = links
        self.links_attached_to_endpoint: dict[str, list[BidirectionalNetworkLink]] = {}
        for link in links:
            self.links_attached_to_endpoint.setdefault(link.endpoint_a, []).append(link)
            self.links_attached_to_endpoint.setdefault(link.endpoint_b, []).append(link)
        for node_id in range(compute_node_count):
            if self.compute_node_endpoint(node_id) not in self.links_attached_to_endpoint and compute_node_count > 1:
                raise ValueError(f"compute node {node_id} has no network link")

    @staticmethod
    def compute_node_endpoint(node_id: int) -> str:
        return f"compute:{node_id}"

    def transfer_bytes_between_compute_nodes(
        self,
        source_node_id: int,
        destination_node_id: int,
        ready_time_ns: float,
        byte_count: float,
    ) -> tuple[float, float]:
        if byte_count < 0:
            raise ValueError("byte_count must be non-negative")
        if source_node_id == destination_node_id or byte_count == 0:
            return ready_time_ns, ready_time_ns

        source_endpoint = self.compute_node_endpoint(source_node_id)
        destination_endpoint = self.compute_node_endpoint(destination_node_id)
        route = self._find_route_with_lowest_uncontended_transfer_time(
            source_endpoint,
            destination_endpoint,
            byte_count,
        )

        first_serialization_start_time_ns: float | None = None
        current_ready_time_ns = ready_time_ns
        current_endpoint = source_endpoint

        for link in route:
            next_endpoint = link.other_endpoint(current_endpoint)
            serialization_resource = link.serialization_resource_for_direction(
                current_endpoint,
                next_endpoint,
            )
            serialization_start_time_ns, serialization_end_time_ns = (
                serialization_resource.reserve_resource_until_work_finishes(
                    current_ready_time_ns,
                    byte_count,
                )
            )
            if first_serialization_start_time_ns is None:
                first_serialization_start_time_ns = serialization_start_time_ns
            current_ready_time_ns = (
                serialization_end_time_ns + link.fixed_latency_ns_per_hop
            )
            current_endpoint = next_endpoint

        return first_serialization_start_time_ns or ready_time_ns, current_ready_time_ns

    def _find_route_with_lowest_uncontended_transfer_time(
        self,
        source_endpoint: str,
        destination_endpoint: str,
        byte_count: float,
    ) -> list[BidirectionalNetworkLink]:
        queue: list[tuple[float, str, tuple[BidirectionalNetworkLink, ...]]] = [
            (0.0, source_endpoint, ())
        ]
        best_cost_by_endpoint = {source_endpoint: 0.0}

        while queue:
            accumulated_cost_ns, endpoint, route = heapq.heappop(queue)
            if endpoint == destination_endpoint:
                return list(route)
            if accumulated_cost_ns > best_cost_by_endpoint.get(endpoint, float("inf")):
                continue

            for link in self.links_attached_to_endpoint.get(endpoint, []):
                next_endpoint = link.other_endpoint(endpoint)
                edge_cost_ns = (
                    link.fixed_latency_ns_per_hop
                    + byte_count / link.bytes_per_ns_each_direction
                )
                next_cost_ns = accumulated_cost_ns + edge_cost_ns
                if next_cost_ns < best_cost_by_endpoint.get(next_endpoint, float("inf")):
                    best_cost_by_endpoint[next_endpoint] = next_cost_ns
                    heapq.heappush(
                        queue,
                        (next_cost_ns, next_endpoint, route + (link,)),
                    )

        raise ValueError(
            f"no network route from {source_endpoint} to {destination_endpoint}"
        )

    def count_physical_links_attached_to_compute_node(self, node_id: int) -> int:
        return len(
            self.links_attached_to_endpoint.get(self.compute_node_endpoint(node_id), [])
        )

    def validate_compute_node_port_limits(
        self,
        maximum_physical_links_by_compute_node: dict[int, int],
    ) -> None:
        for node_id, maximum_link_count in maximum_physical_links_by_compute_node.items():
            actual_link_count = self.count_physical_links_attached_to_compute_node(node_id)
            if actual_link_count > maximum_link_count:
                raise ValueError(
                    f"compute node {node_id} uses {actual_link_count} links but only "
                    f"{maximum_link_count} ports are available"
                )

    def calculate_directional_link_busy_fractions_between_times(
        self,
        measurement_start_time_ns: float,
        measurement_end_time_ns: float,
    ) -> dict[str, float]:
        busy_fraction_by_direction: dict[str, float] = {}
        for link in self.links:
            for resource in (
                link.a_to_b_serialization,
                link.b_to_a_serialization,
            ):
                busy_fraction_by_direction[resource.resource_name] = (
                    resource.calculate_busy_fraction_between_times(
                        measurement_start_time_ns,
                        measurement_end_time_ns,
                    )
                )
        return busy_fraction_by_direction


class NetworkTopologyFactory:
    @staticmethod
    def create_direct_full_mesh_between_compute_nodes(
        compute_node_count: int,
        bytes_per_ns_each_direction: float,
        fixed_latency_ns_per_hop: float,
    ) -> RoutedNetworkTopology:
        links = [
            BidirectionalNetworkLink(
                f"direct_{node_a}_{node_b}",
                RoutedNetworkTopology.compute_node_endpoint(node_a),
                RoutedNetworkTopology.compute_node_endpoint(node_b),
                bytes_per_ns_each_direction,
                fixed_latency_ns_per_hop,
            )
            for node_a in range(compute_node_count)
            for node_b in range(node_a + 1, compute_node_count)
        ]
        return RoutedNetworkTopology(compute_node_count, links)

    @staticmethod
    def create_bidirectional_daisy_chain_between_compute_nodes(
        compute_node_count: int,
        bytes_per_ns_each_direction: float,
        fixed_latency_ns_per_hop: float,
    ) -> RoutedNetworkTopology:
        links = [
            BidirectionalNetworkLink(
                f"chain_{node_id}_{node_id + 1}",
                RoutedNetworkTopology.compute_node_endpoint(node_id),
                RoutedNetworkTopology.compute_node_endpoint(node_id + 1),
                bytes_per_ns_each_direction,
                fixed_latency_ns_per_hop,
            )
            for node_id in range(compute_node_count - 1)
        ]
        return RoutedNetworkTopology(compute_node_count, links)

    @staticmethod
    def create_bidirectional_ring_between_compute_nodes(
        compute_node_count: int,
        bytes_per_ns_each_direction: float,
        fixed_latency_ns_per_hop: float,
    ) -> RoutedNetworkTopology:
        if compute_node_count < 3:
            return NetworkTopologyFactory.create_direct_full_mesh_between_compute_nodes(
                compute_node_count,
                bytes_per_ns_each_direction,
                fixed_latency_ns_per_hop,
            )
        links = [
            BidirectionalNetworkLink(
                f"ring_{node_id}_{(node_id + 1) % compute_node_count}",
                RoutedNetworkTopology.compute_node_endpoint(node_id),
                RoutedNetworkTopology.compute_node_endpoint((node_id + 1) % compute_node_count),
                bytes_per_ns_each_direction,
                fixed_latency_ns_per_hop,
            )
            for node_id in range(compute_node_count)
        ]
        return RoutedNetworkTopology(compute_node_count, links)

    @staticmethod
    def create_nonblocking_central_switch_star(
        compute_node_count: int,
        bytes_per_ns_each_direction_per_node_link: float,
        fixed_latency_ns_per_hop: float,
    ) -> RoutedNetworkTopology:
        switch_endpoint = "switch:0"
        links = [
            BidirectionalNetworkLink(
                f"switch_uplink_{node_id}",
                RoutedNetworkTopology.compute_node_endpoint(node_id),
                switch_endpoint,
                bytes_per_ns_each_direction_per_node_link,
                fixed_latency_ns_per_hop,
            )
            for node_id in range(compute_node_count)
        ]
        return RoutedNetworkTopology(compute_node_count, links)
