from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from .hardware_resources import SerializedThroughputResource


@dataclass
class BidirectionalNetworkLink:
    link_name: str
    endpoint_a: str
    endpoint_b: str
    bytes_per_ns_each_direction: float
    fixed_latency_ns: float

    def __post_init__(self) -> None:
        if self.bytes_per_ns_each_direction <= 0:
            raise ValueError("bytes_per_ns_each_direction must be positive")
        if self.fixed_latency_ns < 0:
            raise ValueError("fixed_latency_ns must be non-negative")
        self.a_to_b = SerializedThroughputResource(
            f"{self.link_name}:{self.endpoint_a}->{self.endpoint_b}",
            self.bytes_per_ns_each_direction,
        )
        self.b_to_a = SerializedThroughputResource(
            f"{self.link_name}:{self.endpoint_b}->{self.endpoint_a}",
            self.bytes_per_ns_each_direction,
        )

    def other_endpoint(self, endpoint: str) -> str:
        if endpoint == self.endpoint_a:
            return self.endpoint_b
        if endpoint == self.endpoint_b:
            return self.endpoint_a
        raise ValueError(f"{endpoint} is not attached to {self.link_name}")

    def directional_resource(
        self,
        source_endpoint: str,
        destination_endpoint: str,
    ) -> SerializedThroughputResource:
        if source_endpoint == self.endpoint_a and destination_endpoint == self.endpoint_b:
            return self.a_to_b
        if source_endpoint == self.endpoint_b and destination_endpoint == self.endpoint_a:
            return self.b_to_a
        raise ValueError("requested direction does not belong to this link")


class RoutedNetworkTopology:
    def __init__(self, compute_node_count: int, links: list[BidirectionalNetworkLink]) -> None:
        if compute_node_count < 1:
            raise ValueError("compute_node_count must be positive")
        self.compute_node_count = compute_node_count
        self.links = links
        self.links_by_endpoint: dict[str, list[BidirectionalNetworkLink]] = {}
        self.cached_route_by_endpoint_pair: dict[
            tuple[str, str],
            tuple[BidirectionalNetworkLink, ...],
        ] = {}

        for link in links:
            self.links_by_endpoint.setdefault(link.endpoint_a, []).append(link)
            self.links_by_endpoint.setdefault(link.endpoint_b, []).append(link)

        if compute_node_count > 1:
            for node_id in range(compute_node_count):
                if self.compute_node_endpoint(node_id) not in self.links_by_endpoint:
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
        route = self._find_and_cache_shortest_hop_route(
            source_endpoint,
            destination_endpoint,
        )

        directional_resources: list[SerializedThroughputResource] = []
        current_endpoint = source_endpoint
        total_fixed_latency_ns = 0.0
        serialization_duration_by_resource: list[float] = []

        for link in route:
            next_endpoint = link.other_endpoint(current_endpoint)
            resource = link.directional_resource(current_endpoint, next_endpoint)
            directional_resources.append(resource)
            serialization_duration_by_resource.append(
                byte_count / resource.work_units_per_ns
            )
            total_fixed_latency_ns += link.fixed_latency_ns
            current_endpoint = next_endpoint

        serialization_start_time_ns = max(
            ready_time_ns,
            *(resource.next_free_time_ns for resource in directional_resources),
        )

        for resource, serialization_duration_ns in zip(
            directional_resources,
            serialization_duration_by_resource,
        ):
            resource.reserve_exact_interval(
                ready_time_ns,
                serialization_start_time_ns,
                serialization_start_time_ns + serialization_duration_ns,
            )

        arrival_time_ns = (
            serialization_start_time_ns
            + max(serialization_duration_by_resource)
            + total_fixed_latency_ns
        )
        return serialization_start_time_ns, arrival_time_ns

    def _find_and_cache_shortest_hop_route(
        self,
        source_endpoint: str,
        destination_endpoint: str,
    ) -> tuple[BidirectionalNetworkLink, ...]:
        cache_key = (source_endpoint, destination_endpoint)
        cached_route = self.cached_route_by_endpoint_pair.get(cache_key)
        if cached_route is not None:
            return cached_route

        queue = deque([(source_endpoint, tuple())])
        visited_endpoints = {source_endpoint}

        while queue:
            endpoint, route = queue.popleft()
            if endpoint == destination_endpoint:
                self.cached_route_by_endpoint_pair[cache_key] = route
                return route

            for link in self.links_by_endpoint.get(endpoint, []):
                next_endpoint = link.other_endpoint(endpoint)
                if next_endpoint in visited_endpoints:
                    continue
                visited_endpoints.add(next_endpoint)
                queue.append((next_endpoint, route + (link,)))

        raise ValueError(
            f"no network route from {source_endpoint} to {destination_endpoint}"
        )

    def validate_compute_node_port_limits(
        self,
        maximum_physical_links_by_compute_node: dict[int, int],
    ) -> None:
        for node_id, maximum_link_count in maximum_physical_links_by_compute_node.items():
            actual_link_count = len(
                self.links_by_endpoint.get(self.compute_node_endpoint(node_id), [])
            )
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
        return {
            resource.resource_name: resource.calculate_busy_fraction_between_times(
                measurement_start_time_ns,
                measurement_end_time_ns,
            )
            for link in self.links
            for resource in (link.a_to_b, link.b_to_a)
        }


class NetworkTopologyFactory:
    @staticmethod
    def create_direct_full_mesh_between_compute_nodes(
        compute_node_count: int,
        bytes_per_ns_each_direction: float,
        fixed_latency_ns_per_link: float,
    ) -> RoutedNetworkTopology:
        return RoutedNetworkTopology(
            compute_node_count,
            [
                BidirectionalNetworkLink(
                    f"direct_{node_a}_{node_b}",
                    RoutedNetworkTopology.compute_node_endpoint(node_a),
                    RoutedNetworkTopology.compute_node_endpoint(node_b),
                    bytes_per_ns_each_direction,
                    fixed_latency_ns_per_link,
                )
                for node_a in range(compute_node_count)
                for node_b in range(node_a + 1, compute_node_count)
            ],
        )

    @staticmethod
    def create_bidirectional_daisy_chain_between_compute_nodes(
        compute_node_count: int,
        bytes_per_ns_each_direction: float,
        fixed_latency_ns_per_link: float,
    ) -> RoutedNetworkTopology:
        return RoutedNetworkTopology(
            compute_node_count,
            [
                BidirectionalNetworkLink(
                    f"chain_{node_id}_{node_id + 1}",
                    RoutedNetworkTopology.compute_node_endpoint(node_id),
                    RoutedNetworkTopology.compute_node_endpoint(node_id + 1),
                    bytes_per_ns_each_direction,
                    fixed_latency_ns_per_link,
                )
                for node_id in range(compute_node_count - 1)
            ],
        )

    @staticmethod
    def create_bidirectional_ring_between_compute_nodes(
        compute_node_count: int,
        bytes_per_ns_each_direction: float,
        fixed_latency_ns_per_link: float,
    ) -> RoutedNetworkTopology:
        if compute_node_count < 3:
            return NetworkTopologyFactory.create_direct_full_mesh_between_compute_nodes(
                compute_node_count,
                bytes_per_ns_each_direction,
                fixed_latency_ns_per_link,
            )
        return RoutedNetworkTopology(
            compute_node_count,
            [
                BidirectionalNetworkLink(
                    f"ring_{node_id}_{(node_id + 1) % compute_node_count}",
                    RoutedNetworkTopology.compute_node_endpoint(node_id),
                    RoutedNetworkTopology.compute_node_endpoint(
                        (node_id + 1) % compute_node_count
                    ),
                    bytes_per_ns_each_direction,
                    fixed_latency_ns_per_link,
                )
                for node_id in range(compute_node_count)
            ],
        )

    @staticmethod
    def create_nonblocking_central_switch_star(
        compute_node_count: int,
        bytes_per_ns_each_direction_per_node_link: float,
        fixed_latency_ns_per_link: float,
    ) -> RoutedNetworkTopology:
        switch_endpoint = "switch:0"
        return RoutedNetworkTopology(
            compute_node_count,
            [
                BidirectionalNetworkLink(
                    f"switch_link_{node_id}",
                    RoutedNetworkTopology.compute_node_endpoint(node_id),
                    switch_endpoint,
                    bytes_per_ns_each_direction_per_node_link,
                    fixed_latency_ns_per_link,
                )
                for node_id in range(compute_node_count)
            ],
        )
