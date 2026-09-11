import unittest

from src.llm_architecture_simulator.network_topology import NetworkTopologyFactory


class NetworkTopologyTest(unittest.TestCase):
    def test_full_mesh_can_validate_compute_node_port_limits(self) -> None:
        topology = NetworkTopologyFactory.create_direct_full_mesh_between_compute_nodes(
            4,
            20.0,
            2_000.0,
        )

        with self.assertRaises(ValueError):
            topology.validate_compute_node_port_limits({0: 2})

        topology.validate_compute_node_port_limits({0: 3})

    def test_daisy_chain_transfer_uses_each_link_on_the_route(self) -> None:
        topology = NetworkTopologyFactory.create_bidirectional_daisy_chain_between_compute_nodes(
            3,
            20.0,
            2_000.0,
        )

        _, arrival_time_ns = topology.transfer_bytes_between_compute_nodes(
            0,
            2,
            0.0,
            20_000.0,
        )
        occupancy = topology.calculate_directional_link_busy_fractions_between_times(
            0.0,
            arrival_time_ns,
        )

        self.assertGreater(occupancy["chain_0_1:compute:0->compute:1"], 0.0)
        self.assertGreater(occupancy["chain_1_2:compute:1->compute:2"], 0.0)

    def test_nonblocking_switch_star_allows_disjoint_node_links_to_serialize_concurrently(self) -> None:
        topology = NetworkTopologyFactory.create_nonblocking_central_switch_star(
            4,
            20.0,
            2_000.0,
        )

        first_start_time_ns, _ = topology.transfer_bytes_between_compute_nodes(
            0,
            1,
            0.0,
            20_000.0,
        )
        second_start_time_ns, _ = topology.transfer_bytes_between_compute_nodes(
            2,
            3,
            0.0,
            20_000.0,
        )

        self.assertEqual(first_start_time_ns, 0.0)
        self.assertEqual(second_start_time_ns, 0.0)


if __name__ == "__main__":
    unittest.main()
