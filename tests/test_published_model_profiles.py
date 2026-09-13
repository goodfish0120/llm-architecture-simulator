import unittest

from src.llm_architecture_simulator.published_model_profiles import (
    BYTES_PER_GB,
    PUBLISHED_MODEL_PROFILES,
)


class PublishedModelProfileTest(unittest.TestCase):
    def test_capacity_thresholds_reserve_ten_percent_of_memory(self) -> None:
        usable_memory_bytes_per_node = int(512 * BYTES_PER_GB * 0.90)
        minimum_nodes = {
            profile.name: profile.minimum_node_count(usable_memory_bytes_per_node)
            for profile in PUBLISHED_MODEL_PROFILES
        }

        self.assertEqual(minimum_nodes["Kimi K3"], 4)
        self.assertEqual(minimum_nodes["DeepSeek V4.1 Flash"], 2)
        self.assertEqual(minimum_nodes["Qwen3.8 Flash Next"], 1)

    def test_profiles_create_valid_full_mesh_simulation_configurations(self) -> None:
        for profile in PUBLISHED_MODEL_PROFILES:
            configuration = profile.create_simulation_configuration(
                node_count=2,
                continuously_active_agent_count=8,
                compute_peak_tflops_per_node=100.0,
                memory_bandwidth_gb_per_second_per_node=1200.0,
                network_link_bandwidth_gb_per_second_each_direction=7.5,
                network_link_fixed_latency_ns=50_000.0,
                random_seed=7,
            )

            configuration.validate()
            self.assertEqual(configuration.network_topology_kind, "full_mesh")
            self.assertEqual(configuration.model_layer_count, profile.layer_count)

            collapsed_configuration = profile.create_simulation_configuration(
                node_count=2,
                continuously_active_agent_count=8,
                compute_peak_tflops_per_node=100.0,
                memory_bandwidth_gb_per_second_per_node=1200.0,
                network_link_bandwidth_gb_per_second_each_direction=7.5,
                network_link_fixed_latency_ns=50_000.0,
                random_seed=7,
                collapse_repeated_layers=True,
                context_token_count=100_000,
                kv_resident_fraction=0.80,
            )
            collapsed_configuration.validate()
            self.assertEqual(collapsed_configuration.model_layer_count, 1)
            self.assertGreater(
                collapsed_configuration.shared_layer_activation_bytes_per_token,
                profile.approximate_kv_cache_bytes_per_context_token * 79_000,
            )


if __name__ == "__main__":
    unittest.main()
