import unittest

from src.llm_architecture_simulator.simulation_configuration import (
    StochasticMoeSimulationConfiguration,
    TransformerLayerConfiguration,
)
from src.llm_architecture_simulator.stochastic_moe_simulation import (
    StochasticMoeArchitectureSimulator,
)


class SimulationSmokeTest(unittest.TestCase):
    def test_small_simulation_completes_tokens_and_observes_expert_batches(self) -> None:
        simulator = StochasticMoeArchitectureSimulator(
            StochasticMoeSimulationConfiguration(
                node_count=2,
                continuously_active_agent_count=16,
                model_layers=(
                    TransformerLayerConfiguration(
                        routed_expert_count=4,
                        selected_expert_count_per_token=2,
                    ),
                    TransformerLayerConfiguration(
                        routed_expert_count=4,
                        selected_expert_count_per_token=2,
                    ),
                ),
                expert_batching_window_ns=100_000.0,
                maximum_expert_batch_size=16,
            )
        )
        observer = simulator.run_for_simulated_nanoseconds(20_000_000.0)

        self.assertTrue(observer.completed_token_times_ns)
        self.assertTrue(observer.executed_expert_batch_sizes)
        self.assertGreater(simulator.next_globally_unique_token_id, 16)

    def test_dense_and_different_sized_moe_layers_can_be_mixed(self) -> None:
        simulator = StochasticMoeArchitectureSimulator(
            StochasticMoeSimulationConfiguration(
                node_count=2,
                continuously_active_agent_count=8,
                model_layers=(
                    TransformerLayerConfiguration(),
                    TransformerLayerConfiguration(
                        routed_expert_count=4,
                        selected_expert_count_per_token=2,
                    ),
                    TransformerLayerConfiguration(
                        routed_expert_count=6,
                        selected_expert_count_per_token=3,
                    ),
                ),
            )
        )
        observer = simulator.run_for_simulated_nanoseconds(20_000_000.0)

        self.assertTrue(observer.completed_token_times_ns)
        self.assertEqual(
            len(simulator.expert_router_by_layer_index[1].relative_expert_popularity),
            4,
        )
        self.assertEqual(
            len(simulator.expert_router_by_layer_index[2].relative_expert_popularity),
            6,
        )

    def test_largest_local_expert_group_rendezvous_policy_completes_tokens(self) -> None:
        simulator = StochasticMoeArchitectureSimulator(
            StochasticMoeSimulationConfiguration(
                node_count=3,
                continuously_active_agent_count=12,
                model_layers=(
                    TransformerLayerConfiguration(
                        routed_expert_count=9,
                        selected_expert_count_per_token=4,
                    ),
                ),
                expert_result_rendezvous_policy="largest_local_expert_group",
            )
        )
        observer = simulator.run_for_simulated_nanoseconds(20_000_000.0)

        self.assertTrue(observer.completed_token_times_ns)

    def test_repeated_runs_advance_by_relative_duration(self) -> None:
        simulator = StochasticMoeArchitectureSimulator(
            StochasticMoeSimulationConfiguration(
                node_count=2,
                continuously_active_agent_count=8,
                model_layers=(
                    TransformerLayerConfiguration(
                        routed_expert_count=4,
                        selected_expert_count_per_token=2,
                    ),
                    TransformerLayerConfiguration(
                        routed_expert_count=4,
                        selected_expert_count_per_token=2,
                    ),
                ),
            )
        )

        simulator.run_for_simulated_nanoseconds(5_000_000.0)
        first_end_time_ns = simulator.event_engine.current_time_ns
        simulator.run_for_simulated_nanoseconds(5_000_000.0)

        self.assertEqual(first_end_time_ns, 5_000_000.0)
        self.assertEqual(simulator.event_engine.current_time_ns, 10_000_000.0)

    def test_resource_idle_reason_report_is_available_after_simulation(self) -> None:
        simulator = StochasticMoeArchitectureSimulator(
            StochasticMoeSimulationConfiguration(
                node_count=1,
                continuously_active_agent_count=4,
                model_layers=(TransformerLayerConfiguration(),),
            )
        )
        simulator.run_for_simulated_nanoseconds(2_000_000.0)

        idle_reasons = simulator.calculate_compute_and_memory_idle_reasons_between_times(
            measurement_start_time_ns=0.0,
            measurement_end_time_ns=2_000_000.0,
        )

        self.assertIn("node[0]", idle_reasons)
        self.assertIn("compute_productive", idle_reasons["node[0]"])
        self.assertIn("compute_idle_without_reserved_kernel", idle_reasons["node[0]"])


if __name__ == "__main__":
    unittest.main()
