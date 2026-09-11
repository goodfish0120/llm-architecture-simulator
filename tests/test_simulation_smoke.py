import unittest

from src.llm_architecture_simulator.stochastic_moe_simulation import (
    StochasticMoeArchitectureSimulator,
    StochasticMoeSimulationConfiguration,
)


class SimulationSmokeTest(unittest.TestCase):
    def test_small_simulation_completes_tokens_and_observes_expert_batches(self) -> None:
        simulator = StochasticMoeArchitectureSimulator(
            StochasticMoeSimulationConfiguration(
                node_count=2,
                continuously_active_agent_count=16,
                model_layer_count=2,
                routed_expert_count_per_moe_layer=4,
                selected_expert_count_per_token=2,
                expert_batching_window_ns=100_000.0,
                maximum_expert_batch_size=16,
            )
        )
        observer = simulator.run_for_simulated_nanoseconds(20_000_000.0)

        self.assertTrue(observer.completed_token_times_ns)
        self.assertTrue(observer.executed_expert_batch_sizes)
        self.assertGreater(simulator.next_globally_unique_token_id, 16)

    def test_dense_and_moe_layers_can_be_mixed(self) -> None:
        simulator = StochasticMoeArchitectureSimulator(
            StochasticMoeSimulationConfiguration(
                node_count=2,
                continuously_active_agent_count=8,
                model_layer_count=3,
                moe_layer_indexes=(1,),
                routed_expert_count_per_moe_layer=4,
                selected_expert_count_per_token=2,
            )
        )
        observer = simulator.run_for_simulated_nanoseconds(20_000_000.0)

        self.assertTrue(observer.completed_token_times_ns)
        self.assertTrue(observer.executed_expert_batch_sizes)

    def test_repeated_runs_advance_by_relative_duration(self) -> None:
        simulator = StochasticMoeArchitectureSimulator(
            StochasticMoeSimulationConfiguration(
                node_count=2,
                continuously_active_agent_count=8,
                model_layer_count=2,
                routed_expert_count_per_moe_layer=4,
                selected_expert_count_per_token=2,
            )
        )

        simulator.run_for_simulated_nanoseconds(5_000_000.0)
        first_end_time_ns = simulator.event_engine.current_time_ns
        simulator.run_for_simulated_nanoseconds(5_000_000.0)

        self.assertEqual(first_end_time_ns, 5_000_000.0)
        self.assertEqual(simulator.event_engine.current_time_ns, 10_000_000.0)


if __name__ == "__main__":
    unittest.main()
