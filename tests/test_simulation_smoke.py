from src.llm_architecture_simulator.stochastic_moe_simulation import (
    StochasticMoeArchitectureSimulator,
    StochasticMoeSimulationConfiguration,
)


def test_small_simulation_completes_tokens_and_observes_expert_batches() -> None:
    simulator = StochasticMoeArchitectureSimulator(
        StochasticMoeSimulationConfiguration(
            node_count=2,
            continuously_active_agent_count=16,
            moe_layer_count=2,
            routed_expert_count_per_layer=4,
            selected_expert_count_per_token=2,
            expert_batching_window_ns=100_000.0,
            maximum_expert_batch_size=16,
        )
    )
    observer = simulator.run_for_simulated_nanoseconds(20_000_000.0)

    assert observer.completed_token_times_ns
    assert observer.executed_expert_batch_sizes
    assert simulator.next_globally_unique_token_id > 16
