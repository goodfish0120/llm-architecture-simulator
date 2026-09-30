"""Generate a small, heterogeneous trace for the local diagnostic viewer."""
import argparse
from pathlib import Path
import subprocess

from src.llm_architecture_simulator.diagnostic_trace import DiagnosticTrace
from src.llm_architecture_simulator.simulation_configuration import (
    StochasticMoeSimulationConfiguration,
    TransformerLayerConfiguration,
)
from src.llm_architecture_simulator.stochastic_moe_simulation import (
    StochasticMoeArchitectureSimulator,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/diagnostic_demo.json")
    arguments = parser.parse_args()
    # Synthetic demonstrator: dense-only L0, MoE L1, dense-only L2, MoE L3.
    configuration = StochasticMoeSimulationConfiguration(
        node_count=3,
        continuously_active_agent_count=12,
        model_layers=(
            TransformerLayerConfiguration(),
            TransformerLayerConfiguration(routed_expert_count=6, selected_expert_count_per_token=2),
            TransformerLayerConfiguration(),
            TransformerLayerConfiguration(routed_expert_count=4, selected_expert_count_per_token=2),
        ),
        maximum_shared_layer_batch_size=6,
        maximum_expert_batch_size=4,
        shared_layer_batching_window_ns=20_000.0,
        expert_batching_window_ns=60_000.0,
        network_topology_kind="daisy_chain",
        random_seed=11,
    )
    try:
        source_revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        source_revision = None
    trace = DiagnosticTrace(configuration, source_revision=source_revision)
    simulator = StochasticMoeArchitectureSimulator(configuration, diagnostic_trace=trace)
    simulator.run_for_simulated_nanoseconds(8_000_000.0)
    trace.finalize(simulator)
    output = Path(arguments.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    trace.write_json(output)
    print(f"wrote {output} ({len(trace.data['events'])} observed events)")


if __name__ == "__main__":
    main()
