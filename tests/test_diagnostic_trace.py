import json
import tempfile
import unittest
from pathlib import Path

from src.llm_architecture_simulator.diagnostic_trace import DiagnosticTrace
from src.llm_architecture_simulator.simulation_configuration import (
    StochasticMoeSimulationConfiguration,
    TransformerLayerConfiguration,
)
from src.llm_architecture_simulator.stochastic_moe_simulation import (
    StochasticMoeArchitectureSimulator,
)


def configuration() -> StochasticMoeSimulationConfiguration:
    return StochasticMoeSimulationConfiguration(
        node_count=2,
        continuously_active_agent_count=8,
        model_layers=(
            TransformerLayerConfiguration(),
            TransformerLayerConfiguration(routed_expert_count=4, selected_expert_count_per_token=2),
        ),
        maximum_expert_batch_size=4,
        expert_batching_window_ns=50_000.0,
        network_topology_kind="full_mesh",
        random_seed=5,
    )


class DiagnosticTraceTest(unittest.TestCase):
    def test_trace_does_not_change_simulation_outcomes_and_preserves_identity(self) -> None:
        plain = StochasticMoeArchitectureSimulator(configuration())
        plain_observer = plain.run_for_simulated_nanoseconds(8_000_000.0)
        trace = DiagnosticTrace(configuration())
        traced = StochasticMoeArchitectureSimulator(configuration(), diagnostic_trace=trace)
        traced_observer = traced.run_for_simulated_nanoseconds(8_000_000.0)
        trace.finalize(traced)

        self.assertEqual(plain_observer.completed_token_latency_records, traced_observer.completed_token_latency_records)
        self.assertEqual(plain_observer.expert_batch_records, traced_observer.expert_batch_records)
        self.assertEqual(plain_observer.expert_route_records, traced_observer.expert_route_records)
        for plain_node, traced_node in zip(
            plain.compute_and_memory_node_by_id, traced.compute_and_memory_node_by_id
        ):
            self.assertEqual(
                plain_node.compute_resource.reserved_busy_intervals_ns,
                traced_node.compute_resource.reserved_busy_intervals_ns,
            )
            self.assertEqual(
                plain_node.memory_resource.productive_work_intervals_ns,
                traced_node.memory_resource.productive_work_intervals_ns,
            )
            self.assertEqual(plain_node.executed_kernels, traced_node.executed_kernels)
        self.assertEqual(
            plain.network_topology.calculate_directional_link_busy_fractions_between_times(0, 8_000_000.0),
            traced.network_topology.calculate_directional_link_busy_fractions_between_times(0, 8_000_000.0),
        )
        self.assertTrue(trace.data["capture_complete"])
        self.assertTrue(any(event["type"] == "shared_layer_work_arrived" for event in trace.data["events"]))
        self.assertTrue(any(batch["kind"] == "expert" for batch in trace.data["batches"]))
        self.assertTrue(any(route["units"][0]["join_id"] for route in trace.data["routes"]))
        self.assertTrue(any(route["source_node_id"] == route["destination_node_id"] for route in trace.data["routes"]))
        self.assertTrue(any(route["source_node_id"] != route["destination_node_id"] for route in trace.data["routes"]))
        self.assertTrue(any(transfer["source_node_id"] != transfer["destination_node_id"] for transfer in trace.data["transfers"]))
        same_time = [event for event in trace.data["events"] if event["time_ns"] == trace.data["events"][0]["time_ns"]]
        self.assertEqual([event["insertion_order"] for event in same_time], sorted(event["insertion_order"] for event in same_time))

    def test_trace_json_manifest_and_resource_records_are_replayable(self) -> None:
        trace = DiagnosticTrace(configuration())
        simulator = StochasticMoeArchitectureSimulator(configuration(), diagnostic_trace=trace)
        simulator.run_for_simulated_nanoseconds(4_000_000.0)
        trace.finalize(simulator)
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "trace.json"
            trace.write_json(destination)
            loaded = json.loads(destination.read_text(encoding="utf-8"))
        self.assertEqual(loaded["schema_version"], 1)
        self.assertTrue(loaded["resources"])
        self.assertIn("configuration", loaded["manifest"])
        self.assertIn("source_revision", loaded["manifest"])

    def test_trace_records_dense_and_moe_layers_without_collapsing_layer_order(self) -> None:
        trace = DiagnosticTrace(configuration())
        simulator = StochasticMoeArchitectureSimulator(configuration(), diagnostic_trace=trace)
        simulator.run_for_simulated_nanoseconds(4_000_000.0)
        trace.finalize(simulator)
        shared_layers = {batch["layer"] for batch in trace.data["batches"] if batch["kind"] == "shared"}
        expert_layers = {batch["layer"] for batch in trace.data["batches"] if batch["kind"] == "expert"}
        self.assertEqual(shared_layers, {0, 1})
        self.assertEqual(expert_layers, {1})
        self.assertEqual(trace.data["manifest"]["configuration"]["model_layers"][0]["routed_expert_count"], 0)

    def test_trace_is_deterministic_and_has_explicit_join_and_transfer_resources(self) -> None:
        traces = []
        for _ in range(2):
            trace = DiagnosticTrace(configuration(), source_revision="test")
            simulator = StochasticMoeArchitectureSimulator(configuration(), diagnostic_trace=trace)
            simulator.run_for_simulated_nanoseconds(4_000_000.0)
            trace.finalize(simulator)
            traces.append(trace.data)
        self.assertEqual(traces[0], traces[1])
        self.assertTrue(any(join["kind"] == "registered" for join in traces[0]["joins"]))
        self.assertTrue(any(join["kind"] == "released" for join in traces[0]["joins"]))
        self.assertTrue(all(transfer["resource_names"] for transfer in traces[0]["transfers"]))
        started = [token for token in traces[0]["tokens"] if token["kind"] == "started"]
        self.assertTrue(all("token_ordinal" in token for token in started))


if __name__ == "__main__":
    unittest.main()
