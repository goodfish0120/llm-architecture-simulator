"""Optional, append-only semantic trace for replaying simulator executions.

The recorder deliberately observes the existing simulation; it does not schedule
events, mutate resources, or consume routing randomness.
"""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any


TRACE_SCHEMA_VERSION = 1


class DiagnosticTrace:
    def __init__(self, configuration: Any, source_revision: str | None = None) -> None:
        self.data: dict[str, Any] = {
            "schema_version": TRACE_SCHEMA_VERSION,
            "capture_complete": False,
            "manifest": {
                "simulator": "llm-architecture-simulator",
                "simulator_version": "0.1.0",
                "source_revision": source_revision or "unrecorded",
                "trace_semantics": "event-engine replay; not measured hardware telemetry",
                "configuration": asdict(configuration),
                "assumptions": [
                    "Resource busy/productive are model-defined reservations, not GPU utilization or MFU.",
                    "Compute productive duration includes startup and the modeled batch-efficiency multiplier.",
                    "KV is represented by costs/capacity only; this trace does not claim physical KV page movement.",
                    "Shared stage exists on every layer; dense-only means zero routed experts, not no shared stage.",
                ],
            },
            "events": [], "tokens": [], "joins": [], "routes": [], "batches": [], "transfers": [], "resources": [],
        }

    @staticmethod
    def units(units: Any) -> list[dict[str, Any]]:
        return [{
            "token_id": unit.globally_unique_token_id,
            "agent_id": unit.workload_agent_id,
            "join_id": unit.dependency_join_id,
            "branch_index": unit.branch_index_inside_dependency_join,
            "sequence_owner_node_id": unit.node_that_owns_sequence_state,
            "rendezvous_node_id": unit.expert_result_rendezvous_node_id,
        } for unit in units]

    def record_event(self, event: Any) -> None:
        particle = event.work_particle
        self.data["events"].append({
            "time_ns": event.scheduled_time_ns, "insertion_order": event.insertion_order,
            "type": event.event_type, "layer": particle.model_layer_index,
            "node_id": particle.current_node_id, "route_name": particle.route_name,
            "units": self.units(particle.logical_work_units),
        })

    def record_token(self, kind: str, time_ns: float, token_id: int, agent_id: int, node_id: int, token_ordinal: int) -> None:
        self.data["tokens"].append({"kind": kind, "time_ns": time_ns, "token_id": token_id, "agent_id": agent_id, "node_id": node_id, "token_ordinal": token_ordinal})

    def record_join(self, kind: str, time_ns: float, join_id: str, expected_branch_count: int, units: Any) -> None:
        self.data["joins"].append({"kind": kind, "time_ns": time_ns, "join_id": join_id, "expected_branch_count": expected_branch_count, "units": self.units(units)})

    def record_route(self, time_ns: float, layer: int, source_node_id: int, expert_index: int, destination_node_id: int, rendezvous_node_id: int, units: Any) -> None:
        self.data["routes"].append({"time_ns": time_ns, "layer": layer, "source_node_id": source_node_id, "expert_index": expert_index, "destination_node_id": destination_node_id, "rendezvous_node_id": rendezvous_node_id, "units": self.units(units)})

    def record_batch(self, kind: str, ready_time_ns: float, service_start_time_ns: float, completion_time_ns: float, layer: int, node_id: int, queue_depth: int, units: Any, expert_index: int | None = None) -> None:
        self.data["batches"].append({"kind": kind, "ready_time_ns": ready_time_ns, "time_ns": service_start_time_ns, "completion_time_ns": completion_time_ns, "layer": layer, "node_id": node_id, "expert_index": expert_index, "queue_depth_before_execution": queue_depth, "units": self.units(units)})

    def record_transfer(self, kind: str, ready_time_ns: float, start_time_ns: float, arrival_time_ns: float, source_node_id: int, destination_node_id: int, layer: int, units: Any, resource_names: tuple[str, ...] = ()) -> None:
        self.data["transfers"].append({"kind": kind, "ready_time_ns": ready_time_ns, "start_time_ns": start_time_ns, "arrival_time_ns": arrival_time_ns, "source_node_id": source_node_id, "destination_node_id": destination_node_id, "layer": layer, "resource_names": list(resource_names), "units": self.units(units)})

    def finalize(self, simulator: Any) -> None:
        for node_id, node in enumerate(simulator.compute_and_memory_node_by_id):
            for kind, resource in (("compute", node.compute_resource), ("memory", node.memory_resource)):
                for start_ns, end_ns in resource.reserved_busy_intervals_ns:
                    self.data["resources"].append({"resource": f"{kind}[{node_id}]", "start_ns": start_ns, "end_ns": end_ns, "kind": "reserved"})
                for start_ns, end_ns in resource.productive_work_intervals_ns:
                    self.data["resources"].append({"resource": f"{kind}[{node_id}]", "start_ns": start_ns, "end_ns": end_ns, "kind": "productive"})
        for link in simulator.network_topology.links:
            for resource in (link.a_to_b, link.b_to_a):
                for start_ns, end_ns in resource.reserved_busy_intervals_ns:
                    self.data["resources"].append({"resource": resource.resource_name, "start_ns": start_ns, "end_ns": end_ns, "kind": "reserved"})
        self.data["capture_complete"] = True

    def write_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.data, indent=2), encoding="utf-8")
