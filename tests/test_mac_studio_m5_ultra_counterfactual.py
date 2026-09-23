import csv
import json

import pytest

from run_mac_studio_m5_ultra_counterfactual import (
    AGENT_COUNT,
    KEYED_ROUTING,
    CONTEXT_TOKENS,
    KV_RESIDENT_FRACTION,
    NODE_COUNT,
    PROFILE_NAME,
    SEED,
    VARIANTS,
    average_idle_reasons,
    calculate_route_prefix_digest,
    kimi_k3_profile,
    run_sweep,
    write_results,
)


def test_variants_change_one_hardware_dimension_from_baseline():
    baseline = VARIANTS[0]
    control = VARIANTS[1]
    assert control == type(baseline)("control_x1")
    assert [variant.name for variant in VARIANTS] == [
        "baseline", "control_x1", "compute_x2", "hbm_bandwidth_x2",
        "network_bandwidth_x2", "network_latency_div2",
    ]
    for variant in VARIANTS[2:]:
        changed = sum(
            getattr(variant, name) != getattr(baseline, name)
            for name in (
                "compute_tflops_per_node", "memory_gbps_per_node",
                "network_gbps_each_direction", "network_latency_ns",
            )
        )
        assert changed == 1
    assert VARIANTS[2].compute_tflops_per_node == baseline.compute_tflops_per_node * 2
    assert VARIANTS[3].memory_gbps_per_node == baseline.memory_gbps_per_node * 2
    assert VARIANTS[4].network_gbps_each_direction == baseline.network_gbps_each_direction * 2
    assert VARIANTS[5].network_latency_ns == baseline.network_latency_ns / 2


def test_idle_reason_aggregation_is_stable_json():
    reasons = {
        "node[0]": {"compute_productive": 1.0, "memory_productive": 0.5},
        "node[1]": {"compute_productive": 0.5, "memory_productive": 1.0},
    }
    assert json.loads(average_idle_reasons(reasons, "compute_")) == {"compute_productive": 0.75}


def test_ten_token_snapshot_reproduces_baseline_control_and_writes_schema(tmp_path):
    results = run_sweep(
        tokens_per_agent=10,
        commit="test-commit",
        routing_randomness="legacy_stream",
    )
    assert len(results) == 6
    baseline, control = results[:2]
    assert baseline.completed_token_count >= AGENT_COUNT * 10
    assert baseline.run_status == control.run_status == "completed_target"
    assert baseline.tokens_per_second == control.tokens_per_second == 1430.9811369323036
    assert baseline.p95_token_latency_ms == control.p95_token_latency_ms == 539.6001001859472
    assert control.delta_throughput_vs_baseline == 0.0
    assert control.delta_p95_vs_baseline == 0.0
    write_results(tmp_path, results, routing_randomness=KEYED_ROUTING)
    with (tmp_path / "mac_studio_m5_ultra_counterfactual.csv").open(newline="", encoding="utf-8") as output:
        rows = list(csv.DictReader(output))
    assert len(rows) == 6
    assert set(rows[0]) == set(results[0].__dataclass_fields__)
    assert json.loads(rows[0]["compute_idle_reasons"])
    metadata = json.loads(
        (tmp_path / "mac_studio_m5_ultra_counterfactual_metadata.json").read_text(
            encoding="utf-8"
        )
    )
    assert metadata["implementation_commit"] == "test-commit"
    assert metadata["workload_control"]["routing_randomness"] == KEYED_ROUTING
    assert metadata["workload_control"]["route_identity_check"] == "passed"
    assert metadata["scenario"] == {
        "profile": PROFILE_NAME,
        "node_count": NODE_COUNT,
        "agent_count": AGENT_COUNT,
        "context_tokens": CONTEXT_TOKENS,
        "kv_resident_fraction": KV_RESIDENT_FRACTION,
        "seed": SEED,
        "tokens_per_agent": 10,
        "warmup": "exclude the first completed token per agent",
        "stop_rule": "target completed tokens or 10 simulated seconds",
    }


def test_keyed_route_prefix_is_identical_across_hardware_variants():
    profile = kimi_k3_profile()
    digests = {
        calculate_route_prefix_digest(profile, variant, KEYED_ROUTING)
        for variant in VARIANTS
    }
    assert len(digests) == 1


def test_sweep_rejects_nonpositive_token_target():
    with pytest.raises(ValueError, match="tokens_per_agent"):
        run_sweep(tokens_per_agent=0, commit="test-commit")
