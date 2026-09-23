from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class TransformerLayerConfiguration:
    routed_expert_count: int = 0
    selected_expert_count_per_token: int = 0

    @property
    def uses_routed_experts(self) -> bool:
        return self.routed_expert_count > 0

    def validate(self, layer_index: int) -> None:
        if self.routed_expert_count < 0:
            raise ValueError(f"layer {layer_index} routed_expert_count cannot be negative")
        if self.routed_expert_count == 0:
            if self.selected_expert_count_per_token != 0:
                raise ValueError(
                    f"layer {layer_index} dense layer must select zero routed experts"
                )
            return
        if not 1 <= self.selected_expert_count_per_token <= self.routed_expert_count:
            raise ValueError(
                f"layer {layer_index} selected_expert_count_per_token is invalid"
            )


def create_default_model_layers() -> tuple[TransformerLayerConfiguration, ...]:
    return tuple(
        TransformerLayerConfiguration(
            routed_expert_count=16,
            selected_expert_count_per_token=2,
        )
        for _ in range(4)
    )


@dataclass
class StochasticMoeSimulationConfiguration:
    node_count: int = 4
    continuously_active_agent_count: int = 128
    model_layers: tuple[TransformerLayerConfiguration, ...] = field(
        default_factory=create_default_model_layers
    )
    random_seed: int = 7
    routing_randomness: str = "legacy_stream"
    maximum_completed_tokens_per_agent: int | None = None

    shared_layer_batching_window_ns: float = 100_000.0
    maximum_shared_layer_batch_size: int = 128
    expert_batching_window_ns: float = 1_000_000.0
    maximum_expert_batch_size: int = 128

    compute_peak_tflops_per_node: float = 100.0
    memory_bandwidth_gb_per_second_per_node: float = 500.0
    network_link_bandwidth_gb_per_second_each_direction: float = 20.0
    network_link_fixed_latency_ns: float = 2_000.0
    network_topology_kind: str = "switch_star"

    shared_layer_flops_per_token: float = 10e6
    shared_layer_weight_bytes_read_per_batch: float = 0.4e6
    shared_layer_activation_bytes_per_token: float = 0.4e6
    expert_flops_per_token: float = 30e6
    expert_weight_bytes_read_per_batch: float = 4e6
    activation_bytes_transferred_per_expert_branch: float = 16_384.0
    expert_popularity_sigma: float = 0.8

    expert_result_rendezvous_policy: str = "sequence_owner"

    @property
    def model_layer_count(self) -> int:
        return len(self.model_layers)

    @property
    def moe_layer_indexes(self) -> tuple[int, ...]:
        return tuple(
            layer_index
            for layer_index, layer_configuration in enumerate(self.model_layers)
            if layer_configuration.uses_routed_experts
        )

    def validate(self) -> None:
        if self.node_count < 1:
            raise ValueError("node_count must be positive")
        if self.continuously_active_agent_count < 1:
            raise ValueError("continuously_active_agent_count must be positive")
        if not self.model_layers:
            raise ValueError("model_layers must contain at least one layer")
        if self.shared_layer_batching_window_ns < 0:
            raise ValueError("shared_layer_batching_window_ns cannot be negative")
        if self.expert_batching_window_ns < 0:
            raise ValueError("expert_batching_window_ns cannot be negative")
        if self.maximum_shared_layer_batch_size < 1:
            raise ValueError("maximum_shared_layer_batch_size must be positive")
        if self.maximum_expert_batch_size < 1:
            raise ValueError("maximum_expert_batch_size must be positive")
        if self.compute_peak_tflops_per_node <= 0:
            raise ValueError("compute_peak_tflops_per_node must be positive")
        if self.memory_bandwidth_gb_per_second_per_node <= 0:
            raise ValueError("memory_bandwidth_gb_per_second_per_node must be positive")
        if self.network_link_bandwidth_gb_per_second_each_direction <= 0:
            raise ValueError(
                "network_link_bandwidth_gb_per_second_each_direction must be positive"
            )
        if self.network_link_fixed_latency_ns < 0:
            raise ValueError("network_link_fixed_latency_ns cannot be negative")
        if self.shared_layer_flops_per_token < 0:
            raise ValueError("shared_layer_flops_per_token cannot be negative")
        if self.shared_layer_weight_bytes_read_per_batch < 0:
            raise ValueError(
                "shared_layer_weight_bytes_read_per_batch cannot be negative"
            )
        if self.shared_layer_activation_bytes_per_token < 0:
            raise ValueError("shared_layer_activation_bytes_per_token cannot be negative")
        if self.expert_flops_per_token < 0:
            raise ValueError("expert_flops_per_token cannot be negative")
        if self.expert_weight_bytes_read_per_batch < 0:
            raise ValueError("expert_weight_bytes_read_per_batch cannot be negative")
        if self.activation_bytes_transferred_per_expert_branch < 0:
            raise ValueError(
                "activation_bytes_transferred_per_expert_branch cannot be negative"
            )
        if self.expert_popularity_sigma < 0:
            raise ValueError("expert_popularity_sigma cannot be negative")
        if self.network_topology_kind not in {
            "switch_star",
            "full_mesh",
            "daisy_chain",
            "ring",
        }:
            raise ValueError(
                f"unknown network_topology_kind: {self.network_topology_kind}"
            )
        if self.expert_result_rendezvous_policy not in {
            "sequence_owner",
            "largest_local_expert_group",
        }:
            raise ValueError(
                "expert_result_rendezvous_policy must be sequence_owner "
                "or largest_local_expert_group"
            )
        if self.routing_randomness not in {"legacy_stream", "keyed_logical_work"}:
            raise ValueError(
                "routing_randomness must be legacy_stream or keyed_logical_work"
            )
        if (
            self.maximum_completed_tokens_per_agent is not None
            and self.maximum_completed_tokens_per_agent < 1
        ):
            raise ValueError("maximum_completed_tokens_per_agent must be positive")

        for layer_index, layer_configuration in enumerate(self.model_layers):
            layer_configuration.validate(layer_index)
