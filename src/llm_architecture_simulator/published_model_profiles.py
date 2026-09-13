from __future__ import annotations

from dataclasses import dataclass
from math import ceil

from .simulation_configuration import (
    StochasticMoeSimulationConfiguration,
    TransformerLayerConfiguration,
)


BYTES_PER_GB = 1_000_000_000


@dataclass(frozen=True)
class PublishedMoeModelProfile:
    name: str
    checkpoint_bytes: int
    transformer_parameter_count: float
    active_transformer_parameter_count_per_decode_token: float
    auxiliary_parameter_count: float
    layer_count: int
    dense_layer_count: int
    routed_expert_count: int
    selected_routed_expert_count_per_token: int
    hidden_size: int
    approximate_kv_cache_bytes_per_context_token: int
    checkpoint_format: str
    source_url: str

    @property
    def moe_layer_count(self) -> int:
        return self.layer_count - self.dense_layer_count

    @property
    def total_profiled_parameter_count(self) -> float:
        return self.transformer_parameter_count + self.auxiliary_parameter_count

    @property
    def approximate_checkpoint_bytes_per_profiled_parameter(self) -> float:
        return self.checkpoint_bytes / self.total_profiled_parameter_count

    def minimum_node_count(self, usable_memory_bytes_per_node: int) -> int:
        if usable_memory_bytes_per_node <= 0:
            raise ValueError("usable_memory_bytes_per_node must be positive")
        return ceil(self.checkpoint_bytes / usable_memory_bytes_per_node)

    def create_simulation_configuration(
        self,
        *,
        node_count: int,
        continuously_active_agent_count: int,
        compute_peak_tflops_per_node: float,
        memory_bandwidth_gb_per_second_per_node: float,
        network_link_bandwidth_gb_per_second_each_direction: float,
        network_link_fixed_latency_ns: float,
        random_seed: int,
        collapse_repeated_layers: bool = False,
        context_token_count: int = 0,
        kv_resident_fraction: float = 0.0,
    ) -> StochasticMoeSimulationConfiguration:
        routed_parameters_per_expert_across_all_moe_layers = (
            self.transformer_parameter_count
            - self.active_transformer_parameter_count_per_decode_token
        ) / (
            self.routed_expert_count
            - self.selected_routed_expert_count_per_token
        )
        shared_transformer_parameter_count = (
            self.active_transformer_parameter_count_per_decode_token
            - self.selected_routed_expert_count_per_token
            * routed_parameters_per_expert_across_all_moe_layers
        )
        if shared_transformer_parameter_count <= 0:
            raise ValueError(f"{self.name} profile implies non-positive shared parameters")

        bytes_per_parameter = self.approximate_checkpoint_bytes_per_profiled_parameter
        if context_token_count < 0:
            raise ValueError("context_token_count cannot be negative")
        if not 0.0 <= kv_resident_fraction <= 1.0:
            raise ValueError("kv_resident_fraction must be between zero and one")
        routed_parameters_per_expert_per_moe_layer = (
            routed_parameters_per_expert_across_all_moe_layers / self.moe_layer_count
        )
        if collapse_repeated_layers:
            shared_parameters_per_simulated_layer = shared_transformer_parameter_count
            routed_parameters_per_expert_per_simulated_layer = (
                routed_parameters_per_expert_across_all_moe_layers
            )
            model_layers = (
                TransformerLayerConfiguration(
                    routed_expert_count=self.routed_expert_count,
                    selected_expert_count_per_token=(
                        self.selected_routed_expert_count_per_token
                    ),
                ),
            )
            activation_and_latency_multiplier = self.moe_layer_count
            kv_bytes_read_per_token_per_simulated_layer = (
                self.approximate_kv_cache_bytes_per_context_token
                * context_token_count
                * kv_resident_fraction
            )
        else:
            shared_parameters_per_simulated_layer = (
                shared_transformer_parameter_count / self.layer_count
            )
            routed_parameters_per_expert_per_simulated_layer = (
                routed_parameters_per_expert_per_moe_layer
            )
            dense_layers = (TransformerLayerConfiguration(),) * self.dense_layer_count
            moe_layers = tuple(
                TransformerLayerConfiguration(
                    routed_expert_count=self.routed_expert_count,
                    selected_expert_count_per_token=(
                        self.selected_routed_expert_count_per_token
                    ),
                )
                for _ in range(self.moe_layer_count)
            )
            model_layers = dense_layers + moe_layers
            activation_and_latency_multiplier = 1
            kv_bytes_read_per_token_per_simulated_layer = (
                self.approximate_kv_cache_bytes_per_context_token
                * context_token_count
                * kv_resident_fraction
                / self.layer_count
            )

        return StochasticMoeSimulationConfiguration(
            node_count=node_count,
            continuously_active_agent_count=continuously_active_agent_count,
            model_layers=model_layers,
            random_seed=random_seed,
            shared_layer_batching_window_ns=100_000.0,
            maximum_shared_layer_batch_size=128,
            expert_batching_window_ns=250_000.0,
            maximum_expert_batch_size=128,
            compute_peak_tflops_per_node=compute_peak_tflops_per_node,
            memory_bandwidth_gb_per_second_per_node=(
                memory_bandwidth_gb_per_second_per_node
            ),
            network_link_bandwidth_gb_per_second_each_direction=(
                network_link_bandwidth_gb_per_second_each_direction
            ),
            network_link_fixed_latency_ns=(
                network_link_fixed_latency_ns * activation_and_latency_multiplier
            ),
            network_topology_kind="full_mesh",
            shared_layer_flops_per_token=(
                2.0 * shared_parameters_per_simulated_layer
            ),
            shared_layer_weight_bytes_read_per_batch=(
                shared_parameters_per_simulated_layer * bytes_per_parameter
            ),
            shared_layer_activation_bytes_per_token=(
                8.0 * self.hidden_size * 2.0
                + kv_bytes_read_per_token_per_simulated_layer
            ),
            expert_flops_per_token=(
                2.0 * routed_parameters_per_expert_per_simulated_layer
            ),
            expert_weight_bytes_read_per_batch=(
                routed_parameters_per_expert_per_simulated_layer * bytes_per_parameter
            ),
            activation_bytes_transferred_per_expert_branch=(
                self.hidden_size * 2.0 * activation_and_latency_multiplier
            ),
            expert_popularity_sigma=0.8,
            expert_result_rendezvous_policy="largest_local_expert_group",
        )


PUBLISHED_MODEL_PROFILES = (
    PublishedMoeModelProfile(
        name="Kimi K3",
        checkpoint_bytes=1_560_936_091_448,
        transformer_parameter_count=2.8e12,
        active_transformer_parameter_count_per_decode_token=104e9,
        auxiliary_parameter_count=0.401e9,
        layer_count=93,
        dense_layer_count=1,
        routed_expert_count=896,
        selected_routed_expert_count_per_token=16,
        hidden_size=7168,
        approximate_kv_cache_bytes_per_context_token=27_648,
        checkpoint_format="official mixed MXFP4/BF16 checkpoint",
        source_url="https://huggingface.co/moonshotai/Kimi-K3",
    ),
    PublishedMoeModelProfile(
        name="DeepSeek V4.1 Flash",
        checkpoint_bytes=510_296_708_312,
        transformer_parameter_count=552e9,
        active_transformer_parameter_count_per_decode_token=16e9,
        auxiliary_parameter_count=196e9,
        layer_count=40,
        dense_layer_count=0,
        routed_expert_count=384,
        selected_routed_expert_count_per_token=6,
        hidden_size=5120,
        approximate_kv_cache_bytes_per_context_token=40_960,
        checkpoint_format="official mixed FP8/FP4 checkpoint",
        source_url="https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash",
    ),
    PublishedMoeModelProfile(
        name="Qwen3.8 Flash Next",
        checkpoint_bytes=360_000_192_888,
        transformer_parameter_count=125e9,
        active_transformer_parameter_count_per_decode_token=6e9,
        auxiliary_parameter_count=55e9,
        layer_count=48,
        dense_layer_count=0,
        routed_expert_count=512,
        selected_routed_expert_count_per_token=10,
        hidden_size=2560,
        approximate_kv_cache_bytes_per_context_token=12_288,
        checkpoint_format="official BF16 checkpoint",
        source_url="https://huggingface.co/Qwen/Qwen3.8-Flash-Next",
    ),
)
