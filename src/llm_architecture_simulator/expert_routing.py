from __future__ import annotations

import math
import random
from hashlib import sha256
from dataclasses import dataclass
from typing import Sequence


@dataclass
class WeightedTopKExpertRouter:
    relative_expert_popularity: Sequence[float]
    selected_expert_count_per_token: int
    random_seed: int = 0

    def __post_init__(self) -> None:
        if not 1 <= self.selected_expert_count_per_token <= len(self.relative_expert_popularity):
            raise ValueError("invalid selected_expert_count_per_token")
        if any(weight < 0 for weight in self.relative_expert_popularity):
            raise ValueError("expert popularity weights must be non-negative")
        if not any(weight > 0 for weight in self.relative_expert_popularity):
            raise ValueError("expert popularity weights need positive mass")
        self.random_number_generator = random.Random(self.random_seed)

    def sample_distinct_expert_indexes_for_one_token(self) -> tuple[int, ...]:
        return self._sample_distinct_expert_indexes(self.random_number_generator)

    def sample_distinct_expert_indexes_for_logical_step(
        self,
        workload_agent_id: int,
        workload_token_ordinal: int,
        model_layer_index: int,
        base_random_seed: int,
    ) -> tuple[int, ...]:
        """Sample without advancing the legacy mutable RNG stream.

        The SHA-256 seed derivation intentionally avoids Python's randomized
        ``hash()`` and makes a logical token's route independent of event order.
        """
        key = (
            f"{base_random_seed}:{workload_agent_id}:"
            f"{workload_token_ordinal}:{model_layer_index}"
        ).encode("ascii")
        local_seed = int.from_bytes(sha256(key).digest()[:16], "big")
        return self._sample_distinct_expert_indexes(random.Random(local_seed))

    def _sample_distinct_expert_indexes(
        self, random_number_generator: random.Random
    ) -> tuple[int, ...]:
        weighted_sampling_keys: list[tuple[float, int]] = []
        for expert_index, popularity_weight in enumerate(self.relative_expert_popularity):
            if popularity_weight <= 0:
                continue
            random_uniform_value = max(random_number_generator.random(), 1e-15)
            weighted_sampling_keys.append(
                (-math.log(random_uniform_value) / popularity_weight, expert_index)
            )
        weighted_sampling_keys.sort()
        return tuple(
            expert_index
            for _, expert_index in weighted_sampling_keys[
                : self.selected_expert_count_per_token
            ]
        )


def generate_normalized_gaussian_expert_popularity(
    expert_count: int,
    popularity_sigma: float,
    random_seed: int = 0,
) -> tuple[float, ...]:
    random_number_generator = random.Random(random_seed)
    unnormalized_popularity = [
        max(
            1e-6,
            1.0 + popularity_sigma * random_number_generator.gauss(0.0, 1.0),
        )
        for _ in range(expert_count)
    ]
    total_popularity = sum(unnormalized_popularity)
    return tuple(
        popularity / total_popularity
        for popularity in unnormalized_popularity
    )
