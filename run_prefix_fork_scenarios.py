from __future__ import annotations

import csv
import math
from dataclasses import asdict, dataclass
from pathlib import Path

from run_mac_studio_m5_ultra_scaling import (
    ConcurrencyObservation,
    create_logarithmic_agent_count_grid,
    simulate_one_concurrency,
)
from src.llm_architecture_simulator.published_model_profiles import (
    BYTES_PER_GB,
    PUBLISHED_MODEL_PROFILES,
    PublishedMoeModelProfile,
)


@dataclass(frozen=True)
class PrefixForkScenario:
    name: str
    system_prefix_duplicate_fraction: float
    family_context_duplicate_fraction: float
    family_size: int = 32
    system_prefix_tokens: int = 20_000
    family_context_tokens: int = 40_000
    unique_suffix_tokens: int = 40_000
    maximum_shared_query_batch: int = 128

    def effective_context_tokens_per_agent(self, agent_count: int) -> float:
        system_group = min(agent_count, self.maximum_shared_query_batch)
        family_group = min(agent_count, self.family_size)
        system = self.system_prefix_tokens * (
            1.0 - self.system_prefix_duplicate_fraction
            + self.system_prefix_duplicate_fraction / system_group
        )
        family = self.family_context_tokens * (
            1.0 - self.family_context_duplicate_fraction
            + self.family_context_duplicate_fraction / family_group
        )
        return system + family + self.unique_suffix_tokens


@dataclass(frozen=True)
class PrefixForkObservation:
    scenario: str
    system_prefix_duplicate_fraction: float
    family_context_duplicate_fraction: float
    effective_context_tokens_per_agent: float
    model: str
    node_count: int
    agent_count: int
    tokens_per_second: float
    mean_token_latency_ms: float
    p95_token_latency_ms: float
    mean_expert_batch_size: float
    maximum_network_link_busy_fraction: float


SCENARIOS = (
    PrefixForkScenario("independent", 0.0, 0.0),
    PrefixForkScenario("system-90-family-70", 0.90, 0.70),
    PrefixForkScenario("system-95-family-70", 0.95, 0.70),
    PrefixForkScenario("system-99-family-70", 0.99, 0.70),
)


def memory_capacity_agent_limit(
    profile: PublishedMoeModelProfile,
    scenario: PrefixForkScenario,
    node_count: int,
    kv_resident_fraction: float,
) -> int:
    usable_bytes = node_count * int(512 * BYTES_PER_GB * 0.90)

    def fits(agent_count: int) -> bool:
        kv_bytes = (
            agent_count
            * scenario.effective_context_tokens_per_agent(agent_count)
            * profile.approximate_kv_cache_bytes_per_context_token
            * kv_resident_fraction
        )
        return profile.checkpoint_bytes + kv_bytes <= usable_bytes

    lower, upper = 0, 1
    while fits(upper) and upper < 100_000:
        lower, upper = upper, upper * 2
    while lower + 1 < upper:
        middle = (lower + upper) // 2
        if fits(middle):
            lower = middle
        else:
            upper = middle
    return lower


def load_independent_baseline(path: Path, model: str) -> list[PrefixForkObservation]:
    rows = []
    with path.open(encoding="utf-8") as file:
        for row in csv.DictReader(file):
            if row["model"] != model or int(row["node_count"]) != 7:
                continue
            rows.append(
                PrefixForkObservation(
                    scenario="independent",
                    system_prefix_duplicate_fraction=0.0,
                    family_context_duplicate_fraction=0.0,
                    effective_context_tokens_per_agent=100_000.0,
                    model=model,
                    node_count=7,
                    agent_count=int(row["agent_count"]),
                    tokens_per_second=float(row["tokens_per_second"]),
                    mean_token_latency_ms=float(row["mean_token_latency_ms"]),
                    p95_token_latency_ms=float(row["p95_token_latency_ms"]),
                    mean_expert_batch_size=float(row["mean_expert_batch_size"]),
                    maximum_network_link_busy_fraction=float(
                        row["maximum_network_link_busy_fraction"]
                    ),
                )
            )
    return rows


def simulate_scenario(
    profile: PublishedMoeModelProfile,
    scenario: PrefixForkScenario,
    node_count: int = 7,
) -> list[PrefixForkObservation]:
    capacity = memory_capacity_agent_limit(profile, scenario, node_count, 0.80)
    agent_counts = create_logarithmic_agent_count_grid(capacity, 6)
    rows = []
    for agent_count in agent_counts:
        effective_context = scenario.effective_context_tokens_per_agent(agent_count)
        observation: ConcurrencyObservation = simulate_one_concurrency(
            profile,
            node_count,
            agent_count,
            7,
            round(effective_context),
            0.80,
            tokens_per_agent=6,
        )
        row = PrefixForkObservation(
            scenario=scenario.name,
            system_prefix_duplicate_fraction=(
                scenario.system_prefix_duplicate_fraction
            ),
            family_context_duplicate_fraction=(
                scenario.family_context_duplicate_fraction
            ),
            effective_context_tokens_per_agent=effective_context,
            model=profile.name,
            node_count=node_count,
            agent_count=agent_count,
            tokens_per_second=observation.tokens_per_second,
            mean_token_latency_ms=observation.mean_token_latency_ms,
            p95_token_latency_ms=observation.p95_token_latency_ms,
            mean_expert_batch_size=observation.mean_expert_batch_size,
            maximum_network_link_busy_fraction=(
                observation.maximum_network_link_busy_fraction
            ),
        )
        rows.append(row)
        print(
            f"{profile.name}, {scenario.name}: {agent_count} agents, "
            f"effective {effective_context:.0f} context -> "
            f"{observation.tokens_per_second:.1f} tok/s",
            flush=True,
        )
    return rows


def write_csv(path: Path, rows: list[PrefixForkObservation]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=asdict(rows[0]).keys())
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def main() -> None:
    output = Path("results/prefix_fork_scenarios.csv")
    baseline = Path("results/mac_studio_m5_ultra_concurrency_scan.csv")
    selected_profiles = [
        profile
        for profile in PUBLISHED_MODEL_PROFILES
        if profile.name in {"DeepSeek V4.1 Flash", "Qwen3.8 Flash Next"}
    ]
    rows = []
    for profile in selected_profiles:
        rows.extend(load_independent_baseline(baseline, profile.name))
        for scenario in SCENARIOS[1:]:
            rows.extend(simulate_scenario(profile, scenario))
    write_csv(output, rows)


if __name__ == "__main__":
    main()
