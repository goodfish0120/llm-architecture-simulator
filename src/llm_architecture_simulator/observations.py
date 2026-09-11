from __future__ import annotations

from statistics import mean


class SimulationObserver:
    def __init__(self) -> None:
        self.completed_token_times_ns: list[float] = []
        self.executed_expert_batch_sizes: list[int] = []
        self.remote_expert_branch_count = 0
        self.local_expert_branch_count = 0
        self.maximum_expert_queue_depth_in_logical_units = 0

    def record_completed_token(self, completion_time_ns: float) -> None:
        self.completed_token_times_ns.append(completion_time_ns)

    def record_expert_batch_execution(
        self,
        executed_batch_size: int,
        queue_depth_before_execution: int,
    ) -> None:
        self.executed_expert_batch_sizes.append(executed_batch_size)
        self.maximum_expert_queue_depth_in_logical_units = max(
            self.maximum_expert_queue_depth_in_logical_units,
            queue_depth_before_execution,
        )

    def record_expert_route(self, is_remote_route: bool, branch_count: int) -> None:
        if is_remote_route:
            self.remote_expert_branch_count += branch_count
        else:
            self.local_expert_branch_count += branch_count

    def calculate_tokens_per_second_for_fixed_time_windows(
        self,
        simulation_end_time_ns: float,
        measurement_window_ns: float,
        warmup_time_ns: float = 0.0,
    ) -> list[float]:
        if measurement_window_ns <= 0 or simulation_end_time_ns <= warmup_time_ns:
            return []

        window_count = int(
            (simulation_end_time_ns - warmup_time_ns) // measurement_window_ns
        )
        completed_tokens_per_window = [0] * window_count

        for completion_time_ns in self.completed_token_times_ns:
            if (
                warmup_time_ns
                <= completion_time_ns
                < warmup_time_ns + window_count * measurement_window_ns
            ):
                window_index = int(
                    (completion_time_ns - warmup_time_ns) // measurement_window_ns
                )
                completed_tokens_per_window[window_index] += 1

        measurement_window_seconds = measurement_window_ns / 1e9
        return [
            token_count / measurement_window_seconds
            for token_count in completed_tokens_per_window
        ]

    def summarize_steady_state_behavior(
        self,
        simulation_end_time_ns: float,
        measurement_window_ns: float,
        warmup_time_ns: float = 0.0,
    ) -> dict[str, float]:
        tokens_per_second = sorted(
            self.calculate_tokens_per_second_for_fixed_time_windows(
                simulation_end_time_ns,
                measurement_window_ns,
                warmup_time_ns,
            )
        )

        def percentile(fraction: float) -> float:
            if not tokens_per_second:
                return 0.0
            index = min(
                len(tokens_per_second) - 1,
                round((len(tokens_per_second) - 1) * fraction),
            )
            return tokens_per_second[index]

        total_expert_branch_count = (
            self.remote_expert_branch_count + self.local_expert_branch_count
        )

        return {
            "measurement_window_count": float(len(tokens_per_second)),
            "mean_tokens_per_second": mean(tokens_per_second) if tokens_per_second else 0.0,
            "p10_tokens_per_second": percentile(0.10),
            "p50_tokens_per_second": percentile(0.50),
            "p90_tokens_per_second": percentile(0.90),
            "mean_expert_batch_size": (
                mean(self.executed_expert_batch_sizes)
                if self.executed_expert_batch_sizes
                else 0.0
            ),
            "remote_expert_route_fraction": (
                self.remote_expert_branch_count / total_expert_branch_count
                if total_expert_branch_count
                else 0.0
            ),
            "maximum_expert_queue_depth_in_logical_units": float(
                self.maximum_expert_queue_depth_in_logical_units
            ),
        }
