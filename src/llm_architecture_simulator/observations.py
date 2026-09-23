from __future__ import annotations

from statistics import mean


class SimulationObserver:
    def __init__(self) -> None:
        self.completed_token_times_ns: list[float] = []
        self.completed_token_latency_records: list[tuple[float, float]] = []
        self.token_start_time_ns_by_token_id: dict[int, float] = {}
        self.expert_batch_records: list[tuple[float, int, int]] = []
        self.expert_route_records: list[tuple[float, bool, int]] = []
        self.selected_expert_indexes_by_logical_identity: dict[
            tuple[int, int, int], tuple[int, ...]
        ] = {}
        self.selected_expert_route_count_by_logical_identity: dict[
            tuple[int, int, int], int
        ] = {}

    @property
    def executed_expert_batch_sizes(self) -> list[int]:
        return [batch_size for _, batch_size, _ in self.expert_batch_records]

    def record_token_started(self, token_id: int, start_time_ns: float) -> None:
        self.token_start_time_ns_by_token_id[token_id] = start_time_ns

    def record_completed_token(self, token_id: int, completion_time_ns: float) -> None:
        self.completed_token_times_ns.append(completion_time_ns)
        start_time_ns = self.token_start_time_ns_by_token_id.pop(token_id)
        self.completed_token_latency_records.append(
            (completion_time_ns, completion_time_ns - start_time_ns)
        )

    def record_expert_batch_execution(
        self,
        execution_time_ns: float,
        executed_batch_size: int,
        queue_depth_before_execution: int,
    ) -> None:
        self.expert_batch_records.append(
            (execution_time_ns, executed_batch_size, queue_depth_before_execution)
        )

    def record_expert_route(
        self,
        routing_time_ns: float,
        is_remote_route: bool,
        branch_count: int,
    ) -> None:
        self.expert_route_records.append(
            (routing_time_ns, is_remote_route, branch_count)
        )

    def record_selected_expert_indexes(
        self,
        workload_agent_id: int,
        workload_token_ordinal: int,
        model_layer_index: int,
        selected_expert_indexes: tuple[int, ...],
    ) -> None:
        identity = (
            workload_agent_id,
            workload_token_ordinal,
            model_layer_index,
        )
        existing = self.selected_expert_indexes_by_logical_identity.get(identity)
        if existing is not None and existing != selected_expert_indexes:
            raise ValueError("logical route identity selected inconsistent experts")
        self.selected_expert_indexes_by_logical_identity[identity] = selected_expert_indexes
        self.selected_expert_route_count_by_logical_identity[identity] = (
            self.selected_expert_route_count_by_logical_identity.get(identity, 0) + 1
        )

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

        expert_batch_records = [
            record
            for record in self.expert_batch_records
            if warmup_time_ns <= record[0] < simulation_end_time_ns
        ]
        expert_route_records = [
            record
            for record in self.expert_route_records
            if warmup_time_ns <= record[0] < simulation_end_time_ns
        ]

        remote_branch_count = sum(
            branch_count
            for _, is_remote, branch_count in expert_route_records
            if is_remote
        )
        total_branch_count = sum(
            branch_count
            for _, _, branch_count in expert_route_records
        )

        return {
            "measurement_window_count": float(len(tokens_per_second)),
            "mean_tokens_per_second": mean(tokens_per_second) if tokens_per_second else 0.0,
            "p10_tokens_per_second": percentile(0.10),
            "p50_tokens_per_second": percentile(0.50),
            "p90_tokens_per_second": percentile(0.90),
            "mean_expert_batch_size": (
                mean(batch_size for _, batch_size, _ in expert_batch_records)
                if expert_batch_records
                else 0.0
            ),
            "remote_expert_route_fraction": (
                remote_branch_count / total_branch_count
                if total_branch_count
                else 0.0
            ),
            "maximum_expert_queue_depth_in_logical_units": float(
                max(
                    (queue_depth for _, _, queue_depth in expert_batch_records),
                    default=0,
                )
            ),
        }
