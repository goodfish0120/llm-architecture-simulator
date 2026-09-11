from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from math import floor
from typing import Callable

from .particle_flow import ParticleBatchingRule, WorkParticle


@dataclass(order=True)
class ScheduledSimulationEvent:
    scheduled_time_ns: float
    insertion_order: int
    event_type: str = field(compare=False)
    work_particle: WorkParticle = field(compare=False)
    event_data: dict = field(default_factory=dict, compare=False)


class DiscreteEventSimulationEngine:
    def __init__(self) -> None:
        self.current_time_ns = 0.0
        self.next_insertion_order = 0
        self.pending_events: list[ScheduledSimulationEvent] = []
        self.event_handler_by_type: dict[
            str,
            Callable[["DiscreteEventSimulationEngine", ScheduledSimulationEvent], None],
        ] = {}

    def register_event_handler(
        self,
        event_type: str,
        handler: Callable[["DiscreteEventSimulationEngine", ScheduledSimulationEvent], None],
    ) -> None:
        self.event_handler_by_type[event_type] = handler

    def schedule_event(
        self,
        scheduled_time_ns: float,
        event_type: str,
        work_particle: WorkParticle,
        **event_data,
    ) -> None:
        if scheduled_time_ns < self.current_time_ns:
            raise ValueError("cannot schedule an event in the past")
        self.next_insertion_order += 1
        heapq.heappush(
            self.pending_events,
            ScheduledSimulationEvent(
                scheduled_time_ns,
                self.next_insertion_order,
                event_type,
                work_particle,
                event_data,
            ),
        )

    def run_until_simulated_time(
        self,
        end_time_ns: float,
        maximum_event_count: int | None = None,
    ) -> int:
        if end_time_ns < self.current_time_ns:
            raise ValueError("end_time_ns cannot move simulation time backwards")

        executed_event_count = 0
        stopped_because_event_limit_was_reached = False

        while self.pending_events:
            if maximum_event_count is not None and executed_event_count >= maximum_event_count:
                stopped_because_event_limit_was_reached = True
                break

            next_event = heapq.heappop(self.pending_events)
            if next_event.scheduled_time_ns > end_time_ns:
                heapq.heappush(self.pending_events, next_event)
                break

            self.current_time_ns = next_event.scheduled_time_ns
            handler = self.event_handler_by_type.get(next_event.event_type)
            if handler is None:
                raise KeyError(f"no handler registered for {next_event.event_type}")
            handler(self, next_event)
            executed_event_count += 1

        if not stopped_because_event_limit_was_reached:
            self.current_time_ns = end_time_ns

        return executed_event_count


class ParticleBatchingGate:
    def __init__(
        self,
        gate_name: str,
        maximum_resource_width_per_service: float,
        batching_rule: ParticleBatchingRule = ParticleBatchingRule.NEVER_BATCH,
        batching_window_ns: float = 0.0,
    ) -> None:
        if maximum_resource_width_per_service <= 0:
            raise ValueError("maximum_resource_width_per_service must be positive")
        self.gate_name = gate_name
        self.maximum_resource_width_per_service = maximum_resource_width_per_service
        self.batching_rule = batching_rule
        self.batching_window_ns = batching_window_ns
        self.waiting_particles: list[WorkParticle] = []

    def enqueue_particle_and_batch_when_allowed(self, particle: WorkParticle) -> None:
        if self.batching_rule != ParticleBatchingRule.NEVER_BATCH:
            for index in range(len(self.waiting_particles) - 1, -1, -1):
                queued_particle = self.waiting_particles[index]
                if queued_particle.can_batch_with(
                    particle,
                    self.batching_rule,
                    self.batching_window_ns,
                ):
                    self.waiting_particles[index] = queued_particle.batch_with(
                        particle,
                        self.batching_rule,
                        self.batching_window_ns,
                    )
                    return
        self.waiting_particles.append(particle)

    def take_next_ready_particle_that_fits_gate_capacity(
        self,
        current_time_ns: float,
    ) -> WorkParticle | None:
        for index, particle in enumerate(self.waiting_particles):
            if particle.earliest_ready_time_ns > current_time_ns:
                continue
            if particle.resource_width_per_unit > self.maximum_resource_width_per_service:
                raise ValueError(f"one unit is wider than gate {self.gate_name}")

            maximum_logical_units = max(
                1,
                floor(
                    self.maximum_resource_width_per_service
                    / particle.resource_width_per_unit
                ),
            )
            selected_particle, remainder = particle.split_to_fit_maximum_logical_units(
                maximum_logical_units
            )
            if remainder is None:
                self.waiting_particles.pop(index)
            else:
                self.waiting_particles[index] = remainder
            return selected_particle

        return None

    @property
    def queued_logical_unit_count(self) -> int:
        return sum(particle.logical_unit_count for particle in self.waiting_particles)
