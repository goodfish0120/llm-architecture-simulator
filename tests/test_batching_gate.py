from src.llm_architecture_simulator.discrete_event_engine import ParticleBatchingGate
from src.llm_architecture_simulator.particle_flow import (
    LogicalWorkUnit,
    ParticleBatchingRule,
    WorkParticle,
)


def make_particle(token_id: int, ready_time_ns: float, route_name: str = "7") -> WorkParticle:
    return WorkParticle(
        logical_work_units=(LogicalWorkUnit(token_id, token_id),),
        operation_name="expert_branch_arrived",
        earliest_ready_time_ns=ready_time_ns,
        current_node_id=0,
        route_name=route_name,
    )


def test_ready_queue_fragments_coalesce_at_service_time_without_extra_wait() -> None:
    gate = ParticleBatchingGate(
        gate_name="expert-7",
        maximum_resource_width_per_service=4,
        batching_rule=ParticleBatchingRule.BATCH_IF_READY_WITHIN_TIME_WINDOW,
        batching_window_ns=10,
    )
    gate.enqueue_particle_and_batch_when_allowed(make_particle(1, 0))
    gate.enqueue_particle_and_batch_when_allowed(make_particle(2, 100))
    gate.enqueue_particle_and_batch_when_allowed(make_particle(3, 200))

    batch = gate.take_next_ready_particle_that_fits_gate_capacity(300)

    assert batch is not None
    assert batch.logical_unit_count == 3
    assert gate.queued_logical_unit_count == 0


def test_service_time_coalescing_respects_identity_and_capacity() -> None:
    gate = ParticleBatchingGate(
        gate_name="expert",
        maximum_resource_width_per_service=2,
        batching_rule=ParticleBatchingRule.BATCH_IF_READY_WITHIN_TIME_WINDOW,
        batching_window_ns=10,
    )
    gate.enqueue_particle_and_batch_when_allowed(make_particle(1, 0, "7"))
    gate.enqueue_particle_and_batch_when_allowed(make_particle(2, 100, "8"))
    gate.enqueue_particle_and_batch_when_allowed(make_particle(3, 200, "7"))
    gate.enqueue_particle_and_batch_when_allowed(make_particle(4, 300, "7"))

    batch = gate.take_next_ready_particle_that_fits_gate_capacity(400)

    assert batch is not None
    assert batch.logical_unit_count == 2
    assert all(unit.globally_unique_token_id in {1, 3} for unit in batch.logical_work_units)
    assert gate.queued_logical_unit_count == 2
