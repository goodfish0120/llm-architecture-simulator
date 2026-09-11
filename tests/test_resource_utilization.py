import unittest

from src.llm_architecture_simulator.hardware_resources import (
    ComputeAndMemoryNode,
    SerializedThroughputResource,
)
from src.llm_architecture_simulator.resource_utilization import (
    calculate_compute_and_memory_idle_reason_fractions_between_times,
)


class ResourceUtilizationTest(unittest.TestCase):
    def test_memory_bound_kernel_exposes_compute_waiting_for_memory(self) -> None:
        node = ComputeAndMemoryNode(
            compute_resource=SerializedThroughputResource(
                resource_name="compute",
                work_units_per_ns=100.0,
            ),
            memory_resource=SerializedThroughputResource(
                resource_name="memory",
                work_units_per_ns=1.0,
            ),
        )
        _, completion_time_ns, _ = node.execute_kernel_with_compute_and_memory_overlap(
            ready_time_ns=0.0,
            floating_point_operations=100.0,
            bytes_accessed=100.0,
        )

        idle_reasons = calculate_compute_and_memory_idle_reason_fractions_between_times(
            node=node,
            measurement_start_time_ns=0.0,
            measurement_end_time_ns=completion_time_ns,
        )

        self.assertGreater(idle_reasons["compute_waiting_for_memory"], 0.9)
        self.assertEqual(idle_reasons["memory_waiting_for_compute"], 0.0)
        self.assertAlmostEqual(idle_reasons["memory_productive"], 1.0)

    def test_compute_bound_kernel_exposes_memory_waiting_for_compute(self) -> None:
        node = ComputeAndMemoryNode(
            compute_resource=SerializedThroughputResource(
                resource_name="compute",
                work_units_per_ns=1.0,
            ),
            memory_resource=SerializedThroughputResource(
                resource_name="memory",
                work_units_per_ns=100.0,
            ),
        )
        _, completion_time_ns, _ = node.execute_kernel_with_compute_and_memory_overlap(
            ready_time_ns=0.0,
            floating_point_operations=100.0,
            bytes_accessed=100.0,
        )

        idle_reasons = calculate_compute_and_memory_idle_reason_fractions_between_times(
            node=node,
            measurement_start_time_ns=0.0,
            measurement_end_time_ns=completion_time_ns,
        )

        self.assertGreater(idle_reasons["memory_waiting_for_compute"], 0.9)
        self.assertEqual(idle_reasons["compute_waiting_for_memory"], 0.0)
        self.assertAlmostEqual(idle_reasons["compute_productive"], 1.0)


if __name__ == "__main__":
    unittest.main()
