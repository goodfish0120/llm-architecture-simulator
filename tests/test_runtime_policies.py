import unittest

from src.llm_architecture_simulator.runtime_policies import (
    choose_rendezvous_node_for_selected_expert_nodes,
)


class ExpertResultRendezvousPolicyTest(unittest.TestCase):
    def test_sequence_owner_policy_always_returns_sequence_owner(self) -> None:
        rendezvous_node_id = choose_rendezvous_node_for_selected_expert_nodes(
            selected_expert_node_ids=(1, 1, 2, 2, 2),
            sequence_owner_node_id=0,
            rendezvous_policy="sequence_owner",
        )

        self.assertEqual(rendezvous_node_id, 0)

    def test_largest_local_expert_group_chooses_node_with_most_selected_experts(
        self,
    ) -> None:
        rendezvous_node_id = choose_rendezvous_node_for_selected_expert_nodes(
            selected_expert_node_ids=(0, 1, 1, 2),
            sequence_owner_node_id=0,
            rendezvous_policy="largest_local_expert_group",
        )

        self.assertEqual(rendezvous_node_id, 1)

    def test_largest_local_expert_group_prefers_sequence_owner_on_tie(self) -> None:
        rendezvous_node_id = choose_rendezvous_node_for_selected_expert_nodes(
            selected_expert_node_ids=(0, 0, 1, 1),
            sequence_owner_node_id=0,
            rendezvous_policy="largest_local_expert_group",
        )

        self.assertEqual(rendezvous_node_id, 0)


if __name__ == "__main__":
    unittest.main()
