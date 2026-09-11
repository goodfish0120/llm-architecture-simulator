from __future__ import annotations

from collections import Counter
from enum import Enum
from typing import Sequence


class ExpertResultRendezvousPolicy(str, Enum):
    SEQUENCE_OWNER = "sequence_owner"
    LARGEST_LOCAL_EXPERT_GROUP = "largest_local_expert_group"


def choose_rendezvous_node_for_selected_expert_nodes(
    selected_expert_node_ids: Sequence[int],
    sequence_owner_node_id: int,
    rendezvous_policy: str,
) -> int:
    """Choose where already-routed expert branches reunite before the token continues.

    Expert routing has already selected the branches and revealed which nodes will
    execute them. This runtime policy uses that known placement to choose the join
    location without changing the model's routing decision.
    """
    if not selected_expert_node_ids:
        raise ValueError("selected_expert_node_ids cannot be empty")

    if rendezvous_policy == ExpertResultRendezvousPolicy.SEQUENCE_OWNER.value:
        return sequence_owner_node_id

    if rendezvous_policy == ExpertResultRendezvousPolicy.LARGEST_LOCAL_EXPERT_GROUP.value:
        expert_count_by_node = Counter(selected_expert_node_ids)
        largest_local_expert_count = max(expert_count_by_node.values())
        tied_node_ids = sorted(
            node_id
            for node_id, local_expert_count in expert_count_by_node.items()
            if local_expert_count == largest_local_expert_count
        )
        if sequence_owner_node_id in tied_node_ids:
            return sequence_owner_node_id
        return tied_node_ids[0]

    raise ValueError(f"unknown rendezvous_policy: {rendezvous_policy}")
