from src.llm_architecture_simulator.expert_routing import WeightedTopKExpertRouter


def test_keyed_routes_are_independent_of_call_order():
    router = WeightedTopKExpertRouter((0.2, 0.3, 0.5), 2, random_seed=99)
    first_a = router.sample_distinct_expert_indexes_for_logical_step(3, 7, 1, 7)
    first_b = router.sample_distinct_expert_indexes_for_logical_step(4, 2, 1, 7)

    reversed_router = WeightedTopKExpertRouter((0.2, 0.3, 0.5), 2, random_seed=99)
    second_b = reversed_router.sample_distinct_expert_indexes_for_logical_step(4, 2, 1, 7)
    second_a = reversed_router.sample_distinct_expert_indexes_for_logical_step(3, 7, 1, 7)

    assert first_a == second_a
    assert first_b == second_b


def test_keyed_sampling_does_not_change_legacy_stream_behavior():
    router = WeightedTopKExpertRouter((0.2, 0.3, 0.5), 2, random_seed=99)
    expected = router.sample_distinct_expert_indexes_for_one_token()

    router_with_keyed_call = WeightedTopKExpertRouter((0.2, 0.3, 0.5), 2, random_seed=99)
    router_with_keyed_call.sample_distinct_expert_indexes_for_logical_step(3, 7, 1, 7)
    assert router_with_keyed_call.sample_distinct_expert_indexes_for_one_token() == expected
