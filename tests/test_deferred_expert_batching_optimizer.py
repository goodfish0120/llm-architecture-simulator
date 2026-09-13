from optimize_deferred_expert_batching import (
    approximate_expected_capped_batch_loads,
    approximate_inclusion_probabilities,
)


def test_inclusion_probabilities_sum_to_top_k() -> None:
    probabilities = approximate_inclusion_probabilities((0.6, 0.3, 0.1), 2)

    assert abs(sum(probabilities) - 2.0) < 1e-9
    assert all(0.0 <= probability <= 1.0 for probability in probabilities)


def test_expected_weight_loads_increase_with_request_count() -> None:
    small = approximate_expected_capped_batch_loads(2.0, 128)
    large = approximate_expected_capped_batch_loads(256.0, 128)

    assert 0.0 < small < large
    assert large > 1.0
