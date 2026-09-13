from run_prefix_fork_scenarios import PrefixForkScenario


def test_one_agent_cannot_reuse_its_own_prefix() -> None:
    scenario = PrefixForkScenario("test", 0.99, 0.70)

    assert scenario.effective_context_tokens_per_agent(1) == 100_000


def test_shared_prefix_reduces_effective_context_at_family_scale() -> None:
    scenario = PrefixForkScenario("test", 0.99, 0.70)

    effective_context = scenario.effective_context_tokens_per_agent(32)

    assert 50_000 < effective_context < 60_000
    assert effective_context < scenario.effective_context_tokens_per_agent(8)
