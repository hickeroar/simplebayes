from simplebayes.runtime.readiness import ReadinessState


def test_readiness_state_transitions():
    readiness = ReadinessState()
    assert readiness.is_ready is True

    readiness.mark_not_ready()
    assert readiness.is_ready is False

    readiness.mark_ready()
    assert readiness.is_ready is True
