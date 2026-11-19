from ctrl_zero.safety import build_safety_decision
from ctrl_zero.traffic_light import TrafficLightLatch


def _stop(latch: TrafficLightLatch, state: str) -> bool:
    return build_safety_decision(traffic_light_state=latch.update(state)).should_stop


def test_red_stops_and_holds_until_green():
    latch = TrafficLightLatch()
    assert not _stop(latch, "unknown")
    assert _stop(latch, "red")
    # red disappears / unlit -> still stopped
    assert _stop(latch, "unknown")
    assert _stop(latch, "unknown")
    assert _stop(latch, "yellow")
    # green releases
    assert not _stop(latch, "green")
    assert not _stop(latch, "unknown")


def test_red_after_green_stops_again():
    latch = TrafficLightLatch()
    _stop(latch, "red")
    _stop(latch, "green")
    assert _stop(latch, "red")
