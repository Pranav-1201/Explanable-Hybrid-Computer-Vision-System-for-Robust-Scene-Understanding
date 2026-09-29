import math

import pytest

from serving.routing import RoutingPolicy, entropy, margin, review_reason

CONF = 0.5


def test_margin_is_top1_minus_top2_regardless_of_order():
    assert margin([0.1, 0.6, 0.3]) == pytest.approx(0.3)


def test_entropy_bounds():
    assert entropy([1.0, 0.0, 0.0, 0.0]) == 0.0
    assert entropy([0.25] * 4) == pytest.approx(1.0)
    assert entropy([0.5, 0.5, 0.0, 0.0]) == pytest.approx(math.log(2) / math.log(4))


def test_default_policy_matches_the_original_two_rules():
    p = RoutingPolicy(conf_min=CONF)
    assert review_reason([0.9, 0.05, 0.05], True, p) is None
    assert review_reason([0.4, 0.35, 0.25], True, p) == "low_confidence"
    assert review_reason([0.9, 0.05, 0.05], False, p) == "out_of_scope"


def test_low_confidence_outranks_out_of_scope():
    assert review_reason([0.4, 0.3, 0.3], False, RoutingPolicy(conf_min=CONF)) == "low_confidence"


def test_confidence_boundary_is_inclusive_of_the_threshold():
    assert review_reason([0.5, 0.3, 0.2], True, RoutingPolicy(conf_min=0.5)) is None


def test_margin_rule_fires_only_when_enabled():
    probs = [0.52, 0.45, 0.03]           # confident enough, but nearly a tie
    assert review_reason(probs, True, RoutingPolicy(conf_min=CONF)) is None
    assert review_reason(probs, True, RoutingPolicy(conf_min=CONF, margin_min=0.2)) == "low_margin"


def test_entropy_rule_fires_only_when_enabled():
    probs = [0.55] + [0.05] * 9          # top-1 passes the gate, the rest is diffuse
    assert review_reason(probs, True, RoutingPolicy(conf_min=CONF)) is None
    assert review_reason(probs, True, RoutingPolicy(conf_min=CONF, entropy_max=0.5)) == "high_entropy"


def test_optional_rules_do_not_apply_to_out_of_scope():
    probs = [0.52, 0.45, 0.03]
    p = RoutingPolicy(conf_min=CONF, margin_min=0.2)
    assert review_reason(probs, False, p) == "out_of_scope"


@pytest.mark.parametrize("kw", [{"conf_min": 1.5}, {"conf_min": 0.5, "margin_min": -0.1},
                                {"conf_min": 0.5, "entropy_max": 2.0}])
def test_policy_rejects_out_of_range_thresholds(kw):
    with pytest.raises(ValueError):
        RoutingPolicy(**kw)
