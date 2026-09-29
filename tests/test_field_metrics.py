import math

import numpy as np
import pytest

from evaluation.field_metrics import field_metrics, wilson
from serving.routing import RoutingPolicy

CLASSES = ["kitchen", "bedroom", "casino"]      # casino is not a home class
HOME = ["kitchen", "bedroom"]
POLICY = RoutingPolicy(conf_min=0.5)


def onehot(k, conf=0.9):
    p = np.full(3, (1 - conf) / 2)
    p[k] = conf
    return p


def run(rows, labels):
    return field_metrics(np.stack(rows), np.array(labels), CLASSES, HOME, POLICY)


def test_right_and_wrong_tags():
    m = run([onehot(0), onehot(1), onehot(0)], [0, 1, 1])   # third: kitchen predicted, bedroom true
    assert (m["auto_tagged"], m["right_tags"], m["wrong_tags"]) == (3, 2, 1)
    assert m["tag_precision"] == pytest.approx(2 / 3)
    assert m["home_tag_recall"] == pytest.approx(2 / 3)


def test_non_home_photo_tagged_home_is_a_false_tag():
    m = run([onehot(0)], [2])                # true casino, model says kitchen confidently
    assert m["non_home_false_tags"] == 1 and m["right_tags"] == 0


def test_correct_non_home_prediction_is_reviewed_and_that_review_is_warranted():
    m = run([onehot(2)], [2])                # casino correctly predicted -> out_of_scope
    assert m["auto_tagged"] == 0 and m["reviewed"] == 1
    assert m["warranted_reviews"] == 1 and m["review_precision"] == 1.0
    assert m["reasons"] == {"out_of_scope": 1}


def test_review_of_a_correct_home_prediction_is_not_warranted():
    m = run([onehot(0, conf=0.4)], [0])      # right answer, below the confidence gate
    assert m["reviewed"] == 1 and m["warranted_reviews"] == 0 and m["review_precision"] == 0.0
    assert m["reasons"] == {"low_confidence": 1}


def test_ood_photos_are_excluded_from_top1_but_counted():
    m = run([onehot(0), onehot(0)], [0, -1])
    assert m["n_ood"] == 1 and m["n_labelled"] == 1 and m["top1_labelled"] == 1.0
    assert m["non_home_false_tags"] == 1     # the OOD photo got a home tag


def test_empty_denominators_are_none_not_zero():
    m = run([onehot(2)], [2])
    assert m["tag_precision"] is None and m["home_tag_recall"] is None


def test_shape_mismatch_fails_loudly():
    with pytest.raises(ValueError):
        field_metrics(np.zeros((2, 3)), np.array([0]), CLASSES, HOME, POLICY)


def test_wilson_known_values():
    lo, hi = wilson(50, 100)
    assert lo == pytest.approx(0.4038, abs=1e-3) and hi == pytest.approx(0.5962, abs=1e-3)
    assert all(math.isnan(x) for x in wilson(0, 0))
    lo, hi = wilson(0, 10)
    assert lo == 0.0 and 0.25 < hi < 0.32
