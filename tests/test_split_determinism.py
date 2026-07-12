"""Regression guard: the seed-42 / 10% validation split must stay deterministic
and identical across trainers. A silently shifted or leaked split invalidates
every reported accuracy number (audit N6), so this locks the exact partition.
"""
import torch
import pytest

from training.train_fusion import split_train_val, SEED, VAL_SPLIT


def _canonical(n, seed=42, val_frac=0.1):
    """The one true split: seed-42 permutation, first 10% = val."""
    idx = torch.randperm(n, generator=torch.Generator().manual_seed(seed)).tolist()
    n_val = int(n * val_frac)
    return idx[n_val:], idx[:n_val]  # (train, val)


@pytest.mark.parametrize("n", [67, 1000, 5360])
def test_split_matches_canonical(n):
    tr, va = split_train_val(n)
    ctr, cva = _canonical(n)
    assert tr == ctr
    assert va == cva


@pytest.mark.parametrize("n", [67, 1000, 5360])
def test_split_is_deterministic(n):
    assert split_train_val(n) == split_train_val(n)


@pytest.mark.parametrize("n", [67, 1000, 5360])
def test_split_partitions_all_indices(n):
    tr, va = split_train_val(n)
    assert set(tr).isdisjoint(set(va))
    assert set(tr) | set(va) == set(range(n))
    assert len(va) == int(n * 0.1)


def test_constants_consistent_across_trainers():
    """train_baseline / train_phase2 / train_fusion must agree on seed & split;
    if one drifts, the fusion val set stops matching the CNN val set."""
    import training.train_phase2 as p2
    import training.train_baseline as base
    assert VAL_SPLIT == p2.VAL_SPLIT == base.VAL_SPLIT == 0.1
    assert SEED == p2.SEED == 42
