"""Tests for the Resettable protocol, written before the implementation."""

from __future__ import annotations

from raglab_common.resettable import Resettable


class _CompliantStore:
    def __init__(self) -> None:
        self.was_reset = False

    def reset(self) -> None:
        self.was_reset = True


class _NonCompliant:
    def clear(self) -> None:  # wrong method name on purpose
        pass


def test_compliant_object_satisfies_the_protocol() -> None:
    store = _CompliantStore()
    assert isinstance(store, Resettable)
    store.reset()
    assert store.was_reset is True


def test_non_compliant_object_does_not_satisfy_the_protocol() -> None:
    assert not isinstance(_NonCompliant(), Resettable)
