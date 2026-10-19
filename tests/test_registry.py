"""The shared backend registry, exercised with invented backends."""

from __future__ import annotations

import pytest

from songscribe import beats, chords
from songscribe.registry import Registry, setting_rules, whole_number


def _stage() -> Registry:
    def plain(audio, sample_rate):
        return []

    def tuned(audio, sample_rate, width=3, *, mode="a"):
        return []

    return Registry("toy", {"plain": plain, "tuned": tuned}, {"plain": 8_000, "tuned": 16_000})


class TestRegistry:
    def test_options_are_read_off_the_signature(self):
        stage = _stage()
        assert stage.options("plain") == set()
        assert stage.options("tuned") == {"width", "mode"}

    def test_errors_name_the_stage(self):
        stage = _stage()
        with pytest.raises(ValueError, match="unknown toy backend 'nope'"):
            stage.check_backend("nope")
        with pytest.raises(ValueError, match="for toy backend 'tuned'"):
            stage.check_options("tuned", ["wdith"])

    def test_sample_rate_follows_the_dict(self):
        assert _stage().sample_rate_for("tuned") == 16_000

    def test_a_backend_added_later_is_seen(self):
        # The dicts are held by reference: a test's fake backend must register.
        stage = _stage()
        stage.backends["late"] = lambda audio, sample_rate, gain=1.0: []
        assert stage.options("late") == {"gain"}


class TestDeclaredRules:
    @staticmethod
    def _ruled() -> Registry:
        @setting_rules(width=whole_number("columns"))
        def ruled(audio, sample_rate, width=3):
            return []

        return Registry("toy", {"ruled": ruled}, {"ruled": 8_000})

    def test_a_rule_checks_the_value_of_its_parameter(self):
        with pytest.raises(ValueError, match="setting 'width' for toy backend 'ruled' must be at"):
            self._ruled().check_options("ruled", {"width": 0})
        self._ruled().check_options("ruled", {"width": 2})

    def test_a_stand_in_does_not_inherit_the_rule(self):
        stage = self._ruled()
        stage.backends["ruled"] = lambda audio, sample_rate, width=3: []
        stage.check_options("ruled", {"width": 0})

    def test_a_rule_on_a_missing_parameter_is_refused_at_definition(self):
        with pytest.raises(TypeError, match="no setting 'depth'"):

            @setting_rules(depth=whole_number("rows"))
            def ruled(audio, sample_rate, width=3):
                return []


def test_each_stage_wraps_its_own_registry():
    # Same code, different dicts and a different word in the error.
    with pytest.raises(ValueError, match="unknown chord backend"):
        chords.sample_rate_for("nope")
    with pytest.raises(ValueError, match="unknown beat backend"):
        beats.sample_rate_for("nope")
