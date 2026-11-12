"""The shared backend registry, exercised with invented backends."""

from __future__ import annotations

import pytest

from songscribe import beats, chords, melody
from songscribe.registry import Registry, positive_number, setting_rules, whole_number


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


class TestPositiveNumber:
    @staticmethod
    def _ruled() -> Registry:
        @setting_rules(cutoff=positive_number("hertz"))
        def ruled(audio, sample_rate, cutoff=100.0):
            return []

        return Registry("toy", {"ruled": ruled}, {"ruled": 8_000})

    @pytest.mark.parametrize("value", [1, 0.5, 20_000.0])
    def test_accepts_a_positive_measure(self, value):
        self._ruled().check_options("ruled", {"cutoff": value})

    @pytest.mark.parametrize("value", [0, -1.0, float("nan"), float("inf")])
    def test_refuses_what_is_not_a_measure(self, value):
        with pytest.raises(ValueError, match="setting 'cutoff'"):
            self._ruled().check_options("ruled", {"cutoff": value})

    def test_refuses_a_boolean(self):
        # True is an int in Python, and 1 Hz is not what the caller meant.
        with pytest.raises(ValueError, match="must be a number of hertz"):
            self._ruled().check_options("ruled", {"cutoff": True})

    def test_refuses_text(self):
        with pytest.raises(ValueError, match="must be a number of hertz"):
            self._ruled().check_options("ruled", {"cutoff": "440"})

    def test_a_fraction_is_fine_where_a_count_would_not_be(self):
        # The difference from whole_number: a frequency is a quantity.
        assert whole_number("samples")(0.5) is not None
        assert positive_number("hertz")(0.5) is None


def test_each_stage_wraps_its_own_registry():
    # Same code, different dicts and a different word in the error.
    with pytest.raises(ValueError, match="unknown chord backend"):
        chords.sample_rate_for("nope")
    with pytest.raises(ValueError, match="unknown beat backend"):
        beats.sample_rate_for("nope")
    with pytest.raises(ValueError, match="unknown melody backend"):
        melody.sample_rate_for("nope")
