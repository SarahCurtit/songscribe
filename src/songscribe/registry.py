"""The backend registry contract, shared by :mod:`~songscribe.chords` and :mod:`~songscribe.beats`.

A stage keeps its own ``BACKENDS`` and ``BACKEND_SAMPLE_RATES`` dicts and wraps them in a
:class:`Registry`, which owns the checks on a name or a setting. Pure Python.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field

#: Parameter kinds a setting can be passed as. A backend's own positional-only
#: or ``*args`` parameters are unreachable through a stage's dispatch function,
#: which forwards settings by keyword.
_KEYWORD_KINDS = (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)

#: Variadic parameter kinds. They carry no default and need none -- an omitted
#: ``*args``/``**kwargs`` is simply empty -- so the rule below that every
#: declared setting be optional does not apply to them.
_VARIADIC_KINDS = (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)

#: A value rule: returns what is wrong with a setting's value, or None.
#: The registry builds the ValueError.
SettingRule = Callable[[object], str | None]


# Holds the stage's dicts by reference rather than copying them: a backend added
# to ``BACKENDS`` after import (a test's fake, a plugin) must be seen here too.
# ``eq=False``: the fields are dicts, so the generated ``__hash__`` would raise.
@dataclass(frozen=True, slots=True, eq=False)
class Registry:
    """One stage's backends, checked the same way as every other stage's.

    ``kind`` is the word the stage's errors use for its backends -- ``"chord"``,
    ``"beat"`` -- so a message names which registry rejected the call.
    """

    kind: str
    backends: dict[str, Callable]
    sample_rates: dict[str, int]
    #: Rules for each backend's setting values, by backend and setting name. A
    #: setting with no rule is passed through unchecked.
    rules: dict[str, dict[str, SettingRule]] = field(default_factory=dict)

    def check_backend(self, backend: str) -> None:
        """Reject an unknown backend by name.

        Never falls back to another model: a typo in a config should fail
        loudly, not quietly downgrade the output to something worse.
        """
        if backend not in self.backends:
            raise ValueError(
                f"unknown {self.kind} backend {backend!r}, expected one of {sorted(self.backends)}"
            )

    def sample_rate_for(self, backend: str) -> int:
        """The sample rate ``backend`` needs its audio loaded at."""
        self.check_backend(backend)
        return self.sample_rates[backend]

    def options(self, backend: str) -> set[str]:
        """The settings ``backend`` accepts, by name.

        Read off the function's signature, after the audio and its rate, which
        dispatch passes positionally. An empty set means no settings; accepting
        *anything* is asked of :meth:`accepts_any_option`. Raises
        :exc:`RuntimeError` if the backend needs more than those two.
        """
        self.check_backend(backend)
        parameters = list(inspect.signature(self.backends[backend]).parameters.values())
        options = set()
        for parameter in parameters[2:]:
            # A setting without a default is not optional, so it cannot be
            # reported as one: dispatch calls a backend with the audio and its
            # rate alone, and a caller who names nothing would get a TypeError
            # from the call -- the one error path this module promises, routed
            # around. Checked before the keyword filter below, so a required
            # positional-only parameter is named here too rather than slipping
            # through as "no settings".
            #
            # A misdeclared backend is this package's own bug, not a caller's
            # bad input, so it is not the ValueError the caller-facing checks
            # raise: a CLI turning ValueError into a usage error must not blame
            # the user for it. Nothing runs when a backend is added to the
            # dict, so the fault surfaces on the first dispatch to it, which is
            # why this check is repeated on every call rather than done once.
            if (
                parameter.default is inspect.Parameter.empty
                and parameter.kind not in _VARIADIC_KINDS
            ):
                raise RuntimeError(
                    f"{self.kind} backend {backend!r} declares setting {parameter.name!r} with "
                    "no default; a backend must be callable with the audio and its rate alone"
                )
            if parameter.kind not in _KEYWORD_KINDS:
                continue
            options.add(parameter.name)
        return options

    def accepts_any_option(self, backend: str) -> bool:
        """Whether ``backend`` keeps a catch-all, and so accepts every setting.

        Such a backend cannot be checked -- it really does take anything -- so
        this reports the exemption instead of folding it into :meth:`options`,
        where "anything" and "nothing" would be two indistinguishable empty
        answers. Nothing registered is exempt, and a test holds that: a
        ``**kwargs`` on a backend is how a silently-ignored setting comes back.
        """
        self.check_backend(backend)
        return any(
            p.kind is inspect.Parameter.VAR_KEYWORD
            for p in inspect.signature(self.backends[backend]).parameters.values()
        )

    def check_options(self, backend: str, options: Iterable[str] | Mapping[str, object]) -> None:
        """Reject settings ``backend`` has no parameter for, or a value it can't use.

        One error path, before dispatch: a misspelled setting that is quietly
        dropped looks like a model that ignored it. Values are checked against
        :attr:`rules` when ``options`` is a mapping; faults are ValueError.
        """
        # Asked first, and of every backend: a catch-all is exempt from the list
        # of accepted names, not from `options`' own rule that each declared
        # setting be optional. Returning early on the exemption would let a
        # backend with both a catch-all and a required setting past this gate
        # and straight into the bare TypeError from the call that this check
        # replaces.
        accepted = self.options(backend)
        if not self.accepts_any_option(backend):
            self._reject_unknown(backend, options, accepted)
        if isinstance(options, Mapping):
            rules = self.rules.get(backend, {})
            for name, value in options.items():
                rule = rules.get(name)
                problem = None if rule is None else rule(value)
                if problem is not None:
                    raise ValueError(
                        f"setting {name!r} for {self.kind} backend {backend!r} {problem}"
                    )

    def _reject_unknown(self, backend: str, options: Iterable[str], accepted: set[str]) -> None:
        unknown = sorted(set(options) - accepted)
        if not unknown:
            return
        named = ", ".join(repr(name) for name in unknown)
        noun = "setting" if len(unknown) == 1 else "settings"
        detail = f"expected one of {sorted(accepted)}" if accepted else "which takes no settings"
        raise ValueError(f"unknown {noun} {named} for {self.kind} backend {backend!r}, {detail}")
