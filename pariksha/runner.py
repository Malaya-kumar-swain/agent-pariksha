"""Run the suite and report reliability. Runs that error are reported, never counted as passes."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass

from .core import Fault, ToolBox
from .scenarios import Scenario, check, suite


@dataclass
class Result:
    scenario: str
    fault: str
    answer: str
    checks: dict[str, bool]
    tool_calls: list[str]
    error: str = ""

    @property
    def passed(self) -> bool:
        return not self.error and all(self.checks.values())


@dataclass
class Report:
    agent: str
    results: list[Result]

    @property
    def completed(self) -> list[Result]:
        return [r for r in self.results if not r.error]

    @property
    def errored(self) -> int:
        return len(self.results) - len(self.completed)

    @property
    def score(self) -> float:
        """Share of all runs that passed. Errored runs count against the score."""
        return sum(r.passed for r in self.results) / len(self.results) if self.results else 0.0

    def by_check(self) -> dict[str, tuple[int, int]]:
        passed, total = Counter(), Counter()
        for r in self.completed:
            for name, ok in r.checks.items():
                total[name] += 1
                passed[name] += ok
        return {name: (passed[name], total[name]) for name in sorted(total)}

    def to_json(self) -> str:
        return json.dumps({"agent": self.agent, "score": self.score, "errored_runs": self.errored,
                           "by_check": self.by_check(), "results": [vars(r) for r in self.results]}, indent=2)

    def to_markdown(self) -> str:
        lines = [f"## Agent-Pariksha report: `{self.agent}`", "",
                 f"**Reliability score: {self.score:.0%}** ({sum(r.passed for r in self.results)} of {len(self.results)} runs passed)"]
        if self.errored:
            lines.append(f"\n**Warning: {self.errored} runs errored and count as failures, so this score is incomplete.**")
        lines += ["", "| Check | Passed |", "|---|---|"]
        lines += [f"| {name} | {p} of {t} |" for name, (p, t) in self.by_check().items()]
        lines += ["", "| Scenario | Fault | Result |", "|---|---|---|"]
        for r in self.results:
            why = "ERROR" if r.error else "pass" if r.passed else "FAIL: " + ", ".join(k for k, v in r.checks.items() if not v)
            lines.append(f"| {r.scenario} | {r.fault} | {why} |")
        return "\n".join(lines)


def run_one(agent, scenario: Scenario, fault: Fault) -> Result:
    effects: Counter = Counter()
    box = ToolBox(scenario.make_tools(effects), fault, effects)
    try:
        answer = agent.run(scenario.task, box)
    except Exception as e:  # keep going, but never treat a crash as a pass
        return Result(scenario.id, fault.label, "", {}, [c.name for c in box.calls], f"{type(e).__name__}: {e}")
    return Result(scenario.id, fault.label, answer, check(scenario, fault, box, answer), [c.name for c in box.calls])


def run(agent) -> Report:
    return Report(getattr(agent, "name", "agent"), [run_one(agent, s, f) for s, f in suite()])
