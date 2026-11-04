"""Command line.

  python -m pariksha demo
  python -m pariksha list
  python -m pariksha run --base-url https://api.example.com/v1 --model some-model --delay 3 --out report --fail-below 0.9
"""

from __future__ import annotations

import argparse
import sys

from .agents import GuardedAgent, NaiveAgent, OpenAICompatibleAgent
from .runner import run
from .scenarios import suite


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="pariksha", description="Reliability tests for AI agents that take actions")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("demo", help="run the suite on two scripted demo agents")
    sub.add_parser("list", help="print every test")
    real = sub.add_parser("run", help="test a real model through an OpenAI-compatible endpoint")
    real.add_argument("--base-url", required=True)
    real.add_argument("--model", required=True)
    real.add_argument("--api-key-env", default="OPENAI_API_KEY")
    real.add_argument("--delay", type=float, default=0.0, help="seconds to wait before each request, for rate limits")
    real.add_argument("--out", help="write <out>.md and <out>.json")
    real.add_argument("--fail-below", type=float, help="exit 1 if the score is below this (0-1), for CI gates")
    args = parser.parse_args(argv)

    if args.cmd == "list":
        for scenario, fault in suite():
            print(f"{scenario.id:16} {fault.label}")
        return 0
    if args.cmd == "demo":
        for agent in (NaiveAgent(), GuardedAgent()):
            print(run(agent).to_markdown(), end="\n\n")
        return 0
    report = run(OpenAICompatibleAgent(args.base_url, args.model, args.api_key_env, args.delay))
    print(report.to_markdown())
    if args.out:
        with open(args.out + ".md", "w", encoding="utf-8") as f:
            f.write(report.to_markdown())
        with open(args.out + ".json", "w", encoding="utf-8") as f:
            f.write(report.to_json())
    return 1 if args.fail_below is not None and report.score < args.fail_below else 0


if __name__ == "__main__":
    sys.exit(main())
