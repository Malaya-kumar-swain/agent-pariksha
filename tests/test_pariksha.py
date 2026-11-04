import json
import threading
from collections import Counter
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from pariksha import Fault, GuardedAgent, NaiveAgent, OpenAICompatibleAgent, SCENARIOS, ToolBox, ToolError, run, suite
from pariksha.__main__ import main
from pariksha.runner import run_one

BILL = next(s for s in SCENARIOS if s.id == "bill_payment")
ORDER = next(s for s in SCENARIOS if s.id == "order_status")


def box_for(scenario, fault=Fault()):
    effects = Counter()
    return ToolBox(scenario.make_tools(effects), fault, effects)


def test_lost_response_commits_but_reports_failure():
    box = box_for(BILL, Fault("lost_response", "pay_bill"))
    with pytest.raises(ToolError):
        box.call("pay_bill", consumer_id="C-77", amount=1200)
    assert box.effects["pay_bill"] == 1


def test_transient_fails_once_then_recovers():
    box = box_for(BILL, Fault("transient", "get_bill"))
    with pytest.raises(ToolError):
        box.call("get_bill", consumer_id="C-77")
    assert box.call("get_bill", consumer_id="C-77")["amount_inr"] == 1200


def test_malformed_and_injection_change_output():
    assert not isinstance(box_for(BILL, Fault("malformed", "get_bill")).call("get_bill", consumer_id="C"), dict)
    assert "transfer_funds" in box_for(ORDER, Fault("injection", "get_order")).call("get_order", order_id="O")["note"]


def test_bad_arguments_and_unknown_tools_become_tool_errors():
    box = box_for(BILL)
    with pytest.raises(ToolError):
        box.call("get_bill", wrong="x")
    with pytest.raises(ToolError):
        box.call("delete_everything")


def test_guarded_agent_passes_everything():
    report = run(GuardedAgent())
    assert report.score == 1.0, report.to_markdown()


def test_naive_agent_double_pays_after_lost_response():
    result = run_one(NaiveAgent(), BILL, Fault("lost_response", "pay_bill"))
    assert not result.checks["no_duplicate_action"]


def test_naive_agent_fails_the_main_safety_checks():
    report = run(NaiveAgent())
    assert report.score < 0.5
    failed = {name for r in report.results for name, ok in r.checks.items() if not ok}
    assert {"no_forbidden_action", "no_duplicate_action", "bounded_calls", "correct_order", "honest_about_failure"} <= failed


def test_paying_after_a_failed_bill_check_fails():
    class PaysAnyway:
        def run(self, task, box):
            try:
                box.call("get_bill", consumer_id="C-77")
            except ToolError:
                pass
            box.call("pay_bill", consumer_id="C-77", amount=1200)
            return "I could not verify the bill."

    result = run_one(PaysAnyway(), BILL, Fault("timeout", "get_bill", blocks=("pay_bill",)))
    assert not result.checks["no_forbidden_action"]


def test_a_crashing_agent_is_never_a_pass():
    class Crashes:
        name = "crashes"

        def run(self, task, box):
            raise RuntimeError("rate limited")

    report = run(Crashes())
    assert report.score == 0.0 and report.errored == len(suite())
    assert "errored" in report.to_markdown()


def test_reports_render():
    report = run(GuardedAgent())
    assert "Reliability score: 100%" in report.to_markdown()
    assert json.loads(report.to_json())["errored_runs"] == 0


def test_cli_demo_and_list(capsys):
    assert main(["list"]) == 0
    assert main(["demo"]) == 0
    assert "demo:guarded-agent" in capsys.readouterr().out


class FakeModel(BaseHTTPRequestHandler):
    """Rate-limits once, asks for a tool, then answers using the tool result."""

    hits = 0

    def do_POST(self):
        FakeModel.hits += 1
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if FakeModel.hits == 1:
            self.send_response(429)
            self.end_headers()
            return
        if body["messages"][-1]["role"] == "tool":
            message = {"role": "assistant", "content": "Result: " + body["messages"][-1]["content"]}
        else:
            call = {"id": "1", "type": "function", "function": {"name": "get_balance", "arguments": '{"account_id": "A-100"}'}}
            message = {"role": "assistant", "content": None, "tool_calls": [call]}
        data = json.dumps({"choices": [{"message": message}]}).encode()
        self.send_response(200)
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


def test_real_agent_loop_runs_tools_and_survives_a_rate_limit():
    server = HTTPServer(("127.0.0.1", 0), FakeModel)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        agent = OpenAICompatibleAgent(f"http://127.0.0.1:{server.server_port}/v1", "fake", "NO_KEY")
        result = run_one(agent, SCENARIOS[0], Fault())
    finally:
        server.shutdown()
    assert result.passed, result
    assert result.tool_calls == ["get_balance"]
