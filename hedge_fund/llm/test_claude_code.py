"""The CLI backend must behave like every other provider, and fail loudly.

These drive a real subprocess against a stub `claude`, not a mock: the parts
that actually break here are the pipes, the encoding, and the stream parse, and
a mock exercises none of them. Nothing reaches the network or the real CLI.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from hedge_fund.llm import make_llm
from hedge_fund.llm.claude_code import (
    ClaudeCodeError,
    ClaudeCodeLLM,
    _alias_of,
    _flatten,
    is_claude_code_model,
)

# A stub that answers with everything the client sent it, so one fake covers
# the argv, the stdin, and the system-prompt file at once.
_ECHO = """
import json, pathlib, sys
# The real CLI is Node, which reads stdin as utf-8 whatever the Windows code
# page says. Python does not, so the stub is told to — otherwise this asserts
# CPython's console defaults rather than the client's encoding.
sys.stdin.reconfigure(encoding="utf-8")
argv = sys.argv[1:]
user = sys.stdin.read()
i = argv.index("--system-prompt-file")
system = pathlib.Path(argv[i + 1]).read_text(encoding="utf-8")
print(json.dumps({"type": "result", "is_error": False, "result": json.dumps(
    {"argv": argv, "user": user, "system": system})}))
"""

_STREAM = """
import json, sys
sys.stdin.read()
events = [
    {"type": "partial_message", "delta": {"type": "text_delta", "text": '{"sig'}},
    {"type": "partial_message", "delta": {"type": "text_delta", "text": 'nal": '}},
    {"type": "partial_message", "delta": {"type": "text_delta", "text": '"buy"}'}},
    {"type": "assistant", "content": [
        {"type": "thinking", "thinking": "weighing margins"},
        {"type": "text", "text": '{"signal": "buy"}'}]},
    {"type": "result", "is_error": False, "result": '{"signal": "buy"}'},
]
for event in events:
    print(json.dumps(event), flush=True)
"""

_STREAM_NO_DELTAS = """
import json, sys
sys.stdin.read()
print(json.dumps({"type": "assistant", "content": [{"type": "text", "text": "whole"}]}), flush=True)
"""

# The event sequence CLI 2.1.232 actually emits, captured verbatim and trimmed
# to the fields that matter. Note the blocks sit under message.content, a level
# deeper than the event, and that no token deltas appear at all.
_STREAM_REAL = """
import json, sys
sys.stdin.read()
events = [
    {"type": "system", "subtype": "init", "model": "claude-sonnet-5", "tools": []},
    {"type": "rate_limit_event",
     "rate_limit_info": {"status": "allowed", "rateLimitType": "five_hour"}},
    {"type": "assistant", "parent_tool_use_id": None, "message": {
        "type": "message", "role": "assistant", "model": "claude-sonnet-5",
        "content": [{"type": "text", "text": "1, 2, 3, 4, 5"}],
        "stop_reason": None}},
    {"type": "result", "subtype": "success", "is_error": False,
     "result": "1, 2, 3, 4, 5", "num_turns": 1},
]
for event in events:
    print(json.dumps(event), flush=True)
"""

# Streams its own argv back, so the flag assertions can run on the same path.
_STREAM_ARGV = """
import json, sys
sys.stdin.read()
print(json.dumps({"type": "assistant",
                  "content": [{"type": "text", "text": json.dumps(sys.argv[1:])}]}), flush=True)
"""

_BOOM = """
import sys
sys.stdin.read()
sys.stderr.write("credit balance too low")
sys.exit(2)
"""

_GARBAGE = """
import sys
sys.stdin.read()
print("Welcome to Claude Code!")
"""

_ERROR_ENVELOPE = """
import json, sys
sys.stdin.read()
print(json.dumps({"type": "result", "is_error": True, "result": "rate limited"}))
"""

# What the real CLI does when it is not logged in: the actionable sentence goes
# to stdout inside the envelope, stderr stays empty, and it still exits 1.
_ERROR_ENVELOPE_AND_EXIT = """
import json, sys
sys.stdin.read()
print(json.dumps({"type": "result", "is_error": True,
                  "result": "Not logged in \\u00b7 Please run /login"}))
sys.exit(1)
"""

_SLOW = """
import sys, time
sys.stdin.read()
time.sleep(30)
"""


@pytest.fixture
def stub(tmp_path):
    """Build a ClaudeCodeLLM whose `claude` is a python script we control."""

    def build(body: str, **kwargs) -> ClaudeCodeLLM:
        script = tmp_path / "fake_claude.py"
        script.write_text(body, encoding="utf-8")
        return ClaudeCodeLLM(cli=[sys.executable, str(script)], **kwargs)

    return build


class TestInvocation:
    """What reaches the CLI decides whether the agents get a usable answer."""

    def test_complete_returns_the_models_text(self, stub):
        echoed = json.loads(stub(_ECHO).complete("SYS", "USER"))
        assert echoed["system"] == "SYS"
        assert echoed["user"] == "USER"

    def test_prompts_never_go_through_argv(self, stub):
        """A persona prompt plus a filing snapshot blows past Windows' ~32k
        command line, so neither prompt may appear as an argument."""
        big = "x" * 50_000
        echoed = json.loads(stub(_ECHO).complete(big, big))
        assert echoed["system"] == big and echoed["user"] == big
        assert not any(big in arg for arg in echoed["argv"])

    def test_tools_are_off_and_the_turn_is_pinned(self, stub):
        """Left enabled, Claude Code reads the working directory instead of
        answering from the fundamentals it was handed."""
        argv = json.loads(stub(_ECHO).complete("s", "u"))["argv"]
        assert argv[argv.index("--disallowed-tools") + 1] == "*"
        assert argv[argv.index("--max-turns") + 1] == "1"

    def test_the_persona_replaces_the_default_system_prompt(self, stub):
        """--append-system-prompt would leave Claude Code's coding-agent
        persona in front of the JSON the agents parse."""
        argv = json.loads(stub(_ECHO).complete("s", "u"))["argv"]
        assert "--system-prompt-file" in argv
        assert "--append-system-prompt" not in argv

    def test_non_ascii_survives_the_pipe(self, stub):
        """Without an explicit utf-8 encoding this decodes to mojibake on
        Windows, where the default is the ANSI code page."""
        echoed = json.loads(stub(_ECHO).complete("한글 시스템", "버핏 판단해줘"))
        assert echoed["system"] == "한글 시스템"
        assert echoed["user"] == "버핏 판단해줘"

    def test_the_alias_reaches_the_model_flag(self, stub):
        argv = json.loads(stub(_ECHO, model="claude-code:opus").complete("s", "u"))["argv"]
        assert argv[argv.index("--model") + 1] == "opus"


class TestFailures:
    """A provider raises; the LLMAgent layer decides to abstain, not us."""

    def test_non_zero_exit_raises_with_the_stderr(self, stub):
        with pytest.raises(ClaudeCodeError, match="credit balance"):
            stub(_BOOM).complete("s", "u")

    def test_non_json_output_raises(self, stub):
        with pytest.raises(ClaudeCodeError, match="did not return JSON"):
            stub(_GARBAGE).complete("s", "u")

    def test_error_envelope_raises_rather_than_returning_prose(self, stub):
        """is_error with an HTTP-ish success is the shape that would otherwise
        reach extract_json as a plausible-looking string."""
        with pytest.raises(ClaudeCodeError, match="rate limited"):
            stub(_ERROR_ENVELOPE).complete("s", "u")

    def test_the_envelope_message_survives_a_non_zero_exit(self, stub):
        """The CLI reports failures in the envelope *and* exits non-zero, with
        stderr empty. Checking the return code first threw away the only
        sentence the user can act on — the reason this test exists."""
        with pytest.raises(ClaudeCodeError, match="Please run /login"):
            stub(_ERROR_ENVELOPE_AND_EXIT).complete("s", "u")

    def test_timeout_raises_and_does_not_hang(self, stub):
        with pytest.raises(ClaudeCodeError, match="timed out"):
            stub(_SLOW, timeout=1.0).complete("s", "u")

    def test_missing_cli_names_the_fix(self, monkeypatch):
        """The error tells you what to install — the only actionable fact."""
        monkeypatch.delenv("CLAUDE_CLI", raising=False)
        monkeypatch.setattr("shutil.which", lambda _: None)
        with pytest.raises(ClaudeCodeError, match="npm install"):
            ClaudeCodeLLM()


class TestStreaming:
    """A listener changes how the text arrives, never what comes back."""

    def test_listener_sees_every_piece_and_the_return_is_whole(self, stub):
        seen: list[str] = []
        result = stub(_STREAM, on_token=seen.append).complete("s", "u")
        assert seen == ['{"sig', 'nal": ', '"buy"}']
        assert result == '{"signal": "buy"}'

    def test_deltas_are_not_counted_twice(self, stub):
        """The assistant message repeats the text its deltas already carried;
        accumulating both would hand extract_json the payload doubled."""
        result = stub(_STREAM, on_token=lambda _: None).complete("s", "u")
        assert result == '{"signal": "buy"}'

    def test_thinking_blocks_reach_neither_the_listener_nor_the_result(self, stub):
        seen: list[str] = []
        result = stub(_STREAM, on_token=seen.append).complete("s", "u")
        assert "weighing margins" not in result
        assert not any("weighing margins" in piece for piece in seen)

    def test_a_stream_without_deltas_still_reaches_the_listener(self, stub):
        """Not every CLI build emits token deltas; message-level streaming has
        to keep the TUI's thesis feed alive."""
        seen: list[str] = []
        assert stub(_STREAM_NO_DELTAS, on_token=seen.append).complete("s", "u") == "whole"
        assert seen == ["whole"]

    def test_the_real_event_sequence_reaches_the_listener(self, stub):
        """CLI 2.1.232 emits no deltas and nests the blocks under
        message.content. Read as a top-level block that wrapper's own
        "type": "message" matches nothing, the listener stays silent, and the
        result-event fallback hides it by still returning the right answer —
        which is exactly how this went unnoticed.
        """
        seen: list[str] = []
        result = stub(_STREAM_REAL, on_token=seen.append).complete("s", "u")
        assert result == "1, 2, 3, 4, 5"
        assert seen == ["1, 2, 3, 4, 5"]

    def test_init_and_rate_limit_events_are_ignored(self, stub):
        """The stream opens with a system init and a rate-limit notice; neither
        is assistant text and neither may reach the listener."""
        seen: list[str] = []
        stub(_STREAM_REAL, on_token=seen.append).complete("s", "u")
        assert not any("five_hour" in piece or "claude-sonnet-5" in piece
                       for piece in seen)

    def test_a_failing_stream_raises(self, stub):
        with pytest.raises(ClaudeCodeError):
            stub(_BOOM, on_token=lambda _: None).complete("s", "u")

    def test_stream_json_carries_verbose(self, stub):
        """--print refuses stream-json without --verbose, so a listener that
        omits the flag fails the run outright rather than degrading."""
        argv = json.loads(stub(_STREAM_ARGV, on_token=lambda _: None).complete("s", "u"))
        assert argv[argv.index("--output-format") + 1] == "stream-json"
        assert "--verbose" in argv

    def test_the_one_shot_path_stays_quiet(self, stub):
        """--verbose only exists to unlock stream-json; it would otherwise add
        noise to the envelope the one-shot path parses."""
        argv = json.loads(stub(_ECHO).complete("s", "u"))["argv"]
        assert argv[argv.index("--output-format") + 1] == "json"
        assert "--verbose" not in argv


class TestRouting:
    """make_llm is the only seam; the CLI backend has to come through it."""

    @pytest.mark.parametrize(
        "model,routed",
        [("claude-code", True), ("claude-code:opus", True),
         ("claude-opus-5", False), ("claude-sonnet-5", False), ("gpt-5.5", False)],
    )
    def test_only_the_prefix_routes(self, model, routed):
        assert is_claude_code_model(model) is routed

    def test_make_llm_routes_without_an_api_key(self, monkeypatch, tmp_path):
        """The point of this backend: no ANTHROPIC_API_KEY in the environment."""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("HEDGE_FUND_LLM_MODEL", raising=False)
        monkeypatch.setenv("CLAUDE_CLI", sys.executable)
        llm = make_llm("claude-code")
        assert isinstance(llm, ClaudeCodeLLM)
        assert llm.model == "claude-code"

    def test_the_env_var_selects_it_like_any_other_model(self, monkeypatch):
        """HEDGE_FUND_LLM_MODEL is the seam the TUI's picker writes to."""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.setenv("CLAUDE_CLI", sys.executable)
        monkeypatch.setenv("HEDGE_FUND_LLM_MODEL", "claude-code:haiku")
        assert isinstance(make_llm(), ClaudeCodeLLM)

    def test_make_llm_passes_the_listener_through(self, monkeypatch):
        monkeypatch.setenv("CLAUDE_CLI", sys.executable)

        def listener(text: str) -> None:
            pass

        assert make_llm("claude-code", on_token=listener)._on_token is listener

    @pytest.mark.parametrize(
        "model,alias",
        [("claude-code", "sonnet"), ("claude-code:opus", "opus"),
         ("claude-code:", "sonnet"), ("claude-code:claude-opus-5", "claude-opus-5")],
    )
    def test_alias_parsing(self, model, alias):
        assert _alias_of(model) == alias


class TestFlatten:
    """Same contract as client._flatten, over the CLI's block shapes."""

    def test_plain_string_passes_through(self):
        assert _flatten('{"signal": "bullish"}') == '{"signal": "bullish"}'

    def test_blocks_join_without_a_separator(self):
        """Stream blocks are fragments of one continuing string; a newline
        would land mid-word."""
        assert _flatten([{"type": "text", "text": "mo"},
                         {"type": "text", "text": "at"}]) == "moat"

    def test_thinking_blocks_are_dropped(self):
        blocks = [{"type": "thinking", "thinking": "Let me weigh the margins..."},
                  {"type": "text", "text": '{"signal": "bearish"}'}]
        assert _flatten(blocks) == '{"signal": "bearish"}'

    def test_unknown_block_types_are_dropped_not_stringified(self):
        assert _flatten([{"type": "tool_use", "id": "x"},
                         {"type": "text", "text": "ok"}]) == "ok"

    def test_none_becomes_empty(self):
        assert _flatten(None) == ""
