"""Claude Code as an LLMClient.

Runs the `claude` CLI in headless (-p) mode, so a Claude Code subscription can
back the investor agents instead of a metered API key. Satisfies the
`LLMClient` protocol structurally — nothing downstream of make_llm() changes.

What you give up against the langchain clients: prompt caching (llm/cache.py
still keys the prompt, but the provider cannot mark a cacheable prefix), and a
process spawn per call. What you gain: no ANTHROPIC_API_KEY, and per-call cost
folded into a subscription. That trade is sized for a live cycle — tens of
calls. A backtest issues tens of thousands and belongs on the API.

Select it by model id: HEDGE_FUND_LLM_MODEL=claude-code, or
claude-code:opus to pin an alias. See make_llm() in client.py.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from tempfile import TemporaryDirectory

# Matches client.TokenListener; duplicated rather than imported to keep this
# module free of an import cycle with client.py, which routes to it.
TokenListener = Callable[[str], None] | None

# The model id prefix that routes here. Real Anthropic ids never start with it.
MODEL_PREFIX = "claude-code"

DEFAULT_ALIAS = "sonnet"

# make_llm's default timeout is 60s, sized for one HTTP round trip. A CLI turn
# pays a process spawn and a full agent turn on top of that, so a floor applies
# unless the caller asks for longer.
MIN_TIMEOUT = float(os.environ.get("HEDGE_FUND_CLAUDE_CODE_TIMEOUT", "180"))


class ClaudeCodeError(RuntimeError):
    """The claude CLI failed, timed out, or returned an unusable response.

    Raised rather than swallowed: the LLMClient contract puts the decision to
    abstain in the LLMAgent layer, not the provider.
    """


class ClaudeCodeLLM:
    """LLMClient backed by the Claude Code CLI in headless mode.

    Every tool is removed and the turn count is pinned to 1. The agents want a
    single completion, not an agent — left enabled, Claude Code will go read
    files in the working directory instead of answering from the fundamentals
    it was handed.

    The persona prompt *replaces* Claude Code's default system prompt rather
    than appending to it. The default is a coding-agent prompt, and the agents
    parse JSON out of the reply (client.extract_json) — a second persona in
    front of the JSON is exactly what breaks that parse.
    """

    def __init__(
        self,
        model: str = MODEL_PREFIX,
        timeout: float = MIN_TIMEOUT,
        on_token: TokenListener = None,
        cli: str | Sequence[str] | None = None,
    ) -> None:
        self.model = model
        self._alias = _alias_of(model)
        self._timeout = timeout
        self._on_token = on_token
        # A sequence lets a caller front the binary with a launcher
        # ("npx", "claude") and lets the tests point at a stub.
        self._argv0: list[str] = (
            [cli] if isinstance(cli, str)
            else list(cli) if cli is not None
            else [_find_cli()]
        )

    def complete(self, system: str, user: str) -> str:
        # Neither prompt goes through argv: a persona prompt plus a filing
        # snapshot blows past Windows' ~32k command-line ceiling. The system
        # prompt goes to a file, the user prompt to stdin.
        with TemporaryDirectory(prefix="hf-claude-") as tmp:
            system_file = Path(tmp) / "system.txt"
            system_file.write_text(system, encoding="utf-8")

            argv = [
                *self._argv0,
                "-p",
                "--system-prompt-file", str(system_file),
                "--model", self._alias,
                "--disallowed-tools", "*",
                "--max-turns", "1",
            ]

            if self._on_token is None:
                return self._run([*argv, "--output-format", "json"], user)

            # --print refuses stream-json unless --verbose rides along
            # ("requires --verbose"), so the flag is not optional here.
            return self._run_streaming(
                [*argv, "--output-format", "stream-json", "--verbose"], user)

    # -- one shot ---------------------------------------------------------

    def _run(self, argv: list[str], user: str) -> str:
        try:
            proc = subprocess.run(
                argv, input=user, capture_output=True, text=True,
                encoding="utf-8", errors="replace", timeout=self._timeout,
            )
        except subprocess.TimeoutExpired as exc:
            raise ClaudeCodeError(f"claude timed out after {self._timeout}s") from exc

        try:
            envelope = json.loads(proc.stdout)
        except json.JSONDecodeError:
            envelope = None

        # The CLI reports its own failures in the envelope *and* exits non-zero
        # ("Not logged in · Please run /login" is the one you meet first), and
        # leaves stderr empty when it does. Reading the envelope before the
        # return code is what keeps the actionable sentence.
        if isinstance(envelope, dict) and envelope.get("is_error"):
            raise ClaudeCodeError(
                f"claude reported an error: "
                f"{_result_text(envelope) or _tail(str(envelope))}")

        if proc.returncode != 0:
            raise ClaudeCodeError(
                f"claude exited {proc.returncode}: "
                f"{_tail(proc.stderr) or _tail(proc.stdout)}")

        if envelope is None:
            raise ClaudeCodeError(
                f"claude did not return JSON: {_tail(proc.stdout)}")

        text = _result_text(envelope)
        if not text:
            raise ClaudeCodeError(f"claude returned no text: {_tail(str(envelope))}")
        return text

    # -- streaming --------------------------------------------------------

    def _run_streaming(self, argv: list[str], user: str) -> str:
        assert self._on_token is not None
        proc = subprocess.Popen(
            argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, encoding="utf-8",
            errors="replace", bufsize=1,
        )

        # stderr drains on its own thread; a full pipe there would deadlock the
        # stdout loop below.
        stderr_lines: list[str] = []
        drain = threading.Thread(
            target=lambda: stderr_lines.extend(proc.stderr or []), daemon=True)
        drain.start()

        deadline = time.monotonic() + self._timeout
        parts: list[str] = []
        fallback = ""
        saw_delta = False

        try:
            proc.stdin.write(user)
            proc.stdin.close()

            for line in proc.stdout:
                if time.monotonic() > deadline:
                    raise ClaudeCodeError(f"claude timed out after {self._timeout}s")

                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue  # heartbeats and other non-JSON noise

                kind = event.get("type")
                if kind in ("partial_message", "stream_event"):
                    # Token deltas. Forwarded to the listener but not
                    # accumulated — the assistant message that closes them out
                    # is the authoritative copy of the same text.
                    text = _event_text(event)
                    if text:
                        saw_delta = True
                        self._on_token(text)
                elif kind == "assistant":
                    text = _event_text(event)
                    if text:
                        parts.append(text)
                        # Deltas for this message already reached the listener.
                        if not saw_delta:
                            self._on_token(text)
                elif kind == "result":
                    if event.get("is_error"):
                        raise ClaudeCodeError(
                            f"claude reported an error: "
                            f"{_result_text(event) or _tail(str(event))}")
                    fallback = _result_text(event)
        finally:
            if proc.poll() is None:
                proc.kill()
            proc.wait()
            drain.join(timeout=1.0)

        if proc.returncode:
            raise ClaudeCodeError(
                f"claude exited {proc.returncode}: {_tail(''.join(stderr_lines))}")

        text = "".join(parts) or fallback
        if not text:
            raise ClaudeCodeError(
                f"claude streamed no text: {_tail(''.join(stderr_lines))}")
        return text


def make_claude_code_llm(
    model: str | None = None,
    timeout: float = MIN_TIMEOUT,
    on_token: TokenListener = None,
) -> ClaudeCodeLLM:
    """Build the CLI client. The seam make_llm() routes into.

    max_tokens has no CLI equivalent, so make_llm's value is dropped here
    rather than silently ignored inside the client.
    """
    return ClaudeCodeLLM(
        model=model or MODEL_PREFIX,
        timeout=max(timeout, MIN_TIMEOUT),
        on_token=on_token,
    )


def is_claude_code_model(model: str) -> bool:
    """Does this model id route to the CLI rather than to a langchain client?"""
    return model == MODEL_PREFIX or model.startswith(f"{MODEL_PREFIX}:")


# -- helpers --------------------------------------------------------------


def _alias_of(model: str) -> str:
    """The --model value out of a routed id.

    "claude-code" -> the default alias; "claude-code:opus" -> "opus". The
    suffix is passed through untouched, so a full model id works too.
    """
    _, _, alias = model.partition(":")
    return alias.strip() or DEFAULT_ALIAS


def _find_cli() -> str:
    """Absolute path to the claude executable.

    Resolved rather than shelled out to, so Windows picks up claude.cmd
    without shell=True.
    """
    exe = os.environ.get("CLAUDE_CLI") or shutil.which("claude")
    if not exe:
        raise ClaudeCodeError(
            "`claude` not found on PATH. Install Claude Code "
            "(npm install -g @anthropic-ai/claude-code), or point CLAUDE_CLI "
            "at the executable."
        )
    return exe


def _result_text(envelope: object) -> str:
    """Assistant text out of a --output-format json envelope.

    `result` is the field today; the alternatives keep this working if a
    future CLI release renames it. Confirm the current shape with
    `claude -p "hi" --output-format json`.
    """
    if isinstance(envelope, str):
        return envelope
    if isinstance(envelope, dict):
        for key in ("result", "text", "content", "response"):
            text = _flatten(envelope.get(key))
            if text:
                return text
    return ""


def _event_text(event: dict) -> str:
    """Assistant text out of one stream-json event.

    The nesting differs by event kind. A whole message arrives as
    {"type": "assistant", "message": {"content": [...]}} — the blocks are a
    level deeper than the event, and the wrapper's own "type" is "message",
    so reading it as a block yields nothing. Deltas carry {"delta": {...}}
    instead. Both are read so the loop survives either shape.
    """
    message = event.get("message")
    if isinstance(message, dict):
        text = _flatten(message.get("content"))
        if text:
            return text
    for key in ("delta", "content", "text"):
        text = _flatten(event.get(key))
        if text:
            return text
    return ""


def _flatten(value: object) -> str:
    """Text out of a string, a content-block list, or a block dict.

    Mirrors client._flatten: only text blocks are the answer. A thinking block
    stringified into the payload is prose in front of the JSON, which is
    exactly what breaks extract_json(). Blocks join without a separator —
    they are fragments of one continuing string, and a newline would land
    mid-word.
    """
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(_flatten(item) for item in value)
    if isinstance(value, dict):
        if value.get("type") in (None, "text", "text_delta"):
            for key in ("text", "content"):
                inner = value.get(key)
                if inner is not None:
                    return _flatten(inner)
    return ""


def _tail(text: str, limit: int = 300) -> str:
    """The last few hundred characters, for an error message."""
    text = (text or "").strip()
    return text[-limit:] if len(text) > limit else text
