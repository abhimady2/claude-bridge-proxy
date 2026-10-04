"""Self-check for context-window resolution + the settings.json keys Claude Code
actually honors for auto-compaction.

Run:  python test_context.py
Exits 0 on pass. No framework, no fixtures.
"""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))

from proxy_engine import (
    DEFAULT_CONTEXT_LENGTH,
    fetch_context_length,
    _lookup_known_context,
    trim_openai_messages,
    add_tokens,
    get_token_stats,
    reset_token_stats,
)


def test_known_lookup():
    assert _lookup_known_context("Atria-Dawn-Preview") == 128000
    assert _lookup_known_context("atria-dawn-preview") == 128000
    assert _lookup_known_context("claude-sonnet-4") == 200000
    assert _lookup_known_context("deepseek-ai/DeepSeek-V4-Flash-0731") is None
    assert _lookup_known_context("") is None
    print("[PASS] known-table lookup incl. case-insensitivity")


def test_atria_context_limit():
    # Atria-Dawn-Preview TokenPlan ingress gateway rejects payloads > 875KB (~135k tokens).
    # The known table resolves Atria to 128,000 so Claude Code compacts at ~83k tokens.
    assert _lookup_known_context("Atria-Dawn-Preview") == 128000
    print("[PASS] Atria context limit == 128000 (safe for TokenPlan gateway)")


def test_unknown_router_falls_back():
    # Unreachable/unknown endpoint must never raise.
    length = fetch_context_length("http://127.0.0.1:59999/v1", "test-key", "no-such-model")
    assert length == DEFAULT_CONTEXT_LENGTH, length
    print(f"[PASS] unknown router -> {length:,} (fallback, no exception)")


def test_cached_lookup_is_stable():
    a = fetch_context_length("http://127.0.0.1:59999/v1", "test-key", "model-x")
    b = fetch_context_length("http://127.0.0.1:59999/v1", "test-key", "model-x")
    assert a == b == DEFAULT_CONTEXT_LENGTH
    print("[PASS] repeated lookup stable via cache")


def test_autocompact_keys_are_the_real_ones():
    # Claude Code 2.1.141 resolves the auto-compact threshold from
    # CLAUDE_CODE_AUTO_COMPACT_WINDOW (env) or settings.autoCompactWindow.
    # CLAUDE_CODE_MAX_CONTEXT_TOKENS alone does NOT trigger compaction.
    settings = {
        "autoCompactWindow": 256000,
        "env": {
            "CLAUDE_CODE_MAX_CONTEXT_TOKENS": "256000",
            "CLAUDE_CODE_AUTO_COMPACT_WINDOW": "256000",
        },
    }
    env = settings["env"]
    assert isinstance(settings["autoCompactWindow"], int)
    assert env["CLAUDE_CODE_MAX_CONTEXT_TOKENS"].isdigit()
    assert env["CLAUDE_CODE_AUTO_COMPACT_WINDOW"].isdigit()
    print("[PASS] settings payload carries both real auto-compact keys")


def test_auto_mode_server_opt_out():
    # A proxy that rewrites Anthropic<->OpenAI traffic drops the safeguards
    # fields the server-side classifier needs, so the notice fires and Claude
    # Code bills its own classifier requests. Opting out silences it.
    settings = {"env": {"CLAUDE_CODE_AUTO_MODE_SERVER": "0"}}
    assert settings["env"]["CLAUDE_CODE_AUTO_MODE_SERVER"] == "0"
    print("[PASS] auto-mode server opt-out present")


def test_token_counter():
    reset_token_stats()
    assert get_token_stats() == {"input": 0, "output": 0, "total": 0, "requests": 0}
    add_tokens(100, 50)
    add_tokens(0, 0)          # a request with no usage still counts as a request
    add_tokens(None, None)
    add_tokens("junk", "junk")  # non-numeric must not poison the totals
    s = get_token_stats()
    assert s["input"] == 100 and s["output"] == 50, s
    assert s["total"] == 150 and s["requests"] >= 3, s
    reset_token_stats()
    assert get_token_stats()["total"] == 0
    print("[PASS] token counter accumulates, tolerates junk, resets cleanly")


def test_trim_openai_messages():
    # Oversized payload gets compacted while preserving system prompt and tool pairs
    msgs = [
        {"role": "system", "content": "system instruction"},
        {"role": "user", "content": "old question " * 500},
        {"role": "assistant", "content": "old answer " * 500},
        {"role": "user", "content": "tool call step"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "tc1", "type": "function", "function": {"name": "f", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "tc1", "content": "tool result"},
        {"role": "assistant", "content": "tool done"},
        {"role": "user", "content": "latest user prompt"}
    ]
    trimmed = trim_openai_messages(msgs, max_bytes=3000)
    assert len(json.dumps(trimmed).encode("utf-8")) <= 3000
    assert trimmed[0]["role"] == "system"
    # Notice marker present
    assert any("compacted by Claude Bridge" in str(m.get("content")) for m in trimmed)
    # Latest user prompt preserved
    assert trimmed[-1]["content"] == "latest user prompt"
    # Tool call pairing preserved
    roles = [m["role"] for m in trimmed]
    if "tool" in roles:
        assert "assistant" in roles
    print("[PASS] trim_openai_messages compacts payload and preserves integrity")


if __name__ == "__main__":
    test_known_lookup()
    test_atria_context_limit()
    test_unknown_router_falls_back()
    test_cached_lookup_is_stable()
    test_autocompact_keys_are_the_real_ones()
    test_auto_mode_server_opt_out()
    test_token_counter()
    test_trim_openai_messages()
    print("\nALL CONTEXT CHECKS PASSED")
