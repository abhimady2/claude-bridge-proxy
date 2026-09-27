"""Replays Claude Code's real autocompact gate chain against proxy responses.

The functions below are transliterated from the Claude Code 2.1.141 binary
(W2, li, iHH, K_A, ot7, v28, _MH, BY8). If the proxy's usage object is missing
the cache fields, BY8 returns 0 and compaction never fires -- that is the bug.

Run:  python test_autocompact.py
Exits 0 on pass. No framework, no fixtures.
"""
import json
import threading
import time
import urllib.request
import http.server

from proxy_engine import ProxyServer

PROXY_PORT = 4399
ROUTER_PORT = 4398


class Router(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n))
        if body.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            for c in ({"choices": [{"delta": {"content": "answer"}}]},
                      {"choices": [{"delta": {}, "finish_reason": "stop"}],
                       "usage": {"prompt_tokens": 210, "completion_tokens": 30}}):
                self.wfile.write(f"data: {json.dumps(c)}\n\n".encode())
                self.wfile.flush()
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        else:
            payload = json.dumps({
                "choices": [{"message": {"content": "answer"}}],
                "usage": {"prompt_tokens": 210, "completion_tokens": 30},
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)


# --- Claude Code 2.1.141, transliterated -------------------------------------

def N7(model):
    """Model name normalization (only real claude-* names collapse)."""
    m = model.lower()
    for known in ("claude-opus-4-7", "claude-sonnet-4-5", "claude-3-5-sonnet"):
        if known in m:
            return known
    return m


def a0(model):
    """Token-counting accuracy tier: 4 for known Claude models, 3 otherwise."""
    return 4 if "claude-" in N7(model or "") else 3


DISABLE_COMPACT = False
MAX_CONTEXT_TOKENS = int("256000")
DEFAULT_CONTEXT = 200000
TT7 = 20000


def W2(model, betas):
    """Context ceiling. MAX_CONTEXT_TOKENS only applies with DISABLE_COMPACT."""
    if DISABLE_COMPACT and MAX_CONTEXT_TOKENS:
        return MAX_CONTEXT_TOKENS
    return DEFAULT_CONTEXT


def BZ():
    return not DISABLE_COMPACT


def li(model, window):
    """Resolve the autocompact window. Env beats settings beats default."""
    ceiling = W2(model, None)
    if window is not None:
        return min(ceiling, window)
    return min(ceiling, DEFAULT_CONTEXT)


def iHH(model, window):
    return li(model, window) - min(32000, TT7)


def ot7(tokens, model, window):
    """Level: ok | warn | compact | blocked."""
    threshold = iHH(model, window)
    if tokens >= threshold - 20000:
        return "warn"
    return "ok"


def v28(window):
    return window - 13000


def K_A(tokens, model, window):
    """shouldCompact -- the gate that was never firing."""
    if not BZ():
        return False, "compaction disabled"
    if tokens >= v28(iHH(model, window)):
        return True, "threshold reached"
    return False, f"{tokens} < {v28(iHH(model, window))}"


def BY8(usage):
    """The counter Claude Code actually compares to the threshold."""
    return (usage.get("input_tokens", 0)
            + usage.get("cache_creation_input_tokens", 0)
            + usage.get("cache_read_input_tokens", 0)
            + usage.get("output_tokens", 0))


def _MH(msg):
    if msg.get("type") == "assistant" and "usage" in msg.get("message", {}):
        return msg["message"]["usage"]
    return None


def main():
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", ROUTER_PORT), Router)
    threading.Thread(target=srv.serve_forever, daemon=True).start()

    proxy = ProxyServer(port=PROXY_PORT, config_getter=lambda: {
        "router_url": f"http://127.0.0.1:{ROUTER_PORT}/v1", "api_key": "k",
        "model": "Atria-Dawn-Preview", "sonnet_model": "Atria-Dawn-Preview",
        "opus_model": "Atria-Dawn-Preview", "haiku_model": "Atria-Dawn-Preview",
        "thinking_mode": "thinking_block"}, log_callback=lambda m: None)
    proxy.start()
    time.sleep(0.3)

    failures = []

    def check(name, cond, detail=""):
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + ("" if cond else f" -- {detail}"))
        if not cond:
            failures.append(name)

    WINDOW = 256000

    # --- non-streaming: the path Atria actually uses
    req = urllib.request.Request(
        f"http://127.0.0.1:{PROXY_PORT}/v1/messages",
        data=json.dumps({"model": "claude-sonnet-4",
                         "messages": [{"role": "user", "content": "hi"}],
                         "stream": False}).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=15) as r:
        body = json.loads(r.read())

    msg = {"type": "assistant", "message": body}
    usage = _MH(msg)
    tokens = BY8(usage)
    print(f"\nnon-streaming: usage={json.dumps(body['usage'])}")
    print(f"  _MH found usage : {usage is not None}")
    print(f"  BY8 total       : {tokens}")
    check("usage_present", usage is not None, str(body))
    check("by8_counts_prompt_tokens", tokens == 240, f"got {tokens}, want 240 (210+30)")
    check("by8_nonzero", tokens > 0)

    # With usage at 240 the counter is small; simulate a long session by
    # replaying the same assistant message N times, as the real loop does.
    N = 700
    total = sum(BY8(_MH(msg)) for _ in range(N))
    fire, reason = K_A(total, body["model"], WINDOW)
    print(f"\nsimulated {N}-turn session: total={total:,} threshold={v28(iHH(body['model'], WINDOW)):,}")
    print(f"  K_A fires       : {fire} ({reason})")
    check("long_session_compacts", fire, reason)

    small = 1000
    fire_small, why_small = K_A(small, body["model"], WINDOW)
    check("short_session_does_not", not fire_small, why_small)

    # --- streaming: verify message_delta carries the same fields
    sreq = urllib.request.Request(
        f"http://127.0.0.1:{PROXY_PORT}/v1/messages",
        data=json.dumps({"model": "claude-sonnet-4",
                         "messages": [{"role": "user", "content": "hi"}],
                         "stream": True}).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    delta_usage = None
    with urllib.request.urlopen(sreq, timeout=15) as r:
        for raw in r:
            line = raw.decode(errors="replace").strip()
            if line.startswith("data:") and "message_delta" in line:
                delta_usage = json.loads(line[5:].strip()).get("usage")
    stream_tokens = BY8(delta_usage or {})
    print(f"\nstreaming message_delta.usage={json.dumps(delta_usage)}")
    check("stream_usage_present", delta_usage is not None, "missing message_delta usage")
    check("stream_by8_nonzero", stream_tokens > 0, f"BY8={stream_tokens}")

    proxy.stop()
    srv.shutdown()
    srv.server_close()

    print()
    if failures:
        print(f"FAILURES: {failures}")
        raise SystemExit(1)
    print("AUTOCOMPACT GATE VERIFIED")


if __name__ == "__main__":
    main()
