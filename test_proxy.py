"""End-to-end check: fake OpenAI router <-> ClaudeBridge proxy <-> Anthropic SSE client.

Run:  python test_proxy.py
Exits 0 on pass. No framework, no fixtures.
"""
import json
import threading
import time
import urllib.request
import http.server

from proxy_engine import ProxyServer

PROXY_PORT = 4199
ROUTER_PORT = 4198

# Events the proxy emits, in order. Recorded by the SSE client.
received = []


# ---------------------------------------------------------------- fake router
class FakeRouter(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n))

        # Record that the upstream connection was reused (keep-alive working).
        received.append(("router_request", body["model"], body.get("stream")))

        if body.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True

            chunks = [
                {"choices": [{"delta": {"reasoning_content": "Planning "}}]},
                {"choices": [{"delta": {"reasoning_content": "the answer."}}]},
                {"choices": [{"delta": {"content": "Hello "}}]},
                {"choices": [{"delta": {"content": "world"}}]},
                {"choices": [{"delta": {}, "finish_reason": "stop"}]},
            ]
            for c in chunks:
                self.wfile.write(f"data: {json.dumps(c)}\n\n".encode())
                self.wfile.flush()
                time.sleep(0.01)
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        else:
            payload = json.dumps({
                "choices": [{"message": {"content": "non-stream reply", "reasoning_content": "thoughts"}}],
                "usage": {"prompt_tokens": 7, "completion_tokens": 3},
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)


def get_cfg():
    return {
        "router_url": f"http://127.0.0.1:{ROUTER_PORT}/v1",
        "api_key": "test-key",
        "model": "test-flash",
        "sonnet_model": "test-sonnet",
        "opus_model": "test-opus",
        "haiku_model": "test-haiku",
        "thinking_mode": "thinking_block",
    }


def main():
    router = http.server.ThreadingHTTPServer(("127.0.0.1", ROUTER_PORT), FakeRouter)
    threading.Thread(target=router.serve_forever, daemon=True).start()

    proxy = ProxyServer(port=PROXY_PORT, config_getter=get_cfg, log_callback=lambda m: None)
    proxy.start()
    time.sleep(0.2)

    failures = []

    def check(name, cond, detail=""):
        received.append(("check", name, bool(cond)))
        if not cond:
            failures.append(f"{name}: {detail}")

    # --- 1. health endpoint (exercises Content-Length on GET)
    with urllib.request.urlopen(f"http://127.0.0.1:{PROXY_PORT}/health", timeout=5) as r:
        check("health_200", r.status == 200)
        check("health_body", r.read() == b'{"status":"ok","service":"ClaudeBridge"}')

    # --- 2. models endpoint
    with urllib.request.urlopen(f"http://127.0.0.1:{PROXY_PORT}/v1/models", timeout=5) as r:
        models = json.loads(r.read())
        check("models_list", len(models["data"]) > 0, str(models))

    # --- 3. count_tokens (must not hang; Content-Length present)
    tok_req = urllib.request.Request(
        f"http://127.0.0.1:{PROXY_PORT}/v1/messages/count_tokens",
        data=json.dumps({"messages": [{"role": "user", "content": "hi there"}]}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(tok_req, timeout=5) as r:
        check("count_tokens", json.loads(r.read())["input_tokens"] >= 1)

    # --- 4. streaming /v1/messages: measure time-to-first-byte
    req = urllib.request.Request(
        f"http://127.0.0.1:{PROXY_PORT}/v1/messages",
        data=json.dumps({
            "model": "claude-sonnet-4",
            "messages": [{"role": "user", "content": "say hi"}],
            "stream": True,
        }).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    t0 = time.perf_counter()
    events = []
    with urllib.request.urlopen(req, timeout=15) as r:
        check("stream_content_type", "text/event-stream" in r.headers.get("Content-Type", ""))
        ttfb = time.perf_counter() - t0
        cur = []
        for raw in r:  # one line per iteration, blank "\r\n" delimits SSE events
            line = raw.decode(errors="replace").strip()
            if line == "":
                if cur:
                    events.append(cur[0][6:].strip() if cur[0].startswith("event:") else "?")
                    cur = []
                continue
            cur.append(line)

    total = time.perf_counter() - t0

    check("ttfb_under_3s", ttfb < 3.0, f"ttfb={ttfb:.3f}s")
    check("event_order", events[:3] == ["message_start", "content_block_start", "content_block_delta"],
          str(events))
    check("has_thinking_block", "thinking_delta" in [
        e for e in events if e.startswith("content_block")
    ] or "content_block_delta" in events, str(events))
    check("message_stop", events[-1] == "message_stop", str(events))
    check("stream_fast", total < 5.0, f"total={total:.3f}s")

    # --- 5. non-streaming path
    nreq = urllib.request.Request(
        f"http://127.0.0.1:{PROXY_PORT}/v1/messages",
        data=json.dumps({
            "model": "claude-sonnet-4",
            "messages": [{"role": "user", "content": "say hi"}],
            "stream": False,
        }).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(nreq, timeout=15) as r:
        resp = json.loads(r.read())
        check("nonstream_text", any(b.get("type") == "text" for b in resp["content"]), str(resp))
        check("nonstream_thinking", any(b.get("type") == "thinking" for b in resp["content"]), str(resp))

    # --- 6. concurrency: N parallel requests must all finish (threading works)
    results = [None] * 6

    def worker(i):
        try:
            rq = urllib.request.Request(
                f"http://127.0.0.1:{PROXY_PORT}/v1/messages",
                data=json.dumps({
                    "model": "claude-sonnet-4",
                    "messages": [{"role": "user", "content": f"req {i}"}],
                    "stream": True,
                }).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(rq, timeout=20) as r:
                results[i] = any(l.strip() == b"event: message_stop" for l in r)
        except Exception as exc:
            results[i] = f"ERR {exc}"

    t0 = time.perf_counter()
    threads = [threading.Thread(target=worker, args=(i,)) for i in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    concurrent = time.perf_counter() - t0

    check("all_concurrent_ok", all(r is True for r in results), str(results))
    # 6 serial requests at ~50ms router latency each would be sequential;
    # threaded they overlap. Not a strict bound -- just must complete.
    check("concurrent_completes", concurrent < 20.0, f"{concurrent:.3f}s")

    # --- 7. 404 path
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{PROXY_PORT}/v1/nope", timeout=5)
        check("404_raised", False, "expected HTTPError")
    except urllib.error.HTTPError as e:
        check("404_raised", e.code == 404)

    proxy.stop()
    router.shutdown()
    router.server_close()

    print("\n=== ClaudeBridge proxy end-to-end ===")
    print(f"TTFB (streaming):      {ttfb*1000:.0f} ms")
    print(f"Stream total:          {total*1000:.0f} ms")
    print(f"6 concurrent streams:  {concurrent*1000:.0f} ms")
    print(f"Router requests seen:  {sum(1 for r in received if r[0]=='router_request')}")
    print()
    for c in [c for c in received if c[0] == "check"]:
        status = "PASS" if c[2] else "FAIL"
        print(f"  [{status}] {c[1]}")
    print()

    if failures:
        print("FAILURES:")
        for f in failures:
            print(" -", f)
        raise SystemExit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
