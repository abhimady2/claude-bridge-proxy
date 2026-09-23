import json
import time
import uuid
import threading
import re
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
import httpx
import anthropic

# ponytail: one shared pooled client instead of one per request. Each request
# otherwise re-does DNS+TCP+TLS (~100-400ms) before the first byte arrives.
# httpx.Client is thread-safe; ThreadingHTTPServer hits it concurrently.
_client = None
_client_lock = threading.Lock()

_anthropic_client = None
_anthropic_cache_key = None
_anthropic_lock = threading.Lock()


def _get_anthropic_client(api_key, base_url):
    global _anthropic_client, _anthropic_cache_key
    key = (api_key, base_url)
    with _anthropic_lock:
        if _anthropic_client is None or _anthropic_cache_key != key:
            if _anthropic_client is not None:
                try:
                    _anthropic_client.close()
                except Exception:
                    pass
            _anthropic_client = anthropic.Anthropic(
                api_key=api_key if api_key else "placeholder",
                base_url=base_url,
                timeout=600.0,
                max_retries=2
            )
            _anthropic_cache_key = key
        return _anthropic_client


def _close_anthropic_client():
    global _anthropic_client, _anthropic_cache_key
    with _anthropic_lock:
        if _anthropic_client is not None:
            try:
                _anthropic_client.close()
            except Exception:
                pass
            _anthropic_client = None
            _anthropic_cache_key = None


def _get_client():
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                _client = httpx.Client(
                    limits=httpx.Limits(
                        max_keepalive_connections=32,
                        max_connections=128,
                        keepalive_expiry=60.0,
                    ),
                    # read is long on purpose: a streamed generation can run for minutes.
                    timeout=httpx.Timeout(connect=10.0, read=600.0, write=60.0, pool=10.0),
                )
    return _client


def _close_client():
    global _client
    with _client_lock:
        if _client is not None:
            _client.close()
            _client = None

def convert_anthropic_to_openai(anthropic_body, target_model):
    """
    Translates Anthropic Messages API request format to OpenAI Chat Completions format.
    """
    messages = []
    
    # 1. System prompt
    system_content = anthropic_body.get("system")
    if system_content:
        if isinstance(system_content, list):
            sys_text = "\n".join(b.get("text", "") for b in system_content if isinstance(b, dict) and b.get("type") == "text")
        else:
            sys_text = str(system_content)
        if sys_text.strip():
            messages.append({"role": "system", "content": sys_text})
            
    # 2. Messages
    for msg in anthropic_body.get("messages", []):
        role = msg.get("role", "user")
        content = msg.get("content")
        
        if isinstance(content, str):
            messages.append({"role": role, "content": content})
        elif isinstance(content, list):
            text_parts = []
            tool_calls = []
            tool_results = []
            
            for block in content:
                if not isinstance(block, dict):
                    continue
                btype = block.get("type")
                if btype == "text":
                    text_parts.append(block.get("text", ""))
                elif btype == "tool_use":
                    tool_calls.append({
                        "id": block.get("id", f"call_{uuid.uuid4().hex[:8]}"),
                        "type": "function",
                        "function": {
                            "name": block.get("name", ""),
                            "arguments": json.dumps(block.get("input", {}))
                        }
                    })
                elif btype == "tool_result":
                    res_content = block.get("content", "")
                    if isinstance(res_content, list):
                        res_content = "\n".join(b.get("text", "") for b in res_content if isinstance(b, dict))
                    tool_results.append({
                        "role": "tool",
                        "tool_call_id": block.get("tool_use_id", ""),
                        "content": str(res_content)
                    })
            
            # If assistant message has tool_calls
            if role == "assistant" and tool_calls:
                m = {
                    "role": "assistant",
                    "content": "\n".join(text_parts) if text_parts else None,
                    "tool_calls": tool_calls
                }
                messages.append(m)
            else:
                for tr in tool_results:
                    messages.append(tr)
                if text_parts:
                    messages.append({"role": role, "content": "\n".join(text_parts)})
                elif not tool_results:
                    messages.append({"role": role, "content": ""})

    # 3. Tools
    tools = []
    for tool in anthropic_body.get("tools", []):
        tools.append({
            "type": "function",
            "function": {
                "name": tool.get("name"),
                "description": tool.get("description", ""),
                "parameters": tool.get("input_schema", {"type": "object", "properties": {}})
            }
        })

    openai_body = {
        "model": target_model,
        "messages": messages,
        "stream": anthropic_body.get("stream", True)
    }
    
    if "max_tokens" in anthropic_body:
        openai_body["max_tokens"] = anthropic_body["max_tokens"]
    if "temperature" in anthropic_body:
        openai_body["temperature"] = anthropic_body["temperature"]
    if tools:
        openai_body["tools"] = tools

    return openai_body


class ProxyRequestHandler(BaseHTTPRequestHandler):
    # HTTP/1.1 so Claude Code can keep-alive one connection to the proxy
    # instead of opening a fresh socket per request.
    protocol_version = "HTTP/1.1"

    config_getter = None  # Function returning dict: {"router_url", "api_key", "model"}
    log_callback = None   # Function(msg: str)

    def log_message(self, format, *args):
        # Override to prevent default stderr logging
        pass

    def emit_log(self, text):
        if self.log_callback:
            self.log_callback(text)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.end_headers()

    def do_GET(self):
        if self.path in ("/", "/health"):
            payload = b'{"status":"ok","service":"ClaudeBridge"}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(payload)
            return

        if self.path.startswith("/v1/models"):
            cfg = self.config_getter() if self.config_getter else {}
            models_list = [
                cfg.get("model", "deepseek-ai/DeepSeek-V4-Flash-0731"),
                cfg.get("sonnet_model", "deepseek-ai/DeepSeek-V4-Flash-0731"),
                cfg.get("opus_model", "MiniMaxAI/MiniMax-M2.7"),
                cfg.get("haiku_model", "zai-org/GLM-5.3-Flash"),
                "claude-3-5-sonnet-20241022",
                "claude-3-opus-20240229",
                "claude-3-5-haiku-20241022"
            ]
            seen = set()
            items = []
            for m in models_list:
                if m and m not in seen:
                    seen.add(m)
                    items.append({"id": m, "object": "model"})
            payload = json.dumps({"data": items}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(payload)
            return

        self.send_response(404)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _handle_anthropic_native(self, anthropic_req, target_model, router_url, api_key, cfg):
        """
        Directly routes requests to Anthropic-compatible providers (like AgentRouter)
        using the official anthropic SDK client, which complies with Aliyun WAF TLS
        fingerprinting and stainless headers.
        """
        base_url = re.sub(r"/v1/?$", "", router_url).rstrip("/")
        if not base_url.startswith("http"):
            base_url = f"https://{base_url}"

        self.emit_log(f"Claude Request -> {target_model} via {base_url} (Anthropic-Native)")

        client = _get_anthropic_client(api_key, base_url)
        is_stream = anthropic_req.get("stream", True)
        thinking_mode = cfg.get("thinking_mode", "thinking_block")

        clean_req = {
            "model": target_model,
            "messages": anthropic_req.get("messages", []),
            "max_tokens": anthropic_req.get("max_tokens", 4096),
        }
        for k in ("system", "tools", "tool_choice", "temperature", "stop_sequences", "metadata"):
            if k in anthropic_req and anthropic_req[k] is not None:
                clean_req[k] = anthropic_req[k]

        if thinking_mode != "strip" and "thinking" in anthropic_req and anthropic_req["thinking"] is not None:
            clean_req["thinking"] = anthropic_req["thinking"]

        def _execute_streaming(req_dict):
            try:
                with client.messages.with_streaming_response.create(**req_dict, stream=True) as resp:
                    if resp.status_code != 200:
                        err_body = resp.read().decode("utf-8", errors="replace")
                        self.emit_log(f"Router Error ({resp.status_code}): {err_body}")
                        self.send_response(resp.status_code)
                        self.send_header("Content-Type", "application/json")
                        self.send_header("Content-Length", str(len(err_body.encode("utf-8"))))
                        self.send_header("Access-Control-Allow-Origin", "*")
                        self.end_headers()
                        self.wfile.write(err_body.encode("utf-8"))
                        return

                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Connection", "close")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.close_connection = True

                    is_filtering_billing = False
                    ignoring_thinking = False
                    for line in resp.iter_lines():
                        if line.startswith("event: billing_summary"):
                            is_filtering_billing = True
                            continue
                        if is_filtering_billing:
                            if not line.strip():
                                is_filtering_billing = False
                            continue

                        # If user requested hiding/stripping thinking
                        if thinking_mode == "strip":
                            if line.startswith("data:"):
                                try:
                                    d = json.loads(line[5:].strip())
                                    b_type = d.get("type")
                                    if b_type == "content_block_start":
                                        cb = d.get("content_block", {})
                                        if cb.get("type") == "thinking":
                                            ignoring_thinking = True
                                            continue
                                        else:
                                            ignoring_thinking = False
                                    elif b_type in ("content_block_delta", "content_block_stop"):
                                        if ignoring_thinking:
                                            if b_type == "content_block_stop":
                                                ignoring_thinking = False
                                            continue
                                except Exception:
                                    pass
                            if ignoring_thinking and line.startswith("event: content_block"):
                                continue

                        try:
                            self.wfile.write((line + "\n").encode("utf-8"))
                            self.wfile.flush()
                        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                            break
                    self.emit_log("Response successfully streamed to Claude.")

            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                # Client closed socket prematurely
                return
            except anthropic.APIStatusError as err:
                status = err.status_code or 500
                err_body = err.response.text if (hasattr(err, "response") and err.response is not None) else str(err)
                if status == 400 and "thinking" in err_body.lower() and "thinking" in req_dict:
                    self.emit_log("Model does not support thinking param; retrying without it...")
                    new_req = dict(req_dict)
                    new_req.pop("thinking", None)
                    return _execute_streaming(new_req)

                self.emit_log(f"Router Error ({status}): {err_body}")
                try:
                    self.send_response(status)
                    self.send_header("Content-Type", "application/json")
                    body_bytes = err_body.encode("utf-8")
                    self.send_header("Content-Length", str(len(body_bytes)))
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.wfile.write(body_bytes)
                except Exception:
                    pass
            except Exception as ex:
                self.emit_log(f"Proxy Connection Error: {str(ex)}")
                try:
                    payload = json.dumps({"error": str(ex)}).encode("utf-8")
                    self.send_response(500)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(payload)))
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.wfile.write(payload)
                except Exception:
                    pass

        def _execute_non_streaming(req_dict):
            try:
                resp = client.messages.create(**req_dict, stream=False)
                payload = resp.model_dump_json().encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(payload)
                self.emit_log("Response completed successfully.")
            except anthropic.APIStatusError as err:
                status = err.status_code or 500
                err_body = err.response.text if (hasattr(err, "response") and err.response is not None) else str(err)
                if status == 400 and "thinking" in err_body.lower() and "thinking" in req_dict:
                    self.emit_log("Model does not support thinking param; retrying without it...")
                    new_req = dict(req_dict)
                    new_req.pop("thinking", None)
                    return _execute_non_streaming(new_req)

                self.emit_log(f"Router Error ({status}): {err_body}")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                body_bytes = err_body.encode("utf-8")
                self.send_header("Content-Length", str(len(body_bytes)))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(body_bytes)
            except Exception as ex:
                self.emit_log(f"Proxy Connection Error: {str(ex)}")
                payload = json.dumps({"error": str(ex)}).encode("utf-8")
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(payload)

        if is_stream:
            _execute_streaming(clean_req)
        else:
            _execute_non_streaming(clean_req)

    def do_POST(self):
        # Claude Code sends Expect: 100-continue on large bodies. Answer it or
        # the client stalls waiting for the go-ahead before sending its payload.
        if self.headers.get("Expect", "").lower() == "100-continue":
            self.send_response_only(100)
            self.end_headers()
            try:
                self.wfile.flush()
            except Exception:
                pass

        if self.path.startswith("/v1/messages/count_tokens"):
            # Estimate tokens roughly
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            tokens = max(1, len(body) // 4)
            payload = json.dumps({"input_tokens": tokens}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(payload)
            return

        if not self.path.startswith("/v1/messages"):
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return

        # Handle /v1/messages
        content_length = int(self.headers.get("Content-Length", 0))
        raw_body = self.rfile.read(content_length)

        try:
            anthropic_req = json.loads(raw_body.decode("utf-8"))
        except Exception as e:
            payload = json.dumps({"error": str(e)}).encode("utf-8")
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return

        cfg = self.config_getter() if self.config_getter else {}
        router_url = cfg.get("router_url", "https://inference.dahl.global/v1").rstrip("/")
        api_key = cfg.get("api_key", "").strip()
        
        # Multi-model resolution
        default_model = cfg.get("model", "deepseek-ai/DeepSeek-V4-Flash-0731").strip()
        sonnet_model = cfg.get("sonnet_model", default_model).strip()
        opus_model = cfg.get("opus_model", "MiniMaxAI/MiniMax-M2.7").strip()
        haiku_model = cfg.get("haiku_model", "zai-org/GLM-5.3-Flash").strip()

        req_model = (anthropic_req.get("model") or "").lower()
        if "opus" in req_model and opus_model:
            target_model = opus_model
        elif "haiku" in req_model and haiku_model:
            target_model = haiku_model
        elif "sonnet" in req_model and sonnet_model:
            target_model = sonnet_model
        elif anthropic_req.get("model") and not any(k in req_model for k in ("claude", "default")):
            target_model = anthropic_req.get("model")
        else:
            target_model = default_model
        
        # Check if router uses Anthropic-native protocol (e.g. AgentRouter)
        is_anthropic_native = ("agentrouter" in router_url.lower()) or ("anthropic.com" in router_url.lower()) or (cfg.get("protocol") == "anthropic")
        if is_anthropic_native:
            self._handle_anthropic_native(anthropic_req, target_model, router_url, api_key, cfg)
            return

        # Build target OpenAI endpoint
        endpoint = f"{router_url}/chat/completions"
        openai_req = convert_anthropic_to_openai(anthropic_req, target_model)
        
        self.emit_log(f"Claude Request -> {target_model} via {router_url}")
        
        req_headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}" if api_key else ""
        }
        
        is_stream = anthropic_req.get("stream", True)
        openai_req["stream"] = is_stream
        
        msg_id = f"msg_{uuid.uuid4().hex[:16]}"
        
        try:
            client = _get_client()
            if is_stream:
                # Stream response
                with client.stream("POST", endpoint, headers=req_headers, json=openai_req) as resp:
                    if resp.status_code != 200:
                        err_body = resp.read().decode("utf-8", errors="replace")
                        self.emit_log(f"Router Error ({resp.status_code}): {err_body}")
                        self.send_response(resp.status_code)
                        self.send_header("Content-Type", "application/json")
                        self.send_header("Content-Length", str(len(err_body.encode("utf-8"))))
                        self.send_header("Access-Control-Allow-Origin", "*")
                        self.end_headers()
                        self.wfile.write(err_body.encode("utf-8"))
                        return

                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Connection", "close")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.close_connection = True

                    # 1. Start message event
                    start_evt = {
                        "type": "message_start",
                        "message": {
                            "id": msg_id,
                            "type": "message",
                            "role": "assistant",
                            "model": anthropic_req.get("model", target_model),
                            "content": [],
                            "stop_reason": None,
                            "stop_sequence": None,
                            "usage": {"input_tokens": 0, "output_tokens": 0}
                        }
                    }
                    self.wfile.write(f"event: message_start\ndata: {json.dumps(start_evt)}\n\n".encode("utf-8"))
                    self.wfile.flush()

                    thinking_mode = cfg.get("thinking_mode", "thinking_block")
                    thinking_block_started = False
                    text_block_started = False
                    tool_blocks = {} # index -> {"id", "name", "args"}
                    current_block_index = 0
                    has_tools = False
                    in_think_tag = False

                    for line in resp.iter_lines():
                        line = line.strip()
                        if not line or not line.startswith("data:"):
                            continue
                        data_str = line[5:].strip()
                        if data_str == "[DONE]":
                            break
                        
                        try:
                            chunk = json.loads(data_str)
                        except Exception:
                            continue

                        choices = chunk.get("choices", [])
                        if not choices:
                            continue
                        choice = choices[0]
                        delta = choice.get("delta", {})

                        reasoning_piece = delta.get("reasoning_content")
                        content_piece = delta.get("content")

                        # 1. Handle reasoning_content
                        if reasoning_piece:
                            if thinking_mode == "thinking_block":
                                if not thinking_block_started:
                                    block_start = {
                                        "type": "content_block_start",
                                        "index": current_block_index,
                                        "content_block": {"type": "thinking", "thinking": ""}
                                    }
                                    self.wfile.write(f"event: content_block_start\ndata: {json.dumps(block_start)}\n\n".encode("utf-8"))
                                    thinking_block_started = True

                                delta_evt = {
                                    "type": "content_block_delta",
                                    "index": current_block_index,
                                    "delta": {"type": "thinking_delta", "thinking": reasoning_piece}
                                }
                                self.wfile.write(f"event: content_block_delta\ndata: {json.dumps(delta_evt)}\n\n".encode("utf-8"))
                                self.wfile.flush()
                            elif thinking_mode == "raw":
                                if not text_block_started:
                                    block_start = {
                                        "type": "content_block_start",
                                        "index": current_block_index,
                                        "content_block": {"type": "text", "text": ""}
                                    }
                                    self.wfile.write(f"event: content_block_start\ndata: {json.dumps(block_start)}\n\n".encode("utf-8"))
                                    text_block_started = True

                                delta_evt = {
                                    "type": "content_block_delta",
                                    "index": current_block_index,
                                    "delta": {"type": "text_delta", "text": reasoning_piece}
                                }
                                self.wfile.write(f"event: content_block_delta\ndata: {json.dumps(delta_evt)}\n\n".encode("utf-8"))
                                self.wfile.flush()
                            # if thinking_mode == "strip", ignore reasoning_piece completely

                        # 2. Handle content
                        if content_piece:
                            # If thinking block was active via reasoning_content, close it before text
                            if thinking_block_started and not in_think_tag:
                                sig_evt = {
                                    "type": "content_block_delta",
                                    "index": current_block_index,
                                    "delta": {"type": "signature_delta", "signature": "bridge_sig"}
                                }
                                self.wfile.write(f"event: content_block_delta\ndata: {json.dumps(sig_evt)}\n\n".encode("utf-8"))
                                stop_evt = {"type": "content_block_stop", "index": current_block_index}
                                self.wfile.write(f"event: content_block_stop\ndata: {json.dumps(stop_evt)}\n\n".encode("utf-8"))
                                self.wfile.flush()
                                current_block_index += 1
                                thinking_block_started = False

                            pending_text = content_piece
                            while pending_text:
                                if not in_think_tag:
                                    if "<think>" in pending_text:
                                        before_think, pending_text = pending_text.split("<think>", 1)
                                        if before_think:
                                            if not text_block_started:
                                                b_start = {
                                                    "type": "content_block_start",
                                                    "index": current_block_index,
                                                    "content_block": {"type": "text", "text": ""}
                                                }
                                                self.wfile.write(f"event: content_block_start\ndata: {json.dumps(b_start)}\n\n".encode("utf-8"))
                                                text_block_started = True
                                            t_evt = {
                                                "type": "content_block_delta",
                                                "index": current_block_index,
                                                "delta": {"type": "text_delta", "text": before_think}
                                            }
                                            self.wfile.write(f"event: content_block_delta\ndata: {json.dumps(t_evt)}\n\n".encode("utf-8"))
                                            self.wfile.flush()

                                        if text_block_started:
                                            stop_evt = {"type": "content_block_stop", "index": current_block_index}
                                            self.wfile.write(f"event: content_block_stop\ndata: {json.dumps(stop_evt)}\n\n".encode("utf-8"))
                                            current_block_index += 1
                                            text_block_started = False

                                        in_think_tag = True
                                        if thinking_mode == "thinking_block" and not thinking_block_started:
                                            b_start = {
                                                "type": "content_block_start",
                                                "index": current_block_index,
                                                "content_block": {"type": "thinking", "thinking": ""}
                                            }
                                            self.wfile.write(f"event: content_block_start\ndata: {json.dumps(b_start)}\n\n".encode("utf-8"))
                                            thinking_block_started = True
                                    else:
                                        if not text_block_started:
                                            b_start = {
                                                "type": "content_block_start",
                                                "index": current_block_index,
                                                "content_block": {"type": "text", "text": ""}
                                            }
                                            self.wfile.write(f"event: content_block_start\ndata: {json.dumps(b_start)}\n\n".encode("utf-8"))
                                            text_block_started = True
                                        t_evt = {
                                            "type": "content_block_delta",
                                            "index": current_block_index,
                                            "delta": {"type": "text_delta", "text": pending_text}
                                        }
                                        self.wfile.write(f"event: content_block_delta\ndata: {json.dumps(t_evt)}\n\n".encode("utf-8"))
                                        self.wfile.flush()
                                        pending_text = ""
                                else:
                                    if "</think>" in pending_text:
                                        think_content, pending_text = pending_text.split("</think>", 1)
                                        if think_content:
                                            if thinking_mode == "thinking_block":
                                                t_evt = {
                                                    "type": "content_block_delta",
                                                    "index": current_block_index,
                                                    "delta": {"type": "thinking_delta", "thinking": think_content}
                                                }
                                                self.wfile.write(f"event: content_block_delta\ndata: {json.dumps(t_evt)}\n\n".encode("utf-8"))
                                                self.wfile.flush()
                                            elif thinking_mode == "raw":
                                                if not text_block_started:
                                                    b_start = {
                                                        "type": "content_block_start",
                                                        "index": current_block_index,
                                                        "content_block": {"type": "text", "text": ""}
                                                    }
                                                    self.wfile.write(f"event: content_block_start\ndata: {json.dumps(b_start)}\n\n".encode("utf-8"))
                                                    text_block_started = True
                                                t_evt = {
                                                    "type": "content_block_delta",
                                                    "index": current_block_index,
                                                    "delta": {"type": "text_delta", "text": think_content}
                                                }
                                                self.wfile.write(f"event: content_block_delta\ndata: {json.dumps(t_evt)}\n\n".encode("utf-8"))
                                                self.wfile.flush()

                                        if thinking_block_started:
                                            sig_evt = {
                                                "type": "content_block_delta",
                                                "index": current_block_index,
                                                "delta": {"type": "signature_delta", "signature": "bridge_sig"}
                                            }
                                            self.wfile.write(f"event: content_block_delta\ndata: {json.dumps(sig_evt)}\n\n".encode("utf-8"))
                                            stop_evt = {"type": "content_block_stop", "index": current_block_index}
                                            self.wfile.write(f"event: content_block_stop\ndata: {json.dumps(stop_evt)}\n\n".encode("utf-8"))
                                            self.wfile.flush()
                                            current_block_index += 1
                                            thinking_block_started = False
                                        in_think_tag = False
                                    else:
                                        if thinking_mode == "thinking_block":
                                            if not thinking_block_started:
                                                b_start = {
                                                    "type": "content_block_start",
                                                    "index": current_block_index,
                                                    "content_block": {"type": "thinking", "thinking": ""}
                                                }
                                                self.wfile.write(f"event: content_block_start\ndata: {json.dumps(b_start)}\n\n".encode("utf-8"))
                                                thinking_block_started = True
                                            t_evt = {
                                                "type": "content_block_delta",
                                                "index": current_block_index,
                                                "delta": {"type": "thinking_delta", "thinking": pending_text}
                                            }
                                            self.wfile.write(f"event: content_block_delta\ndata: {json.dumps(t_evt)}\n\n".encode("utf-8"))
                                            self.wfile.flush()
                                        elif thinking_mode == "raw":
                                            if not text_block_started:
                                                b_start = {
                                                    "type": "content_block_start",
                                                    "index": current_block_index,
                                                    "content_block": {"type": "text", "text": ""}
                                                }
                                                self.wfile.write(f"event: content_block_start\ndata: {json.dumps(b_start)}\n\n".encode("utf-8"))
                                                text_block_started = True
                                            t_evt = {
                                                "type": "content_block_delta",
                                                "index": current_block_index,
                                                "delta": {"type": "text_delta", "text": pending_text}
                                            }
                                            self.wfile.write(f"event: content_block_delta\ndata: {json.dumps(t_evt)}\n\n".encode("utf-8"))
                                            self.wfile.flush()
                                        pending_text = ""

                        # 3. Tool calls
                        tool_calls_chunk = delta.get("tool_calls", [])
                        for tc in tool_calls_chunk:
                            has_tools = True
                            tc_idx = tc.get("index", 0)
                            if tc_idx not in tool_blocks:
                                # If thinking block was active, close it
                                if thinking_block_started:
                                    sig_evt = {"type": "content_block_delta", "index": current_block_index, "delta": {"type": "signature_delta", "signature": "bridge_sig"}}
                                    self.wfile.write(f"event: content_block_delta\ndata: {json.dumps(sig_evt)}\n\n".encode("utf-8"))
                                    stop_evt = {"type": "content_block_stop", "index": current_block_index}
                                    self.wfile.write(f"event: content_block_stop\ndata: {json.dumps(stop_evt)}\n\n".encode("utf-8"))
                                    current_block_index += 1
                                    thinking_block_started = False

                                # If text block was active, stop it
                                if text_block_started:
                                    stop_evt = {"type": "content_block_stop", "index": current_block_index}
                                    self.wfile.write(f"event: content_block_stop\ndata: {json.dumps(stop_evt)}\n\n".encode("utf-8"))
                                    current_block_index += 1
                                    text_block_started = False

                                tool_id = tc.get("id") or f"toolu_{uuid.uuid4().hex[:12]}"
                                tool_name = tc.get("function", {}).get("name", "")
                                tool_blocks[tc_idx] = {
                                    "block_index": current_block_index,
                                    "id": tool_id,
                                    "name": tool_name
                                }
                                tool_start = {
                                    "type": "content_block_start",
                                    "index": current_block_index,
                                    "content_block": {
                                        "type": "tool_use",
                                        "id": tool_id,
                                        "name": tool_name,
                                        "input": {}
                                    }
                                }
                                self.wfile.write(f"event: content_block_start\ndata: {json.dumps(tool_start)}\n\n".encode("utf-8"))
                                current_block_index += 1

                            arg_chunk = tc.get("function", {}).get("arguments", "")
                            if arg_chunk:
                                b_idx = tool_blocks[tc_idx]["block_index"]
                                tool_delta = {
                                    "type": "content_block_delta",
                                    "index": b_idx,
                                    "delta": {"type": "input_json_delta", "partial_json": arg_chunk}
                                }
                                self.wfile.write(f"event: content_block_delta\ndata: {json.dumps(tool_delta)}\n\n".encode("utf-8"))
                                self.wfile.flush()

                    # Close any open blocks
                    if thinking_block_started:
                        sig_evt = {"type": "content_block_delta", "index": current_block_index, "delta": {"type": "signature_delta", "signature": "bridge_sig"}}
                        self.wfile.write(f"event: content_block_delta\ndata: {json.dumps(sig_evt)}\n\n".encode("utf-8"))
                        stop_evt = {"type": "content_block_stop", "index": current_block_index}
                        self.wfile.write(f"event: content_block_stop\ndata: {json.dumps(stop_evt)}\n\n".encode("utf-8"))

                    if text_block_started:
                        stop_evt = {"type": "content_block_stop", "index": current_block_index}
                        self.wfile.write(f"event: content_block_stop\ndata: {json.dumps(stop_evt)}\n\n".encode("utf-8"))

                    for tb in tool_blocks.values():
                        stop_evt = {"type": "content_block_stop", "index": tb["block_index"]}
                        self.wfile.write(f"event: content_block_stop\ndata: {json.dumps(stop_evt)}\n\n".encode("utf-8"))

                    # Message delta
                    msg_delta = {
                        "type": "message_delta",
                        "delta": {
                            "stop_reason": "tool_use" if has_tools else "end_turn",
                            "stop_sequence": None
                        },
                        "usage": {"output_tokens": 50}
                    }
                    self.wfile.write(f"event: message_delta\ndata: {json.dumps(msg_delta)}\n\n".encode("utf-8"))

                    # Message stop
                    self.wfile.write(b"event: message_stop\ndata: {\"type\": \"message_stop\"}\n\n")
                    self.wfile.flush()
                    self.emit_log(f"Response successfully streamed to Claude ({'Tool use' if has_tools else 'Text'})")

            else:
                # Non-streaming
                resp = client.post(endpoint, headers=req_headers, json=openai_req)
                if resp.status_code != 200:
                    self.emit_log(f"Router Error ({resp.status_code}): {resp.text}")
                    self.send_response(resp.status_code)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(resp.content)))
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.wfile.write(resp.content)
                    return

                oresp = resp.json()
                choice = oresp["choices"][0]
                content_blocks = []
                stop_reason = "end_turn"

                raw_reasoning = choice["message"].get("reasoning_content") or ""
                raw_content = choice["message"].get("content") or ""

                # Extract <think> from raw_content if present
                think_matches = re.findall(r"<think>(.*?)</think>", raw_content, flags=re.DOTALL)
                if think_matches:
                    extracted_think = "\n".join(m.strip() for m in think_matches)
                    raw_reasoning = (raw_reasoning + "\n" + extracted_think).strip()
                    raw_content = re.sub(r"<think>.*?</think>", "", raw_content, flags=re.DOTALL).strip()

                thinking_mode = cfg.get("thinking_mode", "thinking_block")
                if thinking_mode == "thinking_block" and raw_reasoning:
                    content_blocks.append({
                        "type": "thinking",
                        "thinking": raw_reasoning,
                        "signature": "bridge_sig"
                    })
                elif thinking_mode == "raw" and raw_reasoning:
                    raw_content = raw_reasoning + ("\n" if raw_content else "") + raw_content

                if raw_content:
                    content_blocks.append({"type": "text", "text": raw_content})

                if choice["message"].get("tool_calls"):
                    stop_reason = "tool_use"
                    for tc in choice["message"]["tool_calls"]:
                        try:
                            args = json.loads(tc["function"]["arguments"])
                        except Exception:
                            args = {}
                        content_blocks.append({
                            "type": "tool_use",
                            "id": tc.get("id", f"toolu_{uuid.uuid4().hex[:12]}"),
                            "name": tc["function"]["name"],
                            "input": args
                        })

                anthropic_resp = {
                    "id": msg_id,
                    "type": "message",
                    "role": "assistant",
                    "model": anthropic_req.get("model", target_model),
                    "content": content_blocks,
                    "stop_reason": stop_reason,
                    "usage": {
                        "input_tokens": oresp.get("usage", {}).get("prompt_tokens", 0),
                        "output_tokens": oresp.get("usage", {}).get("completion_tokens", 0)
                    }
                }
                payload = json.dumps(anthropic_resp).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(payload)
                self.emit_log("Response completed successfully.")

        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            return
        except Exception as ex:
            self.emit_log(f"Proxy Connection Error: {str(ex)}")
            try:
                payload = json.dumps({"error": str(ex)}).encode("utf-8")
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(payload)
            except Exception:
                pass


class ProxyServer:
    def __init__(self, port=4000, config_getter=None, log_callback=None):
        self.port = port
        self.config_getter = config_getter
        self.log_callback = log_callback
        self.server = None
        self.thread = None
        self.is_running = False

    def start(self):
        if self.is_running:
            return
        handler = ProxyRequestHandler
        handler.config_getter = staticmethod(self.config_getter)
        handler.log_callback = staticmethod(self.log_callback)

        # ThreadingHTTPServer: handle each request on its own thread so a slow
        # stream never blocks /count_tokens, /v1/models, or a parallel request.
        self.server = ThreadingHTTPServer(("127.0.0.1", self.port), handler)
        self.server.daemon_threads = True
        self.is_running = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def stop(self):
        if not self.is_running or not self.server:
            return
        self.is_running = False
        srv = self.server
        self.server = None
        def _shutdown():
            try:
                srv.shutdown()
                srv.server_close()
            except Exception:
                pass
        t = threading.Thread(target=_shutdown, daemon=True)
        t.start()
        t.join(timeout=1.0)
        _close_client()
        _close_anthropic_client()
