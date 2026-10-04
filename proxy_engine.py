import json
import time
import uuid
import threading
import re
import os
import base64
import tempfile
import hashlib
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
import httpx
import anthropic

# ponytail: one shared pooled client instead of one per request. Each request
# otherwise re-does DNS+TCP+TLS (~100-400ms) before the first byte arrives.
# httpx.Client is thread-safe; ThreadingHTTPServer hits it concurrently.
_client = None
_client_lock = threading.Lock()

_anthropic_clients = {}
_anthropic_lock = threading.Lock()


def _get_anthropic_client(api_key, base_url):
    key = (api_key, base_url)
    with _anthropic_lock:
        if key not in _anthropic_clients:
            headers = {}
            if "agentrouter" in base_url.lower():
                headers["User-Agent"] = "claude-cli/1.0.0 (external, cli)"
            _anthropic_clients[key] = anthropic.Anthropic(
                api_key=api_key if api_key else "placeholder",
                base_url=base_url,
                default_headers=headers if headers else None,
                timeout=600.0,
                max_retries=2
            )
        return _anthropic_clients[key]


def _close_anthropic_client():
    with _anthropic_lock:
        for client in list(_anthropic_clients.values()):
            try:
                client.close()
            except Exception:
                pass
        _anthropic_clients.clear()


# --- Multi-Key Rotation & Pooling ---------------------------------------------
_key_round_robin = {}
_key_lock = threading.Lock()


def parse_api_keys(key_val):
    """Splits single string, comma-separated, semicolon, or newline-separated API keys."""
    if not key_val:
        return []
    if isinstance(key_val, list):
        return [str(k).strip() for k in key_val if str(k).strip()]
    return [k.strip() for k in re.split(r"[\r\n,;\s]+", str(key_val)) if k.strip()]


def select_api_key(router_url, api_keys_list, multi_key_enabled=True):
    """Round-robin selection of active API key across requests."""
    if not api_keys_list:
        return ""
    if not multi_key_enabled or len(api_keys_list) == 1:
        return api_keys_list[0]
    with _key_lock:
        idx = _key_round_robin.get(router_url, 0)
        key = api_keys_list[idx % len(api_keys_list)]
        _key_round_robin[router_url] = idx + 1
        return key


# --- Image to Local Disk Bridge (Vision for Atria & TokenPlan limit fix) -------
def save_base64_image(media_type, base64_data, save_dir=None):
    """
    Decodes a base64 image and saves it to a persistent local temp folder.
    Returns normalized forward-slash absolute path so agent tools and scripts can read it.
    """
    try:
        if not base64_data:
            return None
        
        # Strip data URL prefix if present (e.g. data:image/png;base64,...)
        if "," in base64_data and "base64" in base64_data[:50]:
            base64_data = base64_data.split(",", 1)[1]
            
        ext_map = {
            "image/jpeg": ".jpg",
            "image/jpg": ".jpg",
            "image/png": ".png",
            "image/webp": ".webp",
            "image/gif": ".gif",
            "image/bmp": ".bmp",
            "image/svg+xml": ".svg",
        }
        ext = ext_map.get((media_type or "").lower().strip(), ".png")
        
        if not save_dir:
            save_dir = os.path.join(tempfile.gettempdir(), "claude_bridge_images")
        os.makedirs(save_dir, exist_ok=True)
        
        filename = f"img_{int(time.time())}_{uuid.uuid4().hex[:6]}{ext}"
        filepath = os.path.join(save_dir, filename)
        
        raw_bytes = base64.b64decode(base64_data)
        with open(filepath, "wb") as f:
            f.write(raw_bytes)
            
        # Clean up files older than 3 days
        try:
            now = time.time()
            for fname in os.listdir(save_dir):
                fpath = os.path.join(save_dir, fname)
                if os.path.isfile(fpath) and (now - os.path.getmtime(fpath)) > 3 * 86400:
                    try:
                        os.remove(fpath)
                    except Exception:
                        pass
        except Exception:
            pass

        return filepath.replace("\\", "/")
    except Exception:
        return None


def process_anthropic_images(messages, auto_save=True):
    """
    Scans Anthropic messages for image blocks.
    If auto_save is True, saves images to disk and converts the image blocks
    into text blocks containing the file path for models that don't support vision
    or have gateway size limits (like Atria).
    """
    if not auto_save or not messages:
        return messages
    processed = []
    for msg in messages:
        if not isinstance(msg, dict):
            processed.append(msg)
            continue
        content = msg.get("content")
        if isinstance(content, list):
            new_content = []
            for b in content:
                if isinstance(b, dict) and b.get("type") == "image":
                    src = b.get("source", {})
                    if src.get("type") == "base64" and src.get("data"):
                        saved_path = save_base64_image(src.get("media_type", "image/png"), src.get("data"))
                        if saved_path:
                            new_content.append({
                                "type": "text",
                                "text": f"[User attached image saved at: {saved_path}. Inspect or read this image from this local file path.]"
                            })
                            continue
                elif isinstance(b, dict) and b.get("type") == "tool_result":
                    tr_c = b.get("content")
                    if isinstance(tr_c, list):
                        new_tr_c = []
                        for sub_b in tr_c:
                            if isinstance(sub_b, dict) and sub_b.get("type") == "image":
                                src = sub_b.get("source", {})
                                if src.get("type") == "base64" and src.get("data"):
                                    p = save_base64_image(src.get("media_type", "image/png"), src.get("data"))
                                    if p:
                                        new_tr_c.append({
                                            "type": "text",
                                            "text": f"[Tool returned image saved at: {p}]"
                                        })
                                        continue
                            new_tr_c.append(sub_b)
                        b_copy = dict(b)
                        b_copy["content"] = new_tr_c
                        new_content.append(b_copy)
                        continue
                        new_content.append(b_copy)
                        continue
                new_content.append(b)
            msg_copy = dict(msg)
            msg_copy["content"] = new_content
            processed.append(msg_copy)
        else:
            processed.append(msg)
    return processed


def _get_client():
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                _client = httpx.Client(
                    limits=httpx.Limits(
                        max_keepalive_connections=32,
                        max_connections=128,
                    ),
                    timeout=httpx.Timeout(connect=10.0, read=600.0, write=60.0, pool=10.0),
                )
    return _client


def _close_client():
    global _client
    with _client_lock:
        if _client is not None:
            _client.close()
            _client = None


# --- Token accounting ---------------------------------------------------------
# ponytail: module-level counters read by the UI. ProxyRequestHandler instances
# are created per request, so an instance field would reset every time.
_tokens_in = 0
_tokens_out = 0
_requests = 0
_token_lock = threading.Lock()


def add_tokens(input_tokens, output_tokens):
    """Accumulate usage from a completed request. Called on the request thread."""
    global _tokens_in, _tokens_out, _requests
    if not isinstance(input_tokens, (int, float)) and input_tokens is not None:
        return
    with _token_lock:
        if input_tokens:
            _tokens_in += int(input_tokens)
        if output_tokens:
            _tokens_out += int(output_tokens)
        _requests += 1


def get_token_stats():
    with _token_lock:
        return {"input": _tokens_in, "output": _tokens_out,
                "total": _tokens_in + _tokens_out, "requests": _requests}


def reset_token_stats():
    global _tokens_in, _tokens_out, _requests
    with _token_lock:
        _tokens_in = 0
        _tokens_out = 0
        _requests = 0


# --- Context window resolution ------------------------------------------------
# Only some routers (OpenRouter) report context_length in /v1/models. The others
# return a bare id list, so a static table is the source of truth for them and
# DEFAULT_CONTEXT_LENGTH is the editable last resort.
DEFAULT_CONTEXT_LENGTH = 256000

# ponytail: hardcoded for models the routers won't describe. Wrong numbers here
# are worse than the default, so this only holds values verified against the
# vendor's own docs; anything uncertain falls through to the fetch/default.
# Note: Atria-Dawn-Preview advertises 256k, but its TokenPlan gateway enforces an ~875KB
# request payload ceiling (~130k-140k tokens). Setting 128k ensures Claude Code compacts
# at ~83k tokens before approaching the gateway body limit.
KNOWN_CONTEXT_LENGTHS = (
    ("atria", 128000),
    ("claude-", 200000),
)

_context_cache = {}
_context_cache_lock = threading.Lock()


def _lookup_known_context(model):
    m = (model or "").strip().lower()
    if not m:
        return None
    for prefix, length in KNOWN_CONTEXT_LENGTHS:
        if prefix in m:
            return length
    return None


def fetch_context_length(router_url, api_key, model, timeout=8.0):
    """Best-effort context window in tokens for `model` at `router_url`.

    Never raises: returns DEFAULT_CONTEXT_LENGTH when the value is unknown, so
    callers always have a number to write into Claude Code settings.
    """
    cache_key = (router_url, model)
    with _context_cache_lock:
        if cache_key in _context_cache:
            return _context_cache[cache_key]

    resolved = _lookup_known_context(model)
    if resolved is None:
        # Ask the router. Most return no context_length; 401/404/timeout just
        # fall through to the default.
        try:
            base = router_url.rstrip("/")
            if not base.endswith("/models"):
                base = f"{base}/models"
            if not base.endswith("/models"):
                base = f"{base}/models"
            headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
            with httpx.Client(timeout=timeout) as probe:
                resp = probe.get(base, headers=headers)
            if resp.status_code == 200:
                payload = resp.json()
                items = payload.get("data") or payload.get("models") or []
                wanted = (model or "").strip().lower()
                for item in items:
                    if not isinstance(item, dict):
                        continue
                    if item.get("id", "").strip().lower() != wanted:
                        continue
                    length = item.get("context_length") or item.get("max_context_length")
                    if isinstance(length, int) and length > 0:
                        resolved = length
                    break
        except Exception:
            resolved = None

    if not resolved:
        resolved = DEFAULT_CONTEXT_LENGTH

    with _context_cache_lock:
        _context_cache[cache_key] = resolved
    return resolved


def heal_anthropic_messages(messages, force_all=False):
    """
    Ensures assistant messages comply with thinking-mode requirements.
    Reasoning providers (e.g. DeepSeek on AgentRouter) require that any assistant
    turn following a tool call includes a content[].thinking block.
    If missing (e.g. model called a tool directly without generating thinking tokens,
    or a client stripped thinking), injects a placeholder thinking block.
    If force_all is True, injects thinking into every assistant message lacking one.
    """
    healed = []
    for msg in messages:
        if isinstance(msg, dict) and msg.get("role") == "assistant":
            content = msg.get("content")
            if isinstance(content, list):
                has_tool_use = any(isinstance(b, dict) and b.get("type") == "tool_use" for b in content)
                has_thinking = any(isinstance(b, dict) and b.get("type") == "thinking" for b in content)
                if (has_tool_use or force_all) and not has_thinking:
                    msg_copy = dict(msg)
                    msg_copy["content"] = [{"type": "thinking", "thinking": "Thinking..."}] + list(content)
                    healed.append(msg_copy)
                    continue
            elif isinstance(content, str) and force_all:
                msg_copy = dict(msg)
                msg_copy["content"] = [{"type": "thinking", "thinking": "Thinking..."}, {"type": "text", "text": content}]
                healed.append(msg_copy)
                continue
        healed.append(msg)
    return healed


def convert_anthropic_to_openai(anthropic_body, target_model, auto_save_images=True):
    """
    Translates Anthropic Messages API request format to OpenAI Chat Completions format.
    Automatically saves attached Anthropic images to disk and provides the local file path
    for models with TokenPlan/request size limits (such as Atria) or models lacking vision endpoints.
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
            reasoning_parts = []
            
            for block in content:
                if not isinstance(block, dict):
                    continue
                btype = block.get("type")
                if btype == "text":
                    text_parts.append(block.get("text", ""))
                elif btype == "thinking":
                    reasoning_parts.append(block.get("thinking", ""))
                elif btype == "image":
                    if auto_save_images:
                        src = block.get("source", {})
                        if src.get("type") == "base64" and src.get("data"):
                            media_type = src.get("media_type", "image/png")
                            saved_path = save_base64_image(media_type, src.get("data"))
                            if saved_path:
                                text_parts.append(
                                    f"[User attached image file: {saved_path}. "
                                    f"Note: This is an image file on disk. You are a text-only model and cannot view image pixels directly with the Read tool. If asked about the image, inform the user or inspect its metadata via python script.]"
                                )
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
                        parts = []
                        for sub_b in res_content:
                            if isinstance(sub_b, dict):
                                if sub_b.get("type") == "text":
                                    parts.append(sub_b.get("text", ""))
                                elif sub_b.get("type") == "image":
                                    if auto_save_images:
                                        src = sub_b.get("source", {})
                                        if src.get("type") == "base64" and src.get("data"):
                                            p = save_base64_image(src.get("media_type", "image/png"), src.get("data"))
                                            if p:
                                                parts.append(f"[Image file content: {p}]")
                        res_content = "\n".join(parts)
                    tool_results.append({
                        "role": "tool",
                        "tool_call_id": block.get("tool_use_id", ""),
                        "content": str(res_content)
                    })
            
            # If assistant message has tool_calls
            if role == "assistant":
                if tool_calls:
                    m = {
                        "role": "assistant",
                        "content": "\n".join(text_parts) if text_parts else None,
                        "tool_calls": tool_calls
                    }
                    if reasoning_parts:
                        m["reasoning_content"] = "\n".join(reasoning_parts)
                    elif "deepseek" in (target_model or "").lower():
                        m["reasoning_content"] = "Thinking..."
                    messages.append(m)
                else:
                    for tr in tool_results:
                        messages.append(tr)
                    if text_parts:
                        m = {"role": role, "content": "\n".join(text_parts)}
                        if reasoning_parts:
                            m["reasoning_content"] = "\n".join(reasoning_parts)
                        messages.append(m)
                    elif not tool_results:
                        m = {"role": role, "content": ""}
                        if reasoning_parts:
                            m["reasoning_content"] = "\n".join(reasoning_parts)
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
        max_toks = anthropic_body["max_tokens"]
        if isinstance(max_toks, (int, float)):
            max_toks = max(1, min(int(max_toks), 65536))
        openai_body["max_tokens"] = max_toks
    if "temperature" in anthropic_body:
        openai_body["temperature"] = anthropic_body["temperature"]
    if tools:
        openai_body["tools"] = tools
    return openai_body


def trim_openai_messages(messages, max_bytes=650000):
    """
    Safely trims older messages from the middle of an OpenAI messages list
    when the total serialized JSON exceeds max_bytes.
    Preserves:
      - System message(s) at index 0.
      - Integrity of tool_calls and tool results (never separates an assistant
        tool_call from its matching tool responses).
      - Recent conversational turns up to the byte budget.
    """
    if not messages:
        return messages

    try:
        payload_bytes = len(json.dumps(messages).encode("utf-8"))
    except Exception:
        return messages

    if payload_bytes <= max_bytes:
        return messages

    system_msgs = [m for m in messages if m.get("role") == "system"]
    other_msgs = [m for m in messages if m.get("role") != "system"]

    if not other_msgs:
        return messages

    notice_msgs = [
        {
            "role": "user",
            "content": "[System Note: Earlier conversation history was automatically compacted by Claude Bridge to fit the upstream router payload limit.]"
        },
        {
            "role": "assistant",
            "content": "Understood. I will continue using the recent context."
        }
    ]

    overhead = len(json.dumps(system_msgs + notice_msgs).encode("utf-8"))
    target_budget = max(1000, max_bytes - overhead)

    # Group messages into conversational turns starting with a user message.
    # Grouping at 'user' boundaries guarantees that assistant tool_calls
    # and their subsequent tool results stay strictly together.
    turns = []
    curr_turn = []
    for m in other_msgs:
        if m.get("role") == "user" and curr_turn:
            turns.append(curr_turn)
            curr_turn = [m]
        else:
            curr_turn.append(m)
    if curr_turn:
        turns.append(curr_turn)

    retained_turns = []
    accum_bytes = 0
    # Collect recent turns backwards from the end
    for turn in reversed(turns):
        turn_bytes = len(json.dumps(turn).encode("utf-8"))
        if accum_bytes + turn_bytes > target_budget and retained_turns:
            break
        retained_turns.insert(0, turn)
        accum_bytes += turn_bytes

    flattened = [m for turn in retained_turns for m in turn]
    return system_msgs + notice_msgs + flattened


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

        if self.path == "/tokens":
            payload = json.dumps(get_token_stats()).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(payload)
            return

        if self.path.startswith("/v1/models"):
            cfg = self.config_getter() if self.config_getter else {}
            router_url = cfg.get("router_url", "https://inference.dahl.global/v1")
            models_list = [
                cfg.get("model", "deepseek-ai/DeepSeek-V4-Flash-0731"),
                cfg.get("sonnet_model", "deepseek-ai/DeepSeek-V4-Flash-0731"),
                cfg.get("opus_model", "MiniMaxAI/MiniMax-M2.7"),
                cfg.get("haiku_model", "zai-org/GLM-5.3-Flash"),
                "claude-3-7-sonnet-20250219",
                "claude-3-5-sonnet-20241022",
                "claude-3-opus-20240229",
                "claude-3-5-haiku-20241022"
            ]
            seen = set()
            items = []
            for m in models_list:
                if m and m not in seen:
                    seen.add(m)
                    items.append({
                        "id": m,
                        "object": "model",
                        "context_length": fetch_context_length(router_url, cfg.get("api_key", ""), m)
                    })
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

    def _handle_anthropic_native(self, anthropic_req, target_model, router_url, api_key, cfg,
                                allow_fallback=False, fallback_cfg=None, fallback_name=""):
        """
        Directly routes requests to Anthropic-compatible providers (like AgentRouter)
        using the official anthropic SDK client, which complies with Aliyun WAF TLS
        fingerprinting and stainless headers.
        """
        base_url = re.sub(r"/v1/?$", "", router_url).rstrip("/")
        if not base_url.startswith("http"):
            base_url = f"https://{base_url}"

        self.emit_log(f"Claude Request -> {target_model} via {base_url} (Anthropic-Native)")

        default_model = cfg.get("model", target_model).strip()
        api_keys = parse_api_keys(api_key)
        multi_key_enabled = cfg.get("multi_key_rotation", True)
        current_key = select_api_key(router_url, api_keys, multi_key_enabled)
        client = _get_anthropic_client(current_key, base_url)
        is_stream = anthropic_req.get("stream", True)
        thinking_mode = cfg.get("thinking_mode", "thinking_block")

        max_toks = anthropic_req.get("max_tokens", 4096)
        if isinstance(max_toks, (int, float)):
            max_toks = max(1, min(int(max_toks), 65536))

        auto_save_images = cfg.get("auto_save_images", True)
        processed_msgs = process_anthropic_images(anthropic_req.get("messages", []), auto_save=auto_save_images)

        clean_req = {
            "model": target_model,
            "messages": heal_anthropic_messages(processed_msgs),
            "max_tokens": max_toks,
        }
        for k in ("system", "tools", "tool_choice", "temperature", "stop_sequences", "metadata"):
            if k in anthropic_req and anthropic_req[k] is not None:
                clean_req[k] = anthropic_req[k]

        if thinking_mode != "strip" and "thinking" in anthropic_req and anthropic_req["thinking"] is not None:
            clean_req["thinking"] = anthropic_req["thinking"]

        def _execute_streaming(req_dict):
            nonlocal client, current_key
            try:
                with client.messages.with_streaming_response.create(**req_dict, stream=True) as resp:
                    # 1. Multi-key failover retry on 429/402
                    if resp.status_code in (429, 402) and multi_key_enabled and len(api_keys) > 1:
                        for alt_key in [k for k in api_keys if k != current_key]:
                            old_m = f"...{current_key[-4:]}" if len(current_key) >= 4 else "key"
                            new_m = f"...{alt_key[-4:]}" if len(alt_key) >= 4 else "key"
                            self.emit_log(f"Key {old_m} hit limit ({resp.status_code}). Rotating to {new_m}...")
                            current_key = alt_key
                            client = _get_anthropic_client(alt_key, base_url)
                            return _execute_streaming(req_dict)

                    if resp.status_code != 200:
                        err_body = resp.read().decode("utf-8", errors="replace")
                        if allow_fallback and fallback_cfg and fallback_cfg.get("router_url"):
                            self.emit_log(f"Anthropic Router Error ({resp.status_code}): Failing over to fallback router [{fallback_name}]...")
                            return self._execute_request(anthropic_req, fallback_cfg, allow_fallback=False)
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
                    stream_in = 0
                    stream_out = 0
                    for line in resp.iter_lines():
                        if line.startswith("event: billing_summary"):
                            is_filtering_billing = True
                            continue
                        if is_filtering_billing:
                            if not line.strip():
                                is_filtering_billing = False
                            continue

                        # Anthropic-native SSE: usage arrives on message_start /
                        # message_delta data lines.
                        if line.startswith("data:"):
                            try:
                                d = json.loads(line[5:].strip())
                                u = d.get("message", {}).get("usage") or d.get("usage")
                                if isinstance(u, dict):
                                    stream_in += u.get("input_tokens", 0) or 0
                                    stream_out += u.get("output_tokens", 0) or 0
                            except Exception:
                                pass

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
                    add_tokens(stream_in, stream_out)
                    self.emit_log("Response successfully streamed to Claude.")

            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                # Client closed socket prematurely
                return
            except anthropic.APIStatusError as err:
                status = err.status_code or 500
                err_body = err.response.text if (hasattr(err, "response") and err.response is not None) else str(err)
                err_lower = err_body.lower()

                # Upstream requires content[].thinking in message history (e.g. DeepSeek on AgentRouter)
                if status == 400 and ("content[].thinking" in err_lower or "passed back to the api" in err_lower):
                    self.emit_log("Upstream router requires content[].thinking for assistant history; healing turns and retrying...")
                    new_req = dict(req_dict)
                    new_req["messages"] = heal_anthropic_messages(req_dict.get("messages", []), force_all=True)
                    return _execute_streaming(new_req)

                if status == 400 and "thinking" in req_dict and (
                    "unrecognized" in err_lower or "unexpected" in err_lower or "extra inputs" in err_lower
                    or "unknown field" in err_lower or "does not support" in err_lower
                ):
                    self.emit_log("Model does not support thinking param; retrying without it...")
                    new_req = dict(req_dict)
                    new_req.pop("thinking", None)
                    return _execute_streaming(new_req)

                # Fallback on 429/402 to next key in pool
                if status in (429, 402) and multi_key_enabled and len(api_keys) > 1:
                    for alt_key in [k for k in api_keys if k != current_key]:
                        old_m = f"...{current_key[-4:]}" if len(current_key) >= 4 else "key"
                        new_m = f"...{alt_key[-4:]}" if len(alt_key) >= 4 else "key"
                        self.emit_log(f"Key {old_m} hit limit ({status}). Failing over to next key {new_m}...")
                        current_key = alt_key
                        client = _get_anthropic_client(alt_key, base_url)
                        return _execute_streaming(req_dict)

                # Fallback to default_model if target_model failed due to quota (402) or unavailable (404/503)
                if status in (400, 402, 404, 503) and default_model and req_dict.get("model") != default_model:
                    self.emit_log(f"Model '{req_dict.get('model')}' failed (HTTP {status}). Falling back to default model '{default_model}'...")
                    new_req = dict(req_dict)
                    new_req["model"] = default_model
                    return _execute_streaming(new_req)

                # Failover to secondary router if available
                if status in (500, 502, 503, 504) and allow_fallback and fallback_cfg and fallback_cfg.get("router_url"):
                    self.emit_log(f"Anthropic Router Error ({status}): Failing over to fallback router [{fallback_name}]...")
                    return self._execute_request(anthropic_req, fallback_cfg, allow_fallback=False)

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
            nonlocal client, current_key
            try:
                resp = client.messages.create(**req_dict, stream=False)
                payload = resp.model_dump_json().encode("utf-8")
                # resp.usage is validated SDK output, so it is trusted-shape here.
                usage = getattr(resp, "usage", None)
                add_tokens(
                    getattr(usage, "input_tokens", 0),
                    getattr(usage, "output_tokens", 0),
                )
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
                err_lower = err_body.lower()

                # Upstream requires content[].thinking in message history (e.g. DeepSeek on AgentRouter)
                if status == 400 and ("content[].thinking" in err_lower or "passed back to the api" in err_lower):
                    self.emit_log("Upstream router requires content[].thinking for assistant history; healing turns and retrying...")
                    new_req = dict(req_dict)
                    new_req["messages"] = heal_anthropic_messages(req_dict.get("messages", []), force_all=True)
                    return _execute_non_streaming(new_req)

                if status == 400 and "thinking" in req_dict and (
                    "unrecognized" in err_lower or "unexpected" in err_lower or "extra inputs" in err_lower
                    or "unknown field" in err_lower or "does not support" in err_lower
                ):
                    self.emit_log("Model does not support thinking param; retrying without it...")
                    new_req = dict(req_dict)
                    new_req.pop("thinking", None)
                    return _execute_non_streaming(new_req)

                # Fallback on 429/402 to next key in pool
                if status in (429, 402) and multi_key_enabled and len(api_keys) > 1:
                    for alt_key in [k for k in api_keys if k != current_key]:
                        old_m = f"...{current_key[-4:]}" if len(current_key) >= 4 else "key"
                        new_m = f"...{alt_key[-4:]}" if len(alt_key) >= 4 else "key"
                        self.emit_log(f"Key {old_m} hit limit ({status}). Failing over to next key {new_m}...")
                        current_key = alt_key
                        client = _get_anthropic_client(alt_key, base_url)
                        return _execute_non_streaming(req_dict)

                # Fallback to default_model if target_model failed due to quota (402) or unavailable (404/503)
                if status in (400, 402, 404, 503) and default_model and req_dict.get("model") != default_model:
                    self.emit_log(f"Model '{req_dict.get('model')}' failed (HTTP {status}). Falling back to default model '{default_model}'...")
                    new_req = dict(req_dict)
                    new_req["model"] = default_model
                    return _execute_non_streaming(new_req)

                # Failover to secondary router if available
                if status in (500, 502, 503, 504) and allow_fallback and fallback_cfg and fallback_cfg.get("router_url"):
                    self.emit_log(f"Anthropic Router Error ({status}): Failing over to fallback router [{fallback_name}]...")
                    return self._execute_request(anthropic_req, fallback_cfg, allow_fallback=False)

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
        is_hybrid = cfg.get("enable_hybrid_router", False) or cfg.get("is_hybrid", False)
        all_providers = cfg.get("all_providers", {})
        req_model = (anthropic_req.get("model") or "").lower()
        is_haiku = "haiku" in req_model

        # Detect if request contains images (vision task)
        has_images = False
        for m in anthropic_req.get("messages", []):
            c = m.get("content")
            if isinstance(c, list):
                for b in c:
                    if isinstance(b, dict):
                        if b.get("type") == "image":
                            has_images = True
                            break
                        if b.get("type") == "tool_result" and isinstance(b.get("content"), list):
                            for sb in b.get("content"):
                                if isinstance(sb, dict) and sb.get("type") == "image":
                                    has_images = True
                                    break
            if has_images:
                break

        fallback_cfg = None
        fallback_name = ""
        allow_fallback = False

        if is_hybrid:
            primary_name = cfg.get("hybrid_primary_provider") or cfg.get("primary_provider") or "Atria ASI"
            secondary_name = cfg.get("hybrid_secondary_provider") or cfg.get("secondary_provider") or "Agent Router"
            primary_cfg = dict(all_providers.get(primary_name, cfg))
            secondary_cfg = dict(all_providers.get(secondary_name, {}))

            for k in ("thinking_mode", "auto_save_images", "multi_key_rotation"):
                if k in cfg:
                    primary_cfg[k] = cfg[k]
                    secondary_cfg[k] = cfg[k]

            allow_fallback = cfg.get("hybrid_fallback", True)
            if has_images and secondary_cfg.get("router_url"):
                active_cfg = secondary_cfg
                fallback_cfg = primary_cfg
                fallback_name = primary_name
                self.emit_log(f"Hybrid Route: Vision Task (Image attached) -> [{secondary_name}] (vision-capable)")
            elif is_haiku and secondary_cfg.get("router_url"):
                active_cfg = secondary_cfg
                fallback_cfg = primary_cfg
                fallback_name = primary_name
                self.emit_log(f"Hybrid Route: Fast Task (Haiku) -> [{secondary_name}]")
            else:
                active_cfg = primary_cfg
                fallback_cfg = secondary_cfg
                fallback_name = secondary_name
                self.emit_log(f"Hybrid Route: Heavy Task (Sonnet/Opus) -> [{primary_name}]")
        else:
            active_cfg = cfg

        router_url = active_cfg.get("router_url", "https://inference.dahl.global/v1").rstrip("/")
        api_key = active_cfg.get("api_key", "").strip()
        
        # Multi-model resolution
        default_model = active_cfg.get("model", "deepseek-ai/DeepSeek-V4-Flash-0731").strip()
        sonnet_model = active_cfg.get("sonnet_model", default_model).strip()
        opus_model = active_cfg.get("opus_model", "MiniMaxAI/MiniMax-M2.7").strip()
        haiku_model = active_cfg.get("haiku_model", "zai-org/GLM-5.3-Flash").strip()

        if "opus" in req_model and opus_model:
            target_model = opus_model
        elif "haiku" in req_model and haiku_model:
            target_model = haiku_model
        elif "sonnet" in req_model and sonnet_model:
            target_model = sonnet_model
        elif anthropic_req.get("model") and any(
            anthropic_req.get("model") == m for m in (default_model, sonnet_model, opus_model, haiku_model) if m
        ):
            target_model = anthropic_req.get("model")
        else:
            target_model = default_model or sonnet_model
        
        # Check if router uses Anthropic-native protocol (e.g. AgentRouter)
        is_anthropic_native = ("agentrouter" in router_url.lower()) or ("anthropic.com" in router_url.lower()) or (active_cfg.get("protocol") == "anthropic")
        if is_anthropic_native:
            self._handle_anthropic_native(anthropic_req, target_model, router_url, api_key, active_cfg,
                                          allow_fallback=allow_fallback, fallback_cfg=fallback_cfg, fallback_name=fallback_name)
            return

        # Build target OpenAI endpoint
        endpoint = f"{router_url}/chat/completions"
        auto_save_images = active_cfg.get("auto_save_images", True)
        openai_req = convert_anthropic_to_openai(anthropic_req, target_model, auto_save_images=auto_save_images)

        # Check payload size for routers with request body limits (e.g. Atria TokenPlan 850KB ceiling)
        is_atria = ("atria" in router_url.lower()) or ("atria" in (target_model or "").lower())
        safe_ceiling = 650_000 if is_atria else 2_000_000
        try:
            original_bytes = len(json.dumps(openai_req).encode("utf-8"))
        except Exception:
            original_bytes = 0
        if original_bytes > safe_ceiling:
            self.emit_log(f"Request payload ({original_bytes:,} bytes) exceeds safe ceiling ({safe_ceiling:,} bytes). Compacting earlier context...")
            openai_req["messages"] = trim_openai_messages(openai_req.get("messages", []), max_bytes=safe_ceiling)
            try:
                new_bytes = len(json.dumps(openai_req).encode("utf-8"))
                self.emit_log(f"Context compacted: {original_bytes:,} -> {new_bytes:,} bytes")
            except Exception:
                pass

        self.emit_log(f"Claude Request -> {target_model} via {router_url}")
        
        api_keys = parse_api_keys(api_key)
        multi_key_enabled = active_cfg.get("multi_key_rotation", True)
        current_key = select_api_key(router_url, api_keys, multi_key_enabled)

        req_headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {current_key}" if current_key else ""
        }
        
        is_stream = anthropic_req.get("stream", True)
        openai_req["stream"] = is_stream
        if is_stream:
            openai_req["stream_options"] = {"include_usage": True}
        
        msg_id = f"msg_{uuid.uuid4().hex[:16]}"
        
        try:
            client = _get_client()
            if is_stream:
                # Stream response
                with client.stream("POST", endpoint, headers=req_headers, json=openai_req) as resp:
                    # 1. Multi-key failover retry on 429/402
                    if resp.status_code in (429, 402) and multi_key_enabled and len(api_keys) > 1:
                        for alt_key in [k for k in api_keys if k != current_key]:
                            old_m = f"...{current_key[-4:]}" if len(current_key) >= 4 else "key"
                            new_m = f"...{alt_key[-4:]}" if len(alt_key) >= 4 else "key"
                            self.emit_log(f"API key {old_m} hit limit ({resp.status_code}). Rotating to {new_m}...")
                            alt_headers = dict(req_headers)
                            alt_headers["Authorization"] = f"Bearer {alt_key}"
                            try:
                                resp.close()
                            except Exception:
                                pass
                            retry_ctx = client.stream("POST", endpoint, headers=alt_headers, json=openai_req)
                            retry_resp = retry_ctx.__enter__()
                            if retry_resp.status_code == 200:
                                resp = retry_resp
                                current_key = alt_key
                                req_headers = alt_headers
                                break
                            elif retry_resp.status_code not in (429, 402):
                                resp = retry_resp
                                current_key = alt_key
                                req_headers = alt_headers
                                break
                            else:
                                try:
                                    retry_ctx.__exit__(None, None, None)
                                except Exception:
                                    pass

                    if resp.status_code != 200:
                        err_body = resp.read().decode("utf-8", errors="replace")
                        is_size_err = ("not supported by TokenPlan" in err_body or resp.status_code == 413 or "too large" in err_body.lower())
                        if is_size_err and len(openai_req.get("messages", [])) > 2:
                            self.emit_log(f"Router rejected payload ({resp.status_code}): TokenPlan limit reached. Rescuing with compacted context...")
                            openai_req["messages"] = trim_openai_messages(openai_req["messages"], max_bytes=450_000)
                            try:
                                resp.close()
                            except Exception:
                                pass
                            retry_ctx = client.stream("POST", endpoint, headers=req_headers, json=openai_req)
                            retry_resp = retry_ctx.__enter__()
                            if retry_resp.status_code == 200:
                                resp = retry_resp
                            else:
                                err_body = retry_resp.read().decode("utf-8", errors="replace")
                                try:
                                    retry_ctx.__exit__(None, None, None)
                                except Exception:
                                    pass

                        if resp.status_code != 200:
                            # Router fallback
                            if allow_fallback and fallback_cfg and fallback_cfg.get("router_url"):
                                self.emit_log(f"Router Error ({resp.status_code}): Failing over to fallback router [{fallback_name}]...")
                                try:
                                    resp.close()
                                except Exception:
                                    pass
                                fb_url = fallback_cfg.get("router_url", "").rstrip("/")
                                fb_key = fallback_cfg.get("api_key", "").strip()
                                fb_model = fallback_cfg.get("model", target_model).strip()
                                fb_is_anthropic = ("agentrouter" in fb_url.lower()) or ("anthropic.com" in fb_url.lower()) or (fallback_cfg.get("protocol") == "anthropic")
                                if fb_is_anthropic:
                                    self._handle_anthropic_native(anthropic_req, fb_model, fb_url, fb_key, fallback_cfg, allow_fallback=False)
                                    return
                                else:
                                    fb_endpoint = f"{fb_url}/chat/completions"
                                    fb_req = convert_anthropic_to_openai(anthropic_req, fb_model, auto_save_images=fallback_cfg.get("auto_save_images", True))
                                    fb_req["stream"] = is_stream
                                    if is_stream:
                                        fb_req["stream_options"] = {"include_usage": True}
                                    fb_keys = parse_api_keys(fb_key)
                                    fb_cur_key = select_api_key(fb_url, fb_keys, fallback_cfg.get("multi_key_rotation", True))
                                    fb_headers = {"Content-Type": "application/json", "Authorization": f"Bearer {fb_cur_key}" if fb_cur_key else ""}
                                    fb_ctx = client.stream("POST", fb_endpoint, headers=fb_headers, json=fb_req)
                                    fb_resp = fb_ctx.__enter__()
                                    if fb_resp.status_code == 200:
                                        resp = fb_resp
                                    else:
                                        fb_err = fb_resp.read().decode("utf-8", errors="replace")
                                        try:
                                            fb_ctx.__exit__(None, None, None)
                                        except Exception:
                                            pass
                                        self.emit_log(f"Fallback Router Error ({fb_resp.status_code}): {fb_err}")
                                        self.send_response(fb_resp.status_code)
                                        self.send_header("Content-Type", "application/json")
                                        self.send_header("Content-Length", str(len(fb_err.encode("utf-8"))))
                                        self.send_header("Access-Control-Allow-Origin", "*")
                                        self.end_headers()
                                        self.wfile.write(fb_err.encode("utf-8"))
                                        return
                            else:
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
                    stream_in = 0
                    stream_out = 0

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

                        # Usage may arrive on the final chunk (usage field) or on a
                        # dedicated [DONE] sentinel line; accumulate whichever appears.
                        if chunk.get("usage"):
                            u = chunk["usage"]
                            stream_in += u.get("prompt_tokens", 0) or 0
                            stream_out += u.get("completion_tokens", 0) or 0

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

                    # Message delta. BY8 sums input + cache_creation + cache_read, so
                    # all three must be present or Claude Code's autocompact counter
                    # stays at zero and compaction never fires.
                    msg_delta = {
                        "type": "message_delta",
                        "delta": {
                            "stop_reason": "tool_use" if has_tools else "end_turn",
                            "stop_sequence": None
                        },
                        "usage": {
                            "input_tokens": stream_in,
                            "output_tokens": stream_out,
                            "cache_read_input_tokens": 0,
                            "cache_creation_input_tokens": 0
                        }
                    }
                    self.wfile.write(f"event: message_delta\ndata: {json.dumps(msg_delta)}\n\n".encode("utf-8"))

                    # Message stop
                    self.wfile.write(b"event: message_stop\ndata: {\"type\": \"message_stop\"}\n\n")
                    self.wfile.flush()
                    add_tokens(stream_in, stream_out)
                    self.emit_log(f"Response successfully streamed to Claude ({'Tool use' if has_tools else 'Text'})")

            else:
                # Non-streaming
                resp = client.post(endpoint, headers=req_headers, json=openai_req)
                # 1. Multi-key failover retry on 429/402
                if resp.status_code in (429, 402) and multi_key_enabled and len(api_keys) > 1:
                    for alt_key in [k for k in api_keys if k != current_key]:
                        old_m = f"...{current_key[-4:]}" if len(current_key) >= 4 else "key"
                        new_m = f"...{alt_key[-4:]}" if len(alt_key) >= 4 else "key"
                        self.emit_log(f"API key {old_m} hit limit ({resp.status_code}). Rotating to {new_m}...")
                        alt_headers = dict(req_headers)
                        alt_headers["Authorization"] = f"Bearer {alt_key}"
                        retry_resp = client.post(endpoint, headers=alt_headers, json=openai_req)
                        if retry_resp.status_code == 200:
                            resp = retry_resp
                            break
                        elif retry_resp.status_code not in (429, 402):
                            resp = retry_resp
                            break

                if resp.status_code != 200:
                    err_body = resp.text
                    is_size_err = ("not supported by TokenPlan" in err_body or resp.status_code == 413 or "too large" in err_body.lower())
                    if is_size_err and len(openai_req.get("messages", [])) > 2:
                        self.emit_log(f"Router rejected non-streaming payload ({resp.status_code}): TokenPlan limit reached. Rescuing with compacted context...")
                        openai_req["messages"] = trim_openai_messages(openai_req["messages"], max_bytes=450_000)
                        resp = client.post(endpoint, headers=req_headers, json=openai_req)

                if resp.status_code != 200:
                    if allow_fallback and fallback_cfg and fallback_cfg.get("router_url"):
                        self.emit_log(f"Router Error ({resp.status_code}): Failing over to fallback router [{fallback_name}]...")
                        fb_url = fallback_cfg.get("router_url", "").rstrip("/")
                        fb_key = fallback_cfg.get("api_key", "").strip()
                        fb_model = fallback_cfg.get("model", target_model).strip()
                        fb_is_anthropic = ("agentrouter" in fb_url.lower()) or ("anthropic.com" in fb_url.lower()) or (fallback_cfg.get("protocol") == "anthropic")
                        if fb_is_anthropic:
                            self._handle_anthropic_native(anthropic_req, fb_model, fb_url, fb_key, fallback_cfg, allow_fallback=False)
                            return
                        else:
                            fb_endpoint = f"{fb_url}/chat/completions"
                            fb_req = convert_anthropic_to_openai(anthropic_req, fb_model, auto_save_images=fallback_cfg.get("auto_save_images", True))
                            fb_req["stream"] = False
                            fb_keys = parse_api_keys(fb_key)
                            fb_cur_key = select_api_key(fb_url, fb_keys, fallback_cfg.get("multi_key_rotation", True))
                            fb_headers = {"Content-Type": "application/json", "Authorization": f"Bearer {fb_cur_key}" if fb_cur_key else ""}
                            fb_resp = client.post(fb_endpoint, headers=fb_headers, json=fb_req)
                            if fb_resp.status_code == 200:
                                resp = fb_resp
                            else:
                                self.emit_log(f"Fallback Router Error ({fb_resp.status_code}): {fb_resp.text}")
                                self.send_response(fb_resp.status_code)
                                self.send_header("Content-Type", "application/json")
                                self.send_header("Content-Length", str(len(fb_resp.content)))
                                self.send_header("Access-Control-Allow-Origin", "*")
                                self.end_headers()
                                self.wfile.write(fb_resp.content)
                                return
                    else:
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

                # The non-Claude router reports no cache tokens, but Claude Code's
                # autocompact counter (BY8) sums input + cache_creation + cache_read.
                # Omitting them zeroes the counter and compaction never fires.
                router_usage = oresp.get("usage", {})
                input_tokens = router_usage.get("prompt_tokens", 0) or 0
                cache_read = router_usage.get("cached_prompt_tokens", 0) or 0
                cache_creation = router_usage.get("cache_creation_tokens", 0) or 0

                anthropic_resp = {
                    "id": msg_id,
                    "type": "message",
                    "role": "assistant",
                    "model": anthropic_req.get("model", target_model),
                    "content": content_blocks,
                    "stop_reason": stop_reason,
                    "usage": {
                        "input_tokens": input_tokens,
                        "output_tokens": router_usage.get("completion_tokens", 0) or 0,
                        "cache_read_input_tokens": cache_read,
                        "cache_creation_input_tokens": cache_creation
                    }
                }
                add_tokens(input_tokens + cache_read + cache_creation,
                           router_usage.get("completion_tokens", 0) or 0)
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

