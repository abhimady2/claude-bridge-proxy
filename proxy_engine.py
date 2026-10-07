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

# ponytail: one shared pooled client instead of one per request; egress
# proxies (per-provider `egress_proxy`) are threaded through the Transport.
_client = None
_client_lock = threading.Lock()

_anthropic_clients = {}
_anthropic_lock = threading.Lock()


def _proxy_arg(proxy_url: str):
    """httpx accepts a proxy string (http/socks5) or a Transport per URL."""
    return proxy_url or None


def _maketransport(proxy_url):
    if not proxy_url:
        return None
    return httpx.HTTPTransport(
        proxy=_proxy_arg(proxy_url),
        limits=httpx.Limits(
            max_keepalive_connections=8,
            max_connections=32,
        ),
        trust_env=False,
    )


def _get_anthropic_client(api_key, base_url, proxy_url=None):
    key = (api_key, base_url, proxy_url)
    with _anthropic_lock:
        if key not in _anthropic_clients:
            headers = {}
            if "agentrouter" in base_url.lower():
                headers["User-Agent"] = "claude-cli/1.0.0 (external, cli)"
            transport = _maketransport(proxy_url)
            _anthropic_clients[key] = anthropic.Anthropic(
                api_key=api_key if api_key else "placeholder",
                base_url=base_url,
                default_headers=headers if headers else None,
                timeout=600.0,
                max_retries=2,
                transport=transport,
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
    """Round-robin selection of active API key across requests.

    Rotation is per router and independent of hybrid routing, so a normal
    (non-hybrid) chat spreads across the whole pool when the rotate-keys option
    is on. The pick is recorded for the app's live key view.
    """
    if not api_keys_list:
        return ""
    if not multi_key_enabled or len(api_keys_list) == 1:
        key = api_keys_list[0]
        _record_key_use(router_url, key, api_keys_list)
        return key
    with _key_lock:
        idx = _key_round_robin.get(router_url, 0)
        key = api_keys_list[idx % len(api_keys_list)]
        _key_round_robin[router_url] = idx + 1
    _record_key_use(router_url, key, api_keys_list)
    return key


def next_untried_key(router_url, api_keys_list, tried):
    """Next key in round-robin order that hasn't been attempted yet for this request.

    Used during 429/402 failover: walks the whole pool (not just one alternate)
    and advances the shared counter so the pool keeps rotating after a limit hit.
    Returns None once every key has been tried, so the caller can hand off to the
    other router instead of erroring.
    """
    if not api_keys_list:
        return None
    with _key_lock:
        start = _key_round_robin.get(router_url, 0)
        for i in range(len(api_keys_list)):
            key = api_keys_list[(start + i) % len(api_keys_list)]
            if key not in tried:
                _key_round_robin[router_url] = start + i + 1
                return key
    return None


# --- Key usage telemetry (live view in the app) -------------------------------
# Counts are taken at selection time, which is the one point every request path
# passes through. A key swapped in by 429 rotation mid-request is not counted
# separately -- the next request's selection corrects the active marker.
_key_stats = {}          # router_url -> {"order": [key, ...], "counts": {key: n}, "active": key}
_key_stats_lock = threading.Lock()


def mask_key(key):
    """Identifiable but not copyable: dahl_…Uz8f. Never returns a whole key."""
    if not key:
        return "(none)"
    if len(key) <= 8:
        return "…" + key[-4:]
    return f"{key[:5]}…{key[-4:]}"


def _record_key_use(router_url, key, pool):
    if not key:
        return
    with _key_stats_lock:
        st = _key_stats.get(router_url)
        if st is None:
            st = {"order": [], "counts": {}, "active": ""}
            _key_stats[router_url] = st
        for k in pool or [key]:
            if k not in st["counts"]:
                st["counts"][k] = 0
                st["order"].append(k)
        st["counts"][key] = st["counts"].get(key, 0) + 1
        st["active"] = key


def get_key_stats():
    """Per-router key rotation snapshot for the app: masked keys, request counts,
    and which one is live. Masked so a screenshot never leaks a key."""
    with _key_stats_lock:
        return {
            url: {
                "active": mask_key(st.get("active", "")),
                "keys": [
                    {"masked": mask_key(k), "requests": st["counts"].get(k, 0)}
                    for k in st["order"]
                ],
            }
            for url, st in _key_stats.items()
        }


# --- Image to Local Disk Bridge (Vision for Atria & TokenPlan limit fix) -------

# Routers that 400-rejected an inline image_url part get remembered here and
# every later request to them strips images to file paths up front, instead of
# paying a failed round-trip each time. Process-lifetime only; a restart resets
# it, which is also the escape hatch if a router fixes itself.
_inline_image_rejects = set()
_inline_image_lock = threading.Lock()


def router_rejects_inline_images(router_url):
    with _inline_image_lock:
        return router_url in _inline_image_rejects


def mark_router_rejects_inline_images(router_url):
    with _inline_image_lock:
        _inline_image_rejects.add(router_url)


def strip_inline_images(openai_req):
    """Replace every inline image_url part in an OpenAI request with a text
    note carrying the file path that save_base64_image already wrote."""
    for msg in openai_req.get("messages", []):
        c = msg.get("content")
        if not isinstance(c, list):
            continue
        kept = []
        for part in c:
            if isinstance(part, dict) and part.get("type") == "image_url":
                url = (part.get("image_url") or {}).get("url", "")
                media = "image"
                if url.startswith("data:"):
                    media = url[5:].split(";", 1)[0] or media
                kept.append({
                    "type": "text",
                    "text": (f"[Image ({media}) omitted: this router does not accept "
                             f"inline image data. Ask the user to describe it or use a "
                             f"local image tool if available.]")
                })
            else:
                kept.append(part)
        msg["content"] = kept
    return openai_req

# Media types OpenAI-compatible vision endpoints accept inline. Anything else
# (svg, bmp, ...) is passed as a disk path rather than risking a 400.
VISION_MEDIA_TYPES = {"image/png", "image/jpeg", "image/jpg", "image/webp", "image/gif"}


def clean_base64(data):
    """Strips a data URL prefix (data:image/png;base64,...) if one is present."""
    if data and "," in data and "base64" in data[:50]:
        return data.split(",", 1)[1]
    return data


def save_base64_image(media_type, base64_data, save_dir=None):
    """
    Decodes a base64 image and saves it to a persistent local temp folder.
    Returns normalized forward-slash absolute path so agent tools and scripts can read it.
    """
    try:
        if not base64_data:
            return None

        base64_data = clean_base64(base64_data)

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


def process_anthropic_images(messages, auto_save=True, strip_images=False):
    """
    Scans Anthropic messages for image blocks.

    auto_save=True writes a copy of each image to disk so agent tools and scripts
    can still reach the file. strip_images=True replaces the image block with that
    path, for routers whose gateway rejects inline image data or enforces a tight
    request body ceiling (e.g. Atria TokenPlan). Otherwise the image passes
    through untouched to the vision-capable target.
    """
    if (not auto_save and not strip_images) or not messages:
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
                    src = b.get("source") or {}
                    if src.get("type") == "base64" and src.get("data"):
                        saved_path = save_base64_image(src.get("media_type", "image/png"), src.get("data")) if auto_save else None
                        if strip_images and saved_path:
                            new_content.append({
                                "type": "text",
                                "text": (f"[An image was attached but this model/router cannot receive "
                                         f"image data, so it has been omitted. Do NOT try to Read or "
                                         f"open {saved_path} -- the result would be stripped the same "
                                         f"way. Tell the user you cannot view the image and ask them "
                                         f"to describe it in words.]")
                            })
                            continue
                elif isinstance(b, dict) and b.get("type") == "tool_result":
                    tr_c = b.get("content")
                    if isinstance(tr_c, list):
                        new_tr_c = []
                        for sub_b in tr_c:
                            if isinstance(sub_b, dict) and sub_b.get("type") == "image":
                                src = sub_b.get("source") or {}
                                if src.get("type") == "base64" and src.get("data"):
                                    p = save_base64_image(src.get("media_type", "image/png"), src.get("data")) if auto_save else None
                                    if strip_images and p:
                                        new_tr_c.append({
                                            "type": "text",
                                            "text": (f"[A tool returned an image, but this model/router "
                                                     f"cannot receive image data so it was omitted. Do NOT "
                                                     f"Read {p} to retry -- it would be stripped again. "
                                                     f"Continue without the image or ask the user to "
                                                     f"describe it.]")
                                        })
                                        continue
                            new_tr_c.append(sub_b)
                        b_copy = dict(b)
                        b_copy["content"] = new_tr_c
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
    """Shared pooled httpx.Client for the OpenAI-style router paths."""
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


# ponytail: per-provider egress proxies. `_get_client_for` hands back either
# the shared proxy-less client or a per-proxy ephemeral httpx.Client; both are
# bound to the request already (the per-proxy one gets closed at end of the
# request that used it). Anthropic-native paths pass the proxy to
# `_get_anthropic_client`'s transport instead.
_proxy_clients = {}
_proxy_clients_lock = threading.Lock()


def _client_for_url(proxy_url):
    """Return a fixed client for a proxy; locks the proxy to its own pool.

    ponytail: most requests never use the proxy-path branch (no proxy set), so
    the pool only ever holds proxies the user actually configured. Since httpx
    cannot re-point an open connection + pool at a differently-proxied host, a
    proxy'd client cannot be shared with the plain one; the proxy path keeps
    its own small pool instead.
    """
    if not proxy_url:
        return _get_client(), False
    with _proxy_clients_lock:
        client = _proxy_clients.get(proxy_url)
    if client is None:
        client = httpx.Client(
            transport=_maketransport(proxy_url),
            limits=httpx.Limits(max_keepalive_connections=8, max_connections=32),
            timeout=httpx.Timeout(connect=10.0, read=600.0, write=60.0, pool=10.0),
        )
        with _proxy_clients_lock:
            _proxy_clients[proxy_url] = client
    return client, True


def close_proxy_clients():
    with _proxy_clients_lock:
        for c in _proxy_clients.values():
            try:
                c.close()
            except Exception:
                pass
        _proxy_clients.clear()


# Upstream SSE streams can stall mid-response with no bytes (dead router, dropped
# TCP, overloaded inference). httpx would then block for the full 600s read timeout
# and Claude Code would sit waiting for a terminal event until manually nudged.
# This bounds the *idle* gap between chunks; a healthy long generation keeps
# sending chunks and is unaffected.
STREAM_IDLE_TIMEOUT = 240.0

# ponytail: `read` bounds the idle gap between chunks, not total generation
# time, so a healthy long response is unaffected -- a dead upstream releases
# the turn in STREAM_IDLE_TIMEOUT instead of the 600s default. Built per call
# (not a constant) so tests can retune STREAM_IDLE_TIMEOUT at runtime.
def _stream_timeout():
    return httpx.Timeout(connect=10.0, read=STREAM_IDLE_TIMEOUT, write=60.0, pool=10.0)


_NONSTREAM_TIMEOUT = httpx.Timeout(connect=10.0, read=300.0, write=60.0, pool=10.0)

# The hybrid fast lane (Haiku/vision routers) is for small cheap turns. A
# compaction request carries the whole conversation as text; sending that to a
# small fast model stalls it, which is the "stuck compacting" failure in hybrid
# mode. Above this much *text*, a request takes the heavy lane regardless of
# model hint. Image bytes don't count -- a screenshot turn is still a small task.
FAST_LANE_MAX_TEXT_BYTES = 200_000

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
    # The cache holds the whole model->length table per router, not one entry
    # per (router, model): routers that report nothing send every listed model
    # its own serial /models round-trip otherwise, and /v1/models lists ~6.
    wanted = (model or "").strip().lower()
    with _context_cache_lock:
        table = _context_cache.get(router_url)
        if table is not None and wanted in table:
            return table[wanted]

    resolved = _lookup_known_context(model)
    if resolved is None:
        # Ask the router. Most return no context_length; 401/404/timeout just
        # fall through to the default. The whole model table is fetched once
        # per router: after that, unknown models resolve to the default from
        # the cache instead of each paying their own round-trip.
        try:
            with _context_cache_lock:
                already_probed = _context_cache.get(router_url, {}).get("__probed__", False)
            if not already_probed:
                base = router_url.rstrip("/")
                if not base.endswith("/models"):
                    base = f"{base}/models"
                headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
                resp = _get_client().get(base, headers=headers, timeout=timeout)
                if resp.status_code == 200:
                    payload = resp.json()
                    items = payload.get("data") or payload.get("models") or []
                    fetched = {"__probed__": True}
                    for item in items:
                        if not isinstance(item, dict):
                            continue
                        length = item.get("context_length") or item.get("max_context_length")
                        if isinstance(length, int) and length > 0:
                            fetched[(item.get("id") or "").strip().lower()] = length
                    with _context_cache_lock:
                        _context_cache.setdefault(router_url, {}).update(fetched)
                else:
                    with _context_cache_lock:
                        _context_cache.setdefault(router_url, {})["__probed__"] = True
        except Exception:
            resolved = None
            # Router unreachable: remember the probe failed so /v1/models
            # doesn't retry it for every listed model this process lifetime.
            with _context_cache_lock:
                _context_cache.setdefault(router_url, {})["__probed__"] = True

    if not resolved:
        resolved = DEFAULT_CONTEXT_LENGTH

    with _context_cache_lock:
        _context_cache.setdefault(router_url, {})[wanted] = resolved
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


def openai_message_content(text_parts, image_parts):
    """OpenAI content is a plain string, or a parts list once images are attached."""
    text = "\n".join(p for p in text_parts if p)
    if not image_parts:
        return text
    return ([{"type": "text", "text": text}] if text else []) + image_parts


def convert_anthropic_to_openai(anthropic_body, target_model, auto_save_images=True, strip_images=False):
    """
    Translates Anthropic Messages API request format to OpenAI Chat Completions format.
    Images are forwarded as inline image_url parts when the target accepts them; the
    disk copy is kept either way. strip_images=True restores the path-only behavior
    for routers with TokenPlan/request size limits (such as Atria) or no vision endpoint.
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
            image_parts = []
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
                    src = block.get("source") or {}
                    data = src.get("data")
                    media_type = (src.get("media_type") or "image/png").lower().strip()
                    if src.get("type") == "base64" and data:
                        saved_path = save_base64_image(media_type, data) if auto_save_images else None
                        if not strip_images and media_type in VISION_MEDIA_TYPES:
                            image_parts.append({
                                "type": "image_url",
                                "image_url": {"url": f"data:{media_type};base64,{clean_base64(data)}"}
                            })
                        elif saved_path:
                            # Must NOT invite the model to read the file back:
                            # this router cannot receive images, so a Read would
                            # return an image block that gets stripped into this
                            # same note again -- an endless read loop.
                            text_parts.append(
                                f"[An image was attached but this model/router cannot receive "
                                f"image data, so it has been omitted from the request. "
                                f"Do NOT try to Read or open {saved_path} -- the result would be "
                                f"stripped the same way. Tell the user you cannot view the image "
                                f"and ask them to describe it in words.]"
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
                                        src = sub_b.get("source") or {}
                                        if src.get("type") == "base64" and src.get("data"):
                                            p = save_base64_image(src.get("media_type", "image/png"), src.get("data"))
                                            if p:
                                                # OpenAI `tool` messages carry text only, so a
                                                # tool-returned image can only be passed as a path.
                                                # Must NOT say "read this file": a Read returns an
                                                # image block that lands right back here, so the
                                                # model would read its own output forever.
                                                parts.append(
                                                    f"[A tool returned an image ({p}). OpenAI tool "
                                                    f"messages cannot carry image data, so it is not "
                                                    f"included here. Do NOT Read {p} to retry -- that "
                                                    f"returns another image and repeats this. Continue "
                                                    f"without the image, or ask the user to describe it.]"
                                                )
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
                if text_parts or image_parts:
                    messages.append({"role": role, "content": openai_message_content(text_parts, image_parts)})
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


def _anthropic_text_bytes(anthropic_req):
    """Size of a request's *text* content, base64 image payloads excluded.

    See FAST_LANE_MAX_TEXT_BYTES for why this matters.
    """
    total = 0
    for m in anthropic_req.get("messages") or []:
        c = m.get("content")
        if isinstance(c, str):
            total += len(c.encode("utf-8"))
        elif isinstance(c, list):
            for b in c:
                if not isinstance(b, dict):
                    continue
                if b.get("type") == "text":
                    total += len((b.get("text") or "").encode("utf-8"))
                elif b.get("type") == "tool_result":
                    rc = b.get("content")
                    if isinstance(rc, str):
                        total += len(rc.encode("utf-8"))
                    elif isinstance(rc, list):
                        for sb in rc:
                            if isinstance(sb, dict) and sb.get("type") == "text":
                                total += len((sb.get("text") or "").encode("utf-8"))
    return total


class ProxyRequestHandler(BaseHTTPRequestHandler):
    # HTTP/1.1 so Claude Code can keep-alive one connection to the proxy
    # instead of opening a fresh socket per request.
    protocol_version = "HTTP/1.1"

    config_getter = None  # Function returning dict: {"router_url", "api_key", "model"}
    log_callback = None   # Function(msg: str)

    # Ordered failover tail for the request being handled: [(name, cfg), ...]
    # after the active router. Per-request because one handler instance is
    # built per request.
    _failover_chain = None

    def log_message(self, format, *args):
        # Override to prevent default stderr logging
        pass

    def emit_log(self, text):
        if self.log_callback:
            self.log_callback(text)

    def _close_sse_stream(self, reason="", input_tokens=0, output_tokens=0):
        """Best-effort terminal SSE sequence for a stream that already sent its
        200 headers and then died mid-response. Without this Claude Code keeps
        the turn open waiting for message_stop, which is the "chat goes dead
        until nudged" symptom.

        The message_delta matters as much as the stop: Claude Code finalizes the
        turn on its stop_reason, and its autocompact counter reads usage from
        this block. A bare message_stop left the counter unsettled, so a turn
        that died during compaction could re-fire compaction on itself.
        """
        if reason:
            self.emit_log(reason)
        try:
            delta = {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {
                    "input_tokens": int(input_tokens or 0),
                    "output_tokens": int(output_tokens or 0),
                    "cache_read_input_tokens": 0,
                    "cache_creation_input_tokens": 0,
                },
            }
            self.wfile.write(f"event: message_delta\ndata: {json.dumps(delta)}\n\n".encode("utf-8"))
            self.wfile.write(b'event: message_stop\ndata: {"type": "message_stop"}\n\n')
            self.wfile.flush()
        except Exception:
            pass

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
                                allow_fallback=False, fallback_cfg=None, fallback_name="", tb=None):
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
        tried_keys = set()  # keys already attempted for this request (429 failover)
        egress_proxy = (cfg.get("egress_proxy") or "").strip()
        client = _get_anthropic_client(current_key, base_url, egress_proxy)
        is_stream = anthropic_req.get("stream", True)
        # True once the 200 + SSE headers went out. An exception after that
        # point cannot send a fresh HTTP status, so the turn must be closed
        # with a terminal SSE event instead or Claude Code waits forever.
        stream_started = False
        thinking_mode = cfg.get("thinking_mode", "thinking_block")

        max_toks = anthropic_req.get("max_tokens", 4096)
        if isinstance(max_toks, (int, float)):
            max_toks = max(1, min(int(max_toks), 65536))

        auto_save_images = cfg.get("auto_save_images", True)
        processed_msgs = process_anthropic_images(anthropic_req.get("messages", []),
                                                  auto_save=auto_save_images,
                                                  strip_images=cfg.get("strip_images", False))

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
            # Visible to the except clauses below even if the upstream dies
            # before the streaming loop starts.
            stream_in = 0
            stream_out = 0
            try:
                with client.messages.with_streaming_response.create(
                        **req_dict, stream=True, timeout=STREAM_IDLE_TIMEOUT) as resp:
                    # 1. Multi-key failover retry on 429/402
                    if resp.status_code in (429, 402) and multi_key_enabled:
                        tried_keys.add(current_key)
                        alt_key = next_untried_key(router_url, api_keys, tried_keys)
                        if alt_key:
                            old_m = f"...{current_key[-4:]}" if len(current_key) >= 4 else "key"
                            new_m = f"...{alt_key[-4:]}" if len(alt_key) >= 4 else "key"
                            self.emit_log(f"Key {old_m} hit limit ({resp.status_code}). Rotating to {new_m}...")
                            current_key = alt_key
                            client = _get_anthropic_client(alt_key, base_url, egress_proxy)
                            tried_keys.add(alt_key)
                            return _execute_streaming(req_dict)
                        if not egress_proxy and len(api_keys) > 1:
                            self.emit_log("Every key hit 429 - looks like an IP rate limit. Set an Egress Proxy on this provider to route around it.")

                    if resp.status_code != 200:
                        err_body = resp.read().decode("utf-8", errors="replace")
                        if allow_fallback and fallback_cfg and fallback_cfg.get("router_url"):
                            self.emit_log(f"Anthropic Router Error ({resp.status_code}): Failing over to fallback router [{fallback_name}]...")
                            return self._failover_request(anthropic_req, fallback_cfg, tb, fallback_name)
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
                    stream_started = True

                    is_filtering_billing = False
                    ignoring_thinking = False
                    saw_stop = False
                    stream_in = 0
                    stream_out = 0
                    # The read timeout bounds the gap between socket reads, but
                    # an upstream can hold the socket open on SSE comment lines
                    # (": keep-alive") while delivering nothing, and then the
                    # client waits forever. Every data: line resets this; only a
                    # run of comments (or silence) trips it.
                    last_progress = time.monotonic()
                    for line in resp.iter_lines():
                        if line.startswith(":"):
                            if time.monotonic() - last_progress > STREAM_IDLE_TIMEOUT:
                                self.emit_log(
                                    f"Upstream sent only SSE keep-alives for "
                                    f"{int(STREAM_IDLE_TIMEOUT)}s; abandoning stalled stream.")
                                break
                            continue
                        if line.startswith("data:"):
                            last_progress = time.monotonic()
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
                            if line.startswith("event: message_stop"):
                                saw_stop = True
                        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                            break
                    add_tokens(stream_in, stream_out)
                    if not saw_stop:
                        # Upstream ended (clean close or truncation) without a
                        # terminal event: close the turn or Claude Code waits
                        # on a stream that will never emit message_stop.
                        self._close_sse_stream(
                            "Upstream stream ended without message_stop; "
                            "closing turn so Claude Code can continue.",
                            stream_in, stream_out)
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
                if status in (429, 402) and multi_key_enabled:
                    tried_keys.add(current_key)
                    alt_key = next_untried_key(router_url, api_keys, tried_keys)
                    if alt_key:
                        old_m = f"...{current_key[-4:]}" if len(current_key) >= 4 else "key"
                        new_m = f"...{alt_key[-4:]}" if len(alt_key) >= 4 else "key"
                        self.emit_log(f"Key {old_m} hit limit ({status}). Failing over to next key {new_m}...")
                        current_key = alt_key
                        client = _get_anthropic_client(alt_key, base_url, egress_proxy)
                        tried_keys.add(alt_key)
                        return _execute_streaming(req_dict)
                    if not egress_proxy and len(api_keys) > 1:
                        self.emit_log("Every key hit 429 - likely an IP rate limit. Set an Egress Proxy on this provider to route around it.")

                # Fallback to default_model if target_model failed due to quota (402) or unavailable (404/503)
                if status in (400, 402, 404, 503) and default_model and req_dict.get("model") != default_model:
                    self.emit_log(f"Model '{req_dict.get('model')}' failed (HTTP {status}). Falling back to default model '{default_model}'...")
                    new_req = dict(req_dict)
                    new_req["model"] = default_model
                    return _execute_streaming(new_req)

                # Failover to secondary router if available
                if status in (429, 500, 502, 503, 504, 520, 521, 522, 523, 524) and allow_fallback and fallback_cfg and fallback_cfg.get("router_url"):
                    self.emit_log(f"Anthropic Router Error ({status}): Failing over to fallback router [{fallback_name}]...")
                    return self._failover_request(anthropic_req, fallback_cfg, tb, fallback_name)

                if stream_started:
                    self._close_sse_stream(
                        f"Upstream stream failed mid-response (HTTP {status}: {err_body[:200]}); "
                        f"closing turn so Claude Code can continue.",
                        stream_in, stream_out)
                    return
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
            except anthropic.APIConnectionError as ex:
                # Timeout or dead socket before the response started: the
                # failover chain exists for exactly this. Without it a hung
                # primary router raises here, the generic handler below 500s,
                # and Claude Code re-sends the same request to the same dead
                # router -- the "stuck compacting" loop in hybrid mode.
                if stream_started:
                    self._close_sse_stream(
                        f"Upstream stream failed mid-response ({type(ex).__name__}: {ex}); "
                        f"closing turn so Claude Code can continue.",
                        stream_in, stream_out,
                    )
                    return
                if allow_fallback and fallback_cfg and fallback_cfg.get("router_url"):
                    self.emit_log(
                        f"Router unreachable ({type(ex).__name__}: {ex}); "
                        f"failing over to [{fallback_name}]...")
                    return self._failover_request(anthropic_req, fallback_cfg, tb, fallback_name)
                self.emit_log(f"Proxy Connection Error: {str(ex)}")
                try:
                    payload = json.dumps({"error": str(ex)}).encode("utf-8")
                    self.send_response(504)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(payload)))
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.wfile.write(payload)
                except Exception:
                    pass
            except Exception as ex:
                if stream_started:
                    self._close_sse_stream(
                        f"Upstream stream failed mid-response ({type(ex).__name__}: {ex}); "
                        f"closing turn so Claude Code can continue.",
                        stream_in, stream_out,
                    )
                    return
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
                resp = client.messages.create(**req_dict, stream=False, timeout=300.0)
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
                if status in (429, 402) and multi_key_enabled:
                    tried_keys.add(current_key)
                    alt_key = next_untried_key(router_url, api_keys, tried_keys)
                    if alt_key:
                        old_m = f"...{current_key[-4:]}" if len(current_key) >= 4 else "key"
                        new_m = f"...{alt_key[-4:]}" if len(alt_key) >= 4 else "key"
                        self.emit_log(f"Key {old_m} hit limit ({status}). Failing over to next key {new_m}...")
                        current_key = alt_key
                        client = _get_anthropic_client(alt_key, base_url, egress_proxy)
                        tried_keys.add(alt_key)
                        return _execute_non_streaming(req_dict)
                    if not egress_proxy and len(api_keys) > 1:
                        self.emit_log("Every key hit 429 - likely an IP rate limit. Set an Egress Proxy on this provider to route around it.")

                # Fallback to default_model if target_model failed due to quota (402) or unavailable (404/503)
                if status in (400, 402, 404, 503) and default_model and req_dict.get("model") != default_model:
                    self.emit_log(f"Model '{req_dict.get('model')}' failed (HTTP {status}). Falling back to default model '{default_model}'...")
                    new_req = dict(req_dict)
                    new_req["model"] = default_model
                    return _execute_non_streaming(new_req)

                # Failover to secondary router if available
                if status in (429, 500, 502, 503, 504, 520, 521, 522, 523, 524) and allow_fallback and fallback_cfg and fallback_cfg.get("router_url"):
                    self.emit_log(f"Anthropic Router Error ({status}): Failing over to fallback router [{fallback_name}]...")
                    return self._failover_request(anthropic_req, fallback_cfg, tb, fallback_name)

                self.emit_log(f"Router Error ({status}): {err_body}")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                body_bytes = err_body.encode("utf-8")
                self.send_header("Content-Length", str(len(body_bytes)))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(body_bytes)
            except anthropic.APIConnectionError as ex:
                # Same rationale as the streaming path: a hung primary must
                # advance the chain, not 500 and get retried against itself.
                if allow_fallback and fallback_cfg and fallback_cfg.get("router_url"):
                    self.emit_log(
                        f"Router unreachable ({type(ex).__name__}: {ex}); "
                        f"failing over to [{fallback_name}]...")
                    return self._failover_request(anthropic_req, fallback_cfg, tb, fallback_name)
                self.emit_log(f"Proxy Connection Error: {str(ex)}")
                payload = json.dumps({"error": str(ex)}).encode("utf-8")
                self.send_response(504)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(payload)
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


    def _route_request(self, anthropic_req, cfg, tb=None):
        """Resolve active + fallback provider configs for this request.

        cfg carries the flat fields the GUI always writes (router_url, api_key,
        models) plus the hybrid options (enable_hybrid_router, all_providers,
        hybrid_chain, hybrid_fallback).

        The chain is an ordered list of provider names. Heavy Sonnet/Opus tasks
        start at chain[0]; fast Haiku/vision tasks start at chain[1] when the
        chain has 2+ entries. Whatever sits after the chosen router, in chain
        order, is the failover tail (recorded on the handler so _failover_request
        can walk it hop by hop instead of the old single secondary router).

        On a failover re-dispatch tb marks this request as already handed off, so
        the lane split is not re-applied: the remaining chain is walked in order.
        """
        all_providers = cfg.get("all_providers", {}) or {}

        chain_raw = cfg.get("hybrid_chain")
        if isinstance(chain_raw, list):
            # An explicit empty list means "no chain" (single router, no failover).
            chain_names = [str(n).strip() for n in chain_raw if str(n).strip()]
        else:
            # Legacy 2-router config: primary then secondary.
            p = cfg.get("hybrid_primary_provider") or cfg.get("primary_provider") or ""
            s = cfg.get("hybrid_secondary_provider") or cfg.get("secondary_provider") or ""
            chain_names = [p] + ([s] if s else [])
        chain_names = [n for n in chain_names if n in all_providers]

        is_hybrid = bool(cfg.get("enable_hybrid_router") or cfg.get("is_hybrid")) and chain_names

        req_model = (anthropic_req.get("model") or "").lower()
        is_haiku = "haiku" in req_model

        # Detect images in the request body (vision task).
        has_images = False
        for m in anthropic_req.get("messages") or []:
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

        if not is_hybrid:
            self._failover_chain = []
            return dict(cfg), None, "", False

        def _cfg_for(name):
            c = dict(all_providers.get(name) or {})
            if not c.get("router_url"):
                return None
            for k in ("thinking_mode", "auto_save_images", "strip_images", "multi_key_rotation"):
                if k in cfg:
                    c[k] = cfg[k]
            return c

        active_cfg = _cfg_for(chain_names[0])
        if active_cfg is None:
            # First chain entry is unusable (no URL); the flat fields are the
            # last resort rather than erroring out.
            self._failover_chain = []
            return dict(cfg), None, "", False

        # A failover re-dispatch carries its own remaining tail; the fast/heavy
        # lane split already happened on the first entry.
        is_redispatch = bool((tb or {}).get("used_fallback"))
        fast_lane = (has_images or is_haiku) and not is_redispatch and len(chain_names) >= 2
        if fast_lane and _anthropic_text_bytes(anthropic_req) >= FAST_LANE_MAX_TEXT_BYTES:
            self.emit_log(
                f"Hybrid Route: large context (>= {FAST_LANE_MAX_TEXT_BYTES:,} text bytes) "
                f"demoted off the fast lane -> heavy router."
            )
            fast_lane = False
        start = 1 if fast_lane else 0
        ordered = chain_names[start:] + chain_names[:start]

        active_name = ordered[0]
        active_cfg = _cfg_for(active_name) or active_cfg

        # Everything after the active router is the failover tail, walked in
        # order on each upstream error. Per-request state on the handler
        # (one ProxyRequestHandler per request) keeps it off other requests.
        tail = [(n, c) for n in ordered[1:] if (c := _cfg_for(n))]
        self._failover_chain = tail

        allow_fallback = bool(cfg.get("hybrid_fallback", True)) and bool(tail)
        if not tail:
            return active_cfg, None, "", allow_fallback

        next_name, next_cfg = tail[0]
        if fast_lane:
            self.emit_log(
                f"Hybrid Route: {'Vision Task (Image attached)' if has_images else 'Fast Task (Haiku)'} "
                f"-> [{active_name}] (failover: {' -> '.join(n for n, _ in tail)})"
            )
        else:
            self.emit_log(
                f"Hybrid Route: Heavy Task (Sonnet/Opus) -> [{active_name}] "
                f"(failover: {' -> '.join(n for n, _ in tail)})"
            )
        return active_cfg, next_cfg, next_name, allow_fallback

    def _failover_request(self, anthropic_req, fallback_cfg, tb=None, fallback_name=""):
        """Re-dispatch a request to the next router in the failover chain.

        The hop being dispatched now is `fallback_cfg`; the rest of the chain
        (set by _route_request on this request) is passed along as the
        re-dispatch's own hybrid_chain, so the next upstream error advances to
        the following hop. The chain strictly shrinks each hop, so a broken
        router cannot cause a loop.
        """
        fb_name = fallback_name or fallback_cfg.get("_name") or "fallback"
        self.emit_log(f"Failing over to router [{fb_name}]...")
        tail = getattr(self, "_failover_chain", None) or []
        # The hop being dispatched now heads the re-dispatch's own chain; what
        # sat after it arms its failover. The chain strictly shrinks each hop,
        # so a broken router cannot cause a loop.
        rest = [(n, c) for n, c in tail[1:] if c and c.get("router_url")]
        try:
            # Rebuild the provider table around the remaining hops so the
            # re-dispatch walks them in order instead of re-running the
            # fast/heavy lane split on the tail.
            redispatch_cfg = dict(fallback_cfg)
            redispatch_cfg.update({
                "enable_hybrid_router": True,
                "hybrid_fallback": True,
                "hybrid_chain": [fb_name] + [n for n, _ in rest],
                "all_providers": {fb_name: fallback_cfg, **{n: c for n, c in rest}},
            })
            return self._handle_messages(anthropic_req, redispatch_cfg, tb={
                **(tb or {}), "used_fallback": True, "fallback_to": fb_name
            })
        except Exception:
            import traceback
            traceback.print_exc()
            err_body = json.dumps({"error": "fallback router failed"}).encode("utf-8")
            try:
                self.send_response(502)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(err_body)))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(err_body)
            except Exception:
                pass

    def _commit_stream_start(self, target_model, anthropic_req, msg_id):
        """Send 200 + SSE headers + message_start. Deferred until the first
        upstream data byte so a pre-first-byte stall can still fail over --
        the caller owns the stream_started flag."""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.close_connection = True

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

    def _handle_messages(self, anthropic_req, cfg, tb=None):
        """Full message dispatch. cfg is the resolved provider config."""
        tb = tb or {}
        active_cfg, fallback_cfg, fallback_name, allow_fallback = self._route_request(anthropic_req, cfg, tb)

        req_model = (anthropic_req.get("model") or "").lower()
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
                                          allow_fallback=allow_fallback, fallback_cfg=fallback_cfg,
                                          fallback_name=fallback_name, tb=tb)
            return

        # Build target OpenAI endpoint
        endpoint = f"{router_url}/chat/completions"
        auto_save_images = active_cfg.get("auto_save_images", True)
        strip_images = active_cfg.get("strip_images", False) or router_rejects_inline_images(router_url)
        openai_req = convert_anthropic_to_openai(anthropic_req, target_model,
                                                 auto_save_images=auto_save_images,
                                                 strip_images=strip_images)

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
        egress_proxy = (active_cfg.get("egress_proxy") or "").strip()

        req_headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {current_key}" if current_key else ""
        }

        is_stream = anthropic_req.get("stream", True)
        openai_req["stream"] = is_stream
        if is_stream:
            openai_req["stream_options"] = {"include_usage": True}
        
        msg_id = f"msg_{uuid.uuid4().hex[:16]}"

        stream_started = False
        last_retry_err = ""
        # Accumulated usage, visible to the except clauses below even when the
        # upstream dies before the streaming loop assigns them.
        stream_in = 0
        stream_out = 0
        try:
            client, _is_proxied = _client_for_url(egress_proxy)
            if is_stream:
                # Stream response
                with client.stream("POST", endpoint, headers=req_headers, json=openai_req,
                                  timeout=_stream_timeout()) as resp:
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
                            retry_ctx = client.stream("POST", endpoint, headers=alt_headers, json=openai_req, timeout=_stream_timeout())
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
                                # Keep the last retry's error body for the report
                                # below; the loop may exhaust with every key
                                # limited, leaving the original resp closed.
                                try:
                                    last_retry_err = retry_resp.read().decode("utf-8", errors="replace")
                                except Exception:
                                    last_retry_err = ""
                                try:
                                    retry_ctx.__exit__(None, None, None)
                                except Exception:
                                    pass
                        else:
                            if not egress_proxy:
                                self.emit_log("Every key hit 429 - likely an IP rate limit. Set an Egress Proxy on this provider to route around it.")

                    if resp.status_code != 200:
                        try:
                            err_body = resp.read().decode("utf-8", errors="replace")
                        except Exception:
                            # Response was closed by the key-rotation loop above.
                            err_body = last_retry_err or json.dumps(
                                {"error": f"Router returned HTTP {resp.status_code}"})
                        is_size_err = ("not supported by TokenPlan" in err_body or resp.status_code == 413 or "too large" in err_body.lower())
                        # Some OpenAI-style routers (e.g. Dahl) reject inline
                        # image_url parts with a 400. Strip to paths and retry once.
                        is_img_err = (resp.status_code == 400 and "image" in err_body.lower()
                                      and not router_rejects_inline_images(router_url))
                        if is_img_err:
                            mark_router_rejects_inline_images(router_url)
                            self.emit_log(f"Router rejected inline images ({resp.status_code}). Retrying with images as file paths; future requests will skip inline images for this router.")
                            openai_req = strip_inline_images(openai_req)
                            try:
                                resp.close()
                            except Exception:
                                pass
                            retry_ctx = client.stream("POST", endpoint, headers=req_headers, json=openai_req, timeout=_stream_timeout())
                            retry_resp = retry_ctx.__enter__()
                            if retry_resp.status_code == 200:
                                resp = retry_resp
                            else:
                                try:
                                    err_body = retry_resp.read().decode("utf-8", errors="replace")
                                except Exception:
                                    err_body = ""
                                try:
                                    retry_ctx.__exit__(None, None, None)
                                except Exception:
                                    pass
                        elif is_size_err and len(openai_req.get("messages", [])) > 2:
                            self.emit_log(f"Router rejected payload ({resp.status_code}): TokenPlan limit reached. Rescuing with compacted context...")
                            openai_req["messages"] = trim_openai_messages(openai_req["messages"], max_bytes=450_000)
                            try:
                                resp.close()
                            except Exception:
                                pass
                            retry_ctx = client.stream("POST", endpoint, headers=req_headers, json=openai_req, timeout=_stream_timeout())
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
                            # Router fallback (429/402/5xx/all) via unified router
                            if allow_fallback and fallback_cfg and fallback_cfg.get("router_url"):
                                try:
                                    resp.close()
                                except Exception:
                                    pass
                                self._failover_request(anthropic_req, fallback_cfg, tb, fallback_name)
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

                    # 200 + message_start are held back until the first upstream
                    # data byte arrives. A router that stalls before producing
                    # anything (the Dahl freeze) then raises with stream_started
                    # still False, and the TransportError handler below fails
                    # over to the next router instead of closing an empty turn.
                    committed = False
                    thinking_mode = cfg.get("thinking_mode", "thinking_block")
                    thinking_block_started = False
                    text_block_started = False
                    tool_blocks = {}  # index -> {"id", "name", "args"}
                    current_block_index = 0
                    has_tools = False
                    in_think_tag = False
                    # Content is buffered up to one tag-length deep. A <think> tag
                    # split across two SSE chunks must never be emitted: a partial
                    # "<thi" printed as text is the leak seen in chat, and a split
                    # *closing* tag used to leave in_think_tag stuck at True, so
                    # every later chunk landed in the hidden thinking block and
                    # the visible answer was empty -- the "chat went dead" turn.
                    pending = ""
                    stream_in = 0
                    stream_out = 0

                    def _close_text_block():
                        nonlocal text_block_started, current_block_index
                        if not text_block_started:
                            return
                        stop_evt = {"type": "content_block_stop", "index": current_block_index}
                        self.wfile.write(f"event: content_block_stop\ndata: {json.dumps(stop_evt)}\n\n".encode("utf-8"))
                        self.wfile.flush()
                        current_block_index += 1
                        text_block_started = False

                    def _emit_text(text):
                        nonlocal text_block_started
                        if not text:
                            return
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
                            "delta": {"type": "text_delta", "text": text}
                        }
                        self.wfile.write(f"event: content_block_delta\ndata: {json.dumps(t_evt)}\n\n".encode("utf-8"))
                        self.wfile.flush()

                    def _close_thinking_block():
                        nonlocal thinking_block_started, current_block_index
                        if not thinking_block_started:
                            return
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

                    def _emit_thinking(text):
                        nonlocal thinking_block_started
                        if not text:
                            return
                        if thinking_mode == "raw":
                            _emit_text(text)
                            return
                        if thinking_mode != "thinking_block":
                            return  # "strip": drop reasoning entirely
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
                            "delta": {"type": "thinking_delta", "thinking": text}
                        }
                        self.wfile.write(f"event: content_block_delta\ndata: {json.dumps(t_evt)}\n\n".encode("utf-8"))
                        self.wfile.flush()

                    def _holdback(buf, tag):
                        """Length of the longest suffix of buf that could still grow
                        into tag -- the part that must wait for the next chunk."""
                        for n in range(min(len(tag) - 1, len(buf)), 0, -1):
                            if buf.endswith(tag[:n]):
                                return n
                        return 0

                    def _feed(content_piece):
                        """Split content into visible text and <think> reasoning.
                        Everything that cannot be part of a split tag is emitted
                        immediately; only an ambiguous tail is held back."""
                        nonlocal pending, in_think_tag
                        pending += content_piece
                        while True:
                            tag = "</think>" if in_think_tag else "<think>"
                            idx = pending.find(tag)
                            if idx == -1:
                                cut = len(pending) - _holdback(pending, tag)
                                if in_think_tag:
                                    _emit_thinking(pending[:cut])
                                else:
                                    _emit_text(pending[:cut])
                                pending = pending[cut:]
                                return
                            if in_think_tag:
                                _emit_thinking(pending[:idx])
                                _close_thinking_block()
                                in_think_tag = False
                            else:
                                _emit_text(pending[:idx])
                                _close_text_block()
                                in_think_tag = True
                            pending = pending[idx + len(tag):]

                    def _commit():
                        nonlocal committed, stream_started
                        if committed:
                            return
                        self._commit_stream_start(target_model, anthropic_req, msg_id)
                        committed = True
                        stream_started = True

                    # Same stall guard as the anthropic-native path: an upstream
                    # can keep the socket warm with SSE comment lines (": ping")
                    # and deliver nothing. data: lines reset the clock.
                    last_progress = time.monotonic()
                    for line in resp.iter_lines():
                        line = line.strip()
                        if not line.startswith("data:"):
                            if line and time.monotonic() - last_progress > STREAM_IDLE_TIMEOUT:
                                self.emit_log(
                                    f"Upstream produced no SSE data for "
                                    f"{int(STREAM_IDLE_TIMEOUT)}s (keep-alive only); "
                                    f"closing the stalled turn.")
                                break
                            continue
                        last_progress = time.monotonic()
                        data_str = line[5:].strip()
                        if data_str == "[DONE]":
                            break

                        # First real upstream byte: the router is alive, so it is
                        # now safe to commit the 200 + message_start.
                        _commit()

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

                        # 1. reasoning_content arrives already typed as reasoning:
                        #    no tag parsing -- straight to a thinking block (or to
                        #    text in "raw" mode, or dropped in "strip" mode).
                        _emit_thinking(reasoning_piece or "")

                        # 2. Visible content, which may carry inline <think> tags.
                        if content_piece:
                            # A thinking block opened by reasoning_content has to
                            # be closed before visible text starts.
                            if thinking_block_started and not in_think_tag:
                                _close_thinking_block()
                            _feed(content_piece)

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

                    # Flush whatever the split-tag buffer was still holding back.
                    if pending:
                        if in_think_tag:
                            _emit_thinking(pending)
                        else:
                            _emit_text(pending)
                        pending = ""

                    if not stream_started:
                        # Not one byte of usable data arrived (the Dahl freeze).
                        if allow_fallback and fallback_cfg and fallback_cfg.get("router_url"):
                            # Nothing has been sent to Claude Code yet, so the
                            # request can be handed to the next router instead of
                            # committing an empty turn -- an empty turn is what
                            # left the chat looking dead however often it was
                            # retried.
                            raise httpx.TransportError("upstream delivered no data")
                        # Single router, nothing else to try: commit the 200 and
                        # close the turn so Claude Code can continue rather than
                        # wait on a stream that will never emit a terminal event.
                        _commit()

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
                resp = client.post(endpoint, headers=req_headers, json=openai_req, timeout=_NONSTREAM_TIMEOUT)
                # 1. Multi-key failover retry on 429/402
                if resp.status_code in (429, 402) and multi_key_enabled and len(api_keys) > 1:
                    for alt_key in [k for k in api_keys if k != current_key]:
                        old_m = f"...{current_key[-4:]}" if len(current_key) >= 4 else "key"
                        new_m = f"...{alt_key[-4:]}" if len(alt_key) >= 4 else "key"
                        self.emit_log(f"API key {old_m} hit limit ({resp.status_code}). Rotating to {new_m}...")
                        alt_headers = dict(req_headers)
                        alt_headers["Authorization"] = f"Bearer {alt_key}"
                        retry_resp = client.post(endpoint, headers=alt_headers, json=openai_req, timeout=_NONSTREAM_TIMEOUT)
                        if retry_resp.status_code == 200:
                            resp = retry_resp
                            break
                        elif retry_resp.status_code not in (429, 402):
                            resp = retry_resp
                            break
                    else:
                        if not egress_proxy:
                            self.emit_log("Every key hit 429 - likely an IP rate limit. Set an Egress Proxy on this provider to route around it.")

                if resp.status_code != 200:
                    err_body = resp.text
                    is_size_err = ("not supported by TokenPlan" in err_body or resp.status_code == 413 or "too large" in err_body.lower())
                    # Inline-image rejection: strip to paths and retry once.
                    is_img_err = (resp.status_code == 400 and "image" in err_body.lower()
                                  and not router_rejects_inline_images(router_url))
                    if is_img_err:
                        mark_router_rejects_inline_images(router_url)
                        self.emit_log(f"Router rejected inline images ({resp.status_code}). Retrying with images as file paths; future requests will skip inline images for this router.")
                        openai_req = strip_inline_images(openai_req)
                        resp = client.post(endpoint, headers=req_headers, json=openai_req, timeout=_NONSTREAM_TIMEOUT)
                    elif is_size_err and len(openai_req.get("messages", [])) > 2:
                        self.emit_log(f"Router rejected non-streaming payload ({resp.status_code}): TokenPlan limit reached. Rescuing with compacted context...")
                        openai_req["messages"] = trim_openai_messages(openai_req["messages"], max_bytes=450_000)
                        resp = client.post(endpoint, headers=req_headers, json=openai_req, timeout=_NONSTREAM_TIMEOUT)

                if resp.status_code != 200:
                    if allow_fallback and fallback_cfg and fallback_cfg.get("router_url"):
                        self._failover_request(anthropic_req, fallback_cfg, tb, fallback_name)
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
        except httpx.TransportError as ex:
            # Connect/read timeout or a dead socket. Before the response
            # started this is exactly what the failover chain is for: without
            # it a hung primary router 500s and Claude Code re-sends the same
            # request to the same dead router -- the "stuck compacting" loop
            # in hybrid mode.
            if stream_started:
                self._close_sse_stream(
                    f"Upstream stream failed mid-response ({type(ex).__name__}: {ex}); "
                    f"closing turn so Claude Code can continue.",
                    stream_in, stream_out,
                )
                return
            if allow_fallback and fallback_cfg and fallback_cfg.get("router_url"):
                self.emit_log(
                    f"Router unreachable ({type(ex).__name__}: {ex}); "
                    f"failing over to [{fallback_name}]...")
                self._failover_request(anthropic_req, fallback_cfg, tb, fallback_name)
                return
            self.emit_log(f"Proxy Connection Error: {str(ex)}")
            try:
                payload = json.dumps({"error": str(ex)}).encode("utf-8")
                self.send_response(504)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(payload)
            except Exception:
                pass
        except Exception as ex:
            # If the 200 + message_start already went out, a 500 is no longer a
            # valid HTTP response: close the SSE turn so Claude Code resumes
            # instead of waiting on a stream that will never emit a stop event.
            if stream_started:
                self._close_sse_stream(
                    f"Upstream stream failed mid-response ({type(ex).__name__}: {ex}); "
                    f"closing turn so Claude Code can continue.",
                    stream_in, stream_out,
                )
                return
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

        try:
            cfg = self.config_getter() if self.config_getter else {}
            self._handle_messages(anthropic_req, cfg)
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

