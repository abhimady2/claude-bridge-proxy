"""Image pass-through check: attached images must reach the model, not just the disk.

Run:  python test_vision.py
Exits 0 on pass. No framework, no fixtures.
"""
from proxy_engine import convert_anthropic_to_openai, process_anthropic_images

# 1x1 PNG
PNG_B64 = ("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==")


def user_msg(blocks):
    return [{"role": "user", "content": blocks}]


def image_block(media_type="image/png", data=PNG_B64):
    return {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": data}}


def text_block(t):
    return {"type": "text", "text": t}


def test_openai_forwards_image():
    req = {"messages": user_msg([text_block("what is this?"), image_block()])}
    content = convert_anthropic_to_openai(req, "claude-3-5-sonnet")["messages"][0]["content"]
    assert isinstance(content, list), f"expected parts list, got {type(content)}"
    url = next(p for p in content if p["type"] == "image_url")["image_url"]["url"]
    assert url.startswith("data:image/png;base64,"), url[:40]
    assert "what is this?" in [p.get("text") for p in content if p["type"] == "text"]


def test_openai_image_only_message_survives():
    req = {"messages": user_msg([image_block()])}
    content = convert_anthropic_to_openai(req, "claude-3-5-sonnet")["messages"][0]["content"]
    assert isinstance(content, list) and content[0]["type"] == "image_url", content


def test_strip_images_falls_back_to_path():
    req = {"messages": user_msg([text_block("hi"), image_block()])}
    content = convert_anthropic_to_openai(req, "Atria-Dawn-Preview", strip_images=True)["messages"][0]["content"]
    assert isinstance(content, str), content
    assert "claude_bridge_images" in content, content
    # the old stub told the model it was blind; that claim must be gone
    assert "text-only model" not in content


def test_non_raster_media_falls_back_to_path():
    req = {"messages": user_msg([image_block(media_type="image/svg+xml")])}
    content = convert_anthropic_to_openai(req, "claude-3-5-sonnet")["messages"][0]["content"]
    assert isinstance(content, str), content


def test_anthropic_native_passthrough():
    msgs = user_msg([text_block("look"), image_block()])
    kept = process_anthropic_images(msgs)
    assert [b["type"] for b in kept[0]["content"]] == ["text", "image"], kept[0]["content"]
    assert kept[0]["content"][1]["source"]["data"] == PNG_B64

    stripped = process_anthropic_images(msgs, strip_images=True)
    assert [b["type"] for b in stripped[0]["content"]] == ["text", "text"], stripped[0]["content"]
    assert "saved at:" in stripped[0]["content"][1]["text"]


def test_noop_when_both_disabled():
    msgs = user_msg([image_block()])
    assert process_anthropic_images(msgs, auto_save=False, strip_images=False) is msgs


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} checks passed.")
