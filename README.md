# ClaudeBridge

A local proxy that lets Claude Code use non-Anthropic models (GLM, DeepSeek, etc.) with auto-compact, vision support, multi-key rotation, and failover.

## Features

- Hybrid routing with failover between providers
- Multi-key rotation
- Model context window detection & real auto-compact keys
- Image-to-disk bridge (vision support)

## Setup

```bash
pip install -r requirements.txt
python claude_bridge_app.py
```

## License

MIT — see [LICENSE](LICENSE).
