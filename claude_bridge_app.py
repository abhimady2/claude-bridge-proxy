import os
import sys
import io
import json
import shutil
import threading
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, simpledialog
from datetime import datetime

# Prevent NoneType write errors in windowed (no-console) mode
if sys.stdout is None:
    sys.stdout = io.StringIO()
if sys.stderr is None:
    sys.stderr = io.StringIO()

from PIL import Image, ImageDraw
import pystray

from proxy_engine import (
    ProxyServer,
    fetch_context_length,
    DEFAULT_CONTEXT_LENGTH,
    get_token_stats,
    reset_token_stats,
)

# Bumped with every behaviour change. Shown in the title bar and logged on start.
APP_VERSION = "1.2.4"

# Configuration paths
CONFIG_DIR = os.path.join(os.environ.get("APPDATA", os.path.expanduser("~")), "ClaudeBridge")
CONFIG_FILE = os.path.join(CONFIG_DIR, "config.json")
CLAUDE_DIR = os.path.join(os.path.expanduser("~"), ".claude")
CLAUDE_SETTINGS_PATH = os.path.join(CLAUDE_DIR, "settings.json")
CLAUDE_BACKUP_PATH = os.path.join(CLAUDE_DIR, "settings.json.bridge_backup")


def get_default_providers():
    return {
        "Dahl Global": {
            "router_url": "https://inference.dahl.global/v1",
            "api_key": "",
            "model": "deepseek-ai/DeepSeek-V4-Flash-0731",
            "sonnet_model": "deepseek-ai/DeepSeek-V4-Flash-0731",
            "opus_model": "MiniMaxAI/MiniMax-M2.7",
            "haiku_model": "zai-org/GLM-5.3-Flash"
        },
        "OpenRouter": {
            "router_url": "https://openrouter.ai/api/v1",
            "api_key": "",
            "model": "anthropic/claude-3.5-sonnet",
            "sonnet_model": "anthropic/claude-3.5-sonnet",
            "opus_model": "anthropic/claude-3-opus",
            "haiku_model": "anthropic/claude-3.5-haiku"
        },
        "OpenCode Zen": {
            "router_url": "https://opencode.ai/zen/v1",
            "api_key": "",
            "model": "minimax-m2.5-free",
            "sonnet_model": "minimax-m2.5-free",
            "opus_model": "minimax-m2.5-free",
            "haiku_model": "minimax-m2.5-free"
        },
        "Atria ASI": {
            "router_url": "https://api.atria-asi.ai/v1",
            "api_key": "",
            "model": "Atria-Dawn-Preview",
            "sonnet_model": "Atria-Dawn-Preview",
            "opus_model": "Atria-Dawn-Preview",
            "haiku_model": "Atria-Dawn-Preview",
            "context_length": 128000
        },
        "Agent Router": {
            "router_url": "https://agentrouter.org",
            "api_key": "",
            "model": "deepseek-v4-flash",
            "sonnet_model": "deepseek-v4-flash",
            "opus_model": "deepseek-v4-flash",
            "haiku_model": "deepseek-v4-flash"
        }
    }


def get_default_config():
    providers = get_default_providers()
    return {
        "active_provider": "Dahl Global",
        "providers": providers,
        "port": 4000,
        "minimize_to_tray": False,
        "auto_start": True,
        "thinking_mode": "thinking_block",
        "context_length": DEFAULT_CONTEXT_LENGTH,
        "auto_compact_window": True
    }


def load_config():
    defaults = get_default_config()
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if "providers" in data and isinstance(data["providers"], dict):
                    defaults["providers"].update(data["providers"])
                if "active_provider" in data and data["active_provider"] in defaults["providers"]:
                    defaults["active_provider"] = data["active_provider"]
                for k in ("port", "minimize_to_tray", "auto_start", "thinking_mode",
                          "context_length", "auto_compact_window"):
                    if k in data:
                        defaults[k] = data[k]
        except Exception:
            pass
    return defaults


def save_config(cfg):
    try:
        os.makedirs(CONFIG_DIR, exist_ok=True)
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
    except Exception:
        pass


def create_tray_image(is_running=False):
    """Generate a clean tray icon dynamically."""
    width = 64
    height = 64
    image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    
    # Outer circle / rounded box
    bg_color = (16, 185, 129, 255) if is_running else (99, 102, 241, 255)
    draw.rounded_rectangle((4, 4, 60, 60), radius=16, fill=bg_color)
    
    # Inner "C" bridge shape
    draw.arc((14, 14, 50, 50), start=45, end=315, fill=(255, 255, 255, 255), width=8)
    
    # Indicator dot
    dot_color = (255, 255, 255, 255) if is_running else (239, 68, 68, 255)
    draw.ellipse((42, 42, 54, 54), fill=dot_color)
    return image


class ClaudeBridgeApp:
    def __init__(self, root):
        self.root = root
        self.root.title(f"Claude Bridge v{APP_VERSION} — Multi-Provider Router Proxy")
        self.root.geometry("660 x 820".replace(" ", ""))
        self.root.minsize(600, 740)
        self.root.configure(bg="#18181b")

        # Set taskbar icon
        self.icon_image = create_tray_image(False)
        self.tray_icon = None
        self.proxy_server = None
        self.cfg = load_config()

        self._setup_styles()
        self._build_ui()
        self._init_tray()

        # Handle close event
        self.root.protocol("WM_DELETE_WINDOW", self._on_close_requested)

    def _setup_styles(self):
        self.style = ttk.Style()
        self.style.theme_use("clam")

        # Configure dark colors
        bg = "#18181b"
        card_bg = "#27272a"
        text = "#f4f4f5"

        self.style.configure(".", background=bg, foreground=text, font=("Segoe UI", 9))
        self.style.configure("Card.TFrame", background=card_bg, relief="flat")
        self.style.configure("Header.TLabel", background=bg, foreground="#ffffff", font=("Segoe UI", 16, "bold"))
        self.style.configure("SubHeader.TLabel", background=bg, foreground="#a1a1aa", font=("Segoe UI", 9))
        self.style.configure("FieldLabel.TLabel", background=card_bg, foreground="#d4d4d8", font=("Segoe UI", 9, "bold"))
        self.style.configure("SectionTitle.TLabel", background=card_bg, foreground="#60a5fa", font=("Segoe UI", 11, "bold"))
        
        # Combobox dark style
        self.style.configure(
            "TCombobox",
            fieldbackground="#18181b",
            background="#3f3f46",
            foreground="#ffffff",
            darkcolor="#3f3f46",
            lightcolor="#3f3f46",
            selectbackground="#3b82f6",
            selectforeground="#ffffff"
        )
        self.style.map("TCombobox", fieldbackground=[("readonly", "#18181b")])

    def _build_ui(self):
        # Main container with padding
        main_frame = tk.Frame(self.root, bg="#18181b", padx=20, pady=16)
        main_frame.pack(fill="both", expand=True)

        # Header Area
        header_frame = tk.Frame(main_frame, bg="#18181b")
        header_frame.pack(fill="x", pady=(0, 12))

        title_box = tk.Frame(header_frame, bg="#18181b")
        title_box.pack(side="left")

        title_lbl = ttk.Label(title_box, text="Claude Bridge", style="Header.TLabel")
        title_lbl.pack(anchor="w")

        sub_lbl = ttk.Label(title_box, text="Multi-Provider OpenAI ➜ Anthropic Messages Router", style="SubHeader.TLabel")
        sub_lbl.pack(anchor="w")

        # Status badge
        self.status_badge = tk.Label(
            header_frame,
            text="● STOPPED",
            font=("Segoe UI", 9, "bold"),
            bg="#ef4444",
            fg="#ffffff",
            padx=10,
            pady=4,
            relief="flat"
        )
        self.status_badge.pack(side="right", pady=4)

        # Configuration Card Frame
        card = tk.Frame(main_frame, bg="#27272a", padx=16, pady=14, highlightbackground="#3f3f46", highlightthickness=1)
        card.pack(fill="x", pady=(0, 10))

        # --- Provider Profiles Selection Bar ---
        profile_frame = tk.Frame(card, bg="#1e1e24", padx=10, pady=8, highlightbackground="#3b82f6", highlightthickness=1)
        profile_frame.pack(fill="x", pady=(0, 10))

        tk.Label(profile_frame, text="Active Provider:", font=("Segoe UI", 9, "bold"), bg="#1e1e24", fg="#60a5fa").pack(side="left", padx=(0, 8))

        provider_names = list(self.cfg.get("providers", {}).keys())
        active_p = self.cfg.get("active_provider")
        if active_p not in provider_names and provider_names:
            active_p = provider_names[0]

        self.provider_var = tk.StringVar(value=active_p)
        self.provider_combo = ttk.Combobox(
            profile_frame, textvariable=self.provider_var, values=provider_names, state="readonly", width=18, font=("Segoe UI", 9)
        )
        self.provider_combo.pack(side="left", padx=(0, 8))
        self.provider_combo.bind("<<ComboboxSelected>>", self._on_provider_selected)

        save_p_btn = tk.Button(
            profile_frame, text="💾 Save", command=self.save_current_provider,
            font=("Segoe UI", 8, "bold"), bg="#2563eb", fg="#ffffff", activebackground="#1d4ed8", activeforeground="#ffffff",
            relief="flat", padx=6, pady=2, cursor="hand2"
        )
        save_p_btn.pack(side="left", padx=2)

        new_p_btn = tk.Button(
            profile_frame, text="➕ New", command=self.add_new_provider,
            font=("Segoe UI", 8), bg="#3f3f46", fg="#ffffff", activebackground="#52525b", activeforeground="#ffffff",
            relief="flat", padx=6, pady=2, cursor="hand2"
        )
        new_p_btn.pack(side="left", padx=2)

        del_p_btn = tk.Button(
            profile_frame, text="🗑 Delete", command=self.delete_current_provider,
            font=("Segoe UI", 8), bg="#dc2626", fg="#ffffff", activebackground="#b91c1c", activeforeground="#ffffff",
            relief="flat", padx=6, pady=2, cursor="hand2"
        )
        del_p_btn.pack(side="left", padx=2)

        # Get initial provider data
        cur_p_data = self.cfg.get("providers", {}).get(active_p, {})

        # 1. Router URL
        url_frame = tk.Frame(card, bg="#27272a")
        url_frame.pack(fill="x", pady=3)
        ttk.Label(url_frame, text="Router URL (OpenAI Base URL):", style="FieldLabel.TLabel").pack(anchor="w")
        self.url_var = tk.StringVar(value=cur_p_data.get("router_url", "https://inference.dahl.global/v1"))
        self.url_entry = tk.Entry(
            url_frame, textvariable=self.url_var, font=("Segoe UI", 10),
            bg="#18181b", fg="#ffffff", insertbackground="#ffffff", relief="flat", highlightbackground="#52525b", highlightthickness=1
        )
        self.url_entry.pack(fill="x", pady=(2, 0), ipady=4)

        # 2. API Key
        key_frame = tk.Frame(card, bg="#27272a")
        key_frame.pack(fill="x", pady=3)
        
        key_header = tk.Frame(key_frame, bg="#27272a")
        key_header.pack(fill="x")
        ttk.Label(key_header, text="API Key:", style="FieldLabel.TLabel").pack(side="left", anchor="w")
        
        self.show_key_var = tk.BooleanVar(value=False)
        show_btn = tk.Checkbutton(
            key_header, text="Show", variable=self.show_key_var, command=self._toggle_key_visibility,
            bg="#27272a", fg="#a1a1aa", activebackground="#27272a", selectcolor="#18181b", relief="flat", font=("Segoe UI", 8)
        )
        show_btn.pack(side="right")

        self.key_var = tk.StringVar(value=cur_p_data.get("api_key", ""))
        self.key_entry = tk.Entry(
            key_frame, textvariable=self.key_var, show="•", font=("Segoe UI", 10),
            bg="#18181b", fg="#ffffff", insertbackground="#ffffff", relief="flat", highlightbackground="#52525b", highlightthickness=1
        )
        self.key_entry.pack(fill="x", pady=(2, 0), ipady=4)

        # 3. Model & Port Row
        row_frame = tk.Frame(card, bg="#27272a")
        row_frame.pack(fill="x", pady=3)

        # Model
        m_col = tk.Frame(row_frame, bg="#27272a")
        m_col.pack(side="left", fill="x", expand=True, padx=(0, 6))
        ttk.Label(m_col, text="Default / Custom Model Name:", style="FieldLabel.TLabel").pack(anchor="w")
        self.model_var = tk.StringVar(value=cur_p_data.get("model", "deepseek-ai/DeepSeek-V4-Flash-0731"))
        self.model_entry = tk.Entry(
            m_col, textvariable=self.model_var, font=("Segoe UI", 10),
            bg="#18181b", fg="#ffffff", insertbackground="#ffffff", relief="flat", highlightbackground="#52525b", highlightthickness=1
        )
        self.model_entry.pack(fill="x", pady=(2, 0), ipady=4)

        # Port
        p_col = tk.Frame(row_frame, bg="#27272a")
        p_col.pack(side="right", padx=(6, 0))
        ttk.Label(p_col, text="Local Port:", style="FieldLabel.TLabel").pack(anchor="w")
        self.port_var = tk.IntVar(value=self.cfg.get("port", 4000))
        self.port_entry = tk.Entry(
            p_col, textvariable=self.port_var, width=8, font=("Segoe UI", 10), justify="center",
            bg="#18181b", fg="#ffffff", insertbackground="#ffffff", relief="flat", highlightbackground="#52525b", highlightthickness=1
        )
        self.port_entry.pack(pady=(2, 0), ipady=4)

        # Multi-model mappings section for Claude Code dropdown
        mapping_frame = tk.Frame(card, bg="#1e1e24", padx=10, pady=8, highlightbackground="#3f3f46", highlightthickness=1)
        mapping_frame.pack(fill="x", pady=(8, 0))
        
        tk.Label(mapping_frame, text="Claude Code Dropdown Model Mappings:", font=("Segoe UI", 9, "bold"), bg="#1e1e24", fg="#60a5fa").pack(anchor="w", pady=(0, 4))
        
        # Sonnet slot
        s_row = tk.Frame(mapping_frame, bg="#1e1e24")
        s_row.pack(fill="x", pady=2)
        tk.Label(s_row, text="Sonnet →", width=9, anchor="w", font=("Segoe UI", 8, "bold"), bg="#1e1e24", fg="#d4d4d8").pack(side="left")
        self.sonnet_var = tk.StringVar(value=cur_p_data.get("sonnet_model", "deepseek-ai/DeepSeek-V4-Flash-0731"))
        tk.Entry(s_row, textvariable=self.sonnet_var, font=("Segoe UI", 9), bg="#18181b", fg="#ffffff", insertbackground="#ffffff", relief="flat").pack(side="left", fill="x", expand=True)
        
        # Opus slot
        o_row = tk.Frame(mapping_frame, bg="#1e1e24")
        o_row.pack(fill="x", pady=2)
        tk.Label(o_row, text="Opus →", width=9, anchor="w", font=("Segoe UI", 8, "bold"), bg="#1e1e24", fg="#d4d4d8").pack(side="left")
        self.opus_var = tk.StringVar(value=cur_p_data.get("opus_model", "MiniMaxAI/MiniMax-M2.7"))
        tk.Entry(o_row, textvariable=self.opus_var, font=("Segoe UI", 9), bg="#18181b", fg="#ffffff", insertbackground="#ffffff", relief="flat").pack(side="left", fill="x", expand=True)

        # Haiku slot
        h_row = tk.Frame(mapping_frame, bg="#1e1e24")
        h_row.pack(fill="x", pady=2)
        tk.Label(h_row, text="Haiku →", width=9, anchor="w", font=("Segoe UI", 8, "bold"), bg="#1e1e24", fg="#d4d4d8").pack(side="left")
        self.haiku_var = tk.StringVar(value=cur_p_data.get("haiku_model", "zai-org/GLM-5.3-Flash"))
        tk.Entry(h_row, textvariable=self.haiku_var, font=("Segoe UI", 9), bg="#18181b", fg="#ffffff", insertbackground="#ffffff", relief="flat").pack(side="left", fill="x", expand=True)

        # Context Window (autodetected from the provider; user-editable)
        ctx_frame = tk.Frame(card, bg="#27272a")
        ctx_frame.pack(fill="x", pady=3)

        ctx_head = tk.Frame(ctx_frame, bg="#27272a")
        ctx_head.pack(fill="x")
        ttk.Label(ctx_head, text="Context Window (tokens):", style="FieldLabel.TLabel").pack(side="left")

        self.auto_ctx_var = tk.BooleanVar(value=self.cfg.get("auto_compact_window", True))
        auto_ctx_check = tk.Checkbutton(
            ctx_head, text="Apply to Claude Code (auto-compact at this limit)",
            variable=self.auto_ctx_var,
            bg="#27272a", fg="#a1a1aa", activebackground="#27272a", selectcolor="#18181b",
            relief="flat", font=("Segoe UI", 8)
        )
        auto_ctx_check.pack(side="right")

        ctx_row = tk.Frame(ctx_frame, bg="#27272a")
        ctx_row.pack(fill="x", pady=(2, 0))

        self.ctx_var = tk.StringVar(value=str(self.cfg.get("context_length", DEFAULT_CONTEXT_LENGTH)))
        self.ctx_entry = tk.Entry(
            ctx_row, textvariable=self.ctx_var, width=14, font=("Segoe UI", 10), justify="center",
            bg="#18181b", fg="#ffffff", insertbackground="#ffffff", relief="flat",
            highlightbackground="#52525b", highlightthickness=1
        )
        self.ctx_entry.pack(side="left", ipady=4)
        self.ctx_entry.bind("<FocusOut>", lambda e: self._normalize_ctx_entry())

        self.ctx_source_lbl = tk.Label(
            ctx_row, text="", font=("Segoe UI", 8), bg="#27272a", fg="#71717a"
        )
        self.ctx_source_lbl.pack(side="left", padx=(8, 0))

        fetch_ctx_btn = tk.Button(
            ctx_row, text="🔄 Auto-detect", command=self.fetch_context_length,
            font=("Segoe UI", 8), bg="#3f3f46", fg="#ffffff", activebackground="#52525b",
            activeforeground="#ffffff", relief="flat", padx=8, pady=2, cursor="hand2"
        )
        fetch_ctx_btn.pack(side="right")

        # Thinking & Reasoning Handling inside card
        t_row = tk.Frame(card, bg="#27272a")
        t_row.pack(fill="x", pady=(8, 2))
        ttk.Label(t_row, text="Thinking / Reasoning:", style="FieldLabel.TLabel").pack(side="left", padx=(0, 6))
        self.thinking_mode_var = tk.StringVar()
        self.thinking_combo = ttk.Combobox(
            t_row, textvariable=self.thinking_mode_var, state="readonly", font=("Segoe UI", 9)
        )
        self.thinking_combo["values"] = [
            "Separate Thinking Block (Claude UI Collapsible)",
            "Hide / Strip Thinking (Clean Chat Only)",
            "Raw in Chat (No Filtering)"
        ]
        current_mode = self.cfg.get("thinking_mode", "thinking_block")
        mode_label_map = {
            "thinking_block": "Separate Thinking Block (Claude UI Collapsible)",
            "strip": "Hide / Strip Thinking (Clean Chat Only)",
            "raw": "Raw in Chat (No Filtering)"
        }
        self.thinking_mode_var.set(mode_label_map.get(current_mode, "Separate Thinking Block (Claude UI Collapsible)"))
        self.thinking_combo.pack(side="left", fill="x", expand=True)
        self.thinking_combo.bind("<<ComboboxSelected>>", self._on_thinking_mode_changed)

        # Action Buttons Row
        action_frame = tk.Frame(main_frame, bg="#18181b")
        action_frame.pack(fill="x", pady=(0, 8))

        self.start_btn = tk.Button(
            action_frame, text="▶  START PROXY", command=self.toggle_proxy,
            font=("Segoe UI", 11, "bold"), bg="#10b981", fg="#ffffff", activebackground="#059669", activeforeground="#ffffff",
            relief="flat", pady=8, cursor="hand2"
        )
        self.start_btn.pack(side="left", fill="x", expand=True, padx=(0, 6))

        self.tray_btn = tk.Button(
            action_frame, text="🗕 Minimize to Tray", command=self.minimize_to_tray,
            font=("Segoe UI", 9), bg="#3f3f46", fg="#ffffff", activebackground="#52525b", activeforeground="#ffffff",
            relief="flat", padx=10, pady=8, cursor="hand2"
        )
        self.tray_btn.pack(side="left", padx=(0, 6))

        self.quit_btn = tk.Button(
            action_frame, text="✕ Close & Exit", command=self.quit_app,
            font=("Segoe UI", 9), bg="#dc2626", fg="#ffffff", activebackground="#b91c1c", activeforeground="#ffffff",
            relief="flat", padx=10, pady=8, cursor="hand2"
        )
        self.quit_btn.pack(side="right")

        # Claude Code Connection Bar
        claude_frame = tk.Frame(main_frame, bg="#27272a", padx=12, pady=8, highlightbackground="#3f3f46", highlightthickness=1)
        claude_frame.pack(fill="x", pady=(0, 8))

        ttk.Label(claude_frame, text="Claude Code Setup:", style="FieldLabel.TLabel").pack(side="left", padx=(0, 8))

        sync_btn = tk.Button(
            claude_frame, text="⚡ Configure Claude Code Settings", command=self.configure_claude_settings,
            font=("Segoe UI", 8, "bold"), bg="#3b82f6", fg="#ffffff", activebackground="#2563eb", activeforeground="#ffffff",
            relief="flat", padx=8, pady=3, cursor="hand2"
        )
        sync_btn.pack(side="left", padx=3)

        restore_btn = tk.Button(
            claude_frame, text="🔄 Restore Original", command=self.restore_claude_settings,
            font=("Segoe UI", 8), bg="#52525b", fg="#ffffff", activebackground="#71717a", activeforeground="#ffffff",
            relief="flat", padx=8, pady=3, cursor="hand2"
        )
        restore_btn.pack(side="left", padx=3)

        # Options Row
        opts_frame = tk.Frame(main_frame, bg="#18181b")
        opts_frame.pack(fill="x", pady=(0, 6))

        self.min_on_close_var = tk.BooleanVar(value=self.cfg.get("minimize_to_tray", True))
        min_check = tk.Checkbutton(
            opts_frame, text="Minimize to tray on close", variable=self.min_on_close_var,
            bg="#18181b", fg="#a1a1aa", activebackground="#18181b", activeforeground="#ffffff", selectcolor="#27272a",
            relief="flat", font=("Segoe UI", 8)
        )
        min_check.pack(side="left", padx=(0, 10))

        self.auto_start_var = tk.BooleanVar(value=self.cfg.get("auto_start", True))
        auto_check = tk.Checkbutton(
            opts_frame, text="Auto-start proxy on launch", variable=self.auto_start_var,
            bg="#18181b", fg="#a1a1aa", activebackground="#18181b", activeforeground="#ffffff", selectcolor="#27272a",
            relief="flat", font=("Segoe UI", 8)
        )
        auto_check.pack(side="left")

        clear_btn = tk.Button(
            opts_frame, text="Clear Log", command=self.clear_log,
            bg="#18181b", fg="#71717a", activebackground="#18181b", activeforeground="#ffffff",
            relief="flat", font=("Segoe UI", 8), cursor="hand2"
        )
        clear_btn.pack(side="right")

        # Token Counter Bar
        token_frame = tk.Frame(main_frame, bg="#27272a", padx=12, pady=6, highlightbackground="#3f3f46", highlightthickness=1)
        token_frame.pack(fill="x", pady=(0, 8))

        tk.Label(
            token_frame, text="Tokens:", font=("Segoe UI", 9, "bold"), bg="#27272a", fg="#60a5fa"
        ).pack(side="left", padx=(0, 8))

        self.token_display = tk.Label(
            token_frame, text="↑ 0 · ↓ 0 · Σ 0 · 0 reqs", font=("Segoe UI", 9),
            bg="#27272a", fg="#d4d4d8"
        )
        self.token_display.pack(side="left")

        reset_tokens_btn = tk.Button(
            token_frame, text="↺ Reset", command=self.reset_token_counter,
            font=("Segoe UI", 8), bg="#3f3f46", fg="#ffffff", activebackground="#52525b",
            activeforeground="#ffffff", relief="flat", padx=8, pady=2, cursor="hand2"
        )
        reset_tokens_btn.pack(side="right")

        # Activity Log Console
        log_frame = tk.Frame(main_frame, bg="#27272a", highlightbackground="#3f3f46", highlightthickness=1)
        log_frame.pack(fill="both", expand=True)

        self.log_area = scrolledtext.ScrolledText(
            log_frame, wrap="word", font=("Consolas", 9),
            bg="#0f0f11", fg="#a1a1aa", insertbackground="#ffffff", relief="flat", padx=8, pady=6
        )
        self.log_area.pack(fill="both", expand=True)
        self.log_area.configure(state="disabled")

        self.log(f"Claude Bridge v{APP_VERSION} initialized. Ready to start.")

        # ponytail: 2s poll of a module-level dict. No per-request work and no
        claude_frame.pack(fill="x", pady=(0, 8))

        ttk.Label(claude_frame, text="Claude Code Setup:", style="FieldLabel.TLabel").pack(side="left", padx=(0, 8))

        sync_btn = tk.Button(
            claude_frame, text="⚡ Configure Claude Code Settings", command=self.configure_claude_settings,
            font=("Segoe UI", 8, "bold"), bg="#3b82f6", fg="#ffffff", activebackground="#2563eb", activeforeground="#ffffff",
            relief="flat", padx=8, pady=3, cursor="hand2"
        )
        sync_btn.pack(side="left", padx=3)

        restore_btn = tk.Button(
            claude_frame, text="🔄 Restore Original", command=self.restore_claude_settings,
            font=("Segoe UI", 8), bg="#52525b", fg="#ffffff", activebackground="#71717a", activeforeground="#ffffff",
            relief="flat", padx=8, pady=3, cursor="hand2"
        )
        restore_btn.pack(side="left", padx=3)

        # Options Row

        # ponytail: 2s poll of a module-level dict. No per-request work and no
        # second event loop; the cost is one label update per tick.
        self._refresh_token_display()
        self.root.after(2000, self._poll_tokens)

        # Auto-start if enabled
        if self.cfg.get("auto_start", True):
            self.root.after(300, self.start_proxy)

        p_data = self.cfg.get("providers", {}).get(p_name, {})
        if not p_data:
            return

        self.url_var.set(p_data.get("router_url", ""))
        self.key_var.set(p_data.get("api_key", ""))
        self.model_var.set(p_data.get("model", ""))
        self.sonnet_var.set(p_data.get("sonnet_model", p_data.get("model", "")))
        self.opus_var.set(p_data.get("opus_model", p_data.get("model", "")))
        self.haiku_var.set(p_data.get("haiku_model", p_data.get("model", "")))

        self.cfg["active_provider"] = p_name
        save_config(self.cfg)
        self.log(f"Switched provider profile to: {p_name} ({p_data.get('router_url')})")

        # Context window depends on the model, so re-resolve for the new profile.
        if self.auto_ctx_var.get():
            self.fetch_context_length()

    def _normalize_ctx_entry(self):
        """Clamp whatever the user typed to a sane positive int, defaulting on junk."""
        try:
            value = int(self.ctx_var.get().strip())
        except (TypeError, ValueError):
            value = DEFAULT_CONTEXT_LENGTH
        if not isinstance(value, int) or value <= 0:
            value = DEFAULT_CONTEXT_LENGTH
        self.ctx_var.set(str(value))
        return value

    def get_context_length(self):
        return self._normalize_ctx_entry()

    def fetch_context_length(self):
        """Resolve the context window for the currently selected model and fill the field."""
        model = self.model_var.get().strip()
        router_url = self.url_var.get().strip()
        if not model or not router_url:
            return

        self.ctx_source_lbl.configure(text="detecting…", fg="#a1a1aa")
        self.root.update_idletasks()

        length = fetch_context_length(router_url, self.key_var.get().strip(), model)

        # A value the user set by hand wins over detection; only overwrite when
        # detection actually knows something the static default doesn't.
        if length == DEFAULT_CONTEXT_LENGTH:
            stored = self.cfg.get("context_length")
            if isinstance(stored, int) and stored > 0 and stored != DEFAULT_CONTEXT_LENGTH:
                self.ctx_source_lbl.configure(
                    text=f"manual {stored:,} kept (router has no data for {model})", fg="#71717a"
                )
                return

        self.ctx_var.set(str(length))
        self.ctx_source_lbl.configure(
            text=(f"{length:,} for {model}" if length != DEFAULT_CONTEXT_LENGTH
                  else f"{length:,} default (router did not report)"),
            fg="#71717a"
        )
        self.log(f"Context window for {model}: {length:,} tokens")

    def _get_thinking_mode_key(self):
        val = self.thinking_mode_var.get()
        if "Strip" in val:
            return "strip"
        if "Raw" in val:
            return "raw"
        return "thinking_block"

    def _on_thinking_mode_changed(self, event=None):
        mode_key = self._get_thinking_mode_key()
        self.cfg["thinking_mode"] = mode_key
        save_config(self.cfg)
        self.log(f"Thinking mode set to: {self.thinking_mode_var.get()}")

    def _save_config_silently(self):
        p_name = self.provider_var.get().strip()
        if not p_name:
            return
        if "providers" not in self.cfg:
            self.cfg["providers"] = {}
        self.cfg["providers"][p_name] = {
            "router_url": self.url_var.get().strip(),
            "api_key": self.key_var.get().strip(),
            "model": self.model_var.get().strip(),
            "sonnet_model": self.sonnet_var.get().strip(),
            "opus_model": self.opus_var.get().strip(),
            "haiku_model": self.haiku_var.get().strip()
        }
        self.cfg["active_provider"] = p_name
        self.cfg["port"] = self.port_var.get()
        self.cfg["minimize_to_tray"] = self.min_on_close_var.get()
        self.cfg["auto_start"] = self.auto_start_var.get()
        self.cfg["thinking_mode"] = self._get_thinking_mode_key()
        self.cfg["context_length"] = self.get_context_length()
        self.cfg["auto_compact_window"] = self.auto_ctx_var.get()
        save_config(self.cfg)

    def save_current_provider(self):
        p_name = self.provider_var.get().strip()
        if not p_name:
            messagebox.showwarning("Warning", "Provider name cannot be empty.")
            return

        self._save_config_silently()
        self.provider_combo["values"] = list(self.cfg["providers"].keys())
        self.log(f"Saved changes to provider profile '{p_name}'.")
        messagebox.showinfo("Saved", f"Provider profile '{p_name}' saved successfully!")

    def add_new_provider(self):
        name = simpledialog.askstring("New Provider", "Enter a name for the new router provider (e.g. DeepSeek, Groq):")
        if not name or not name.strip():
            return
        name = name.strip()

        if "providers" not in self.cfg:
            self.cfg["providers"] = {}

        if name in self.cfg["providers"]:
            messagebox.showwarning("Exists", f"Provider '{name}' already exists.")
            self.provider_var.set(name)
            self._on_provider_selected()
            return

        # Create new profile copying current entries
        self.cfg["providers"][name] = {
            "router_url": self.url_var.get().strip(),
            "api_key": "",
            "model": self.model_var.get().strip(),
            "sonnet_model": self.sonnet_var.get().strip(),
            "opus_model": self.opus_var.get().strip(),
            "haiku_model": self.haiku_var.get().strip()
        }
        self.cfg["active_provider"] = name
        save_config(self.cfg)

        self.provider_combo["values"] = list(self.cfg["providers"].keys())
        self.provider_var.set(name)
        self._on_provider_selected()
        self.log(f"Created new provider profile '{name}'. Fill in your URL/API key and click Save.")

    def delete_current_provider(self):
        p_name = self.provider_var.get()
        providers = self.cfg.get("providers", {})
        if len(providers) <= 1:
            messagebox.showwarning("Warning", "Cannot delete the only remaining provider profile.")
            return

        confirm = messagebox.askyesno("Delete Provider", f"Are you sure you want to delete profile '{p_name}'?")
        if not confirm:
            return

        del self.cfg["providers"][p_name]
        remaining = list(self.cfg["providers"].keys())
        self.cfg["active_provider"] = remaining[0]
        save_config(self.cfg)

        self.provider_combo["values"] = remaining
        self.provider_var.set(remaining[0])
        self._on_provider_selected()
        self.log(f"Deleted provider profile '{p_name}'. Switched to '{remaining[0]}'.")

    def _toggle_key_visibility(self):
        if self.show_key_var.get():
            self.key_entry.configure(show="")
        else:
            self.key_entry.configure(show="•")

    def log(self, text):
        def _append():
            ts = datetime.now().strftime("%H:%M:%S")
            self.log_area.configure(state="normal")
            self.log_area.insert("end", f"[{ts}] {text}\n")
            self.log_area.see("end")
            self.log_area.configure(state="disabled")
        self.root.after(0, _append)

    def clear_log(self):
        self.log_area.configure(state="normal")
        self.log_area.delete("1.0", "end")
        self.log_area.configure(state="disabled")

    def get_current_config(self):
        return {
            "router_url": self.url_var.get().strip(),
            "api_key": self.key_var.get().strip(),
            "model": self.model_var.get().strip(),
            "sonnet_model": self.sonnet_var.get().strip(),
            "opus_model": self.opus_var.get().strip(),
            "haiku_model": self.haiku_var.get().strip(),
            "port": self.port_var.get(),
            "minimize_to_tray": self.min_on_close_var.get(),
            "auto_start": self.auto_start_var.get(),
            "thinking_mode": self._get_thinking_mode_key(),
            "context_length": self.get_context_length(),
            "auto_compact_window": self.auto_ctx_var.get()
        }

    def toggle_proxy(self):
        if self.proxy_server and self.proxy_server.is_running:
            self.stop_proxy()
        else:
            self.start_proxy()

    def start_proxy(self):
        cfg = self.get_current_config()
        # Save back to active profile
        p_name = self.provider_var.get()
        if "providers" in self.cfg and p_name in self.cfg["providers"]:
            self.cfg["providers"][p_name].update({
                "router_url": cfg["router_url"],
                "api_key": cfg["api_key"],
                "model": cfg["model"],
                "sonnet_model": cfg["sonnet_model"],
                "opus_model": cfg["opus_model"],
                "haiku_model": cfg["haiku_model"]
            })
            self.cfg["active_provider"] = p_name
            save_config(self.cfg)

        port = cfg["port"]

        try:
            self.proxy_server = ProxyServer(
                port=port,
                config_getter=self.get_current_config,
                log_callback=self.log
            )
            self.proxy_server.start()

            self.status_badge.configure(text=f"● RUNNING :{port}", bg="#10b981")
            self.start_btn.configure(text="⏹  STOP PROXY", bg="#ef4444", activebackground="#dc2626")
            self.log(f"Proxy successfully started on http://127.0.0.1:{port}")
            self.log(f"Active Provider: [{p_name}] -> {cfg['router_url']}")
            self.log(f"Mappings: Sonnet->{cfg['sonnet_model']}, Opus->{cfg['opus_model']}, Haiku->{cfg['haiku_model']}")
            self._update_tray_menu(is_running=True)

        except Exception as e:
            self.log(f"Failed to start proxy: {e}")
            messagebox.showerror("Proxy Error", f"Could not start proxy on port {port}:\n{e}")

    def stop_proxy(self):
        if self.proxy_server:
            try:
                self.proxy_server.stop()
            except Exception:
                pass
            self.proxy_server = None

        self.status_badge.configure(text="● STOPPED", bg="#ef4444")
        self.start_btn.configure(text="▶  START PROXY", bg="#10b981", activebackground="#059669")
        self.log("Proxy stopped.")
        self._update_tray_menu(is_running=False)

    def restore_claude_settings(self):
        """Restores original settings.json from backup."""
        try:
            if os.path.exists(CLAUDE_BACKUP_PATH):
                shutil.copy2(CLAUDE_BACKUP_PATH, CLAUDE_SETTINGS_PATH)
                self.log("Restored original Claude settings from backup.")
                messagebox.showinfo("Restored", "Original Claude Code settings have been restored.")
            else:
                # Remove custom env
                if os.path.exists(CLAUDE_SETTINGS_PATH):
                    with open(CLAUDE_SETTINGS_PATH, "r", encoding="utf-8") as f:
                        settings = json.load(f)
                    if "env" in settings:
                        for k in (
                            "ANTHROPIC_BASE_URL", "CLAUDE_CODE_USE_AUTH_TOKEN",
                            "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_MODEL",
                            "ANTHROPIC_DEFAULT_OPUS_MODEL", "ANTHROPIC_DEFAULT_SONNET_MODEL",
                            "ANTHROPIC_DEFAULT_HAIKU_MODEL",
                            "CLAUDE_CODE_MAX_CONTEXT_TOKENS", "CLAUDE_CODE_AUTO_COMPACT_WINDOW",
                            "CLAUDE_CODE_AUTO_MODE_SERVER",
                        ):
                            settings["env"].pop(k, None)
                    settings.pop("autoCompactWindow", None)
                    with open(CLAUDE_SETTINGS_PATH, "w", encoding="utf-8") as f:
                        json.dump(settings, f, indent=2)
                self.log("Cleared Claude Bridge overrides from settings.json.")
                messagebox.showinfo("Restored", "Claude Bridge overrides removed from Claude settings.")
        except Exception as e:
            self.log(f"Failed to restore Claude settings: {e}")
            messagebox.showerror("Error", f"Could not restore settings:\n{e}")

    # --- System Tray Implementation ---
    def _init_tray(self):
        try:
            menu = pystray.Menu(
                pystray.MenuItem("Open Claude Bridge", self.show_window, default=True),
                pystray.MenuItem("Toggle Proxy", self.toggle_proxy),
                pystray.MenuItem("Configure Claude Code", self.configure_claude_settings),
                pystray.Menu.SEPARATOR,
                pystray.MenuItem("Quit", self.quit_app)
            )
            self.tray_icon = pystray.Icon("ClaudeBridge", self.icon_image, "Claude Bridge Proxy", menu)
            self.tray_icon.run_detached()
        except Exception as e:
            self.log(f"Tray initialization notice: {e}")

    def _update_tray_menu(self, is_running):
        if not self.tray_icon:
            return
        try:
            p_name = self.provider_var.get()
            status_text = f"Status: Running [{p_name}]" if is_running else "Status: Stopped"
            self.tray_icon.icon = create_tray_image(is_running)
            self.tray_icon.menu = pystray.Menu(
                pystray.MenuItem(status_text, lambda: None, enabled=False),
                pystray.Menu.SEPARATOR,
                pystray.MenuItem("Open Claude Bridge", self.show_window, default=True),
                pystray.MenuItem("Stop Proxy" if is_running else "Start Proxy", self.toggle_proxy),
                pystray.MenuItem("Configure Claude Code", self.configure_claude_settings),
                pystray.Menu.SEPARATOR,
                pystray.MenuItem("Quit", self.quit_app)
            )
        except Exception:
            pass

    def minimize_to_tray(self):
        self.root.withdraw()
        if self.tray_icon:
            try:
                self.tray_icon.notify("Claude Bridge is running in the background.", "Minimized to Tray")
            except Exception:
                pass

    def show_window(self):
        self.root.after(0, self._restore_window)

    def _restore_window(self):
        self.root.deiconify()
        self.root.lift()
        self.root.focus_force()

    def _on_close_requested(self):
        if self.min_on_close_var.get():
            self.minimize_to_tray()
        else:
            self.quit_app()

    def quit_app(self, *args, **kwargs):
        # 1. Stop proxy server
        try:
            self.stop_proxy()
        except Exception:
            pass

        # 2. Silently save config (no blocking popups)
        try:
            self._save_config_silently()
        except Exception:
            pass

        # 3. Stop system tray icon
        if self.tray_icon:
            try:
                self.tray_icon.visible = False
                self.tray_icon.stop()
            except Exception:
                pass
            self.tray_icon = None

        # 4. Destroy Tk root if running
        try:
            self.root.quit()
            self.root.destroy()
        except Exception:
            pass

        # 5. Immediate hard exit to guarantee no orphaned background processes
        os._exit(0)


def main():
    import multiprocessing
    multiprocessing.freeze_support()
    try:
        root = tk.Tk()
        app = ClaudeBridgeApp(root)
        root.mainloop()
    except Exception:
        import traceback
        err_file = os.path.join(os.path.dirname(os.path.abspath(sys.executable)), "claude_bridge_crash.log")
        with open(err_file, "w", encoding="utf-8") as f:
            traceback.print_exc(file=f)


if __name__ == "__main__":
    main()
