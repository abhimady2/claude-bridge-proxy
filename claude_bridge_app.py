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

from PIL import Image, ImageDraw, ImageTk
import pystray

def get_resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller."""
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_path, relative_path)

from proxy_engine import (
    ProxyServer,
    fetch_context_length,
    DEFAULT_CONTEXT_LENGTH,
    get_token_stats,
    reset_token_stats,
    get_key_stats,
    parse_api_keys,
)

# Bumped with every behaviour change. Shown in the title bar and logged on start.
APP_VERSION = "1.4.7"

# Per-model reasoning effort. Claude Code no longer exposes this for custom
# models, so the bridge injects it upstream: reasoning_effort on OpenAI-style
# routers, output_config.effort on Anthropic-native ones. "Default" = omit.
# Levels match what routers actually accept (Token Juice rejects "medium").
EFFORT_LEVELS = ["Default", "None", "Low", "High", "Max"]
EFFORT_KEYS = {"None": "none", "Low": "low", "High": "high", "Max": "max"}
EFFORT_LABELS = {v: k for k, v in EFFORT_KEYS.items()}


def effort_key(label):
    return EFFORT_KEYS.get(label, "")


def effort_label(value):
    return EFFORT_LABELS.get((value or "").strip().lower(), "Default")

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
            "haiku_model": "Atria-Dawn-Preview"
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
        "auto_compact_window": True,
        # True = replace attached images with a disk path (routers that reject
        # inline image data or cap the request body, e.g. Atria TokenPlan).
        "strip_images": False,
        # Hybrid routing between provider profiles. Off by default: enabling it
        # sends Haiku/vision to the second router in the chain and Sonnet/Opus
        # to the first; the rest of the chain is the failover order.
        "enable_hybrid_router": False,
        "hybrid_chain": [],
        "hybrid_primary_provider": "",
        "hybrid_secondary_provider": "",
        "hybrid_fallback": True,
        "multi_key_rotation": True,
        "boot_to_tray": True
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
                          "context_length", "auto_compact_window", "strip_images",
                          "enable_hybrid_router", "hybrid_chain",
                          "hybrid_primary_provider", "hybrid_secondary_provider",
                          "hybrid_fallback", "multi_key_rotation", "boot_to_tray"):
                    if k in data:
                        defaults[k] = data[k]
        except Exception:
            pass
    # Migrate the old primary/secondary pair into the ordered chain. Runs once:
    # after this the chain is the source of truth.
    if not defaults.get("hybrid_chain"):
        legacy = []
        p = defaults.get("hybrid_primary_provider")
        s = defaults.get("hybrid_secondary_provider")
        provs = defaults.get("providers", {})
        if p in provs:
            legacy.append(p)
        if s in provs and s not in legacy:
            legacy.append(s)
        if legacy:
            defaults["hybrid_chain"] = legacy
    return defaults


def save_config(cfg):
    try:
        os.makedirs(CONFIG_DIR, exist_ok=True)
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
    except Exception:
        pass


def create_tray_image(is_running=False):
    """Generate a clean tray icon dynamically or from logo asset."""
    logo_path = get_resource_path(os.path.join("assets", "logo.png"))
    if os.path.exists(logo_path):
        try:
            base_img = Image.open(logo_path).convert("RGBA").resize((64, 64), Image.Resampling.LANCZOS)
            draw = ImageDraw.Draw(base_img)
            dot_color = (16, 185, 129, 255) if is_running else (239, 68, 68, 255)
            # Crisp outline ring and status dot in bottom-right corner
            draw.ellipse((42, 42, 62, 62), fill=(255, 255, 255, 255))
            draw.ellipse((44, 44, 60, 60), fill=dot_color)
            return base_img
        except Exception:
            pass

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


class _ColorBtn(tk.Frame):
    """A Frame+Label composite that renders with a solid background on ALL
    platforms including macOS dark mode.

    On macOS, tk.Button completely ignores its `bg` option when the system
    is in dark mode — the OS draws its own native button and the background
    colour is never visible.  tk.Frame and tk.Label DO respect `bg`, so
    wrapping a Label inside a Frame gives us full colour control.

    The public API is intentionally compatible with tk.Button's .configure()
    so that call sites (e.g. toggling the Start/Stop button colour) need zero
    changes.
    """

    def __init__(self, parent, text="", command=None,
                 bg="#3f3f46", fg="#ffffff", abg=None,
                 font=("Segoe UI", 9), padx=10, pady=5,
                 cursor="hand2", **frame_kw):
        super().__init__(parent, bg=bg, cursor=cursor, **frame_kw)
        self._bg  = bg
        self._abg = abg or bg
        self._cmd = command
        self._lbl = tk.Label(
            self, text=text, bg=bg, fg=fg, font=font,
            padx=padx, pady=pady, cursor=cursor
        )
        self._lbl.pack(fill="both", expand=True)
        for w in (self, self._lbl):
            w.bind("<Button-1>", self._on_click)
            w.bind("<Enter>",    self._on_enter)
            w.bind("<Leave>",    self._on_leave)

    # ---- event handlers ---------------------------------------------------
    def _on_click(self, _e=None):
        if self._cmd:
            self._cmd()

    def _on_enter(self, _e=None):
        self._lbl.config(bg=self._abg)
        super().configure(bg=self._abg)

    def _on_leave(self, _e=None):
        self._lbl.config(bg=self._bg)
        super().configure(bg=self._bg)

    # ---- public API -------------------------------------------------------
    def configure(self, **kw):
        """Intercept bg/fg/text; silently drop inapplicable tk.Button kwargs."""
        bg   = kw.pop("bg",   None)
        fg   = kw.pop("fg",   None)
        text = kw.pop("text", None)
        # Swallow Button-only kwargs that Labels/Frames don't understand
        for _k in ("activebackground", "activeforeground",
                   "highlightbackground", "highlightthickness",
                   "disabledforeground", "state", "relief", "bd"):
            kw.pop(_k, None)
        if bg is not None:
            self._bg = bg
            self._lbl.config(bg=bg)
            super().configure(bg=bg)
        if fg is not None:
            self._lbl.config(fg=fg)
        if text is not None:
            self._lbl.config(text=text)
        if kw:
            super().configure(**kw)

    # keep configure accessible via the standard name alias
    config = configure


class ClaudeBridgeApp:
    def __init__(self, root):
        self.root = root
        self.root.title(f"Claude Bridge v{APP_VERSION} — Multi-Provider Router Proxy")
        self.root.geometry("1200x760")
        self.root.minsize(870, 600)
        self.root.configure(bg="#18181b")

        # Set window icon and taskbar icon
        ico_path = get_resource_path(os.path.join("assets", "icon.ico"))
        logo_path = get_resource_path(os.path.join("assets", "logo.png"))
        if os.path.exists(ico_path):
            try:
                self.root.iconbitmap(default=ico_path)
            except Exception:
                try:
                    self.root.iconbitmap(ico_path)
                except Exception:
                    pass

        if os.path.exists(logo_path):
            try:
                logo_pil = Image.open(logo_path).convert("RGBA")
                self._app_window_icon = ImageTk.PhotoImage(logo_pil)
                self.root.iconphoto(True, self._app_window_icon)
            except Exception:
                pass

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
        # Horizontal layout: config column on the left, live log column on the
        # right. The log previously sat at the bottom of one long stack and got
        # squeezed to a sliver by every card above it.
        main_frame = tk.Frame(self.root, bg="#18181b")
        main_frame.pack(fill="both", expand=True)

        # ---- Header row -------------------------------------------------
        header_frame = tk.Frame(main_frame, bg="#18181b")
        header_frame.pack(fill="x", padx=16, pady=(12, 6))

        # Logo thumbnail in header
        logo_path = get_resource_path(os.path.join("assets", "logo.png"))
        if os.path.exists(logo_path):
            try:
                logo_pil = Image.open(logo_path).convert("RGBA").resize((38, 38), Image.Resampling.LANCZOS)
                self.logo_photo = ImageTk.PhotoImage(logo_pil)
                logo_lbl = tk.Label(header_frame, image=self.logo_photo, bg="#18181b", bd=0, highlightthickness=0)
                logo_lbl.pack(side="left", padx=(0, 10))
            except Exception:
                pass

        title_box = tk.Frame(header_frame, bg="#18181b")
        title_box.pack(side="left")

        title_lbl = ttk.Label(title_box, text="Claude Bridge", style="Header.TLabel")
        title_lbl.pack(anchor="w")

        sub_lbl = ttk.Label(title_box, text="Multi-Provider OpenAI → Anthropic Messages Router", style="SubHeader.TLabel")
        sub_lbl.pack(anchor="w")

        self.status_badge = tk.Label(
            header_frame, text="● STOPPED",
            font=("Segoe UI", 9, "bold"), bg="#ef4444", fg="#ffffff",
            padx=10, pady=4, relief="flat"
        )
        self.status_badge.pack(side="right", pady=4)

        # ---- Toolbar row: start/stop + Claude Code wiring ----------------
        toolbar = tk.Frame(main_frame, bg="#18181b")
        toolbar.pack(fill="x", padx=16, pady=(0, 6))

        self.start_btn = _ColorBtn(
            toolbar, text="Start Proxy", command=self.toggle_proxy,
            font=("Segoe UI", 10, "bold"), bg="#10b981", fg="#ffffff", abg="#059669",
            padx=0, pady=6
        )
        self.start_btn.pack(side="left", fill="x", expand=True, padx=(0, 6))

        sync_btn = _ColorBtn(
            toolbar, text="Configure Claude Code", command=self.configure_claude_settings,
            font=("Segoe UI", 9, "bold"), bg="#3b82f6", fg="#ffffff", abg="#2563eb",
            padx=10, pady=6
        )
        sync_btn.pack(side="left", padx=(0, 6))

        restore_btn = _ColorBtn(
            toolbar, text="Restore Original", command=self.restore_claude_settings,
            font=("Segoe UI", 9), bg="#52525b", fg="#ffffff", abg="#71717a",
            padx=10, pady=6
        )
        restore_btn.pack(side="left", padx=(0, 6))

        self.tray_btn = _ColorBtn(
            toolbar, text="Minimize to Tray", command=self.minimize_to_tray,
            font=("Segoe UI", 9), bg="#3f3f46", fg="#ffffff", abg="#52525b",
            padx=10, pady=6
        )
        self.tray_btn.pack(side="left", padx=(0, 6))

        self.quit_btn = _ColorBtn(
            toolbar, text="Close & Exit", command=self.quit_app,
            font=("Segoe UI", 9), bg="#dc2626", fg="#ffffff", abg="#b91c1c",
            padx=10, pady=6
        )
        self.quit_btn.pack(side="right")

        # ---- Two-column body --------------------------------------------
        # Settings side is fixed/uncontracted; log side absorbs resizing and
        # can contract as much as possible down to a usable limit. A draggable
        # divider also allows manual adjustment within the same safety limits.
        _sash_cursor = "size_we" if sys.platform == "win32" else "sb_h_double_arrow"
        body = tk.PanedWindow(
            main_frame, orient="horizontal", bg="#18181b", bd=0,
            sashwidth=6, sashpad=1, sashrelief="flat", sashcursor=_sash_cursor, opaqueresize=True
        )
        body.pack(fill="both", expand=True, padx=16, pady=(0, 8))

        # LEFT: scrollable configuration column (uncontracted, fixed comfortable width)
        left_pane = tk.Frame(body, bg="#18181b", width=530)
        canvas = tk.Canvas(left_pane, bg="#18181b", highlightthickness=0, bd=0, width=514)
        vbar = ttk.Scrollbar(left_pane, orient="vertical", command=canvas.yview)

        def _on_yscroll(lo, hi):
            flo, fhi = float(lo), float(hi)
            if flo <= 0.0 and fhi >= 1.0:
                vbar.pack_forget()
            else:
                if not vbar.winfo_ismapped():
                    vbar.pack(side="right", fill="y")
            vbar.set(lo, hi)

        canvas.configure(yscrollcommand=_on_yscroll)

        cfg_inner = tk.Frame(canvas, bg="#18181b", padx=2, pady=0)
        canvas.create_window((0, 0), window=cfg_inner, anchor="nw", tags="inner")

        def _on_inner_configure(_e):
            canvas.configure(scrollregion=canvas.bbox("all"))
        cfg_inner.bind("<Configure>", _on_inner_configure)

        def _on_canvas_configure(e):
            canvas.itemconfig("inner", width=e.width)
        canvas.bind("<Configure>", _on_canvas_configure)

        # Wheel scrolling only while the pointer is over the config column, so
        # the log keeps its own scroll.
        def _wheel(e):
            try:
                x, y = e.x_root, e.y_root
                lx = left_pane.winfo_rootx()
                ly = left_pane.winfo_rooty()
                lw = left_pane.winfo_width()
                lh = left_pane.winfo_height()
                if lx <= x <= lx + lw and ly <= y <= ly + lh:
                    canvas.yview_scroll(int(-e.delta / 120), "units")
            except Exception:
                pass
        canvas.bind_all("<MouseWheel>", _wheel)

        canvas.pack(side="left", fill="both", expand=True)
        body.add(left_pane, minsize=530, stretch="never")

        card = cfg_inner  # keep the rest of the builder readable

        # --- Provider Profiles Selection Bar ---
        profile_frame = tk.Frame(card, bg="#1e1e24", padx=10, pady=8, highlightbackground="#3b82f6", highlightthickness=1)
        profile_frame.pack(fill="x", pady=(0, 8))

        # Row 1: label + combobox
        p_row1 = tk.Frame(profile_frame, bg="#1e1e24")
        p_row1.pack(fill="x")
        tk.Label(p_row1, text="Active Provider:", font=("Segoe UI", 9, "bold"), bg="#1e1e24", fg="#60a5fa").pack(side="left", padx=(0, 8))

        provider_names = list(self.cfg.get("providers", {}).keys())
        active_p = self.cfg.get("active_provider")
        if active_p not in provider_names and provider_names:
            active_p = provider_names[0]

        self.provider_var = tk.StringVar(value=active_p)
        self.provider_combo = ttk.Combobox(
            p_row1, textvariable=self.provider_var, values=provider_names, state="readonly", width=22, font=("Segoe UI", 9)
        )
        self.provider_combo.pack(side="left", padx=(0, 8))
        self.provider_combo.bind("<<ComboboxSelected>>", self._on_provider_selected)

        # Row 2: action buttons always visible on their own line
        p_row2 = tk.Frame(profile_frame, bg="#1e1e24")
        p_row2.pack(fill="x", pady=(6, 0))

        save_p_btn = _ColorBtn(
            p_row2, text="Save", command=self.save_current_provider,
            font=("Segoe UI", 8, "bold"), bg="#2563eb", fg="#ffffff", abg="#1d4ed8",
            padx=8, pady=3
        )
        save_p_btn.pack(side="left", padx=(0, 6))

        new_p_btn = _ColorBtn(
            p_row2, text="+ New Provider", command=self.add_new_provider,
            font=("Segoe UI", 8), bg="#3f3f46", fg="#ffffff", abg="#52525b",
            padx=8, pady=3
        )
        new_p_btn.pack(side="left", padx=(0, 6))

        del_p_btn = _ColorBtn(
            p_row2, text="Delete", command=self.delete_current_provider,
            font=("Segoe UI", 8), bg="#dc2626", fg="#ffffff", abg="#b91c1c",
            padx=8, pady=3
        )
        del_p_btn.pack(side="left")

        # Get initial provider data
        cur_p_data = self.cfg.get("providers", {}).get(active_p, {})

        # 1. Router URL
        url_frame = tk.Frame(card, bg="#1e1e24", padx=10, pady=8, highlightbackground="#3f3f46", highlightthickness=1)
        url_frame.pack(fill="x", pady=(0, 8))
        ttk.Label(url_frame, text="Router URL (OpenAI Base URL):", style="FieldLabel.TLabel").pack(anchor="w")
        self.url_var = tk.StringVar(value=cur_p_data.get("router_url", "https://inference.dahl.global/v1"))
        self.url_entry = tk.Entry(
            url_frame, textvariable=self.url_var, font=("Segoe UI", 10),
            bg="#18181b", fg="#ffffff", insertbackground="#ffffff", relief="flat", highlightbackground="#52525b", highlightthickness=1
        )
        self.url_entry.pack(fill="x", pady=(2, 0), ipady=4)

        # 2. API Key
        key_frame = tk.Frame(card, bg="#1e1e24", padx=10, pady=8, highlightbackground="#3f3f46", highlightthickness=1)
        key_frame.pack(fill="x", pady=(0, 8))

        key_header = tk.Frame(key_frame, bg="#1e1e24")
        key_header.pack(fill="x")
        ttk.Label(key_header, text="API Key  (multiple: comma-separated, round-robin)", style="FieldLabel.TLabel").pack(side="left", anchor="w")

        self.show_key_var = tk.BooleanVar(value=False)
        show_btn = tk.Checkbutton(
            key_header, text="Show", variable=self.show_key_var, command=self._toggle_key_visibility,
            bg="#1e1e24", fg="#a1a1aa", activebackground="#1e1e24", selectcolor="#18181b", relief="flat", font=("Segoe UI", 8)
        )
        show_btn.pack(side="right")

        self.key_var = tk.StringVar(value=cur_p_data.get("api_key", ""))
        self.key_entry = tk.Entry(
            key_frame, textvariable=self.key_var, show="•", font=("Segoe UI", 10),
            bg="#18181b", fg="#ffffff", insertbackground="#ffffff", relief="flat", highlightbackground="#52525b", highlightthickness=1
        )
        self.key_entry.pack(fill="x", pady=(2, 0), ipady=4)

        # Live key-pool readout: confirms the comma-separated keys are actually
        # being picked up and rotated, rather than guessing from the masked box.
        self.key_pool_lbl = tk.Label(key_frame, text="", font=("Segoe UI", 8), bg="#1e1e24", fg="#71717a")
        self.key_pool_lbl.pack(anchor="w", pady=(2, 0))

        def _refresh_key_pool(*_a):
            keys = parse_api_keys(self.key_var.get())
            n = len(keys)
            if n <= 1:
                self.key_pool_lbl.configure(text="1 key in pool" if n else "No API key set")
            else:
                self.key_pool_lbl.configure(
                    text=f"{n} keys in round-robin pool - active now: ...{keys[0][-4:]}"
                )

        self.key_var.trace_add("write", _refresh_key_pool)
        _refresh_key_pool()

        # 2b. Egress Proxy (route this provider's traffic through a VPN/proxy)
        proxy_frame = tk.Frame(card, bg="#1e1e24", padx=10, pady=8, highlightbackground="#3f3f46", highlightthickness=1)
        proxy_frame.pack(fill="x", pady=(0, 8))
        proxy_head = tk.Frame(proxy_frame, bg="#1e1e24")
        proxy_head.pack(fill="x")
        ttk.Label(proxy_head, text="Egress Proxy (http://127.0.0.1:7890, socks5://127.0.0.1:1080, or empty)", style="FieldLabel.TLabel").pack(side="left", anchor="w", expand=True)
        self.proxy_url_var = tk.StringVar(value=cur_p_data.get("egress_proxy", ""))
        self.proxy_url_entry = tk.Entry(
            proxy_frame, textvariable=self.proxy_url_var, font=("Segoe UI", 10),
            bg="#18181b", fg="#ffffff", insertbackground="#ffffff", relief="flat",
            highlightbackground="#52525b", highlightthickness=1
        )
        self.proxy_url_entry.pack(fill="x", pady=(2, 0), ipady=4)
        ttk.Label(proxy_frame, text="If all keys 429 at once, your IP is rate-limited. Point this at a VPN/proxy (e.g. Clash/WARP local port) to egress on another IP.", style="FieldLabel.TLabel", font=("Segoe UI", 7)).pack(anchor="w", pady=(2, 0))

        # 3. Model & Port Row
        row_frame = tk.Frame(card, bg="#1e1e24", padx=10, pady=8, highlightbackground="#3f3f46", highlightthickness=1)
        row_frame.pack(fill="x", pady=(0, 8))

        m_col = tk.Frame(row_frame, bg="#1e1e24")
        m_col.pack(side="left", fill="x", expand=True, padx=(0, 6))
        ttk.Label(m_col, text="Default / Custom Model Name:", style="FieldLabel.TLabel").pack(anchor="w")
        self.model_var = tk.StringVar(value=cur_p_data.get("model", "deepseek-ai/DeepSeek-V4-Flash-0731"))
        self.model_entry = tk.Entry(
            m_col, textvariable=self.model_var, font=("Segoe UI", 10),
            bg="#18181b", fg="#ffffff", insertbackground="#ffffff", relief="flat", highlightbackground="#52525b", highlightthickness=1
        )
        self.model_entry.pack(fill="x", pady=(2, 0), ipady=4)

        eff_row = tk.Frame(m_col, bg="#1e1e24")
        eff_row.pack(fill="x", pady=(2, 0))
        ttk.Label(eff_row, text="Reasoning effort:", style="FieldLabel.TLabel").pack(side="left")
        self.model_effort_var = tk.StringVar(value=effort_label(cur_p_data.get("model_effort", "")))
        self._effort_combo(eff_row, self.model_effort_var).pack(side="left", padx=(6, 0))

        p_col = tk.Frame(row_frame, bg="#1e1e24")
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
        mapping_frame.pack(fill="x", pady=(0, 8))

        tk.Label(mapping_frame, text="Claude Code Dropdown Model Mappings:", font=("Segoe UI", 9, "bold"), bg="#1e1e24", fg="#60a5fa").pack(anchor="w", pady=(0, 4))

        s_row = tk.Frame(mapping_frame, bg="#1e1e24")
        s_row.pack(fill="x", pady=2)
        tk.Label(s_row, text="Sonnet →", width=9, anchor="w", font=("Segoe UI", 8, "bold"), bg="#1e1e24", fg="#d4d4d8").pack(side="left")
        self.sonnet_var = tk.StringVar(value=cur_p_data.get("sonnet_model", "deepseek-ai/DeepSeek-V4-Flash-0731"))
        self.sonnet_effort_var = tk.StringVar(value=effort_label(cur_p_data.get("sonnet_effort", "")))
        self._effort_combo(s_row, self.sonnet_effort_var).pack(side="right", padx=(4, 0))
        tk.Entry(s_row, textvariable=self.sonnet_var, font=("Segoe UI", 9), bg="#18181b", fg="#ffffff", insertbackground="#ffffff", relief="flat").pack(side="left", fill="x", expand=True)

        o_row = tk.Frame(mapping_frame, bg="#1e1e24")
        o_row.pack(fill="x", pady=2)
        tk.Label(o_row, text="Opus →", width=9, anchor="w", font=("Segoe UI", 8, "bold"), bg="#1e1e24", fg="#d4d4d8").pack(side="left")
        self.opus_var = tk.StringVar(value=cur_p_data.get("opus_model", "MiniMaxAI/MiniMax-M2.7"))
        self.opus_effort_var = tk.StringVar(value=effort_label(cur_p_data.get("opus_effort", "")))
        self._effort_combo(o_row, self.opus_effort_var).pack(side="right", padx=(4, 0))
        tk.Entry(o_row, textvariable=self.opus_var, font=("Segoe UI", 9), bg="#18181b", fg="#ffffff", insertbackground="#ffffff", relief="flat").pack(side="left", fill="x", expand=True)

        h_row = tk.Frame(mapping_frame, bg="#1e1e24")
        h_row.pack(fill="x", pady=2)
        tk.Label(h_row, text="Haiku →", width=9, anchor="w", font=("Segoe UI", 8, "bold"), bg="#1e1e24", fg="#d4d4d8").pack(side="left")
        self.haiku_var = tk.StringVar(value=cur_p_data.get("haiku_model", "zai-org/GLM-5.3-Flash"))
        self.haiku_effort_var = tk.StringVar(value=effort_label(cur_p_data.get("haiku_effort", "")))
        self._effort_combo(h_row, self.haiku_effort_var).pack(side="right", padx=(4, 0))
        tk.Entry(h_row, textvariable=self.haiku_var, font=("Segoe UI", 9), bg="#18181b", fg="#ffffff", insertbackground="#ffffff", relief="flat").pack(side="left", fill="x", expand=True)

        # --- Hybrid Multi-Router ---
        hybrid_frame = tk.Frame(card, bg="#1e1e24", padx=10, pady=8, highlightbackground="#3f3f46", highlightthickness=1)
        hybrid_frame.pack(fill="x", pady=(0, 8))
        self._hybrid_frame = hybrid_frame

        h_head = tk.Frame(hybrid_frame, bg="#1e1e24")
        h_head.pack(fill="x")
        tk.Label(h_head, text="Hybrid Multi-Router:", font=("Segoe UI", 9, "bold"), bg="#1e1e24", fg="#60a5fa").pack(side="left")

        self.hybrid_var = tk.BooleanVar(value=self.cfg.get("enable_hybrid_router", False))
        tk.Checkbutton(
            h_head, text="Enable", variable=self.hybrid_var, command=self._persist_options, bg="#1e1e24", fg="#a1a1aa",
            activebackground="#1e1e24", selectcolor="#18181b", relief="flat", font=("Segoe UI", 8)
        ).pack(side="left", padx=(10, 0))

        # Ordered router chain. Heavy tasks start at router 1, fast (Haiku/vision)
        # tasks at router 2 when the chain has 2+ entries; every router after the
        # active one is a failover target, in list order.
        self._chain_vars = []
        self._chain_rows = []
        self.chain_add_var = tk.StringVar()

        add_row = tk.Frame(hybrid_frame, bg="#1e1e24")
        add_row.pack(fill="x", pady=(4, 0))
        self.chain_add_combo = ttk.Combobox(add_row, textvariable=self.chain_add_var, state="readonly", width=16, font=("Segoe UI", 8))
        self.chain_add_combo.pack(side="left")
        _ColorBtn(
            add_row, text="+ Add to chain", command=self._add_chain_provider,
            font=("Segoe UI", 8), bg="#3f3f46", fg="#ffffff", abg="#52525b",
            padx=6, pady=1
        ).pack(side="left", padx=(4, 0))

        # Live summary of what the chain means for this request's routing.
        self.chain_summary_lbl = tk.Label(hybrid_frame, text="", font=("Segoe UI", 8), bg="#1e1e24", fg="#71717a", anchor="w", justify="left", wraplength=260)
        self.chain_summary_lbl.pack(fill="x", pady=(4, 0))

        # Rows go last: _rebuild_chain_ui writes the Add picker and the summary.
        self.chain_rows_frame = tk.Frame(hybrid_frame, bg="#1e1e24")
        self.chain_rows_frame.pack(fill="x", pady=(4, 0))
        self._rebuild_chain_ui()

        h_opts = tk.Frame(hybrid_frame, bg="#1e1e24")
        h_opts.pack(fill="x", pady=(4, 0))
        self.hybrid_fallback_var = tk.BooleanVar(value=self.cfg.get("hybrid_fallback", True))
        tk.Checkbutton(
            h_opts, text="Failover on error (incl. 520-524)", variable=self.hybrid_fallback_var, command=self._persist_options,
            bg="#1e1e24", fg="#a1a1aa", activebackground="#1e1e24", selectcolor="#18181b",
            relief="flat", font=("Segoe UI", 8)
        ).pack(side="left")
        self.multi_key_var = tk.BooleanVar(value=self.cfg.get("multi_key_rotation", True))
        tk.Checkbutton(
            h_opts, text="Rotate API keys (all modes)", variable=self.multi_key_var, command=self._persist_options,
            bg="#1e1e24", fg="#a1a1aa", activebackground="#1e1e24", selectcolor="#18181b",
            relief="flat", font=("Segoe UI", 8)
        ).pack(side="left", padx=(8, 0))
        self.strip_images_var = tk.BooleanVar(value=self.cfg.get("strip_images", False))
        tk.Checkbutton(
            h_opts, text="Strip images", variable=self.strip_images_var, command=self._persist_options,
            bg="#1e1e24", fg="#a1a1aa", activebackground="#1e1e24", selectcolor="#18181b",
            relief="flat", font=("Segoe UI", 8)
        ).pack(side="left", padx=(8, 0))

        # Context Window (autodetected from the provider; user-editable)
        ctx_frame = tk.Frame(card, bg="#1e1e24", padx=10, pady=8, highlightbackground="#3f3f46", highlightthickness=1)
        ctx_frame.pack(fill="x", pady=(0, 8))

        ctx_head = tk.Frame(ctx_frame, bg="#1e1e24")
        ctx_head.pack(fill="x")
        ttk.Label(ctx_head, text="Context Window (tokens):", style="FieldLabel.TLabel").pack(side="left")

        self.auto_ctx_var = tk.BooleanVar(value=self.cfg.get("auto_compact_window", True))
        auto_ctx_check = tk.Checkbutton(
            ctx_head, text="Auto-compact at this limit",
            variable=self.auto_ctx_var, command=self._persist_options,
            bg="#1e1e24", fg="#a1a1aa", activebackground="#1e1e24", selectcolor="#18181b",
            relief="flat", font=("Segoe UI", 8)
        )
        auto_ctx_check.pack(side="right")

        ctx_row = tk.Frame(ctx_frame, bg="#1e1e24")
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
            ctx_row, text="", font=("Segoe UI", 8), bg="#1e1e24", fg="#71717a", wraplength=220, justify="left"
        )
        self.ctx_source_lbl.pack(side="left", padx=(8, 0), fill="x", expand=True)

        fetch_ctx_btn = _ColorBtn(
            ctx_row, text="Auto-detect", command=self.fetch_context_length,
            font=("Segoe UI", 8), bg="#3f3f46", fg="#ffffff", abg="#52525b",
            padx=8, pady=2
        )
        fetch_ctx_btn.pack(side="right")

        # Thinking & Reasoning Handling
        t_row = tk.Frame(card, bg="#1e1e24", padx=10, pady=8, highlightbackground="#3f3f46", highlightthickness=1)
        t_row.pack(fill="x", pady=(0, 8))
        ttk.Label(t_row, text="Thinking / Reasoning:", style="FieldLabel.TLabel").pack(anchor="w")
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
        self.thinking_combo.pack(fill="x", pady=(2, 0))
        self.thinking_combo.bind("<<ComboboxSelected>>", self._on_thinking_mode_changed)

        # App options
        opts_frame = tk.Frame(card, bg="#1e1e24", padx=10, pady=8, highlightbackground="#3f3f46", highlightthickness=1)
        opts_frame.pack(fill="x", pady=(0, 8))

        self.min_on_close_var = tk.BooleanVar(value=self.cfg.get("minimize_to_tray", True))
        tk.Checkbutton(
            opts_frame, text="Minimize to tray on close", variable=self.min_on_close_var, command=self._persist_options,
            bg="#1e1e24", fg="#a1a1aa", activebackground="#1e1e24", activeforeground="#ffffff", selectcolor="#18181b",
            relief="flat", font=("Segoe UI", 8)
        ).pack(anchor="w")

        self.auto_start_var = tk.BooleanVar(value=self.cfg.get("auto_start", True))
        tk.Checkbutton(
            opts_frame, text="Start proxy when app launches", variable=self.auto_start_var, command=self._persist_options,
            bg="#1e1e24", fg="#a1a1aa", activebackground="#1e1e24", activeforeground="#ffffff", selectcolor="#18181b",
            relief="flat", font=("Segoe UI", 8)
        ).pack(anchor="w")

        # Boot registration. "auto_start" above only covers proxy start once the
        # app is running; without this nothing launches the app itself at login,
        # so the checkbox silently did nothing for boot. (Run key on Windows,
        # LaunchAgent plist on macOS.)
        self.launch_on_boot_var = tk.BooleanVar(value=self._boot_launch_enabled())
        tk.Checkbutton(
            opts_frame, text=f"Launch Claude Bridge when {'Windows' if sys.platform == 'win32' else 'the OS'} starts", variable=self.launch_on_boot_var,
            command=self._toggle_boot_launch,
            bg="#1e1e24", fg="#a1a1aa", activebackground="#1e1e24", activeforeground="#ffffff", selectcolor="#18181b",
            relief="flat", font=("Segoe UI", 8)
        ).pack(anchor="w")

        self.boot_tray_var = tk.BooleanVar(value=self.cfg.get("boot_to_tray", True))
        boot_tray_check = tk.Checkbutton(
            opts_frame, text="When launched at startup, stay minimized in tray",
            variable=self.boot_tray_var,
            command=self._sync_boot_launch_flag,
            bg="#1e1e24", fg="#a1a1aa", activebackground="#1e1e24", activeforeground="#ffffff", selectcolor="#18181b",
            relief="flat", font=("Segoe UI", 8)
        )
        # Only meaningful once boot launch is on.
        if not self.launch_on_boot_var.get():
            boot_tray_check.configure(state="disabled")
        boot_tray_check.pack(anchor="w")

        # RIGHT: live log column (absorbs extra space, can contract down to limit)
        right_pane = tk.Frame(body, bg="#27272a", padx=12, pady=10, highlightbackground="#3f3f46", highlightthickness=1)

        log_head = tk.Frame(right_pane, bg="#27272a")
        log_head.pack(fill="x", pady=(0, 6))

        tk.Label(log_head, text="Activity Log", font=("Segoe UI", 10, "bold"), bg="#27272a", fg="#60a5fa").pack(side="left")

        clear_btn = _ColorBtn(
            log_head, text="Clear", command=self.clear_log,
            bg="#3f3f46", fg="#a1a1aa", abg="#52525b",
            font=("Segoe UI", 8), padx=8, pady=1
        )
        clear_btn.pack(side="right")

        # Auto-scroll: follow the newest line unless the user is reading back.
        self.auto_scroll_var = tk.BooleanVar(value=True)
        tk.Checkbutton(
            log_head, text="Auto-scroll", variable=self.auto_scroll_var,
            bg="#27272a", fg="#a1a1aa", activebackground="#27272a", selectcolor="#18181b",
            relief="flat", font=("Segoe UI", 8)
        ).pack(side="right", padx=(0, 8))

        self.log_status_lbl = tk.Label(log_head, text="", font=("Segoe UI", 8), bg="#27272a", fg="#71717a", anchor="e")
        self.log_status_lbl.pack(side="right", padx=(0, 8), fill="x", expand=True)

        self.log_area = scrolledtext.ScrolledText(
            right_pane, wrap="word", font=("Consolas", 9),
            bg="#0f0f11", fg="#a1a1aa", insertbackground="#ffffff", relief="flat",
            padx=8, pady=6, bd=0, highlightthickness=0
        )
        self.log_area.tag_configure("info", foreground="#a1a1aa")
        self.log_area.tag_configure("ok", foreground="#4ade80")
        self.log_area.tag_configure("err", foreground="#f87171")
        self.log_area.tag_configure("route", foreground="#60a5fa")
        self.log_area.pack(fill="both", expand=True)
        self.log_area.configure(state="disabled")
        body.add(right_pane, minsize=300, stretch="always")

        # Token counter bar under the log
        token_frame = tk.Frame(right_pane, bg="#27272a")
        token_frame.pack(fill="x", pady=(6, 0))

        reset_tokens_btn = _ColorBtn(
            token_frame, text="Reset", command=self.reset_token_counter,
            font=("Segoe UI", 8), bg="#3f3f46", fg="#ffffff", abg="#52525b",
            padx=8, pady=2
        )
        reset_tokens_btn.pack(side="right")

        tk.Label(
            token_frame, text="Tokens:", font=("Segoe UI", 9, "bold"), bg="#27272a", fg="#60a5fa"
        ).pack(side="left", padx=(0, 8))

        self.token_display = tk.Label(
            token_frame, text="↑ 0 · ↓ 0 · Σ 0 · 0 reqs", font=("Segoe UI", 9),
            bg="#27272a", fg="#d4d4d8"
        )
        self.token_display.pack(side="left")

        # Live API-key rotation view: one row per router, ● marks the key the
        # next request will use and the count shows how the load is spread.
        key_frame = tk.Frame(right_pane, bg="#27272a")
        key_frame.pack(fill="x", pady=(4, 0))
        tk.Label(
            key_frame, text="Keys:", font=("Segoe UI", 9, "bold"), bg="#27272a", fg="#60a5fa"
        ).pack(side="left", padx=(0, 8))
        self.key_display = tk.Label(
            key_frame, text="no requests yet", font=("Consolas", 8),
            bg="#27272a", fg="#d4d4d8", justify="left", anchor="w"
        )
        self.key_display.pack(side="left", fill="x", expand=True)

        self.log(f"Claude Bridge v{APP_VERSION} initialized. Ready to start.")

        # ponytail: 2s poll of a module-level dict. No per-request work and no
        # second event loop; the cost is one label update per tick.
        self._refresh_token_display()
        self.root.after(2000, self._poll_tokens)

        # Auto-start if enabled
        if self.cfg.get("auto_start", True):
            self.root.after(300, self.start_proxy)

    def _refresh_token_display(self):
        s = get_token_stats()
        self.token_display.configure(
            text=f"↑ {s['input']:,} · ↓ {s['output']:,} · Σ {s['total']:,} · {s['requests']} reqs"
        )

    def _refresh_key_display(self):
        """Render which key each router is rotating through, and its request
        share. Fed by the same 2s poll as the token counter."""
        stats = get_key_stats()
        if not stats:
            self.key_display.configure(text="no requests yet")
            return
        # Resolve router URLs back to profile names so the rows read like the
        # provider dropdown instead of raw endpoints.
        name_by_url = {}
        for name, p in self.cfg.get("providers", {}).items():
            url = (p.get("router_url") or "").rstrip("/")
            if url:
                name_by_url.setdefault(url, name)

        rows = []
        for url, st in stats.items():
            if not st["keys"]:
                continue
            label = name_by_url.get(url.rstrip("/"), url)
            chips = [
                ("●" if k["masked"] == st["active"] else "○") + f"{k['masked']} {k['requests']}"
                for k in st["keys"]
            ]
            rows.append(f"{label}:  " + "   ".join(chips))
        self.key_display.configure(text="\n".join(rows) if rows else "no requests yet")

    def _poll_tokens(self):
        """Background counter refresh. Re-arms itself until the window dies."""
        try:
            self._refresh_token_display()
            self._refresh_key_display()
        except Exception:
            return
        self.root.after(2000, self._poll_tokens)

    def reset_token_counter(self):
        reset_token_stats()
        self._refresh_token_display()
        self.log("Token counter reset to zero.")

    def _on_provider_selected(self, event=None):
        p_name = self.provider_var.get()
        p_data = self.cfg.get("providers", {}).get(p_name, {})
        if not p_data:
            return

        self.url_var.set(p_data.get("router_url", ""))
        self.key_var.set(p_data.get("api_key", ""))
        if hasattr(self, "proxy_url_var"):
            self.proxy_url_var.set(p_data.get("egress_proxy", ""))
        self.model_var.set(p_data.get("model", ""))
        self.sonnet_var.set(p_data.get("sonnet_model", p_data.get("model", "")))
        self.opus_var.set(p_data.get("opus_model", p_data.get("model", "")))
        self.haiku_var.set(p_data.get("haiku_model", p_data.get("model", "")))
        self.model_effort_var.set(effort_label(p_data.get("model_effort", "")))
        self.sonnet_effort_var.set(effort_label(p_data.get("sonnet_effort", "")))
        self.opus_effort_var.set(effort_label(p_data.get("opus_effort", "")))
        self.haiku_effort_var.set(effort_label(p_data.get("haiku_effort", "")))

        # Legacy hybrid profiles carry their routing setup inside the profile
        # (is_hybrid/primary_provider/secondary_provider). Mirror it into the
        # Hybrid panel so selecting such a profile actually enables hybrid mode;
        # the global checkbox alone previously stayed off and every request
        # went to a single router.
        if p_data.get("is_hybrid"):
            self.hybrid_var.set(True)
            provs = list(self.cfg.get("providers", {}).keys())
            legacy_chain = []
            if p_data.get("primary_provider") in provs:
                legacy_chain.append(p_data["primary_provider"])
            if p_data.get("secondary_provider") in provs and p_data["secondary_provider"] not in legacy_chain:
                legacy_chain.append(p_data["secondary_provider"])
            if legacy_chain:
                self.cfg["hybrid_chain"] = legacy_chain
                self._rebuild_chain_ui()
            self.log(f"Profile '{p_name}' is a hybrid profile: chain={legacy_chain}")

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

    def _effort_combo(self, parent, var, width=8):
        cb = ttk.Combobox(
            parent, textvariable=var, values=EFFORT_LEVELS,
            state="readonly", width=width, font=("Segoe UI", 8)
        )
        cb.bind("<<ComboboxSelected>>", self._on_effort_changed)
        return cb

    def _on_effort_changed(self, event=None):
        """Persist + log on change. The running proxy reads these vars live via
        get_current_config(), so no restart is needed -- only the Mappings line
        printed at start-up is a snapshot."""
        p_name = self.provider_var.get().strip()
        if p_name and p_name in self.cfg.get("providers", {}):
            self.cfg["providers"][p_name].update({
                "model_effort": effort_key(self.model_effort_var.get()),
                "sonnet_effort": effort_key(self.sonnet_effort_var.get()),
                "opus_effort": effort_key(self.opus_effort_var.get()),
                "haiku_effort": effort_key(self.haiku_effort_var.get()),
            })
            save_config(self.cfg)
        self.log(
            f"Effort set: Custom={self.model_effort_var.get()}, "
            f"Sonnet={self.sonnet_effort_var.get()}, Opus={self.opus_effort_var.get()}, "
            f"Haiku={self.haiku_effort_var.get()} (applies to next request, no restart)"
        )

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
        # Preserve legacy in-profile hybrid fields (is_hybrid, primary/secondary
        # provider, fallback_on_error) that this form doesn't edit; a blind
        # overwrite would silently strip them and kill the routing setup.
        merged = dict(self.cfg["providers"].get(p_name, {}))
        merged.update({
            "router_url": self.url_var.get().strip(),
            "api_key": self.key_var.get().strip(),
            "model": self.model_var.get().strip(),
            "sonnet_model": self.sonnet_var.get().strip(),
            "opus_model": self.opus_var.get().strip(),
            "haiku_model": self.haiku_var.get().strip(),
            "model_effort": effort_key(self.model_effort_var.get()),
            "sonnet_effort": effort_key(self.sonnet_effort_var.get()),
            "opus_effort": effort_key(self.opus_effort_var.get()),
            "haiku_effort": effort_key(self.haiku_effort_var.get()),
            "egress_proxy": self.proxy_url_var.get().strip() if hasattr(self, "proxy_url_var") else self.cfg["providers"].get(p_name, {}).get("egress_proxy", "")
        })
        self.cfg["providers"][p_name] = merged
        self.cfg["active_provider"] = p_name
        self.cfg["port"] = self.port_var.get()
        self.cfg["minimize_to_tray"] = self.min_on_close_var.get()
        self.cfg["auto_start"] = self.auto_start_var.get()
        self.cfg["boot_to_tray"] = self.boot_tray_var.get()
        self.cfg["thinking_mode"] = self._get_thinking_mode_key()
        self.cfg["context_length"] = self.get_context_length()
        self.cfg["auto_compact_window"] = self.auto_ctx_var.get()
        self.cfg["strip_images"] = self.strip_images_var.get()
        self.cfg["enable_hybrid_router"] = self.hybrid_var.get()
        self.cfg["hybrid_chain"] = self._get_chain_names()
        self.cfg["hybrid_fallback"] = self.hybrid_fallback_var.get()
        self.cfg["multi_key_rotation"] = self.multi_key_var.get()
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
        """Open a custom Toplevel dialog to get the new provider name.
        Using simpledialog.askstring can fail silently on macOS when called from
        a widget embedded inside a Canvas (the scrollable config column)."""
        result = {"name": None}

        dlg = tk.Toplevel(self.root)
        dlg.title("New Provider")
        dlg.configure(bg="#18181b")
        dlg.resizable(False, False)
        dlg.transient(self.root)

        # Centre the dialog over the main window
        self.root.update_idletasks()
        rx = self.root.winfo_x() + self.root.winfo_width() // 2
        ry = self.root.winfo_y() + self.root.winfo_height() // 2
        dlg.geometry(f"380x160+{rx - 190}+{ry - 80}")
        dlg.lift()
        dlg.focus_force()

        tk.Label(
            dlg, text="New Provider Profile",
            font=("Segoe UI", 11, "bold"), bg="#18181b", fg="#ffffff"
        ).pack(pady=(16, 4))
        tk.Label(
            dlg, text="Enter a name (e.g. DeepSeek, Groq, My Router):",
            font=("Segoe UI", 9), bg="#18181b", fg="#a1a1aa"
        ).pack()

        name_var = tk.StringVar()
        entry = tk.Entry(
            dlg, textvariable=name_var, font=("Segoe UI", 10),
            bg="#27272a", fg="#ffffff", insertbackground="#ffffff",
            relief="flat", highlightbackground="#3b82f6", highlightthickness=1,
            width=32
        )
        entry.pack(pady=(6, 0), ipady=4)
        entry.focus_set()

        btn_row = tk.Frame(dlg, bg="#18181b")
        btn_row.pack(pady=(10, 0))

        def _confirm(event=None):
            result["name"] = name_var.get()
            dlg.destroy()

        def _cancel(event=None):
            dlg.destroy()

        _ColorBtn(
            btn_row, text="Create", command=_confirm,
            font=("Segoe UI", 9, "bold"), bg="#2563eb", fg="#ffffff", abg="#1d4ed8",
            padx=12, pady=4
        ).pack(side="left", padx=(0, 8))
        _ColorBtn(
            btn_row, text="Cancel", command=_cancel,
            font=("Segoe UI", 9), bg="#3f3f46", fg="#ffffff", abg="#52525b",
            padx=12, pady=4
        ).pack(side="left")

        entry.bind("<Return>", _confirm)
        entry.bind("<Escape>", _cancel)

        dlg.grab_set()
        self.root.wait_window(dlg)

        name = result["name"]
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
            "haiku_model": self.haiku_var.get().strip(),
            "model_effort": effort_key(self.model_effort_var.get()),
            "sonnet_effort": effort_key(self.sonnet_effort_var.get()),
            "opus_effort": effort_key(self.opus_effort_var.get()),
            "haiku_effort": effort_key(self.haiku_effort_var.get())
        }
        self.cfg["active_provider"] = name
        save_config(self.cfg)

        self.provider_combo["values"] = list(self.cfg["providers"].keys())
        self.provider_var.set(name)
        self._on_provider_selected()
        self._rebuild_chain_ui()
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
        # A deleted provider can no longer be a hop; drop it from the chain
        # (and the legacy keys) so the engine doesn't resolve a dead name.
        self.cfg["hybrid_chain"] = [n for n in self.cfg.get("hybrid_chain", []) if n != p_name]
        self.cfg.pop("hybrid_primary_provider", None)
        self.cfg.pop("hybrid_secondary_provider", None)
        save_config(self.cfg)

        self.provider_combo["values"] = remaining
        self.provider_var.set(remaining[0])
        self._on_provider_selected()
        self._rebuild_chain_ui()
        self.log(f"Deleted provider profile '{p_name}'. Switched to '{remaining[0]}'.")

    def _toggle_key_visibility(self):
        if self.show_key_var.get():
            self.key_entry.configure(show="")
        else:
            self.key_entry.configure(show="•")

    # --- Boot launch: HKCU Run key on Windows, LaunchAgent on macOS ---
    RUN_KEY_NAME = "ClaudeBridge"
    MACOS_PLIST_PATH = os.path.expanduser("~/Library/LaunchAgents/com.claudebridge.app.plist")

    @classmethod
    def _boot_launch_command(cls):
        # Frozen exe: launch the binary directly. From source: use the
        # current interpreter so it doesn't depend on file association.
        if getattr(sys, "frozen", False):
            target = f'"{sys.executable}"'
        else:
            target = f'"{sys.executable}" "{os.path.abspath(sys.argv[0])}"'
        return target

    @classmethod
    def _run_key_path(cls):
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\Microsoft\Windows\CurrentVersion\Run") as k:
            val, _ = winreg.QueryValueEx(k, cls.RUN_KEY_NAME)
            return val

    def _boot_launch_enabled(self):
        """Whether our HKCU Run value / LaunchAgent plist exists."""
        if sys.platform == "darwin":
            return os.path.exists(self.MACOS_PLIST_PATH)
        try:
            self._run_key_path()
            return True
        except Exception:
            return False

    def _toggle_boot_launch(self):
        try:
            if self.launch_on_boot_var.get():
                target = self._boot_launch_command()
                if self.boot_tray_var.get():
                    target += " --tray"
                if sys.platform == "darwin":
                    self._write_macos_plist(target)
                    self.log(f"Will launch at macOS login: {target}")
                else:
                    self._write_windows_run_key(target)
                    self.log(f"Will launch at Windows login: {target}")
            else:
                if sys.platform == "darwin":
                    if os.path.exists(self.MACOS_PLIST_PATH):
                        os.remove(self.MACOS_PLIST_PATH)
                    self.log("Removed macOS login launch entry.")
                else:
                    self._delete_windows_run_key()
                    self.log("Removed Windows login launch entry.")
        except Exception as e:
            self.log(f"Could not update startup entry: {e}")
            messagebox.showerror("Startup Error", f"Could not update the startup entry:\n{e}")

    def _write_macos_plist(self, target):
        # ponytail: shlex.split of a quoted command string, macOS needs args as
        # array elements — fine for our two shapes (exe / exe + script).
        import shlex
        args = shlex.split(target)
        plist = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key><string>com.claudebridge.app</string>
    <key>ProgramArguments</key>
    <array>
{''.join(f'        <string>{a}</string>\n' for a in args)}    </array>
    <key>RunAtLoad</key><true/>
</dict>
</plist>
"""
        os.makedirs(os.path.dirname(self.MACOS_PLIST_PATH), exist_ok=True)
        with open(self.MACOS_PLIST_PATH, "w", encoding="utf-8") as f:
            f.write(plist)

    def _write_windows_run_key(self, target):
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER,
                            r"Software\Microsoft\Windows\CurrentVersion\Run", 0,
                            winreg.KEY_SET_VALUE) as k:
            winreg.SetValueEx(k, self.RUN_KEY_NAME, 0, winreg.REG_SZ, target)

    def _delete_windows_run_key(self):
        import winreg
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER,
                                r"Software\Microsoft\Windows\CurrentVersion\Run", 0,
                                winreg.KEY_SET_VALUE) as k:
                winreg.DeleteValue(k, self.RUN_KEY_NAME)
        except FileNotFoundError:
            pass

    def _persist_options(self, *_a):
        """Persist every option checkbox/combobox the moment it changes.

        The engine reads these live via get_current_config(), but without this
        handler a toggle was lost on restart — the value only reached disk when
        the user happened to click Save or quit cleanly.
        """
        self.cfg["minimize_to_tray"] = self.min_on_close_var.get()
        self.cfg["auto_start"] = self.auto_start_var.get()
        self.cfg["boot_to_tray"] = self.boot_tray_var.get()
        self.cfg["enable_hybrid_router"] = self.hybrid_var.get()
        self.cfg["hybrid_chain"] = self._get_chain_names()
        self.cfg["hybrid_fallback"] = self.hybrid_fallback_var.get()
        self.cfg["multi_key_rotation"] = self.multi_key_var.get()
        self.cfg["strip_images"] = self.strip_images_var.get()
        self.cfg["auto_compact_window"] = self.auto_ctx_var.get()
        save_config(self.cfg)

    # --- Failover chain UI ---------------------------------------------------
    def _get_chain_names(self):
        """Chain order straight from the row comboboxes (skips empty rows)."""
        return [v.get().strip() for v in self._chain_vars if v.get().strip()]

    def _rebuild_chain_ui(self):
        """Re-render the chain rows from cfg.

        Called on startup, provider add/delete, and any chain edit. Rebuilding
        on every change is what keeps the combobox values fresh after providers
        are added or deleted -- the old Primary/Secondary boxes went stale.
        """
        for row in self._chain_rows:
            row.destroy()
        self._chain_rows = []
        self._chain_vars = []

        prov_names = list(self.cfg.get("providers", {}).keys())
        chain = [n for n in self.cfg.get("hybrid_chain", []) if n in prov_names]
        self.cfg["hybrid_chain"] = chain

        for idx, name in enumerate(chain):
            row = tk.Frame(self.chain_rows_frame, bg="#1e1e24")
            row.pack(fill="x", pady=(2, 0))
            self._chain_rows.append(row)

            tk.Label(row, text=f"{idx + 1}.", width=3, anchor="w", font=("Segoe UI", 8, "bold"),
                     bg="#1e1e24", fg="#71717a").pack(side="left")
            var = tk.StringVar(value=name)
            self._chain_vars.append(var)
            cb = ttk.Combobox(row, textvariable=var, values=prov_names, state="readonly",
                              width=14, font=("Segoe UI", 8))
            cb.pack(side="left", padx=(0, 4))
            # ttk has no command=; selection is the persistence hook. Rebuilding
            # happens here too so the arrows/summary follow a manual rename.
            cb.bind("<<ComboboxSelected>>", lambda _e: self._on_chain_edited())

            _ColorBtn(row, text="▲", command=lambda i=idx: self._move_chain_entry(i, -1),
                      font=("Segoe UI", 7), bg="#3f3f46", fg="#ffffff", abg="#52525b",
                      padx=4, pady=0).pack(side="left", padx=1)
            _ColorBtn(row, text="▼", command=lambda i=idx: self._move_chain_entry(i, 1),
                      font=("Segoe UI", 7), bg="#3f3f46", fg="#ffffff", abg="#52525b",
                      padx=4, pady=0).pack(side="left", padx=1)
            _ColorBtn(row, text="x", command=lambda i=idx: self._remove_chain_entry(i),
                      font=("Segoe UI", 7), bg="#dc2626", fg="#ffffff", abg="#b91c1c",
                      padx=4, pady=0).pack(side="left", padx=1)

        # The Add combobox only offers providers not already in the chain.
        remaining = [p for p in prov_names if p not in chain]
        self.chain_add_combo["values"] = remaining
        self.chain_add_var.set(remaining[0] if remaining else "")

        self._update_chain_summary()

    def _on_chain_edited(self):
        """A row's provider was changed via its dropdown: persist + refresh."""
        # Re-render so index labels and the Add picker stay consistent.
        names = self._get_chain_names()
        self.cfg["hybrid_chain"] = names
        self._rebuild_chain_ui()
        self._persist_options()

    def _add_chain_provider(self):
        name = self.chain_add_var.get().strip()
        if not name or name in self._get_chain_names():
            return
        self.cfg.setdefault("hybrid_chain", []).append(name)
        self._rebuild_chain_ui()
        self._persist_options()

    def _move_chain_entry(self, idx, delta):
        chain = self._get_chain_names()
        new_idx = idx + delta
        if not (0 <= new_idx < len(chain)):
            return
        chain[idx], chain[new_idx] = chain[new_idx], chain[idx]
        self.cfg["hybrid_chain"] = chain
        self._rebuild_chain_ui()
        self._persist_options()

    def _remove_chain_entry(self, idx):
        chain = self._get_chain_names()
        if not (0 <= idx < len(chain)):
            return
        del chain[idx]
        self.cfg["hybrid_chain"] = chain
        self._rebuild_chain_ui()
        self._persist_options()

    def _update_chain_summary(self):
        chain = self._get_chain_names()
        if not chain:
            self.chain_summary_lbl.config(
                text="Chain empty: hybrid routing is off until you add routers.")
            return
        arrow = " → ".join(chain)
        if len(chain) == 1:
            self.chain_summary_lbl.config(
                text=f"Chain: {arrow} (no failover target — add a 2nd router)")
        else:
            fast = chain[1] if len(chain) >= 2 else chain[0]
            self.chain_summary_lbl.config(
                text=f"Failover order: {arrow}\n"
                     f"Sonnet/Opus start at '{chain[0]}'; Haiku/vision start at '{fast}'.")

    def _sync_boot_launch_flag(self):
        """The --tray flag lives inside the Run key's command, so flipping the
        tray preference has to rewrite that entry for it to take effect."""
        self._persist_options()
        if self.launch_on_boot_var.get():
            self._toggle_boot_launch()  # rewrites the key with the current flag

    def log(self, text):
        def _append():
            ts = datetime.now().strftime("%H:%M:%S")
            lower = text.lower()
            # Routing decisions are informational even though the line names the
            # failover chain -- keep them blue rather than letting the word
            # "failover" paint every route as an error.
            if text.startswith("Hybrid Route:"):
                tag = "route"
            elif "error" in lower or "failed" in lower or "failover" in lower or "rotating" in lower:
                tag = "err"
            elif "successfully" in lower or "started" in lower:
                tag = "ok"
            else:
                tag = "info"
            self.log_area.configure(state="normal")
            self.log_area.insert("end", f"[{ts}] ", ("info",))
            self.log_area.insert("end", f"{text}\n", (tag,))
            if self.auto_scroll_var.get():
                self.log_area.see("end")
            self.log_area.configure(state="disabled")
            # The log is the status surface now: surface the newest line up top.
            self.log_status_lbl.configure(
                text=("⚠ " if tag == "err" else "") + text[:60]
            )
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
            "model_effort": effort_key(self.model_effort_var.get()),
            "sonnet_effort": effort_key(self.sonnet_effort_var.get()),
            "opus_effort": effort_key(self.opus_effort_var.get()),
            "haiku_effort": effort_key(self.haiku_effort_var.get()),
            "port": self.port_var.get(),
            "minimize_to_tray": self.min_on_close_var.get(),
            "auto_start": self.auto_start_var.get(),
            "thinking_mode": self._get_thinking_mode_key(),
            "context_length": self.get_context_length(),
            "auto_compact_window": self.auto_ctx_var.get(),
            "strip_images": self.strip_images_var.get(),
            # The engine reads the egress proxy off the top-level cfg when the
            # hybrid router is off (or its chain resolves empty) and falls back
            # to `dict(cfg)` -- without this key the proxy configured on the
            # provider is silently dropped and every request goes out on the
            # raw IP the proxy exists to hide.
            "egress_proxy": self.proxy_url_var.get().strip() if hasattr(self, "proxy_url_var") else "",
            # Hybrid routing: the engine reads `all_providers` to resolve the
            # chain by name. Without these keys it silently stays single-router
            # and never fails over.
            "all_providers": self.cfg.get("providers", {}),
            "enable_hybrid_router": self.hybrid_var.get(),
            "hybrid_chain": self._get_chain_names(),
            "hybrid_fallback": self.hybrid_fallback_var.get(),
            "multi_key_rotation": self.multi_key_var.get()
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
                "haiku_model": cfg["haiku_model"],
                "model_effort": cfg["model_effort"],
                "sonnet_effort": cfg["sonnet_effort"],
                "opus_effort": cfg["opus_effort"],
                "haiku_effort": cfg["haiku_effort"]
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
            self.start_btn.configure(text="Stop Proxy", bg="#ef4444", activebackground="#dc2626", highlightbackground="#ef4444")
            self.log(f"Proxy successfully started on http://127.0.0.1:{port}")
            if cfg.get("enable_hybrid_router"):
                chain = cfg.get("hybrid_chain") or []
                self.log(
                    f"Hybrid ON: chain = {' -> '.join(chain) if chain else '(empty)'}"
                    + (", failover armed" if cfg.get("hybrid_fallback", True) else ", failover OFF")
                )
            else:
                self.log(f"Active Provider: [{p_name}] -> {cfg['router_url']}")
            self.log(f"Mappings: Sonnet->{cfg['sonnet_model']} ({cfg['sonnet_effort'] or 'default effort'}), Opus->{cfg['opus_model']} ({cfg['opus_effort'] or 'default effort'}), Haiku->{cfg['haiku_model']} ({cfg['haiku_effort'] or 'default effort'}), Custom->{cfg['model']} ({cfg['model_effort'] or 'default effort'})")
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
        self.start_btn.configure(text="Start Proxy", bg="#10b981", activebackground="#059669", highlightbackground="#10b981")
        self.log("Proxy stopped.")
        self._update_tray_menu(is_running=False)

    def configure_claude_settings(self):
        r"""Points Claude Code's settings.json at the local proxy."""
        cfg = self.get_current_config()
        port = cfg["port"]
        local_base_url = f"http://127.0.0.1:{port}"

        try:
            os.makedirs(CLAUDE_DIR, exist_ok=True)
            settings = {}
            if os.path.exists(CLAUDE_SETTINGS_PATH):
                # Make backup if not already present
                if not os.path.exists(CLAUDE_BACKUP_PATH):
                    shutil.copy2(CLAUDE_SETTINGS_PATH, CLAUDE_BACKUP_PATH)
                    self.log(f"Created Claude settings backup at {CLAUDE_BACKUP_PATH}")
                try:
                    with open(CLAUDE_SETTINGS_PATH, "r", encoding="utf-8") as f:
                        settings = json.load(f)
                except Exception:
                    settings = {}

            if "env" not in settings:
                settings["env"] = {}

            settings["env"]["ANTHROPIC_BASE_URL"] = local_base_url
            settings["env"]["CLAUDE_CODE_USE_AUTH_TOKEN"] = "true"
            settings["env"]["ANTHROPIC_AUTH_TOKEN"] = "claude-bridge-local-token"
            settings["env"]["ANTHROPIC_MODEL"] = cfg["model"]
            settings["env"]["ANTHROPIC_DEFAULT_OPUS_MODEL"] = cfg["opus_model"]
            settings["env"]["ANTHROPIC_DEFAULT_SONNET_MODEL"] = cfg["sonnet_model"]
            settings["env"]["ANTHROPIC_DEFAULT_HAIKU_MODEL"] = cfg["haiku_model"]

            # Auto-compact: for an unrecognized non-claude-* model ID (this
            # proxy's usual target) CLAUDE_CODE_MAX_CONTEXT_TOKENS sets the
            # assumed window directly. AUTO_COMPACT_WINDOW is the highest-
            # precedence threshold and is capped at that model window.
            context_length = cfg["context_length"]
            settings["env"]["CLAUDE_CODE_MAX_CONTEXT_TOKENS"] = str(context_length)
            settings["env"]["CLAUDE_CODE_AUTO_COMPACT_WINDOW"] = str(context_length)
            # A [1m] model alias makes Claude Code assume a 1M window and ignore
            # MAX_CONTEXT_TOKENS; disable that assumption when the user's
            # verified window is smaller.
            if context_length < 1_000_000:
                settings["env"]["CLAUDE_CODE_DISABLE_1M_CONTEXT"] = "1"
            else:
                settings["env"].pop("CLAUDE_CODE_DISABLE_1M_CONTEXT", None)
            # This proxy rewrites Anthropic<->OpenAI traffic, which strips the
            # safeguards request/response fields the auto-mode classifier needs,
            # so Claude Code falls back to its own billed classifier requests.
            # Asking for server checks here can never succeed.
            settings["env"]["CLAUDE_CODE_AUTO_MODE_SERVER"] = "0"
            settings["env"].pop("DISABLE_COMPACT", None)
            if cfg.get("auto_compact_window", True):
                settings["autoCompactWindow"] = context_length
            else:
                settings.pop("autoCompactWindow", None)

            with open(CLAUDE_SETTINGS_PATH, "w", encoding="utf-8") as f:
                json.dump(settings, f, indent=2)

            self.log(f"Updated Claude Code settings for provider [{self.provider_var.get()}].")
            compact_note = (
                f"Auto-compact at {context_length:,} tokens (matches this model's context window)."
                if cfg.get("auto_compact_window", True) else
                "Auto-compact disabled — Claude Code will use its own defaults."
            )
            messagebox.showinfo(
                "Claude Code Configured",
                f"Claude Code settings updated successfully!\n\n"
                f"Active Provider: {self.provider_var.get()}\n"
                f"Base URL: {local_base_url}\n"
                f"Sonnet: {cfg['sonnet_model']}\n"
                f"Opus: {cfg['opus_model']}\n"
                f"Haiku: {cfg['haiku_model']}\n"
                f"Custom: {cfg['model']}\n\n"
                f"{compact_note}\n\n"
                f"Claude Code is now connected through Claude Bridge!"
            )
        except Exception as e:
            self.log(f"Failed to configure Claude Code settings: {e}")
            messagebox.showerror("Error", f"Could not update Claude Code settings:\n{e}")

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
                            "CLAUDE_CODE_DISABLE_1M_CONTEXT", "CLAUDE_CODE_AUTO_MODE_SERVER",
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

    # Crucial on Windows: Explicitly register AppUserModelID so Windows Taskbar
    # groups this process as Claude Bridge and renders our custom icon rather than the Python logo.
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(f"anthropic.claudebridge.app.{APP_VERSION}")
        except Exception:
            pass

    # --tray: launched at OS login with "start minimized in the tray".
    start_in_tray = "--tray" in sys.argv[1:]
    try:
        root = tk.Tk()
        app = ClaudeBridgeApp(root)
        if start_in_tray:
            app.minimize_to_tray()
        root.mainloop()
    except Exception:
        import traceback
        err_file = os.path.join(os.path.dirname(os.path.abspath(sys.executable)), "claude_bridge_crash.log")
        with open(err_file, "w", encoding="utf-8") as f:
            traceback.print_exc(file=f)


if __name__ == "__main__":
    main()
