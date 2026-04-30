"""
Configuration loader for the War Room voice server.

Resolves the project root, loads agent voice mappings from voices.json,
and exposes environment variable helpers.
"""

import json
import os
import subprocess
from pathlib import Path


def get_project_root() -> Path:
    """Resolve the ClaudeClaw project root via git or file path fallback."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True, text=True, check=True,
            cwd=Path(__file__).parent,
        )
        return Path(result.stdout.strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Fallback: warroom/ sits one level below project root
        return Path(__file__).resolve().parent.parent


PROJECT_ROOT = get_project_root()
WARROOM_DIR = PROJECT_ROOT / "warroom"
VOICES_FILE = WARROOM_DIR / "voices.json"


def load_voices() -> dict:
    """Load agent voice configs from voices.json.

    Returns a dict mapping agent_id to {voice_id, name}.
    """
    if not VOICES_FILE.exists():
        raise FileNotFoundError(f"Voice config not found at {VOICES_FILE}")

    with open(VOICES_FILE, "r") as f:
        return json.load(f)


# Pre-load at import time so other modules can use it directly
AGENT_VOICES = load_voices()

# Default agent if routing can't determine who should respond
DEFAULT_AGENT = "main"


# ── Dynamic agent roster ─────────────────────────────────────────────────
#
# The Node.js bot writes /tmp/warroom-agents.json on startup with the
# configured agents (from agent.yaml files). If that file is missing or
# empty, we fall back to this default 5-agent demo roster.

import tempfile

_ROSTER_PATH = Path(tempfile.gettempdir()) / "warroom-agents.json"

DEFAULT_ROSTER: list[dict] = [
    {"id": "main", "name": "Main", "description": "General ops and triage", "role": "The Hand of the King"},
    {"id": "research", "name": "Research", "description": "Deep web research, academic sources, competitive intel, trend analysis", "role": "Grand Maester"},
    {"id": "comms", "name": "Comms", "description": "Email, Slack, Telegram, WhatsApp, customer comms, inbox triage", "role": "Master of Whisperers"},
    {"id": "content", "name": "Content", "description": "Writing, YouTube scripts, LinkedIn posts, blog copy, creative direction", "role": "The Royal Bard"},
    {"id": "ops", "name": "Ops", "description": "Calendar, scheduling, cron, system operations, MCP tool work, automations", "role": "Master of War"},
]


def load_dynamic_roster() -> list[dict]:
    """Load the agent roster written by Node.js. Falls back to DEFAULT_ROSTER."""
    try:
        data = json.loads(_ROSTER_PATH.read_text())
        if data and isinstance(data, list) and len(data) > 0:
            return data
    except Exception:
        pass
    return DEFAULT_ROSTER


def get_valid_agent_ids() -> set[str]:
    """Return the set of valid agent IDs from the dynamic roster."""
    return {a["id"] for a in load_dynamic_roster()}
