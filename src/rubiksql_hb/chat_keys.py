"""Hash-only refusal policy shared by chat credential loaders; embeddings are separate."""
from __future__ import annotations

import hashlib
from importlib.resources import files
import logging
import os
from pathlib import Path
import re

_LOGGER = logging.getLogger(__name__)
DENYLIST_ENV = "RUBIKSQL_CHAT_KEY_DENYLIST"


class DeniedChatKey(RuntimeError):
    """A retired chat credential that must never reach a provider."""


def denylist() -> frozenset[str]:
    """Load the permanent package hashes plus an optional local denylist.

    Args:
        None.

    Returns:
        frozenset[str]: SHA-256 digests. Local entries can only add bans.

    Raises:
        RuntimeError: A present policy file is unreadable or malformed.
    """
    local = Path(os.environ.get(DENYLIST_ENV) or Path.home() / ".config/rubiksql-hb/keys/denylist.sha256").expanduser()
    try:
        texts = [files("rubiksql_hb").joinpath("chat_key_denylist.sha256").read_text(encoding="ascii")]
        if local.exists():
            texts.append(local.read_text(encoding="ascii"))
        hashes = [line.strip().lower() for text in texts for line in text.splitlines() if line.strip()]
        if any(re.fullmatch(r"[0-9a-f]{64}", value) is None for value in hashes):
            raise ValueError
        return frozenset(hashes)
    except (OSError, ValueError):
        raise RuntimeError("Cannot load the chat key denylist; chat credentials refused.") from None


def refused(secret: str, label: str, hashes: frozenset[str]) -> bool:
    """Identify a banned chat key and warn using only a caller-owned filename or index.

    Args:
        secret (str): In-memory credential; never rendered.
        label (str): File basename or account index, never credential content.
        hashes (frozenset[str]): Resolved hash-only refusal policy.

    Returns:
        bool: True when the key is permanently retired by policy.
    """
    if hashlib.sha256(secret.strip().encode("utf-8")).hexdigest() not in hashes:
        return False
    _LOGGER.warning("[chat-key-denylist] Retired %s; credential refused.", label)
    return True
