"""Standard-library value types shared by authority and execution bootstrap."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


class AuthorityError(RuntimeError):
    """An authority operation failed an identity, integrity or lifecycle check."""


@dataclass(frozen=True)
class AuthorityRevision:
    """Count and last id of events ordered by timestamp and object id."""

    event_count: int
    latest_event_id: str | None


@dataclass(frozen=True)
class AuthorityBinding:
    """Historical ECS authority identity, retained for export provenance."""

    schema: str
    authority_uuid: str
    instance_id: str
    db_path: Path


@dataclass(frozen=True)
class GitAuthorityBinding:
    """Canonical Git authority identity; local databases are disposable projections."""

    schema: str
    authority_uuid: str
    remote: str
    ref: str


@dataclass(frozen=True)
class VerifiedSnapshot:
    """An execution snapshot whose file map and authority identity were verified."""

    path: Path
    digest: str
    release: str
    root: str
    revision: AuthorityRevision
    bundle_digest: str | None
    tiers: Mapping[str, str]
    authority_commit: str | None = None


@dataclass(frozen=True)
class AuthorityHead:
    """Current pointer and generation, with live event revision and export state."""

    release: str
    generation: str
    decision: str | None
    revision: AuthorityRevision
    exported: bool
    export_commit: str | None
    authority_commit: str | None = None
