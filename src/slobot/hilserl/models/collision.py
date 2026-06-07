"""Collision check results."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CollisionResult:
    valid: bool
    contact_count: int = 0
