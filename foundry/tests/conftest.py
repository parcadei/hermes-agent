"""Shared Foundry test fixtures and canned conversations."""

from __future__ import annotations


PETS_CONVERSATION: list[tuple[str, str]] = [
    ("User", "I have a cat named Luna."),
    ("Assistant", "Luna is a tabby cat."),
    ("User", "I have a dog named Mochi."),
]


CONTRADICTORY_CONVERSATION: list[tuple[str, str]] = [
    ("User", "I moved to Seattle."),
    ("Assistant", "Noted."),
    ("User", "I moved to Portland."),
]
