"""Shared fixtures and constants for Foundry tests."""

from __future__ import annotations


PETS_CONVERSATION = [
    ("User", "I have a cat named Luna."),
    ("Assistant", "Luna is a tabby cat."),
]


CONTRADICTORY_CONVERSATION = [
    ("User", "I live in Seattle."),
    ("User", "Actually, I live in Portland."),
]
