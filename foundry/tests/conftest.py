"""Shared Foundry test fixtures used by package-relative test imports."""

PETS_CONVERSATION = [
    ("User", "I have a cat named Luna."),
    ("Assistant", "Luna is a tabby cat."),
]

CONTRADICTORY_CONVERSATION = [
    ("User", "I live in Portland."),
    ("Assistant", "Got it."),
    ("User", "Actually I moved to Seattle."),
]
