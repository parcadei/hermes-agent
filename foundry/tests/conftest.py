"""Shared fixtures and sample conversations for Foundry tests."""

PETS_CONVERSATION = [
    ("User", "I have a cat named Luna."),
    ("Assistant", "Luna is a tabby cat."),
    ("User", "Luna lives in Portland."),
]

CONTRADICTORY_CONVERSATION = [
    ("User", "I live in Seattle."),
    ("Assistant", "You live in Seattle."),
    ("User", "I moved to Portland."),
    ("Assistant", "You now live in Portland."),
]
