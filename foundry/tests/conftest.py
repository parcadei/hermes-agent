"""Shared fixtures and test data for the Foundry test suite.

Provides realistic conversation data and common fixtures used across all
test files. Each conversation scenario exercises different aspects of the
memory system.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pytest


PETS_CONVERSATION: list[tuple[str, str]] = [
    ("user", "I have two cats named Luna and Mochi. Luna is a tabby and Mochi is a black cat."),
    ("assistant", "That's lovely! Tell me more about your cats."),
    ("user", "Luna is 3 years old, I adopted her in 2021. Mochi is just a kitten, about 6 months old."),
    ("assistant", "How sweet! Do you have any other pets?"),
    ("user", "Yes, I also have a golden retriever named Max. He's 5 years old and very friendly."),
    ("user", "Oh and I recently got three neon tetras for my aquarium. And a betta fish."),
    ("assistant", "Quite the menagerie! So you have cats, a dog, and fish."),
    ("user", "I'm thinking about getting a parrot too, but haven't decided yet."),
]

TRAVEL_CONVERSATION: list[tuple[str, str]] = [
    ("user", "I visited Rome last June. Amazing trip, spent two weeks there."),
    ("assistant", "Rome is wonderful! What did you see?"),
    ("user", "We visited the Colosseum on June 5th and the Vatican on June 8th."),
    ("user", "Before that I was in Paris in March 2024. Just a long weekend, Friday to Monday."),
    ("assistant", "How did Paris compare to Rome?"),
    ("user", "Paris was beautiful but more expensive. We stayed at Hotel Marais."),
    ("user", "Next summer I'm planning a trip to Tokyo. My friend Yuki lives there."),
    ("user", "We also went to Florence on June 12th, a day trip from Rome."),
]

WORK_CONVERSATION: list[tuple[str, str]] = [
    ("user", "I work at TechCorp as a software engineer. Been there since 2022."),
    ("assistant", "What kind of work do you do there?"),
    ("user", "Mostly backend development in Python. My manager is Sarah Chen."),
    ("user", "Actually, I just got promoted to senior engineer last month."),
    ("assistant", "Congratulations! That's a great achievement."),
    ("user", "Thanks! Before TechCorp, I was at StartupXYZ for about a year. Left because they ran out of funding."),
    ("user", "My salary at TechCorp is $145,000 now with the promotion. Was $120,000 before."),
]

HOBBIES_CONVERSATION: list[tuple[str, str]] = [
    ("user", "I love cycling. I have three bikes - a road bike, a mountain bike, and a gravel bike."),
    ("assistant", "That's impressive! Do you ride often?"),
    ("user", "I ride about 5 times a week, usually 30-50 miles per ride."),
    ("user", "My road bike is a Specialized Tarmac, cost me $3,200. The mountain bike is a Trek Fuel EX."),
    ("user", "I also do photography as a hobby. I use a Sony A7 IV camera."),
    ("user", "I have three lenses - a 24-70mm, a 70-200mm, and a 50mm prime."),
    ("assistant", "Do you combine cycling and photography?"),
    ("user", "Sometimes! I take my camera on gravel rides. The gravel bike has a handlebar bag for it."),
]

CONTRADICTORY_CONVERSATION: list[tuple[str, str]] = [
    ("user", "I live in San Francisco. Been here for about 3 years."),
    ("assistant", "San Francisco is a great city!"),
    ("user", "Yeah, I work downtown near the Embarcadero."),
    ("user", "Actually, we just moved to Portland last week. My partner got a new job there."),
    ("assistant", "That's a big change! How do you like Portland?"),
    ("user", "It's nice, much more affordable. We bought a house in the Pearl District."),
    ("user", "My favorite restaurant used to be Nopa in SF, but now I'm trying places here."),
    ("user", "I found this great ramen place called Afuri in Portland. It's my new favorite."),
]

COMPLEX_MULTI_TOPIC_CONVERSATION: list[tuple[str, str]] = [
    ("user", "Chat Time: 2024-01-15 10:00:00"),
    ("user", "I just adopted a rescue dog named Buddy from the local shelter."),
    ("assistant", "That's wonderful! What kind of dog is Buddy?"),
    ("user", "He's a Labrador mix, about 2 years old. The shelter said he was found as a stray."),
    ("user", "Chat Time: 2024-02-20 14:30:00"),
    ("user", "Buddy is settling in well. He loves going to Riverside Park for walks."),
    ("user", "I've been taking him to Dr. Martinez at Happy Paws Vet Clinic for checkups."),
    ("assistant", "How are his checkups going?"),
    ("user", "Great! Dr. Martinez says he's very healthy. He weighs about 65 pounds."),
    ("user", "Chat Time: 2024-03-10 09:15:00"),
    ("user", "I started a new job at DataFlow Inc as a data scientist. Pretty excited about it."),
    ("user", "My commute is 45 minutes by bus. I used to drive but sold my car."),
    ("user", "Chat Time: 2024-04-05 16:00:00"),
    ("user", "Buddy and I went to the beach last weekend. He loves swimming."),
    ("user", "I also signed up for a marathon in October. Training starts next month."),
    ("user", "My running shoes are Nike Vaporfly, I bought them for $250."),
]

PETS_EXPECTED_ENTITIES: dict[str, dict[str, Any]] = {
    "luna": {
        "types": {"cat"},
        "facts": {
            "breed": "tabby",
            "age": "3 years old",
            "adoption_year": "2021",
        },
    },
    "mochi": {
        "types": {"cat"},
        "facts": {
            "color": "black",
            "age": "6 months old",
        },
    },
    "max": {
        "types": {"golden_retriever", "dog"},
        "facts": {
            "age": "5 years old",
            "temperament": "friendly",
        },
    },
}

TRAVEL_EXPECTED_ENTITIES: dict[str, dict[str, Any]] = {
    "rome": {
        "types": {"city", "destination"},
    },
    "paris": {
        "types": {"city", "destination"},
    },
    "colosseum": {
        "types": {"landmark", "attraction"},
    },
    "hotel_marais": {
        "types": {"hotel"},
    },
}


@pytest.fixture
def pets_conversation() -> list[tuple[str, str]]:
    return PETS_CONVERSATION


@pytest.fixture
def travel_conversation() -> list[tuple[str, str]]:
    return TRAVEL_CONVERSATION


@pytest.fixture
def work_conversation() -> list[tuple[str, str]]:
    return WORK_CONVERSATION


@pytest.fixture
def hobbies_conversation() -> list[tuple[str, str]]:
    return HOBBIES_CONVERSATION


@pytest.fixture
def contradictory_conversation() -> list[tuple[str, str]]:
    return CONTRADICTORY_CONVERSATION


@pytest.fixture
def complex_conversation() -> list[tuple[str, str]]:
    return COMPLEX_MULTI_TOPIC_CONVERSATION


@pytest.fixture
def sample_datetime() -> datetime:
    return datetime(2024, 6, 15, 12, 0, 0)


@pytest.fixture
def date_range() -> tuple[datetime, datetime]:
    start = datetime(2024, 6, 1)
    end = datetime(2024, 6, 30)
    return start, end


@pytest.fixture
def empty_store():
    from foundry.storage import FoundryStore

    return FoundryStore()


@pytest.fixture
def empty_foundry():
    from foundry.core import Foundry, FoundryConfig

    return Foundry(FoundryConfig())
