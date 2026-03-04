"""Longitudinal evaluation protocol for dream parameter optimization.

Exercises every dream mechanism:
- Importance differentiation (repeated access vs single mention)
- Contradiction replacement (fact updates)
- Pruning (single-mention decay)
- Merging (related preferences consolidate)
- Cross-domain bridges (association discovery)

Unlike MABench FactConsolidation (which bulk-loads and measures SubEM), this
protocol simulates realistic memory dynamics over time where dreams actually
help. It becomes the objective function for CMA-ES optimization of dream
parameters.

Architecture:
  1. LongitudinalDataset -- 200 sessions + 100 eval questions
  2. LongitudinalEvaluator -- runs a HermesMemoryAgent through the protocol
  3. generate_dataset() -- generates the dataset programmatically (deterministic)

The dataset is fully self-contained: no LLM calls during generation. All facts
and questions are deterministically generated from the seed. The evaluator DOES
need an LLM for answer generation (via HermesMemoryAgent).
"""

from __future__ import annotations

import logging
import os
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LLM-as-judge scorer (OpenRouter / GPT-4o)
# ---------------------------------------------------------------------------

_LLM_JUDGE_CATEGORIES = frozenset({
    "cross_domain", "graceful_forgetting", "reinforced_recall",
})

_JUDGE_SYSTEM_PROMPT = """\
You are an expert judge evaluating a personal memory assistant's answers.
Score how well the answer demonstrates the knowledge described by the expected keywords.

Rules:
- Score 1.0 if the answer clearly demonstrates knowledge of ALL expected concepts, \
even if it uses different words or paraphrases.
- Score 0.5 if the answer demonstrates knowledge of SOME but not all expected concepts.
- Score 0.0 if the answer fails to demonstrate knowledge of the expected concepts, \
gives a generic response, or says it doesn't know.
- If rejected keywords are provided and the answer contains those outdated/wrong concepts, \
reduce the score by 0.3 per rejected concept present.
- Clamp the final score to [0.0, 1.0].

Respond with ONLY a JSON object: {"score": <float>, "reason": "<one sentence>"}
"""


def _get_llm_client():
    """Return an OpenAI client pointed at OpenRouter, or None if unavailable."""
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        # Try loading from .env in the same directory
        try:
            from pathlib import Path
            import dotenv
            env_path = Path(__file__).parent / ".env"
            if env_path.exists():
                dotenv.load_dotenv(env_path)
                api_key = os.environ.get("OPENROUTER_API_KEY", "")
        except Exception:
            pass
    if not api_key:
        return None
    try:
        from openai import OpenAI
        return OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
    except Exception:
        return None


def llm_judge_score(
    question: str,
    answer: str,
    expected: list[str],
    rejected: list[str],
    category: str,
    client=None,
) -> float | None:
    """Score an answer using GPT-4o as judge via OpenRouter.

    Returns a float in [0, 1] on success, or None if the LLM call fails
    (caller should fall back to SubEM).
    """
    if client is None:
        client = _get_llm_client()
    if client is None:
        return None

    user_msg = (
        f"Category: {category}\n"
        f"Question: {question}\n"
        f"Answer: {answer}\n"
        f"Expected concepts: {expected}\n"
    )
    if rejected:
        user_msg += f"Rejected (outdated) concepts: {rejected}\n"

    try:
        resp = client.chat.completions.create(
            model="openai/gpt-4o",
            messages=[
                {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.0,
            max_tokens=100,
        )
        import json
        text = resp.choices[0].message.content.strip()
        # Handle markdown-wrapped JSON
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        parsed = json.loads(text)
        score = float(parsed["score"])
        return max(0.0, min(1.0, score))
    except Exception as e:
        logger.warning("LLM judge failed: %s", e)
        return None


def score_question(
    question: str,
    answer: str,
    expected: list[str],
    rejected: list[str],
    category: str,
    llm_client=None,
) -> float:
    """Score a question using the appropriate method for its category.

    - current_fact: always SubEM (keyword matching) — binary correct/wrong.
    - cross_domain, graceful_forgetting, reinforced_recall: LLM judge when
      available, SubEM fallback.
    """
    if category not in _LLM_JUDGE_CATEGORIES:
        return score_answer(answer, expected, rejected)

    llm_score = llm_judge_score(
        question, answer, expected, rejected, category, client=llm_client,
    )
    if llm_score is not None:
        return llm_score

    # Fallback to SubEM
    return score_answer(answer, expected, rejected)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class Session:
    """A single interaction session in the longitudinal protocol.

    Attributes:
        day: Simulated day (0-180) when this session occurs.
        session_type: One of 'preference', 'update', 'repeated',
                      'single', 'cross_domain'.
        facts: List of fact strings the user mentions in this session.
    """

    day: int
    session_type: str
    facts: list[str]


@dataclass
class EvalQuestion:
    """An evaluation question with scoring criteria.

    Attributes:
        question: The question text to ask the agent.
        expected_keywords: Keywords that MUST appear (case-insensitive)
                           for full credit.
        rejected_keywords: Keywords that must NOT appear; each incurs a
                           0.5 penalty divided by len(rejected).
        category: One of 'current_fact', 'graceful_forgetting',
                  'reinforced_recall', 'cross_domain'.
    """

    question: str
    expected_keywords: list[str]
    rejected_keywords: list[str]
    category: str


@dataclass
class LongitudinalDataset:
    """Complete dataset for the longitudinal evaluation protocol.

    Attributes:
        sessions: 200 sessions in chronological order.
        questions: 100 evaluation questions across 4 categories.
        seed: The RNG seed used to generate this dataset.
    """

    sessions: list[Session]
    questions: list[EvalQuestion]
    seed: int


# ---------------------------------------------------------------------------
# Scoring functions
# ---------------------------------------------------------------------------


def score_answer(answer: str, expected: list[str], rejected: list[str]) -> float:
    """Score a single answer against expected and rejected keywords.

    Args:
        answer: The agent's answer text.
        expected: Keywords that should appear (case-insensitive).
        rejected: Keywords that should NOT appear.

    Returns:
        Score in [0.0, 1.0]. Partial credit for subset of expected keywords.
        Penalty of 0.5 per rejected keyword found, divided by max(len(rejected), 1).
    """
    answer_lower = answer.lower()
    # All expected keywords present -> 1.0, partial credit for subset
    expected_score = sum(
        1 for kw in expected if kw.lower() in answer_lower
    ) / len(expected)
    # Any rejected keyword present -> penalty
    rejection_penalty = sum(
        0.5 for kw in rejected if kw.lower() in answer_lower
    ) / max(len(rejected), 1)
    return max(0.0, expected_score - rejection_penalty)


def composite_score(category_scores: dict[str, float]) -> float:
    """Compute weighted composite score across categories.

    Weights:
        current_fact:       0.30
        graceful_forgetting: 0.20
        reinforced_recall:  0.25
        cross_domain:       0.25

    Args:
        category_scores: Mapping from category name to mean score [0, 1].

    Returns:
        Weighted composite score in [0, 1].
    """
    weights = {
        "current_fact": 0.30,
        "graceful_forgetting": 0.20,
        "reinforced_recall": 0.25,
        "cross_domain": 0.25,
    }
    return sum(weights[cat] * score for cat, score in category_scores.items())


# ---------------------------------------------------------------------------
# Dataset generation -- world-model architecture
# ---------------------------------------------------------------------------


@dataclass
class _FactChain:
    """A chain of evolving values for a single domain (e.g. food preferences).

    values is a list of (day, display_text, keyword) tuples in chronological
    order. The terminal state is values[-1].
    """

    domain: str
    values: list[tuple[int, str, str]]  # [(day, display_text, keyword), ...]

    @property
    def terminal_text(self) -> str:
        return self.values[-1][1]

    @property
    def terminal_keyword(self) -> str:
        return self.values[-1][2]

    @property
    def original_keyword(self) -> str:
        return self.values[0][2]

    @property
    def all_previous_keywords(self) -> list[str]:
        """All keywords EXCEPT the terminal one."""
        return [kw for _, _, kw in self.values[:-1]]

    @property
    def chain_length(self) -> int:
        return len(self.values)


@dataclass
class _CrossDomainBridge:
    """A semantic bridge connecting facts from two separate domains."""

    domain_a: str
    fact_a: str
    keyword_a: str
    domain_b: str
    fact_b: str
    keyword_b: str
    seed_sessions: list[tuple[int, str]]  # [(day, fact_text), ...]
    bridge_questions: list[tuple[str, list[str]]]  # [(question, expected_kws)]


@dataclass
class _ReinforcedTopic:
    """A topic revisited multiple times with concrete detail additions."""

    domain: str
    base_keyword: str
    detail_sentences: list[str]  # concrete details for each revisit


# --- Fact chain definitions (15 chains, 2-4 values each) ---

_FACT_CHAIN_DEFS: list[tuple[str, list[tuple[str, str]]]] = [
    ("food", [
        ("Japanese food", "japanese"),
        ("Mexican food", "mexican"),
        ("Thai food", "thai"),
    ]),
    ("city", [
        ("San Francisco", "san francisco"),
        ("Berlin", "berlin"),
        ("London", "london"),
    ]),
    ("job_company", [
        ("Google", "google"),
        ("Anthropic", "anthropic"),
    ]),
    ("job_role", [
        ("software engineer", "software engineer"),
        ("data scientist", "data scientist"),
        ("research lead", "research lead"),
    ]),
    ("clothing", [
        ("formal attire", "formal"),
        ("casual wear", "casual"),
        ("athleisure", "athleisure"),
    ]),
    ("car", [
        ("a sedan", "sedan"),
        ("a bicycle", "bicycle"),
        ("an electric car", "electric car"),
    ]),
    ("pet_name", [
        ("Bella", "bella"),
        ("Max", "max"),
    ]),
    ("color", [
        ("blue", "blue"),
        ("green", "green"),
        ("red", "red"),
    ]),
    ("season", [
        ("spring", "spring"),
        ("autumn", "autumn"),
        ("winter", "winter"),
    ]),
    ("sport", [
        ("running", "running"),
        ("swimming", "swimming"),
        ("climbing", "climbing"),
    ]),
    ("relationship", [
        ("single", "single"),
        ("dating someone", "dating"),
        ("married", "married"),
    ]),
    ("language", [
        ("Python", "python"),
        ("Go", "go"),
        ("Rust", "rust"),
    ]),
    ("editor", [
        ("VS Code", "vs code"),
        ("Neovim", "neovim"),
        ("Emacs", "emacs"),
    ]),
    ("music", [
        ("pop music", "pop"),
        ("jazz", "jazz"),
        ("electronic music", "electronic"),
    ]),
    ("hobby_secondary", [
        ("painting", "painting"),
        ("photography", "photography"),
        ("gardening", "gardening"),
    ]),
]

# --- Cross-domain bridge definitions (8 bridges) ---

_BRIDGE_DEFS: list[dict] = [
    {
        "domain_a": "hobby",
        "fact_a": "hiking",
        "keyword_a": "hiking",
        "domain_b": "city",
        "fact_b": "Switzerland",
        "keyword_b": "switzerland",
        "seed_facts": [
            (15, "I love hiking and being outdoors in the mountains"),
            (92, "I moved to Switzerland last month"),
        ],
        "questions": [
            ("What outdoor activities might be good near where I live?",
             ["hiking", "switzerland"]),
            ("Given my hobbies, how well does my location suit me?",
             ["hiking", "switzerland"]),
            ("What kind of weekend activities could I enjoy in my area?",
             ["hiking", "switzerland"]),
        ],
    },
    {
        "domain_a": "food",
        "fact_a": "Thai food",
        "keyword_a": "thai",
        "domain_b": "travel",
        "fact_b": "Bangkok",
        "keyword_b": "bangkok",
        "seed_facts": [
            (22, "I have been really into Thai food lately, especially pad thai and green curry"),
            (88, "I booked a trip to Bangkok for next month"),
        ],
        "questions": [
            ("What food should I try on my upcoming trip?",
             ["thai", "bangkok"]),
            ("How does my travel destination relate to my food preferences?",
             ["thai", "bangkok"]),
        ],
    },
    {
        "domain_a": "job_role",
        "fact_a": "research lead",
        "keyword_a": "research",
        "domain_b": "hobby_secondary",
        "fact_b": "gardening",
        "keyword_b": "gardening",
        "seed_facts": [
            (30, "I lead a research team and spend a lot of time analyzing data and running experiments"),
            (75, "I started gardening this spring, it is so satisfying to watch things grow"),
        ],
        "questions": [
            ("How might my analytical work skills apply to my hobby?",
             ["research", "gardening"]),
            ("What parallels exist between my job and my personal interests?",
             ["research", "gardening"]),
        ],
    },
    {
        "domain_a": "music",
        "fact_a": "jazz",
        "keyword_a": "jazz",
        "domain_b": "relationship",
        "fact_b": "married",
        "keyword_b": "married",
        "seed_facts": [
            (18, "I have been listening to a lot of jazz lately, especially Bill Evans and Miles Davis"),
            (55, "My partner and I got married last spring"),
        ],
        "questions": [
            ("What date night activities might my partner and I enjoy?",
             ["jazz", "married"]),
            ("What kind of events would suit my music taste and social life?",
             ["jazz", "married"]),
            ("Where might I take my spouse for a special evening out?",
             ["jazz", "married"]),
        ],
    },
    {
        "domain_a": "pet_name",
        "fact_a": "Max the dog",
        "keyword_a": "max",
        "domain_b": "city",
        "fact_b": "London",
        "keyword_b": "london",
        "seed_facts": [
            (12, "My dog Max loves running around outside, he needs lots of space"),
            (95, "I just moved to London for work"),
        ],
        "questions": [
            ("Are there good parks near where I live for my dog?",
             ["max", "london"]),
            ("How suitable is my living situation for my pet?",
             ["max", "london"]),
        ],
    },
    {
        "domain_a": "language",
        "fact_a": "Rust",
        "keyword_a": "rust",
        "domain_b": "job_company",
        "fact_b": "Anthropic",
        "keyword_b": "anthropic",
        "seed_facts": [
            (25, "I have been learning Rust in my spare time, really enjoying the ownership model"),
            (60, "I started working at Anthropic on AI safety infrastructure"),
        ],
        "questions": [
            ("How does what I am learning relate to my work?",
             ["rust", "anthropic"]),
            ("Could my personal programming interests be useful at my job?",
             ["rust", "anthropic"]),
            ("What technical skills am I building outside of work?",
             ["rust", "anthropic"]),
        ],
    },
    {
        "domain_a": "hobby_secondary",
        "fact_a": "photography",
        "keyword_a": "photography",
        "domain_b": "travel",
        "fact_b": "Japan",
        "keyword_b": "japan",
        "seed_facts": [
            (35, "I got a new mirrorless camera and have been doing street photography on weekends"),
            (80, "Planning a two-week trip to Japan, visiting Tokyo and Kyoto"),
        ],
        "questions": [
            ("What should I photograph on my upcoming trip?",
             ["photography", "japan"]),
            ("How can I combine my creative hobby with my travel plans?",
             ["photography", "japan"]),
        ],
    },
    {
        "domain_a": "sport",
        "fact_a": "climbing",
        "keyword_a": "climbing",
        "domain_b": "season",
        "fact_b": "winter",
        "keyword_b": "winter",
        "seed_facts": [
            (40, "I have been climbing three times a week at the gym and started leading outdoor routes"),
            (70, "Winter is my favorite season, I love the cold crisp air"),
        ],
        "questions": [
            ("When is the best time for my favorite sport given my season preference?",
             ["climbing", "winter"]),
            ("How does the weather affect my ability to do my sport?",
             ["climbing", "winter"]),
            ("What should I plan around my sport and the seasons?",
             ["climbing", "winter"]),
        ],
    },
]

# --- Reinforcement topic definitions (10 topics, 4-8 details each) ---

_REINFORCEMENT_DEFS: list[dict] = [
    {
        "domain": "hiking",
        "base_keyword": "hiking",
        "details": [
            "Went hiking at Uetliberg this weekend, the views of Lake Zurich were incredible",
            "Finally bought proper hiking boots, Salomon X Ultra with much better grip on wet rocks",
            "Did the Eiger Trail yesterday, 12km with 800m elevation gain",
            "Joined a local hiking group that goes out every Saturday morning",
            "Tried night hiking for the first time with a headlamp, surprisingly peaceful",
            "Mapped out a new route through the Jura mountains for next weekend",
        ],
    },
    {
        "domain": "cooking",
        "base_keyword": "cooking",
        "details": [
            "Tried making pad thai from scratch last night, turned out better than expected",
            "Bought a carbon steel wok, it makes such a difference for stir fry",
            "Had friends over for a dinner party, made a three-course Thai meal",
            "Found an amazing spice shop downtown, stocked up on lemongrass and galangal",
            "Took an online cooking class on knife skills, really improved my speed",
            "Started meal prepping on Sundays, saves so much time during the week",
            "Attempted homemade ramen with 12-hour bone broth, worth every minute",
        ],
    },
    {
        "domain": "rust_programming",
        "base_keyword": "rust",
        "details": [
            "Used Rust at work today to build a new parser, the type system caught three bugs at compile time",
            "Discovered the tokio crate for async runtime, rewrote my server code to use it",
            "Watched a conference talk by Jon Gjengset on lifetime annotations, really clarified things",
            "My Rust side project hit 1000 stars on GitHub, feels great",
            "Finished reading the Rustonomicon, now I understand unsafe blocks much better",
            "Started contributing to a Rust open source project, submitted my first PR",
            "Built a CLI tool in Rust that parses log files, runs 50x faster than my Python version",
            "Paired with a colleague on Rust ownership patterns, we both learned a lot",
        ],
    },
    {
        "domain": "dog_max",
        "base_keyword": "max",
        "details": [
            "My dog Max learned a new trick yesterday, he can roll over on command now",
            "Took Max to the vet for his annual checkup, he is healthy and gained a kilo",
            "Max and I discovered a great off-leash park near the river",
            "Started training Max with a clicker, he responds really well to it",
            "Max met a golden retriever at the park and they played for an hour",
            "Bought Max a new harness for our hikes, much more comfortable for him",
        ],
    },
    {
        "domain": "anthropic_work",
        "base_keyword": "anthropic",
        "details": [
            "Working on a new safety evaluation framework at Anthropic, really interesting challenges",
            "Had a great team offsite discussing our research roadmap for the next quarter",
            "Published an internal paper on constitutional AI improvements, got good feedback",
            "My team at Anthropic shipped a new feature to production today",
            "Attended an AI safety reading group at work, debated alignment approaches",
            "Mentoring a new hire at Anthropic, teaching them about our infrastructure",
        ],
    },
    {
        "domain": "reading",
        "base_keyword": "reading",
        "details": [
            "Just finished Project Hail Mary by Andy Weir, could not put it down",
            "Started a book club with coworkers, we are reading Thinking Fast and Slow",
            "Added ten books to my reading list after browsing the bookstore this weekend",
            "Read a fascinating paper on transformer architecture improvements",
            "Re-read Godel Escher Bach, understanding so much more the second time through",
            "Listened to the audiobook of Sapiens during my commute, really thought-provoking",
        ],
    },
    {
        "domain": "switzerland_living",
        "base_keyword": "switzerland",
        "details": [
            "Getting used to my new neighborhood in Zurich, found a great bakery nearby",
            "The train commute here is incredibly punctual, never more than two minutes late",
            "Learned some Swiss German phrases, my neighbors appreciate the effort",
            "Went to a local cheese fondue night, such a cozy tradition",
            "Exploring the Swiss recycling system, it is impressively thorough",
        ],
    },
    {
        "domain": "climbing",
        "base_keyword": "climbing",
        "details": [
            "Sent my first V6 boulder problem at the gym today after weeks of working on it",
            "Went outdoor climbing at Magic Wood, the granite there is incredible",
            "Bought new climbing shoes, La Sportiva Solutions, much better for overhang",
            "Started hangboard training to improve my finger strength",
            "Climbed with a friend who showed me better footwork technique",
            "Joined a climbing competition at the local gym, placed third in my category",
            "Tried lead climbing for the first time, the exposure was exhilarating",
        ],
    },
    {
        "domain": "gardening",
        "base_keyword": "gardening",
        "details": [
            "My tomato plants are finally bearing fruit, picked the first ripe ones today",
            "Built a raised bed from reclaimed wood, planted herbs and lettuces",
            "Learned about companion planting, put basil next to the tomatoes for pest control",
            "The sunflowers I planted in spring are now taller than me",
            "Started composting kitchen scraps, should have great soil by next season",
            "Harvested my first batch of homegrown peppers, made hot sauce with them",
        ],
    },
    {
        "domain": "emacs",
        "base_keyword": "emacs",
        "details": [
            "Customized my Emacs config with org-mode for all my note-taking",
            "Installed magit and it completely changed how I interact with git",
            "Set up LSP mode in Emacs for Rust development, autocomplete works great now",
            "Wrote my first Elisp function to automate my daily standup notes",
            "Switched to evil-mode for vim keybindings inside Emacs, best of both worlds",
            "Discovered org-roam for building a personal knowledge graph",
        ],
    },
]

# --- Single-mention templates and fillers ---

_SINGLE_MENTION_TEMPLATES = [
    "I saw a good movie yesterday called {title}",
    "I tried a new restaurant called {name} last week",
    "I read an interesting article about {topic}",
    "I heard a song called {title} on the radio",
    "I visited {place} over the weekend",
    "I bought a new {item} recently",
    "I attended a talk about {topic} yesterday",
    "I met someone who works in {field}",
    "I discovered a new app called {name}",
    "I tried {activity} for the first time",
]

_SINGLE_FILLERS = {
    "title": [
        "Interstellar", "Inception", "Arrival", "Dune", "Oppenheimer",
        "Everything Everywhere", "The Matrix", "Blade Runner",
        "Parasite", "Moonlight", "La La Land", "Gravity",
        "Her", "Ex Machina", "Annihilation", "Tenet",
        "Nomadland", "Minari", "Soul", "Coco",
        "The Martian", "Contact", "Gattaca", "Solaris",
        "Stalker", "2001 Space Odyssey", "Alien", "Predator",
        "Terminator", "RoboCop",
    ],
    "name": [
        "Noma", "Osteria", "Sukiyabashi", "El Celler", "Alinea",
        "Eleven Madison", "Masa", "Blue Hill", "Chez Panisse", "Nobu",
        "Per Se", "Le Bernardin", "Atelier Crenn", "Gaggan", "Mirazur",
        "Central", "Geranium", "Steirereck", "Den", "Narisawa",
        "Asador Etxebarri", "Ticket", "DiverXO", "Maido", "Florilege",
        "Odette", "Burnt Ends", "Ultraviolet", "Quintonil", "Pujol",
    ],
    "topic": [
        "quantum computing", "deep learning", "climate change",
        "space exploration", "gene editing", "renewable energy",
        "blockchain", "neuroscience", "robotics", "nanotechnology",
        "fusion energy", "synthetic biology", "asteroid mining",
        "ocean currents", "dark matter", "gravitational waves",
        "CRISPR", "photonics", "superconductors", "metamaterials",
        "topological insulators", "quantum entanglement",
        "protein folding", "microbiome", "epigenetics",
        "neuroplasticity", "consciousness", "panpsychism",
        "multiverse theory", "string theory",
    ],
    "place": [
        "the botanical garden", "a local art gallery",
        "the science museum", "a jazz club", "the farmers market",
        "a ceramics workshop", "the observatory",
        "a vintage bookstore", "the aquarium", "a rooftop bar",
        "the planetarium", "a craft brewery", "the national park",
        "a hot spring", "the lighthouse", "a sculpture garden",
        "the night market", "a vinyl record shop",
        "the historical district", "a floating restaurant",
        "an underground cave", "a butterfly sanctuary",
        "the old library", "a windmill", "the watchtower",
        "a treehouse cafe", "the harbor", "a lavender field",
        "the clock tower", "a bamboo forest",
    ],
    "item": [
        "mechanical keyboard", "espresso machine", "telescope",
        "vinyl record player", "smart watch", "e-reader",
        "standing desk", "noise-canceling headphones",
        "ergonomic chair", "portable projector",
        "drone", "3D printer", "sous vide cooker",
        "air purifier", "smart thermostat", "robot vacuum",
        "portable monitor", "drawing tablet", "microphone",
        "camera lens", "backpack", "bike light", "camp stove",
        "water filter", "solar charger", "hammock",
        "binoculars", "pocket knife", "compass", "lantern",
    ],
    "field": [
        "marine biology", "astrophysics", "archaeology",
        "ethnomusicology", "computational linguistics",
        "behavioral economics", "urban planning",
        "conservation biology", "forensic science", "glaciology",
        "volcanology", "paleontology", "cryptography",
        "game theory", "topology", "set theory",
        "number theory", "combinatorics", "algebraic geometry",
        "category theory", "functional analysis",
        "measure theory", "ergodic theory", "dynamical systems",
        "fluid dynamics", "plasma physics", "optics",
        "acoustics", "thermodynamics", "electrodynamics",
    ],
    "activity": [
        "pottery", "archery", "scuba diving",
        "rock climbing indoors", "salsa dancing",
        "woodworking", "calligraphy", "beekeeping",
        "bread baking", "stargazing",
        "bird watching", "fencing", "kite surfing",
        "skateboarding", "tai chi", "parkour",
        "ice skating", "horseback riding",
        "glass blowing", "blacksmithing",
        "origami", "juggling", "unicycling",
        "tightrope walking", "trapeze",
        "aerial yoga", "pole dancing",
        "bouldering", "slacklining", "canyoneering",
    ],
}

# --- Stable (non-evolving) biographical facts ---

_STABLE_BIOGRAPHICAL = [
    ("school", "I studied at ETH Zurich", "eth"),
    ("degree", "I have a degree in Computer Science", "computer science"),
    ("birthday_month", "My birthday is in September", "september"),
    ("name", "My name is Alex", "alex"),
    ("hometown", "I grew up in Portland, Oregon", "portland"),
    ("sibling", "I have a younger sister named Emma", "emma"),
    ("alma_mater_2", "I did my masters at Stanford", "stanford"),
    ("childhood_pet", "I had a cat named Whiskers growing up", "whiskers"),
    ("first_language", "My first language is English", "english"),
    ("height", "I am six feet tall", "six feet"),
]

# --- Natural update sentence templates per domain ---

_UPDATE_TEMPLATES: dict[str, list[str]] = {
    "food": [
        "I have been getting into {new} lately, especially {detail}",
        "Had amazing {new} at a new place downtown, think it is my new favorite",
        "My tastes have changed, I am really into {new} these days",
    ],
    "city": [
        "I just moved to {new} for a new opportunity",
        "Relocated to {new} last month, still getting settled",
        "Made the move to {new}, loving the city so far",
    ],
    "job_company": [
        "Started a new job at {new}, really excited about the work",
        "I switched to {new}, the team there is fantastic",
        "Just joined {new}, working on some interesting problems",
    ],
    "job_role": [
        "Got promoted to {new}, more responsibility but I enjoy it",
        "Transitioned into a {new} role, it suits me better",
        "My title changed to {new}, reflecting the work I have been doing",
    ],
    "clothing": [
        "Overhauled my wardrobe, going all in on {new} these days",
        "Switched to {new}, much more comfortable for my lifestyle",
        "Been wearing {new} almost exclusively now",
    ],
    "car": [
        "Bought {new} last month, much better for my commute",
        "Switched to {new}, really happy with the change",
        "Got rid of my old ride and got {new}",
    ],
    "pet_name": [
        "Got a new dog named {new}, he is a golden retriever",
        "We adopted {new} from the shelter last week",
    ],
    "color": [
        "Been gravitating toward {new} lately, redecorated my room in that palette",
        "My favorite color shifted to {new}, even bought a {new} jacket",
    ],
    "season": [
        "I have come to really love {new}, there is something special about it",
        "Changed my mind about seasons, {new} is definitely my favorite now",
    ],
    "sport": [
        "Picked up {new} recently and I am hooked",
        "Switched from my old sport to {new}, much more fun",
        "Been doing {new} three times a week now",
    ],
    "relationship": [
        "Life update: I am {new} now",
        "Things changed recently, I am {new}",
    ],
    "language": [
        "Made the switch to {new} for most of my projects",
        "Been writing everything in {new} lately, really productive",
        "Converted my main codebase to {new}",
    ],
    "editor": [
        "Switched to {new} and never looking back",
        "Finally made the jump to {new}, so much more efficient",
        "Using {new} full time now for all my coding",
    ],
    "music": [
        "Been listening to a lot of {new} recently, great for focus",
        "My music taste shifted to {new}, discovering amazing artists",
        "Cannot stop listening to {new}, it is all I play these days",
    ],
    "hobby_secondary": [
        "Picked up {new} as a hobby, finding it really relaxing",
        "Gotten into {new} lately, spending most weekends on it",
        "Started {new} and it has become my main creative outlet",
    ],
}

# --- Natural preference sentence templates per domain ---

_INITIAL_TEMPLATES: dict[str, str] = {
    "food": "I really love {val}, it is my favorite cuisine",
    "city": "I live in {val}",
    "job_company": "I work at {val}",
    "job_role": "I work as a {val}",
    "clothing": "I tend to wear {val} most of the time",
    "car": "I drive {val}",
    "pet_name": "I have a dog named {val}",
    "color": "My favorite color is {val}",
    "season": "My favorite season is {val}",
    "sport": "I really enjoy {val}",
    "relationship": "I am {val}",
    "language": "I mostly code in {val}",
    "editor": "I use {val} as my main editor",
    "music": "I listen to a lot of {val}",
    "hobby_secondary": "I spend my free time doing {val}",
}

# --- Natural detail fragments for update templates ---

_UPDATE_DETAILS: dict[str, list[str]] = {
    "food": [
        "the curries and stir fries", "the street food style dishes",
        "the spicy noodle soups", "the fresh ingredients and bold flavors",
    ],
    "city": [], "job_company": [], "job_role": [], "clothing": [],
    "car": [], "pet_name": [], "color": [], "season": [],
    "sport": [], "relationship": [], "language": [], "editor": [],
    "music": [], "hobby_secondary": [],
}

# --- Stable biographical question templates ---

_STABLE_BIO_QUESTIONS: dict[str, str] = {
    "school": "Where did I study?",
    "degree": "What did I study in university?",
    "birthday_month": "When is my birthday?",
    "name": "What is my name?",
    "hometown": "Where did I grow up?",
    "sibling": "Do I have any siblings?",
    "alma_mater_2": "Where did I do my masters?",
    "childhood_pet": "Did I have a pet growing up?",
    "first_language": "What is my first language?",
    "height": "How tall am I?",
}

# --- Chain domain question templates ---

_CHAIN_DOMAIN_QUESTIONS: dict[str, str] = {
    "food": "What is my favorite cuisine?",
    "city": "Where do I currently live?",
    "job_company": "Where do I work?",
    "job_role": "What is my current job title?",
    "clothing": "How would you describe my clothing style?",
    "car": "What do I drive?",
    "pet_name": "What is my pet's name?",
    "color": "What is my favorite color?",
    "season": "What is my favorite season?",
    "sport": "What sport do I do?",
    "relationship": "What is my relationship status?",
    "language": "What programming language do I use most?",
    "editor": "What text editor do I use?",
    "music": "What kind of music do I listen to?",
    "hobby_secondary": "What is my main creative hobby?",
}

_GRACEFUL_QUESTIONS: dict[str, str] = {
    "food": "Have I always liked {terminal}?",
    "city": "Have I always lived in {terminal}?",
    "job_company": "Have I always worked at {terminal}?",
    "job_role": "Was I always a {terminal}?",
    "clothing": "Have I always dressed in {terminal}?",
    "car": "Have I always driven {terminal}?",
    "pet_name": "Was {terminal} my first pet?",
    "color": "Has {terminal} always been my favorite color?",
    "season": "Was {terminal} always my favorite season?",
    "sport": "Have I always done {terminal}?",
    "relationship": "Have I always been {terminal}?",
    "language": "Did I always code in {terminal}?",
    "editor": "Did I always use {terminal}?",
    "music": "Did I always listen to {terminal}?",
    "hobby_secondary": "Was {terminal} always my hobby?",
}


def _generate_active_days(rng, n_days: int = 181) -> list[int]:
    """Generate active days with clustering pattern.

    Creates clusters where the user is active for 1-3 days, with gaps of
    2-8 days between clusters. Multiple sessions can land on the same day.
    Returns sorted list of unique day indices in range [0, n_days).
    """
    active_days: list[int] = []
    day = 0
    while day < n_days:
        cluster_size = rng.integers(1, 4)
        for _ in range(cluster_size):
            if day < n_days:
                active_days.append(day)
                day += 1
        gap = rng.integers(2, 9)
        day += gap
    return sorted(set(active_days))


def _assign_sessions_to_days(
    rng,
    active_days: list[int],
    n_sessions: int,
) -> list[int]:
    """Assign n_sessions to active_days, spreading across the full range.

    Returns a list of day values (length = n_sessions), sorted.
    """
    if not active_days:
        return [0] * n_sessions
    day_assignments = [
        active_days[rng.integers(0, len(active_days))]
        for _ in range(n_sessions)
    ]
    return sorted(day_assignments)


def generate_dataset(seed: int = 42) -> LongitudinalDataset:
    """Generate a complete longitudinal evaluation dataset.

    Uses a two-phase world-model architecture:
      Phase 1: Build fact chains, cross-domain bridges, reinforcement topics,
               and generate all sessions from the world model.
      Phase 2: Generate questions from TERMINAL states only.

    Deterministic: the same seed always produces the same dataset.
    No LLM calls -- all facts and questions are programmatically generated.

    Args:
        seed: RNG seed for reproducibility.

    Returns:
        LongitudinalDataset with 200 sessions and 100 eval questions.
    """
    import numpy as np

    rng = np.random.default_rng(seed)

    # --- Generate active day schedule ---
    day_rng = np.random.default_rng(seed + 1000)
    active_days = _generate_active_days(day_rng)

    # ===================================================================
    # PHASE 1: Build the world model
    # ===================================================================

    # --- 1a. Build fact chains from definitions ---
    fact_chains: list[_FactChain] = []
    for domain, value_list in _FACT_CHAIN_DEFS:
        # Assign days: first value early (day 0-30), subsequent values
        # spread across the timeline with increasing gaps.
        chain_values: list[tuple[int, str, str]] = []
        base_day = int(rng.integers(0, 20))
        for step_idx, (display, keyword) in enumerate(value_list):
            if step_idx == 0:
                day = base_day
            else:
                # Each update happens 15-40 days after the previous
                gap = int(rng.integers(15, 41))
                day = chain_values[-1][0] + gap
                day = min(day, 170)  # keep within range
            chain_values.append((day, display, keyword))
        fact_chains.append(_FactChain(domain=domain, values=chain_values))

    # --- 1b. Build cross-domain bridges from definitions ---
    bridges: list[_CrossDomainBridge] = []
    for bdef in _BRIDGE_DEFS:
        seed_sessions = [
            (day, text) for day, text in bdef["seed_facts"]
        ]
        bridge_questions = [
            (q, kws) for q, kws in bdef["questions"]
        ]
        bridges.append(_CrossDomainBridge(
            domain_a=bdef["domain_a"],
            fact_a=bdef["fact_a"],
            keyword_a=bdef["keyword_a"],
            domain_b=bdef["domain_b"],
            fact_b=bdef["fact_b"],
            keyword_b=bdef["keyword_b"],
            seed_sessions=seed_sessions,
            bridge_questions=bridge_questions,
        ))

    # --- 1c. Build reinforced topics from definitions ---
    reinforced_topics: list[_ReinforcedTopic] = []
    for rdef in _REINFORCEMENT_DEFS:
        reinforced_topics.append(_ReinforcedTopic(
            domain=rdef["domain"],
            base_keyword=rdef["base_keyword"],
            detail_sentences=list(rdef["details"]),
        ))

    # ===================================================================
    # PHASE 1 continued: Generate sessions from the world model
    # ===================================================================

    # --- Preference sessions (50 sessions): initial values of chains + stable bio ---
    preference_sessions: list[Session] = []

    # Each chain's initial value produces 1 preference session
    for chain in fact_chains:
        initial_day, initial_text, _ = chain.values[0]
        template = _INITIAL_TEMPLATES.get(chain.domain, "I like {val}")
        fact = template.replace("{val}", initial_text)
        preference_sessions.append(Session(
            day=initial_day,
            session_type="preference",
            facts=[fact],
        ))

    # Stable biographical facts (one session per fact)
    for bio_key, bio_fact, bio_kw in _STABLE_BIOGRAPHICAL:
        day = int(rng.integers(0, 25))
        preference_sessions.append(Session(
            day=day,
            session_type="preference",
            facts=[bio_fact],
        ))

    # Fill remaining preference sessions by grouping small facts together
    while len(preference_sessions) < 50:
        # Pick a random chain and restate its current initial value with variation
        chain = fact_chains[int(rng.integers(0, len(fact_chains)))]
        initial_text = chain.values[0][1]
        domain = chain.domain
        variations = [
            f"By the way, I should mention I really like {initial_text}",
            f"One thing about me is that I enjoy {initial_text}",
            f"Just so you know, I am into {initial_text}",
        ]
        fact = variations[int(rng.integers(0, len(variations)))]
        day = int(rng.integers(0, 30))
        preference_sessions.append(Session(
            day=day,
            session_type="preference",
            facts=[fact],
        ))

    preference_sessions = preference_sessions[:50]

    # --- Update sessions (40 sessions): evolve fact chains ---
    update_sessions: list[Session] = []

    for chain in fact_chains:
        for step_idx in range(1, chain.chain_length):
            update_day, new_text, new_kw = chain.values[step_idx]
            domain = chain.domain
            templates = _UPDATE_TEMPLATES.get(domain, [
                "I switched to {new} recently",
                "Changed to {new}, much better",
            ])
            template = templates[int(rng.integers(0, len(templates)))]
            fact = template.replace("{new}", new_text)
            # Add detail if available
            details = _UPDATE_DETAILS.get(domain, [])
            if details and "{detail}" in fact:
                detail = details[int(rng.integers(0, len(details)))]
                fact = fact.replace("{detail}", detail)
            elif "{detail}" in fact:
                fact = fact.replace(", especially {detail}", "")
                fact = fact.replace("{detail}", "")
            update_sessions.append(Session(
                day=update_day,
                session_type="update",
                facts=[fact],
            ))

    # Pad update sessions to reach 40
    while len(update_sessions) < 40:
        # Pick a chain with multiple values and create additional context sessions
        multi_chains = [c for c in fact_chains if c.chain_length >= 2]
        chain = multi_chains[int(rng.integers(0, len(multi_chains)))]
        # Add a session reinforcing the update
        last_text = chain.terminal_text
        domain = chain.domain
        reinforce_updates = [
            f"Still really enjoying {last_text}, glad I made the switch",
            f"Confirming that {last_text} is working out great for me",
            f"More and more into {last_text} as time goes on",
        ]
        fact = reinforce_updates[int(rng.integers(0, len(reinforce_updates)))]
        # Place it after the last update in the chain
        last_day = chain.values[-1][0]
        day = min(last_day + int(rng.integers(5, 20)), 175)
        update_sessions.append(Session(
            day=day,
            session_type="update",
            facts=[fact],
        ))

    update_sessions = update_sessions[:40]

    # --- Repeated/reinforcement sessions (60 sessions) ---
    repeated_sessions: list[Session] = []
    detail_indices: dict[str, int] = {
        rt.domain: 0 for rt in reinforced_topics
    }

    for i in range(60):
        # Pick a reinforced topic (round-robin with randomization)
        topic = reinforced_topics[int(rng.integers(0, len(reinforced_topics)))]
        domain = topic.domain
        idx = detail_indices[domain]
        details = topic.detail_sentences
        # Use the next unused detail, cycling if we run out
        fact = details[idx % len(details)]
        detail_indices[domain] = idx + 1
        repeated_sessions.append(Session(
            day=0,  # assigned later
            session_type="repeated",
            facts=[fact],
        ))

    # --- Single-mention sessions (30 sessions) ---
    single_sessions: list[Session] = []
    used_fillers: set[str] = set()

    for i in range(30):
        template = _SINGLE_MENTION_TEMPLATES[
            int(rng.integers(0, len(_SINGLE_MENTION_TEMPLATES)))
        ]
        for placeholder, fillers in _SINGLE_FILLERS.items():
            tag = "{" + placeholder + "}"
            if tag in template:
                available = [f for f in fillers if f not in used_fillers]
                if not available:
                    available = fillers
                filler_idx = int(rng.integers(0, len(available)))
                filler = available[filler_idx]
                used_fillers.add(filler)
                fact = template.replace(tag, filler)
                break
        else:
            fact = template
        single_sessions.append(Session(
            day=0,
            session_type="single",
            facts=[fact],
        ))

    # --- Cross-domain sessions (20 sessions): plant seeds in separate domains ---
    cross_domain_sessions: list[Session] = []

    # Each bridge produces 2-3 seed sessions (each planting one domain fact)
    for bridge in bridges:
        for seed_day, seed_text in bridge.seed_sessions:
            cross_domain_sessions.append(Session(
                day=seed_day,
                session_type="cross_domain",
                facts=[seed_text],
            ))

    # We have 8 bridges x 2 seed_sessions = 16 sessions. Pad to 20.
    while len(cross_domain_sessions) < 20:
        bridge = bridges[int(rng.integers(0, len(bridges)))]
        # Add an additional seed session for one of the domains
        extra_facts = [
            f"I have been thinking more about {bridge.fact_a} lately",
            f"Still very much into {bridge.fact_b}",
            f"Spent some time on {bridge.fact_a} this weekend",
            f"Learning more about {bridge.fact_b} every day",
        ]
        fact = extra_facts[int(rng.integers(0, len(extra_facts)))]
        day = int(rng.integers(50, 170))
        cross_domain_sessions.append(Session(
            day=day,
            session_type="cross_domain",
            facts=[fact],
        ))

    cross_domain_sessions = cross_domain_sessions[:20]

    # --- Assign days to sessions that need them ---
    # Preferences already have days; ensure they are in early period
    pref_active = [d for d in active_days if d <= 60]
    if len(pref_active) < 10:
        pref_active = active_days[:max(len(active_days) // 3, 10)]
    # Re-assign preference session days to active days for clustering
    pref_days = _assign_sessions_to_days(rng, pref_active, 50)
    for s, d in zip(preference_sessions, pref_days):
        s.day = d

    # Update sessions already have approximate days from chains;
    # snap them to active days
    update_active = [d for d in active_days if 20 <= d <= 170]
    if len(update_active) < 10:
        update_active = [d for d in active_days if d >= 15]
    update_days = _assign_sessions_to_days(rng, update_active, 40)
    for s, d in zip(update_sessions, update_days):
        s.day = d

    # Repeated sessions
    repeated_active = [d for d in active_days if 15 <= d <= 175]
    if len(repeated_active) < 15:
        repeated_active = active_days[3:]
    repeated_days = _assign_sessions_to_days(rng, repeated_active, 60)
    for s, d in zip(repeated_sessions, repeated_days):
        s.day = d

    # Single sessions
    single_active = [d for d in active_days if 10 <= d <= 170]
    if len(single_active) < 10:
        single_active = active_days[3:]
    single_days = _assign_sessions_to_days(rng, single_active, 30)
    for s, d in zip(single_sessions, single_days):
        s.day = d

    # Cross-domain sessions: keep their specific days but snap to active days
    cross_active = [d for d in active_days if 10 <= d <= 175]
    if len(cross_active) < 10:
        cross_active = active_days[len(active_days) // 4:]
    cross_days = _assign_sessions_to_days(rng, cross_active, 20)
    for s, d in zip(cross_domain_sessions, cross_days):
        s.day = d

    # --- Merge and sort all sessions chronologically ---
    all_sessions = (
        preference_sessions
        + update_sessions
        + repeated_sessions
        + single_sessions
        + cross_domain_sessions
    )
    all_sessions.sort(key=lambda s: s.day)

    # ===================================================================
    # PHASE 2: Generate questions from TERMINAL states only
    # ===================================================================

    questions: list[EvalQuestion] = []
    used_question_texts: set[str] = set()

    def _add_question(q: EvalQuestion) -> bool:
        """Add question if its text is unique. Returns True if added."""
        if q.question in used_question_texts:
            return False
        used_question_texts.add(q.question)
        questions.append(q)
        return True

    # --- Category 1: Current fact retrieval (30 questions) ---
    # One per chain (terminal value), plus stable biographical facts

    # Questions from fact chains (15 chains -> 15 questions)
    for chain in fact_chains:
        domain = chain.domain
        question_text = _CHAIN_DOMAIN_QUESTIONS.get(
            domain, f"What is my {domain}?"
        )
        rejected = chain.all_previous_keywords
        _add_question(EvalQuestion(
            question=question_text,
            expected_keywords=[chain.terminal_keyword],
            rejected_keywords=rejected,
            category="current_fact",
        ))

    # Questions from stable biographical facts (10 facts -> 10 questions)
    for bio_key, bio_fact, bio_kw in _STABLE_BIOGRAPHICAL:
        question_text = _STABLE_BIO_QUESTIONS.get(
            bio_key, f"What is my {bio_key}?"
        )
        _add_question(EvalQuestion(
            question=question_text,
            expected_keywords=[bio_kw],
            rejected_keywords=[],
            category="current_fact",
        ))

    # Fill to 30 with additional chain-based questions (varied phrasing)
    cf_count = sum(1 for q in questions if q.category == "current_fact")
    chain_idx = 0
    alt_templates = [
        "Can you remind me about my {domain}?",
        "What did I say about my {domain} preference?",
        "Tell me what you know about my {domain}.",
        "What was the last thing I said about my {domain}?",
        "Do you remember my {domain}?",
    ]
    while cf_count < 30:
        chain = fact_chains[chain_idx % len(fact_chains)]
        template = alt_templates[
            (chain_idx // len(fact_chains)) % len(alt_templates)
        ]
        q_text = template.replace("{domain}", chain.domain)
        added = _add_question(EvalQuestion(
            question=q_text,
            expected_keywords=[chain.terminal_keyword],
            rejected_keywords=chain.all_previous_keywords,
            category="current_fact",
        ))
        if added:
            cf_count += 1
        chain_idx += 1
        if chain_idx > len(fact_chains) * len(alt_templates):
            break  # safety: avoid infinite loop

    # --- Category 2: Graceful forgetting (20 questions) ---
    # One per chain with length >= 2, expecting BOTH original and terminal

    multi_chains = [c for c in fact_chains if c.chain_length >= 2]
    for chain in multi_chains:
        domain = chain.domain
        terminal = chain.terminal_text
        template = _GRACEFUL_QUESTIONS.get(
            domain, "Has my {terminal} always been my preference?"
        )
        q_text = template.replace("{terminal}", terminal)
        _add_question(EvalQuestion(
            question=q_text,
            expected_keywords=[chain.original_keyword, chain.terminal_keyword],
            rejected_keywords=[],
            category="graceful_forgetting",
        ))

    # Fill to 20 with varied phrasing
    gf_count = sum(1 for q in questions if q.category == "graceful_forgetting")
    gf_alt_templates = [
        "Has my {domain} always stayed the same?",
        "Tell me the history of my {domain} preferences.",
        "What changes have happened with my {domain}?",
        "Did my {domain} ever change over time?",
        "Walk me through how my {domain} evolved.",
    ]
    gf_chain_idx = 0
    while gf_count < 20:
        chain = multi_chains[gf_chain_idx % len(multi_chains)]
        template = gf_alt_templates[
            (gf_chain_idx // len(multi_chains)) % len(gf_alt_templates)
        ]
        q_text = template.replace("{domain}", chain.domain)
        added = _add_question(EvalQuestion(
            question=q_text,
            expected_keywords=[chain.original_keyword, chain.terminal_keyword],
            rejected_keywords=[],
            category="graceful_forgetting",
        ))
        if added:
            gf_count += 1
        gf_chain_idx += 1
        if gf_chain_idx > len(multi_chains) * len(gf_alt_templates):
            break

    # --- Category 3: Reinforced recall (25 questions) ---
    # One per reinforced topic, no duplicates

    rr_templates = [
        "What do I frequently talk about regarding {domain}?",
        "What is a topic I keep bringing up about {domain}?",
        "What details have I shared about {domain}?",
    ]
    for t_idx, topic in enumerate(reinforced_topics):
        template = rr_templates[t_idx % len(rr_templates)]
        q_text = template.replace("{domain}", topic.domain)
        _add_question(EvalQuestion(
            question=q_text,
            expected_keywords=[topic.base_keyword],
            rejected_keywords=[],
            category="reinforced_recall",
        ))

    # Fill to 25 with varied phrasing
    rr_count = sum(1 for q in questions if q.category == "reinforced_recall")
    rr_alt_templates = [
        "What have I mentioned multiple times about {domain}?",
        "Tell me about my ongoing interest in {domain}.",
        "What recurring theme have I discussed about {domain}?",
        "Summarize what I have said about {domain} over time.",
        "What is the topic I keep revisiting about {domain}?",
    ]
    rr_topic_idx = 0
    while rr_count < 25:
        topic = reinforced_topics[rr_topic_idx % len(reinforced_topics)]
        template = rr_alt_templates[
            (rr_topic_idx // len(reinforced_topics)) % len(rr_alt_templates)
        ]
        q_text = template.replace("{domain}", topic.domain)
        added = _add_question(EvalQuestion(
            question=q_text,
            expected_keywords=[topic.base_keyword],
            rejected_keywords=[],
            category="reinforced_recall",
        ))
        if added:
            rr_count += 1
        rr_topic_idx += 1
        if rr_topic_idx > len(reinforced_topics) * len(rr_alt_templates):
            break

    # --- Category 4: Cross-domain inference (25 questions) ---
    # 2-3 per bridge, requiring facts from both domains

    for bridge in bridges:
        for q_text, expected_kws in bridge.bridge_questions:
            _add_question(EvalQuestion(
                question=q_text,
                expected_keywords=expected_kws,
                rejected_keywords=[],
                category="cross_domain",
            ))

    # Fill to 25 if needed
    cd_count = sum(1 for q in questions if q.category == "cross_domain")
    cd_idx = 0
    cd_alt_templates = [
        "Based on what you know about me, how do {domain_a} and {domain_b} connect?",
        "Can you make a recommendation combining my {domain_a} and {domain_b}?",
    ]
    while cd_count < 25:
        bridge = bridges[cd_idx % len(bridges)]
        template = cd_alt_templates[
            (cd_idx // len(bridges)) % len(cd_alt_templates)
        ]
        q_text = template.replace(
            "{domain_a}", bridge.domain_a
        ).replace(
            "{domain_b}", bridge.domain_b
        )
        added = _add_question(EvalQuestion(
            question=q_text,
            expected_keywords=[bridge.keyword_a, bridge.keyword_b],
            rejected_keywords=[],
            category="cross_domain",
        ))
        if added:
            cd_count += 1
        cd_idx += 1
        if cd_idx > len(bridges) * len(cd_alt_templates):
            break

    return LongitudinalDataset(
        sessions=all_sessions,
        questions=questions,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


class LongitudinalEvaluator:
    """Runs a HermesMemoryAgent through the longitudinal protocol.

    The evaluator:
    1. Creates an agent via the provided factory
    2. Processes sessions in chronological order
    3. Between session clusters (gaps > 1 day), calls agent.coupled_engine.dream()
    4. After all sessions, runs all 100 eval questions
    5. Scores and returns results

    The key insight: dream runs during IDLE PERIODS between session clusters,
    not every N stores. The evaluator controls when dreams happen based on
    simulated time gaps.
    """

    def __init__(
        self,
        dataset: LongitudinalDataset,
        dream_idle_threshold: float = 1.0,
        use_llm_judge: bool = True,
        retrieval_practice: bool = False,
        cross_domain_probes: bool = False,
        probe_frequency: int = 10,
        probes_per_session: int = 3,
        capacity_gating: bool = False,
    ):
        self.dataset = dataset
        self.dream_idle_threshold = dream_idle_threshold
        self.use_llm_judge = use_llm_judge
        self.retrieval_practice = retrieval_practice
        self.cross_domain_probes = cross_domain_probes
        self.probe_frequency = probe_frequency
        self.probes_per_session = probes_per_session
        self.capacity_gating = capacity_gating
        self._llm_client = None
        if use_llm_judge:
            self._llm_client = _get_llm_client()
            if self._llm_client is None:
                logger.info(
                    "No OPENROUTER_API_KEY found; falling back to SubEM scoring"
                )

    def _generate_cross_domain_probes(
        self,
        dataset: LongitudinalDataset,
        n_probes: int,
        session_index: int,
    ) -> list[str]:
        """Generate n_probes synthetic cross-domain query strings.

        Uses bridge definitions from the dataset generation logic to create
        queries that span domain pairs. Each probe is a question from one of
        the pre-defined cross-domain bridges.

        Deterministic per session_index (uses it as RNG seed), but produces
        different probes for different session indices.

        Args:
            dataset: The longitudinal dataset (used to access bridge structure).
            n_probes: Number of probe strings to generate.
            session_index: Current session index, used as seed for determinism.

        Returns:
            List of probe query strings.
        """
        import random

        rng = random.Random(session_index * 31 + 7)

        # Collect all bridge questions from _BRIDGE_DEFS
        all_bridge_questions: list[str] = []
        for bridge in _BRIDGE_DEFS:
            for question_text, _keywords in bridge["questions"]:
                all_bridge_questions.append(question_text)

        # Also generate template-based probes from bridge domain pairs
        templates = [
            "What should I cook for my {activity}?",
            "How does {domain_a} relate to {domain_b}?",
            "What {domain_a} would suit my {domain_b}?",
            "Does my {domain_a} connect to my {domain_b}?",
            "What activities combine {domain_a} and {domain_b}?",
        ]
        template_probes: list[str] = []
        for bridge in _BRIDGE_DEFS:
            da = bridge["fact_a"]
            db = bridge["fact_b"]
            for tmpl in templates:
                try:
                    probe = tmpl.format(
                        activity=da,
                        domain_a=da,
                        domain_b=db,
                    )
                    template_probes.append(probe)
                except KeyError:
                    pass

        combined = all_bridge_questions + template_probes

        probes = []
        for _ in range(n_probes):
            probes.append(rng.choice(combined))

        return probes

    def evaluate(self, agent_factory: Callable) -> dict:
        """Run full evaluation. Returns detailed scores + composite.

        Args:
            agent_factory: Callable that returns a fresh HermesMemoryAgent
                           (or any object with send_message and
                           coupled_engine.dream interfaces).

        Returns:
            Dictionary with:
                - category_scores: dict mapping category -> mean score
                - per_question: list of per-question score details
                - composite: weighted composite score
                - n_dreams: number of dream cycles triggered
                - sessions_processed: number of sessions processed
                - scoring_method: 'hybrid' if LLM judge active, 'subem' otherwise
        """
        agent = agent_factory()
        n_dreams = 0

        # --- Phase 1: Process sessions in chronological order ---
        sessions = self.dataset.sessions
        prev_day = -1

        for i, session in enumerate(sessions):
            # Check if there's an idle period before this session
            if prev_day >= 0 and session.day - prev_day > self.dream_idle_threshold:
                # Idle period detected -- trigger dream (with optional capacity gating)
                if self.capacity_gating and hasattr(agent, 'coupled_engine'):
                    if agent.coupled_engine.should_dream():
                        agent.coupled_engine.dream()
                        n_dreams += 1
                else:
                    agent.coupled_engine.dream()
                    n_dreams += 1

            # Store each fact in this session
            for fact in session.facts:
                agent.send_message(fact, memorizing=True)

            # Retrieval practice: after storing, probe memory with each
            # session fact to build co-retrieval edges. This simulates
            # encoding-time associative recall — when you learn something
            # new, your brain automatically activates related memories.
            # The probe is silent (no LLM call), just cosine retrieval.
            if self.retrieval_practice and hasattr(agent, 'coupled_engine'):
                scorer = getattr(agent, '_scorer', None)
                if scorer is not None:
                    for fact in session.facts:
                        emb = scorer.embed(fact)
                        agent.coupled_engine.query(emb, top_k=10)

            # Cross-domain probes: periodically fire synthetic cross-domain
            # queries via query_readonly() to build co-retrieval edges
            # without corrupting scoring signals (access_count, importance).
            if self.cross_domain_probes and hasattr(agent, 'coupled_engine'):
                if (i + 1) % self.probe_frequency == 0:
                    scorer = getattr(agent, '_scorer', None)
                    if scorer is not None:
                        probes = self._generate_cross_domain_probes(
                            dataset=self.dataset,
                            n_probes=self.probes_per_session,
                            session_index=i,
                        )
                        for probe_text in probes:
                            probe_emb = scorer.embed(probe_text)
                            agent.coupled_engine.query_readonly(
                                probe_emb, top_k=10,
                            )

            prev_day = session.day

        # --- Phase 2: Run evaluation questions ---
        per_question: list[dict] = []
        category_totals: dict[str, float] = {
            "current_fact": 0.0,
            "graceful_forgetting": 0.0,
            "reinforced_recall": 0.0,
            "cross_domain": 0.0,
        }
        category_counts: dict[str, int] = {
            "current_fact": 0,
            "graceful_forgetting": 0,
            "reinforced_recall": 0,
            "cross_domain": 0,
        }

        llm_client = self._llm_client if self.use_llm_judge else None
        used_llm = False

        for q in self.dataset.questions:
            response = agent.send_message(q.question, memorizing=False)

            # Extract answer text from response
            if isinstance(response, dict):
                answer_text = response.get("output", "")
            else:
                answer_text = str(response)

            q_score = score_question(
                q.question,
                answer_text,
                q.expected_keywords,
                q.rejected_keywords,
                q.category,
                llm_client=llm_client,
            )
            if llm_client and q.category in _LLM_JUDGE_CATEGORIES:
                used_llm = True

            per_question.append({
                "question": q.question,
                "answer": answer_text,
                "score": q_score,
                "category": q.category,
                "expected": q.expected_keywords,
                "rejected": q.rejected_keywords,
            })

            category_totals[q.category] += q_score
            category_counts[q.category] += 1

        # Compute mean scores per category
        category_scores = {}
        for cat in category_totals:
            count = category_counts[cat]
            category_scores[cat] = (
                category_totals[cat] / count if count > 0 else 0.0
            )

        comp = composite_score(category_scores)

        return {
            "category_scores": category_scores,
            "per_question": per_question,
            "composite": comp,
            "n_dreams": n_dreams,
            "sessions_processed": len(sessions),
            "scoring_method": "hybrid" if used_llm else "subem",
        }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _print_dataset_summary(ds: LongitudinalDataset) -> None:
    """Print summary statistics for the dataset."""
    print(f"{'=' * 70}")
    print("LONGITUDINAL EVALUATION DATASET SUMMARY")
    print(f"{'=' * 70}")
    print(f"Seed: {ds.seed}")
    print(f"Total sessions: {len(ds.sessions)}")
    print(f"Total questions: {len(ds.questions)}")

    # Session type distribution
    type_counts = Counter(s.session_type for s in ds.sessions)
    print(f"\nSession type distribution:")
    for stype, count in sorted(type_counts.items()):
        print(f"  {stype:<20s}: {count}")

    # Question category distribution
    cat_counts = Counter(q.category for q in ds.questions)
    print(f"\nQuestion category distribution:")
    for cat, count in sorted(cat_counts.items()):
        print(f"  {cat:<25s}: {count}")

    # Day range and clustering
    days = [s.day for s in ds.sessions]
    day_counts = Counter(days)
    print(f"\nDay range: {min(days)} - {max(days)}")
    print(f"Unique active days: {len(day_counts)}")
    multi_days = sum(1 for c in day_counts.values() if c > 1)
    print(f"Days with multiple sessions: {multi_days}")

    # Fact count distribution
    fact_counts = [len(s.facts) for s in ds.sessions]
    print(f"\nFacts per session: min={min(fact_counts)}, "
          f"max={max(fact_counts)}, "
          f"mean={sum(fact_counts)/len(fact_counts):.1f}")

    total_facts = sum(fact_counts)
    print(f"Total facts: {total_facts}")

    # Questions with rejected keywords
    with_rejected = sum(1 for q in ds.questions if q.rejected_keywords)
    print(f"\nQuestions with rejected keywords: {with_rejected}")

    # Sample facts from each type
    print(f"\n{'─' * 70}")
    print("Sample facts by session type:")
    for stype in ["preference", "update", "repeated", "single", "cross_domain"]:
        sessions = [s for s in ds.sessions if s.session_type == stype]
        if sessions:
            sample = sessions[0]
            print(f"\n  [{stype}] (day {sample.day}):")
            for fact in sample.facts[:2]:
                print(f"    - {fact}")

    # Sample questions from each category
    print(f"\n{'─' * 70}")
    print("Sample questions by category:")
    for cat in ["current_fact", "graceful_forgetting",
                 "reinforced_recall", "cross_domain"]:
        qs = [q for q in ds.questions if q.category == cat]
        if qs:
            q = qs[0]
            print(f"\n  [{cat}]")
            print(f"    Q: {q.question}")
            print(f"    Expected: {q.expected_keywords}")
            if q.rejected_keywords:
                print(f"    Rejected: {q.rejected_keywords}")

    print(f"\n{'=' * 70}")


if __name__ == "__main__":
    ds = generate_dataset(seed=42)
    _print_dataset_summary(ds)
