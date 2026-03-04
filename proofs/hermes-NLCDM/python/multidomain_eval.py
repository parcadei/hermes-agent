"""Multi-domain longitudinal evaluation for testing Hopfield retrieval at scale.

Hypothesis: with genuinely diverse embeddings (code, personal, work, travel,
food, learning), spreading activation outperforms direct cosine on cross-domain
retrieval while dreams improve all categories.

This eval tests the full Hopfield theory:
- Dreams carve energy basins for each domain
- Spreading activation traverses between well-separated basins
- Perturbation-response discovers implicit bridges
- Phase 13 switching criterion flips when delta-separation is real

Architecture:
  - Reuses Session, EvalQuestion, LongitudinalDataset from longitudinal_eval
  - Reuses LongitudinalEvaluator (domain-agnostic)
  - Defines generate_multidomain_dataset() with diverse content
  - 200 sessions across 6 domains, 100 questions with 4 categories

Content design principle: each domain has a distinct REGISTER and LENGTH,
producing embeddings in genuinely different regions of the space.
  - Code: technical, specific (race conditions, architecture decisions)
  - Personal: conversational, emotional (relationships, health, stress)
  - Work: professional, structured (meetings, deadlines, processes)
  - Travel: descriptive, geographic (itineraries, destinations)
  - Food: sensory, experiential (recipes, restaurants, ingredients)
  - Learning: analytical, conceptual (books, courses, theorems)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from longitudinal_eval import (
    EvalQuestion,
    LongitudinalDataset,
    Session,
    _generate_active_days,
    _assign_sessions_to_days,
)

logger = logging.getLogger(__name__)


# =====================================================================
# Domain definitions — fact chains with updates
# =====================================================================

# Each chain: (domain, sub_topic, [(display_text, keyword), ...])
# display_text is the FULL session content (long, distinctive register)
# keyword is what the eval checks for in answers

_MULTIDOMAIN_CHAINS: list[tuple[str, str, list[tuple[str, str]]]] = [
    # --- CODE DOMAIN ---
    ("code", "architecture", [
        (
            "Decided to use a microservices architecture for the new payment system. "
            "Each service owns its database and communicates through an event bus. "
            "Started with three services: billing, invoicing, and ledger reconciliation",
            "microservices",
        ),
        (
            "Rethinking the architecture. The event bus is causing too much latency "
            "for the billing flow. Moving to a modular monolith with clear module "
            "boundaries instead. Same logical separation, single deployment",
            "monolith",
        ),
    ]),
    ("code", "debugging", [
        (
            "Spent four hours debugging a race condition in the worker pool. The issue "
            "was a shared mutex between the HTTP handler and the background task runner. "
            "Threads were deadlocking under high load when both tried to acquire the "
            "connection pool lock simultaneously",
            "race condition",
        ),
        (
            "Found a better approach to the concurrency problem. Replaced the shared "
            "mutex with channel-based communication using Go channels. Each worker now "
            "receives jobs through a buffered channel, eliminating the deadlock entirely. "
            "Throughput went from 200 to 1400 requests per second",
            "channels",
        ),
    ]),
    ("code", "database", [
        (
            "Chose PostgreSQL for the analytics service. The relational model works "
            "well for the aggregation queries we need, and the JSONB column type "
            "handles the semi-structured event data without needing a separate document store",
            "postgresql",
        ),
        (
            "Added ClickHouse as the analytics backend alongside PostgreSQL. Postgres "
            "handles transactional writes but ClickHouse processes the analytical queries "
            "ten times faster with its columnar storage. Materialized views sync between them",
            "clickhouse",
        ),
    ]),
    ("code", "testing", [
        (
            "Set up property-based testing with Hypothesis for the serialization layer. "
            "Found three edge cases in the first run that unit tests missed: empty nested "
            "objects, unicode surrogate pairs, and timestamps before epoch",
            "property-based",
        ),
    ]),

    # --- PERSONAL DOMAIN ---
    ("personal", "stress", [
        (
            "Been feeling really stressed from the project deadline at work. Not sleeping "
            "well, waking up at 4am thinking about the deployment timeline. My partner "
            "noticed I have been irritable and suggested I find a way to decompress",
            "stressed",
        ),
        (
            "Started doing ten minutes of morning meditation using the Waking Up app. "
            "Sam Harris guides you through noting practice where you observe thoughts "
            "without engaging. After two weeks my sleep improved significantly and "
            "I feel calmer during the day",
            "meditation",
        ),
    ]),
    ("personal", "relationship", [
        (
            "Had a long conversation with my partner about our future plans. We both "
            "want to travel more in the next year before thinking about settling down. "
            "Agreed to do one international trip per quarter if our schedules allow it",
            "travel together",
        ),
        (
            "My partner got a remote work arrangement approved. This changes everything "
            "for our travel plans since we can now do extended trips of two to three "
            "weeks instead of just long weekends",
            "remote work",
        ),
    ]),
    ("personal", "health", [
        (
            "Doctor said my blood work looks good overall but my cholesterol is borderline. "
            "She recommended reducing saturated fat and adding more omega-3 through fish "
            "or supplements. Also suggested regular cardio at least three times per week",
            "cholesterol",
        ),
        (
            "Follow-up blood work after three months of diet changes. Cholesterol dropped "
            "from borderline to normal range. The combination of cutting dairy and adding "
            "salmon twice a week made the biggest difference according to my doctor",
            "normal range",
        ),
    ]),

    # --- WORK DOMAIN ---
    ("work", "promotion", [
        (
            "One-on-one with my manager went well. She mentioned there might be a "
            "tech lead opening in Q2 and encouraged me to prepare. Need to work on "
            "system design interview skills and document my leadership contributions",
            "tech lead",
        ),
        (
            "Got the tech lead promotion. Starting next month I will be responsible "
            "for the platform team of six engineers. First priority is establishing "
            "architecture review meetings and a technical decision log",
            "promoted",
        ),
    ]),
    ("work", "process", [
        (
            "Sprint retrospective identified a recurring pattern: we keep underestimating "
            "the testing phase. Going to add a dedicated buffer day for QA and exploratory "
            "testing in future sprints. Also introducing test plan reviews before implementation starts",
            "testing buffer",
        ),
    ]),
    ("work", "project", [
        (
            "The data migration project kicked off today. Moving from the legacy Oracle "
            "database to the new PostgreSQL cluster. Timeline is eight weeks with a "
            "parallel running period of two weeks for validation",
            "oracle migration",
        ),
        (
            "Data migration hit week four. Discovered that 15 percent of records have "
            "encoding issues from the original Latin-1 to UTF-8 conversion. Writing a "
            "custom cleaner to handle the edge cases before the parallel run",
            "encoding issues",
        ),
        (
            "Migration complete. All data validated, parallel run showed zero discrepancies "
            "after the encoding fix. Decommissioning the Oracle instance next month. "
            "Total cost savings will be around forty thousand dollars per year in licensing",
            "migration complete",
        ),
    ]),

    # --- TRAVEL DOMAIN ---
    ("travel", "lisbon", [
        (
            "Booked flights to Lisbon for the second week of April. Found a great "
            "Airbnb in Alfama near the Castelo Sao Jorge. The neighborhood looks "
            "charming with narrow cobblestone streets and traditional fado music venues",
            "lisbon",
        ),
        (
            "Researched day trips from Lisbon. Sintra is a must-see with the Pena Palace "
            "and Moorish Castle. Takes about forty minutes by train from Rossio station. "
            "Also want to visit the Belem district for the pasteis de nata and the "
            "Jeronimos Monastery",
            "sintra",
        ),
    ]),
    ("travel", "japan", [
        (
            "Planning a three-week trip to Japan in November for the autumn foliage. "
            "Tokyo for five days, then Kyoto for a week, and finishing with Osaka "
            "and a side trip to Naoshima island for the art installations",
            "japan",
        ),
    ]),
    ("travel", "logistics", [
        (
            "Need to figure out the transit systems for our trips. In Lisbon the "
            "viva viagem card works on metro, tram, and ferries. In Japan we should "
            "get the JR Pass for the shinkansen between cities",
            "transit",
        ),
    ]),

    # --- FOOD DOMAIN ---
    ("food", "cooking_skill", [
        (
            "Made mushroom risotto from scratch tonight. The key is toasting the "
            "arborio rice in butter before adding warm stock one ladle at a time. "
            "Used dried porcini soaked in hot water and added the soaking liquid "
            "for extra umami depth. Finished with real parmigiano reggiano",
            "risotto",
        ),
        (
            "Attempting homemade ramen after watching the Ramen King documentary. "
            "The tonkotsu broth needs twelve hours of rolling boil to extract the "
            "collagen from the pork bones. Made tare from soy sauce, mirin, and "
            "kombu. The chashu is braised pork belly rolled and tied with twine",
            "ramen",
        ),
    ]),
    ("food", "diet", [
        (
            "Going to try cutting out dairy for a month after reading research "
            "suggesting it might help with the skin issues I have been having. "
            "Switching to oat milk for coffee and finding vegan cheese alternatives "
            "for cooking. My doctor supports the experiment",
            "dairy-free",
        ),
        (
            "Two months dairy-free and my skin has cleared up noticeably. Also "
            "lost four pounds without trying. Going to keep it up but allow "
            "exceptions for special occasions and high-quality aged cheese "
            "which is lower in lactose anyway",
            "skin cleared",
        ),
    ]),
    ("food", "restaurant", [
        (
            "Tried the new ramen place on Valencia Street. Their tonkotsu broth "
            "is incredibly rich and creamy, clearly slow-cooked for hours. The "
            "ajitama egg was perfectly soft-boiled with a jammy center. Going back "
            "with friends next weekend",
            "valencia ramen",
        ),
    ]),

    # --- LEARNING DOMAIN ---
    ("learning", "systems_thinking", [
        (
            "Reading Thinking in Systems by Donella Meadows. The concept of leverage "
            "points is fascinating: small interventions at the right place in a system "
            "can have outsized effects. She ranks twelve leverage points from weakest "
            "like adjusting parameters to strongest like changing paradigms",
            "leverage points",
        ),
    ]),
    ("learning", "distributed_systems", [
        (
            "Took detailed notes from the distributed systems lecture. The CAP theorem "
            "trade-offs are clearer now. We chose AP for our event store because "
            "partition tolerance and availability matter more than strict consistency "
            "for our use case. Eventually consistent reads are acceptable",
            "cap theorem",
        ),
        (
            "Deep dive into consensus algorithms. Raft is more understandable than "
            "Paxos but they solve the same problem: getting distributed nodes to agree "
            "on a single value. The leader election mechanism in Raft uses randomized "
            "timeouts to break symmetry",
            "raft consensus",
        ),
    ]),
    ("learning", "formal_methods", [
        (
            "Working through the Lean 4 tutorial. Dependent types are mind-bending "
            "but powerful for encoding invariants directly in the type system. Proved "
            "my first theorem about list concatenation associativity. The tactic mode "
            "feels like a dialogue with the proof assistant",
            "lean",
        ),
    ]),
    ("learning", "book_club", [
        (
            "Book club chose Godel Escher Bach by Douglas Hofstadter for this month. "
            "The strange loop concept connects to ideas about self-reference in "
            "formal systems. Godel numbering is essentially a way to make a formal "
            "system talk about its own statements",
            "godel escher bach",
        ),
    ]),
]


# =====================================================================
# Cross-domain bridges — implicit semantic connections
# =====================================================================

@dataclass
class _MultidomainBridge:
    domain_a: str
    fact_a_summary: str
    keyword_a: str
    domain_b: str
    fact_b_summary: str
    keyword_b: str
    seed_sessions: list[tuple[int, str]]
    bridge_questions: list[tuple[str, list[str]]]


_BRIDGE_DEFS: list[dict] = [
    # Bridge 1: Code stress → Personal meditation
    {
        "domain_a": "code",
        "fact_a": "race condition debugging",
        "keyword_a": "race condition",
        "domain_b": "personal",
        "fact_b": "meditation for stress",
        "keyword_b": "meditation",
        "seed_facts": [
            (20, "The race condition debugging this week has been incredibly stressful. "
                 "Four hours staring at thread dumps and deadlock traces. My shoulders "
                 "are tense and I barely slept last night thinking about the deployment"),
            (50, "Morning meditation has become my anchor. When I notice work stress "
                 "building up, I do a quick five-minute body scan. It reminds me that "
                 "the debugging problems are temporary and solvable"),
        ],
        "questions": [
            ("What am I doing to manage the stress from my technical work?",
             ["debugging", "meditation"]),
            ("How has my approach to work-life balance changed recently?",
             ["stress", "meditation"]),
            ("What connection is there between my work challenges and my new habits?",
             ["race condition", "meditation"]),
        ],
    },
    # Bridge 2: Personal relationship travel → Travel Lisbon
    {
        "domain_a": "personal",
        "fact_a": "partner wants to travel",
        "keyword_a": "partner",
        "domain_b": "travel",
        "fact_b": "Lisbon trip",
        "keyword_b": "lisbon",
        "seed_facts": [
            (25, "My partner and I talked about wanting to explore Europe together "
                 "this year. We made a pact to do at least one trip per quarter"),
            (65, "Finally booked the Lisbon trip for both of us. My partner is excited "
                 "about the food scene there, especially the seafood and pasteis de nata"),
        ],
        "questions": [
            ("How does my upcoming trip relate to my relationship goals?",
             ["partner", "lisbon"]),
            ("What travel plans have my partner and I made together?",
             ["partner", "lisbon"]),
        ],
    },
    # Bridge 3: Food dairy-free → Personal health cholesterol
    {
        "domain_a": "food",
        "fact_a": "dairy-free diet",
        "keyword_a": "dairy",
        "domain_b": "personal",
        "fact_b": "cholesterol improvement",
        "keyword_b": "cholesterol",
        "seed_facts": [
            (35, "Cutting dairy from my diet was initially about skin issues but "
                 "my doctor mentioned it could also help with my borderline cholesterol"),
            (90, "The dairy-free experiment worked on multiple fronts. My skin is "
                 "clear and at my latest checkup my cholesterol numbers improved too. "
                 "The diet change addressed both concerns simultaneously"),
        ],
        "questions": [
            ("How has my dietary change affected my overall health?",
             ["dairy", "cholesterol"]),
            ("What unexpected health benefits came from my food choices?",
             ["dairy", "skin"]),
            ("How are my diet and medical results connected?",
             ["dairy", "cholesterol"]),
        ],
    },
    # Bridge 4: Learning distributed systems → Code architecture
    {
        "domain_a": "learning",
        "fact_a": "CAP theorem",
        "keyword_a": "cap theorem",
        "domain_b": "code",
        "fact_b": "architecture decisions",
        "keyword_b": "monolith",
        "seed_facts": [
            (30, "The distributed systems course is directly relevant to my work. "
                 "Understanding the CAP theorem helped me articulate why our eventual "
                 "consistency model is the right trade-off"),
            (75, "Applied what I learned about consensus to the architecture review. "
                 "The monolith decision actually sidesteps a lot of distributed consensus "
                 "problems because we avoid the event bus entirely"),
        ],
        "questions": [
            ("How has what I am learning academically influenced my work decisions?",
             ["cap theorem", "monolith"]),
            ("What is the connection between my studies and my architecture choices?",
             ["distributed", "monolith"]),
        ],
    },
    # Bridge 5: Work promotion → Code testing leadership
    {
        "domain_a": "work",
        "fact_a": "tech lead promotion",
        "keyword_a": "tech lead",
        "domain_b": "code",
        "fact_b": "property-based testing initiative",
        "keyword_b": "property-based",
        "seed_facts": [
            (40, "Preparing for the tech lead role by documenting my technical "
                 "contributions. The property-based testing initiative I introduced "
                 "caught three critical serialization bugs before production"),
            (85, "As new tech lead, my first initiative is making property-based "
                 "testing standard practice across all teams. Already drafted the "
                 "testing policy document and scheduled training sessions"),
        ],
        "questions": [
            ("How am I leveraging my technical expertise in my new leadership role?",
             ["tech lead", "property-based"]),
            ("What testing improvements am I driving as a leader?",
             ["tech lead", "property-based"]),
        ],
    },
    # Bridge 6: Travel Japan → Food ramen
    {
        "domain_a": "travel",
        "fact_a": "Japan trip",
        "keyword_a": "japan",
        "domain_b": "food",
        "fact_b": "ramen cooking",
        "keyword_b": "ramen",
        "seed_facts": [
            (45, "Part of the motivation for the Japan trip is food. I have been "
                 "making ramen at home and want to taste the real thing in Tokyo, "
                 "especially at Fuunji for their tsukemen"),
            (80, "Making a food itinerary for Japan. Want to compare my homemade "
                 "tonkotsu with authentic Hakata-style ramen in Fukuoka. Also "
                 "planning to take a cooking class in Kyoto"),
        ],
        "questions": [
            ("How does my cooking hobby connect to my travel plans?",
             ["ramen", "japan"]),
            ("What food experiences am I planning for my trip?",
             ["ramen", "japan"]),
            ("How is my interest in cooking shaping my travel decisions?",
             ["ramen", "japan"]),
        ],
    },
    # Bridge 7: Learning formal methods → Code database
    {
        "domain_a": "learning",
        "fact_a": "Lean theorem proving",
        "keyword_a": "lean",
        "domain_b": "code",
        "fact_b": "database correctness",
        "keyword_b": "postgresql",
        "seed_facts": [
            (55, "Learning Lean has changed how I think about software correctness. "
                 "I now want formal guarantees for our database migration, not just "
                 "tests. The type-level encoding of invariants could prevent the "
                 "encoding bugs we hit"),
            (95, "Used ideas from dependent types to design the validation layer "
                 "for the PostgreSQL migration. Each transformation step has explicit "
                 "preconditions and postconditions, like a lightweight proof"),
        ],
        "questions": [
            ("How is my interest in formal methods affecting my engineering work?",
             ["lean", "postgresql"]),
            ("What proof-related concepts am I applying to practical coding?",
             ["lean", "migration"]),
        ],
    },
    # Bridge 8: Work process → Learning systems thinking
    {
        "domain_a": "work",
        "fact_a": "sprint process improvement",
        "keyword_a": "sprint",
        "domain_b": "learning",
        "fact_b": "systems thinking",
        "keyword_b": "leverage points",
        "seed_facts": [
            (60, "The sprint retro pattern keeps repeating: testing takes longer "
                 "than planned. Reading Meadows made me realize this is a systems "
                 "problem, not just a scheduling problem"),
            (100, "Applied leverage point analysis to our sprint process. The buffer "
                  "day was a weak intervention. The real leverage point is moving test "
                  "plan reviews BEFORE implementation starts, changing the information "
                  "flow in the system"),
        ],
        "questions": [
            ("How am I applying ideas from my reading to improve work processes?",
             ["leverage points", "sprint"]),
            ("What systematic thinking am I bringing to my team?",
             ["systems", "sprint"]),
        ],
    },
    # Bridge 9: Personal health → Food cooking
    {
        "domain_a": "personal",
        "fact_a": "omega-3 recommendation",
        "keyword_a": "omega-3",
        "domain_b": "food",
        "fact_b": "cooking with fish",
        "keyword_b": "salmon",
        "seed_facts": [
            (42, "My doctor recommended more omega-3 for cholesterol. I am learning "
                 "to cook fish properly instead of just taking supplements. Started "
                 "with pan-seared salmon, getting the skin crispy"),
            (88, "Made salmon three different ways this month: pan-seared with crispy "
                 "skin, baked with miso glaze, and as poke bowls. My cholesterol is "
                 "improving and I actually enjoy the cooking process"),
        ],
        "questions": [
            ("How have medical recommendations changed my cooking habits?",
             ["omega-3", "salmon"]),
            ("What is the connection between my health goals and what I cook?",
             ["cholesterol", "salmon"]),
        ],
    },
    # Bridge 10: Travel → Work remote
    {
        "domain_a": "travel",
        "fact_a": "extended trip planning",
        "keyword_a": "lisbon",
        "domain_b": "work",
        "fact_b": "work flexibility",
        "keyword_b": "remote",
        "seed_facts": [
            (70, "Now that my partner can work remotely, we are extending the Lisbon "
                 "trip to three weeks. I will need to arrange remote work for the "
                 "extra time. My new tech lead role should give me more flexibility"),
            (105, "Talked to my manager about working from Lisbon for a week. Since "
                  "the time zone overlap with our US team is limited, I offered to "
                  "shift my hours. She approved it as a trial"),
        ],
        "questions": [
            ("How is my work arrangement enabling my travel plans?",
             ["remote", "lisbon"]),
            ("What work flexibility am I leveraging for my trips?",
             ["tech lead", "lisbon"]),
        ],
    },
]


# =====================================================================
# Reinforcement topics — detailed, domain-specific repeated mentions
# =====================================================================

_REINFORCEMENT_DEFS: list[dict] = [
    {
        "domain": "go_programming",
        "base_keyword": "go",
        "details": [
            "Refactored the worker pool to use Go channels instead of mutexes. The "
            "select statement makes multiplexing clean. Benchmarked at 1400 req/s",
            "Wrote a custom Go linter that checks for unbuffered channel sends in "
            "goroutines. Found two potential deadlocks in our codebase already",
            "Go 1.22 generics are maturing. Rewrote our collection utilities with "
            "type parameters, eliminated 200 lines of interface{} assertions",
            "Profiled the Go service with pprof. The HTTP middleware chain was "
            "allocating 4KB per request. Switched to sync.Pool and cut GC pauses by 60%",
            "Implemented graceful shutdown in Go with context cancellation. Each "
            "goroutine checks ctx.Done() and drains its work before exiting",
        ],
    },
    {
        "domain": "meditation_practice",
        "base_keyword": "meditation",
        "details": [
            "Day 30 of daily meditation. Tried a new technique: noting practice where "
            "you silently label each thought as thinking, feeling, or sensing",
            "Found that five minutes of meditation before code review makes me less "
            "reactive to feedback. I notice my ego response and let it pass",
            "Switched from guided meditation to silent sits. Setting a timer for ten "
            "minutes and just following the breath. The silence feels different",
            "Had my first experience of non-dual awareness during meditation. For "
            "about thirty seconds the sense of a separate observer dissolved",
            "Started a meditation log in my journal. Tracking sleep quality, focus "
            "during work, and emotional reactivity. Clear correlation with consistency",
        ],
    },
    {
        "domain": "cooking_exploration",
        "base_keyword": "cooking",
        "details": [
            "Made fresh pasta for the first time. The dough is just flour, eggs, and "
            "olive oil but getting the hydration right took three attempts",
            "Tried fermenting my own kimchi. Napa cabbage, gochugaru, fish sauce, and "
            "garlic. Needs to ferment for five days at room temperature",
            "Bought a carbon steel wok and seasoned it properly. The patina is building "
            "up after a dozen stir fries. Wok hei is real and it changes everything",
            "Experimented with sous vide chicken breast at 63 Celsius for 90 minutes. "
            "Most consistently juicy result I have ever achieved",
            "Learned to make a proper French omelette. The trick is high heat, constant "
            "stirring with chopsticks, and folding in one motion before it sets",
            "Started a sourdough starter named Steve. Day seven, it finally doubled in "
            "four hours. First bake attempt tomorrow",
        ],
    },
    {
        "domain": "lisbon_planning",
        "base_keyword": "lisbon",
        "details": [
            "Mapped out our Lisbon walking routes. Day one: Alfama and Castelo Sao Jorge. "
            "Day two: Belem for the monastery and tower. Day three: LX Factory and "
            "the Christo Rei viewpoint across the river",
            "Found the best pasteis de nata spots: Pasteis de Belem is the famous one "
            "but locals say Manteigaria in Chiado is actually better",
            "Booked a fado dinner show in Alfama at Clube de Fado. Traditional Portuguese "
            "music with a three-course meal. Should be an incredible atmosphere",
            "Researched the Lisbon earthquake of 1755 that destroyed most of the city. "
            "The Baixa district was rebuilt in a grid pattern by the Marquis de Pombal. "
            "Want to see the archaeological ruins under the Rua Augusta arch",
        ],
    },
    {
        "domain": "platform_team",
        "base_keyword": "platform",
        "details": [
            "First week as tech lead. Set up weekly architecture review meetings. Each "
            "team presents their design docs before implementation starts",
            "Created a technical decision log using ADRs. Architecture Decision Records "
            "capture the context, decision, and consequences of each choice",
            "Onboarded two new engineers to the platform team. Built a runbook covering "
            "deployment, monitoring, and incident response procedures",
            "Proposed a developer experience initiative: standardizing CI pipelines, "
            "adding preview environments, and automating dependency updates",
            "Ran the team's first blameless postmortem after a production incident. "
            "Root cause was a missing circuit breaker on the payment API. Action item: "
            "add circuit breakers to all external service calls",
        ],
    },
    {
        "domain": "distributed_study",
        "base_keyword": "distributed",
        "details": [
            "Worked through the Raft paper section by section. The log replication "
            "mechanism is elegant: the leader appends entries and followers replicate "
            "in order. Safety comes from the election restriction",
            "Implemented a toy Raft node in Go for the course project. Leader election "
            "works with randomized timeouts. Got two nodes to agree on a value",
            "Studied vector clocks for causality tracking. Lamport timestamps give "
            "partial ordering but vector clocks capture the happens-before relation "
            "precisely for N processes",
            "Read the Dynamo paper. Their consistent hashing ring with virtual nodes "
            "and quorum reads/writes is still the foundation for DynamoDB today",
        ],
    },
    {
        "domain": "health_journey",
        "base_keyword": "health",
        "details": [
            "Started tracking my macros with Cronometer. Hitting 130g protein per day "
            "which is right for my body weight. Carbs and fats naturally balanced",
            "Three months into the new exercise routine. Alternating between running "
            "and strength training. Resting heart rate dropped from 72 to 64 bpm",
            "Sleep has improved since cutting evening screen time. Using a red light "
            "after 9pm and reading actual books in bed instead of scrolling",
            "My morning routine is dialed in: wake at 6, meditate for 10 minutes, "
            "cold shower, high-protein breakfast, then work. The consistency helps",
        ],
    },
    {
        "domain": "lean_proving",
        "base_keyword": "lean",
        "details": [
            "Proved list reverse distributes over append in Lean 4. The key insight "
            "was structural induction on the first list with a generalized lemma",
            "Learning Lean's tactic mode: intro, apply, exact, simp, and omega handle "
            "most basic proofs. Ring and linarith for arithmetic goals",
            "Discovered the Mathlib library for Lean. It has formalized huge chunks "
            "of undergraduate mathematics. Using it for topology definitions",
            "Built a small verified stack implementation in Lean. Push, pop, and peek "
            "all have correctness proofs. The dependent types ensure you cannot pop "
            "from an empty stack — it is a type error at compile time",
        ],
    },
]


# =====================================================================
# Single-mention filler (noise) — diverse domain-flavored ephemera
# =====================================================================

_SINGLE_MENTIONS: list[str] = [
    "Saw an incredible sunset from the office rooftop today, the sky was orange and purple for twenty minutes",
    "The coffee shop downstairs changed their espresso blend and honestly it is worse now",
    "My neighbor asked me to water their plants while they are on vacation next week",
    "Found a great podcast about the history of the internet, listened to three episodes on my commute",
    "The new standing desk arrived but the assembly instructions are completely wrong",
    "Watched a documentary about octopus intelligence last night, they can solve puzzles and use tools",
    "Traffic was terrible today, took ninety minutes instead of the usual forty",
    "Discovered a shortcut through the park that saves ten minutes on my walk to the office",
    "The team lunch place was closed so we tried the Indian restaurant next door, surprisingly good biryani",
    "My noise-canceling headphones died mid-meeting, ordered replacement ear pads",
    "Saw a street performer playing Bach cello suites on the subway platform, stopped to listen for ten minutes",
    "The building fire alarm went off during standup, false alarm from construction dust",
    "Found a twenty dollar bill in my winter coat pocket, a nice surprise",
    "My roommate is learning ukulele, practicing the same three chords every evening",
    "The gym was packed at 6pm so I went for a run in the park instead, saw deer near the lake",
    "Tried a new laundromat with app-controlled machines, much better than the old coin-operated one",
    "The sunset was incredible from the bridge today, took a photo that actually turned out well",
    "Spilled coffee on my keyboard, had to switch to the backup while it dried out",
    "A colleague recommended a documentary about deep sea creatures, adding it to my watch list",
    "The elevator in our building broke down, been taking stairs all week which is actually good exercise",
    "Found a beautiful used bookstore tucked behind the pharmacy, bought three paperbacks",
    "My phone screen cracked when it fell on the concrete, repair appointment on Thursday",
    "The cherry blossoms in the park are at peak bloom this week, walked through at lunch",
    "Overheard an interesting conversation about urban farming while waiting for the train",
    "The parking meter app charged me double, spent fifteen minutes on hold getting a refund",
    "Saw a hawk catch a pigeon right outside the office window, nature is metal",
    "The quarterly company meeting was actually useful this time, good transparency about financials",
    "My plant in the office finally bloomed after eight months, a small white orchid",
    "The power went out for two hours, used the time to reorganize my desk drawer",
    "A friend sent me a playlist of ambient music for focused work, it actually helps",
]


# =====================================================================
# Stable biographical facts (diverse, longer form)
# =====================================================================

_STABLE_BIOGRAPHICAL = [
    ("education", "I studied computer science and math at ETH Zurich, graduated in 2019 with a focus on distributed systems", "eth"),
    ("hometown", "I grew up in Portland Oregon, lived there until I was eighteen before moving to Switzerland for university", "portland"),
    ("name", "My name is Alex Chen", "alex"),
    ("sibling", "I have a younger sister Emma who is a marine biologist working in Monterey Bay", "emma"),
    ("language_native", "My first language is English but I speak conversational German and basic Portuguese", "english"),
    ("birthday", "My birthday is September fifteenth, I always celebrate with a hike if the weather permits", "september"),
    ("apartment", "I live in a two-bedroom apartment in Zurich near the Limmat river, been here since 2020", "zurich"),
    ("workspace", "I have a home office setup with dual monitors, a Herman Miller chair, and a split keyboard", "home office"),
]


# =====================================================================
# Templates per domain for initial, update, and question generation
# =====================================================================

_DOMAIN_LABELS = {
    "code": "programming and software engineering",
    "personal": "personal life and wellbeing",
    "work": "work and career",
    "travel": "travel and trips",
    "food": "food and cooking",
    "learning": "learning and studies",
}

_CHAIN_QUESTIONS: dict[str, dict[str, str]] = {
    ("code", "architecture"): "What architecture am I currently using for the payment system?",
    ("code", "debugging"): "How did I solve the concurrency problem in the worker pool?",
    ("code", "database"): "What database am I using for analytics?",
    ("code", "testing"): "What testing approach did I introduce for the serialization layer?",
    ("personal", "stress"): "How am I managing my stress?",
    ("personal", "relationship"): "What are my partner and I planning?",
    ("personal", "health"): "What did my latest health checkup show?",
    ("work", "promotion"): "What is my current role?",
    ("work", "process"): "What process change did I introduce for sprints?",
    ("work", "project"): "What is the status of the data migration project?",
    ("travel", "lisbon"): "What are my plans for the Lisbon trip?",
    ("travel", "japan"): "What are my Japan travel plans?",
    ("travel", "logistics"): "What transit options am I researching for travel?",
    ("food", "cooking_skill"): "What dish have I been learning to cook recently?",
    ("food", "diet"): "What dietary change am I making?",
    ("food", "restaurant"): "What restaurant did I try recently?",
    ("learning", "systems_thinking"): "What book about systems am I reading?",
    ("learning", "distributed_systems"): "What am I studying about distributed systems?",
    ("learning", "formal_methods"): "What theorem proving language am I learning?",
    ("learning", "book_club"): "What book is my book club reading?",
}

_GRACEFUL_QUESTIONS_MULTI: dict[str, str] = {
    ("code", "architecture"): "Has my architecture approach always stayed the same?",
    ("code", "debugging"): "How has my approach to the concurrency issue evolved?",
    ("code", "database"): "Have I always used the same database for analytics?",
    ("personal", "stress"): "How has my approach to stress management changed?",
    ("personal", "relationship"): "How have my relationship plans evolved?",
    ("personal", "health"): "How have my health metrics changed over time?",
    ("work", "promotion"): "What has my career progression been like?",
    ("work", "project"): "Walk me through the phases of the data migration project.",
    ("food", "cooking_skill"): "How has my cooking evolved over time?",
    ("food", "diet"): "How has my diet changed and what were the results?",
    ("learning", "distributed_systems"): "How has my understanding of distributed systems developed?",
}


# =====================================================================
# Dataset generator
# =====================================================================


def generate_multidomain_dataset(seed: int = 42) -> LongitudinalDataset:
    """Generate a multi-domain longitudinal evaluation dataset.

    Same temporal structure as generate_dataset() but with genuinely diverse
    content across 6 domains, producing embeddings in different regions of
    the space.

    Args:
        seed: RNG seed for reproducibility.

    Returns:
        LongitudinalDataset with 200 sessions and 100 eval questions.
    """
    import numpy as np

    rng = np.random.default_rng(seed)

    # --- Generate active day schedule ---
    day_rng = np.random.default_rng(seed + 2000)
    active_days = _generate_active_days(day_rng)

    # ===================================================================
    # PHASE 1: Build sessions from the world model
    # ===================================================================

    # --- 1a. Build fact chains ---
    chain_data: list[dict] = []
    for domain, sub_topic, values in _MULTIDOMAIN_CHAINS:
        base_day = int(rng.integers(0, 25))
        chain_values = []
        for step_idx, (text, keyword) in enumerate(values):
            if step_idx == 0:
                day = base_day
            else:
                gap = int(rng.integers(15, 45))
                day = chain_values[-1]["day"] + gap
                day = min(day, 170)
            chain_values.append({"day": day, "text": text, "keyword": keyword})
        chain_data.append({
            "domain": domain,
            "sub_topic": sub_topic,
            "values": chain_values,
        })

    # --- Preference sessions (50): initial values + stable bio ---
    preference_sessions: list[Session] = []

    for chain in chain_data:
        initial = chain["values"][0]
        preference_sessions.append(Session(
            day=initial["day"],
            session_type="preference",
            facts=[initial["text"]],
        ))

    for bio_key, bio_fact, bio_kw in _STABLE_BIOGRAPHICAL:
        day = int(rng.integers(0, 25))
        preference_sessions.append(Session(
            day=day,
            session_type="preference",
            facts=[bio_fact],
        ))

    # Fill to 50 with restated initial facts
    while len(preference_sessions) < 50:
        chain = chain_data[int(rng.integers(0, len(chain_data)))]
        initial = chain["values"][0]
        fact = f"Just to mention again: {initial['text'][:120]}"
        day = int(rng.integers(0, 30))
        preference_sessions.append(Session(
            day=day, session_type="preference", facts=[fact],
        ))
    preference_sessions = preference_sessions[:50]

    # --- Update sessions (40): evolve fact chains ---
    update_sessions: list[Session] = []

    for chain in chain_data:
        for step_idx in range(1, len(chain["values"])):
            val = chain["values"][step_idx]
            update_sessions.append(Session(
                day=val["day"],
                session_type="update",
                facts=[val["text"]],
            ))

    # Pad with reinforcement of terminal values
    while len(update_sessions) < 40:
        multi = [c for c in chain_data if len(c["values"]) >= 2]
        chain = multi[int(rng.integers(0, len(multi)))]
        terminal = chain["values"][-1]
        variations = [
            f"Still on track with the {chain['sub_topic']} change. {terminal['text'][:80]}",
            f"Confirming the {chain['sub_topic']} update is working well. {terminal['keyword']} approach is solid",
        ]
        fact = variations[int(rng.integers(0, len(variations)))]
        day = min(terminal["day"] + int(rng.integers(5, 20)), 175)
        update_sessions.append(Session(
            day=day, session_type="update", facts=[fact],
        ))
    update_sessions = update_sessions[:40]

    # --- Repeated/reinforcement sessions (60) ---
    repeated_sessions: list[Session] = []
    detail_indices = {rd["domain"]: 0 for rd in _REINFORCEMENT_DEFS}

    for i in range(60):
        topic = _REINFORCEMENT_DEFS[int(rng.integers(0, len(_REINFORCEMENT_DEFS)))]
        domain = topic["domain"]
        idx = detail_indices[domain]
        details = topic["details"]
        fact = details[idx % len(details)]
        detail_indices[domain] = idx + 1
        repeated_sessions.append(Session(
            day=0, session_type="repeated", facts=[fact],
        ))

    # --- Single-mention sessions (30) ---
    single_sessions: list[Session] = []
    shuffled_singles = list(_SINGLE_MENTIONS)
    rng.shuffle(shuffled_singles)
    for i in range(30):
        single_sessions.append(Session(
            day=0, session_type="single",
            facts=[shuffled_singles[i % len(shuffled_singles)]],
        ))

    # --- Cross-domain seed sessions (20) ---
    cross_sessions: list[Session] = []
    for bdef in _BRIDGE_DEFS:
        for seed_day, seed_text in bdef["seed_facts"]:
            cross_sessions.append(Session(
                day=seed_day,
                session_type="cross_domain",
                facts=[seed_text],
            ))

    # Pad to 20
    while len(cross_sessions) < 20:
        bdef = _BRIDGE_DEFS[int(rng.integers(0, len(_BRIDGE_DEFS)))]
        fact_a = bdef["fact_a"]
        fact_b = bdef["fact_b"]
        extras = [
            f"Thinking more about the connection between {fact_a} and {fact_b}",
            f"The overlap between {bdef['domain_a']} and {bdef['domain_b']} keeps coming up",
        ]
        fact = extras[int(rng.integers(0, len(extras)))]
        day = int(rng.integers(50, 170))
        cross_sessions.append(Session(
            day=day, session_type="cross_domain", facts=[fact],
        ))
    cross_sessions = cross_sessions[:20]

    # --- Assign days ---
    pref_active = [d for d in active_days if d <= 60]
    if len(pref_active) < 10:
        pref_active = active_days[:max(len(active_days) // 3, 10)]
    pref_days = _assign_sessions_to_days(rng, pref_active, 50)
    for s, d in zip(preference_sessions, pref_days):
        s.day = d

    update_active = [d for d in active_days if 20 <= d <= 170]
    if len(update_active) < 10:
        update_active = [d for d in active_days if d >= 15]
    update_days = _assign_sessions_to_days(rng, update_active, 40)
    for s, d in zip(update_sessions, update_days):
        s.day = d

    repeated_active = [d for d in active_days if 15 <= d <= 175]
    if len(repeated_active) < 15:
        repeated_active = active_days[3:]
    repeated_days = _assign_sessions_to_days(rng, repeated_active, 60)
    for s, d in zip(repeated_sessions, repeated_days):
        s.day = d

    single_active = [d for d in active_days if 10 <= d <= 170]
    if len(single_active) < 10:
        single_active = active_days[3:]
    single_days = _assign_sessions_to_days(rng, single_active, 30)
    for s, d in zip(single_sessions, single_days):
        s.day = d

    cross_active = [d for d in active_days if 10 <= d <= 175]
    if len(cross_active) < 10:
        cross_active = active_days[len(active_days) // 4:]
    cross_days = _assign_sessions_to_days(rng, cross_active, 20)
    for s, d in zip(cross_sessions, cross_days):
        s.day = d

    # --- Merge and sort ---
    all_sessions = (
        preference_sessions + update_sessions + repeated_sessions
        + single_sessions + cross_sessions
    )
    all_sessions.sort(key=lambda s: s.day)

    # ===================================================================
    # PHASE 2: Generate questions from terminal states
    # ===================================================================

    questions: list[EvalQuestion] = []
    used_texts: set[str] = set()

    def _add(q: EvalQuestion) -> bool:
        if q.question in used_texts:
            return False
        used_texts.add(q.question)
        questions.append(q)
        return True

    # --- Category 1: Current fact retrieval (30 questions) ---
    for chain in chain_data:
        key = (chain["domain"], chain["sub_topic"])
        terminal = chain["values"][-1]
        q_text = _CHAIN_QUESTIONS.get(key, f"What is the latest on my {chain['sub_topic']}?")
        rejected = [v["keyword"] for v in chain["values"][:-1]]
        _add(EvalQuestion(
            question=q_text,
            expected_keywords=[terminal["keyword"]],
            rejected_keywords=rejected,
            category="current_fact",
        ))

    # Stable bio questions
    _bio_questions = {
        "education": "Where did I study?",
        "hometown": "Where did I grow up?",
        "name": "What is my name?",
        "sibling": "Do I have siblings?",
        "language_native": "What languages do I speak?",
        "birthday": "When is my birthday?",
        "apartment": "Where do I currently live?",
        "workspace": "What is my home office setup like?",
    }
    for bio_key, bio_fact, bio_kw in _STABLE_BIOGRAPHICAL:
        q_text = _bio_questions.get(bio_key, f"What do you know about my {bio_key}?")
        _add(EvalQuestion(
            question=q_text,
            expected_keywords=[bio_kw],
            rejected_keywords=[],
            category="current_fact",
        ))

    # Fill to 30
    cf_count = sum(1 for q in questions if q.category == "current_fact")
    alt_idx = 0
    cf_alts = [
        "Can you remind me about my {topic}?",
        "What did I last say about {topic}?",
        "Tell me what you know about my {topic} situation.",
    ]
    while cf_count < 30:
        chain = chain_data[alt_idx % len(chain_data)]
        template = cf_alts[(alt_idx // len(chain_data)) % len(cf_alts)]
        q_text = template.replace("{topic}", chain["sub_topic"])
        terminal = chain["values"][-1]
        if _add(EvalQuestion(
            question=q_text,
            expected_keywords=[terminal["keyword"]],
            rejected_keywords=[v["keyword"] for v in chain["values"][:-1]],
            category="current_fact",
        )):
            cf_count += 1
        alt_idx += 1
        if alt_idx > len(chain_data) * len(cf_alts):
            break

    # --- Category 2: Graceful forgetting (20 questions) ---
    for key, q_text in _GRACEFUL_QUESTIONS_MULTI.items():
        matching = [c for c in chain_data
                    if (c["domain"], c["sub_topic"]) == key and len(c["values"]) >= 2]
        for chain in matching:
            original_kw = chain["values"][0]["keyword"]
            terminal_kw = chain["values"][-1]["keyword"]
            _add(EvalQuestion(
                question=q_text,
                expected_keywords=[original_kw, terminal_kw],
                rejected_keywords=[],
                category="graceful_forgetting",
            ))

    gf_count = sum(1 for q in questions if q.category == "graceful_forgetting")
    gf_alts = [
        "How has my {topic} evolved over time?",
        "What changes have happened with my {topic}?",
        "Tell me the history of my {topic} situation.",
        "Walk me through the progression of my {topic}.",
    ]
    gf_idx = 0
    multi_chains = [c for c in chain_data if len(c["values"]) >= 2]
    while gf_count < 20:
        chain = multi_chains[gf_idx % len(multi_chains)]
        template = gf_alts[(gf_idx // len(multi_chains)) % len(gf_alts)]
        q_text = template.replace("{topic}", chain["sub_topic"])
        if _add(EvalQuestion(
            question=q_text,
            expected_keywords=[
                chain["values"][0]["keyword"],
                chain["values"][-1]["keyword"],
            ],
            rejected_keywords=[],
            category="graceful_forgetting",
        )):
            gf_count += 1
        gf_idx += 1
        if gf_idx > len(multi_chains) * len(gf_alts):
            break

    # --- Category 3: Reinforced recall (25 questions) ---
    rr_templates = [
        "What do I frequently talk about regarding {domain}?",
        "What details have I shared about {domain}?",
        "What is a recurring topic in our conversations about {domain}?",
    ]
    for t_idx, topic in enumerate(_REINFORCEMENT_DEFS):
        template = rr_templates[t_idx % len(rr_templates)]
        q_text = template.replace("{domain}", topic["domain"].replace("_", " "))
        _add(EvalQuestion(
            question=q_text,
            expected_keywords=[topic["base_keyword"]],
            rejected_keywords=[],
            category="reinforced_recall",
        ))

    rr_count = sum(1 for q in questions if q.category == "reinforced_recall")
    rr_alts = [
        "What have I mentioned multiple times about {domain}?",
        "Summarize my ongoing activities in {domain}.",
        "What recurring theme comes up about {domain}?",
        "Tell me about my regular {domain} updates.",
    ]
    rr_idx = 0
    while rr_count < 25:
        topic = _REINFORCEMENT_DEFS[rr_idx % len(_REINFORCEMENT_DEFS)]
        template = rr_alts[(rr_idx // len(_REINFORCEMENT_DEFS)) % len(rr_alts)]
        q_text = template.replace("{domain}", topic["domain"].replace("_", " "))
        if _add(EvalQuestion(
            question=q_text,
            expected_keywords=[topic["base_keyword"]],
            rejected_keywords=[],
            category="reinforced_recall",
        )):
            rr_count += 1
        rr_idx += 1
        if rr_idx > len(_REINFORCEMENT_DEFS) * len(rr_alts):
            break

    # --- Category 4: Cross-domain inference (25 questions) ---
    for bdef in _BRIDGE_DEFS:
        for q_text, expected_kws in bdef["questions"]:
            _add(EvalQuestion(
                question=q_text,
                expected_keywords=expected_kws,
                rejected_keywords=[],
                category="cross_domain",
            ))

    cd_count = sum(1 for q in questions if q.category == "cross_domain")
    cd_alts = [
        "What connections exist between my {da} and {db} activities?",
        "How do my {da} and {db} interests overlap?",
    ]
    cd_idx = 0
    while cd_count < 25:
        bdef = _BRIDGE_DEFS[cd_idx % len(_BRIDGE_DEFS)]
        template = cd_alts[(cd_idx // len(_BRIDGE_DEFS)) % len(cd_alts)]
        q_text = template.replace(
            "{da}", bdef["domain_a"]
        ).replace(
            "{db}", bdef["domain_b"]
        )
        if _add(EvalQuestion(
            question=q_text,
            expected_keywords=[bdef["keyword_a"], bdef["keyword_b"]],
            rejected_keywords=[],
            category="cross_domain",
        )):
            cd_count += 1
        cd_idx += 1
        if cd_idx > len(_BRIDGE_DEFS) * len(cd_alts):
            break

    return LongitudinalDataset(
        sessions=all_sessions,
        questions=questions,
        seed=seed,
    )


# =====================================================================
# CLI entry point
# =====================================================================

if __name__ == "__main__":
    from longitudinal_eval import _print_dataset_summary

    ds = generate_multidomain_dataset(seed=42)
    _print_dataset_summary(ds)
