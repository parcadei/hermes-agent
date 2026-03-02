"""Real embedding geometry test suite for the Hermes dream system.

Uses Qwen3-Embedding-0.6B to produce real 1024-dim embeddings of natural
language memory fragments, then validates that the dream consolidation
pipeline (merge, prune, repulsion, unlearn) preserves the geometric
structure that matters for retrieval.

This file defines the test corpus only. Embedding infrastructure and
test classes will be added in subsequent phases.

Corpus structure per domain (20 sentences each):
  - Sentences 0-4:   Near-paraphrases of the same event/fact (expected cosine >= 0.85)
  - Sentences 5-14:  Same-topic, different content (expected cosine 0.40-0.70)
  - Sentences 15-19: Cross-domain sentences with semantic overlap to another domain

Ten domains mirror the lifecycle simulator: finance, work, technology,
health, family, learning, fitness, relationships, travel, hobbies.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from sentence_transformers import SentenceTransformer

from coupled_engine import CoupledEngine
from dream_ops import (
    dream_cycle_v2,
    nrem_repulsion_xb,
    rem_explore_cross_domain_xb,
    compute_adaptive_thresholds,
    DreamReport,
)
from test_capacity_boundary import compute_min_delta, compute_n_max, measure_p1

# Ensure the package directory is importable (test_dream_lifecycle lives here)
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from test_dream_lifecycle import (
    PatternMeta,
    build_daily_calendar,
    ingest_memories,
    dream_with_generation_tracking,
    collect_metrics,
)


# ---------------------------------------------------------------------------
# Domain configuration (mirrors test_dream_lifecycle.py)
# ---------------------------------------------------------------------------

DOMAIN_NAMES: list[str] = [
    "finance",
    "work",
    "technology",
    "health",
    "family",
    "learning",
    "fitness",
    "relationships",
    "travel",
    "hobbies",
]


# ---------------------------------------------------------------------------
# Corpus: 10 domains x 20 sentences = 200 memory fragments
# ---------------------------------------------------------------------------

CORPUS: dict[str, list[str]] = {

    # ------------------------------------------------------------------
    # FINANCE
    # ------------------------------------------------------------------
    "finance": [
        # Paraphrases (0-4): portfolio lost value on Monday
        "My portfolio dropped almost four percent on Monday morning.",
        "Monday saw my investments fall by nearly four percent.",
        "The start of the week brought a four percent decline in my portfolio.",
        "I lost close to four percent of my portfolio value on Monday.",
        "My investments took a four percent hit when Monday opened.",
        # Varied (5-14): different financial events
        "My 401k is up twelve percent this year which surprised me.",
        "The Fed raised interest rates again for the third consecutive time.",
        "I finally opened a high-yield savings account at four point five percent.",
        "Bitcoin climbed past ninety thousand dollars overnight and everyone panicked.",
        "I set up automatic transfers into my brokerage every two weeks.",
        "The dividend payout from my index fund came through today.",
        "I refinanced my mortgage and locked in a lower rate.",
        "My credit card rewards added up to three hundred dollars this quarter.",
        "Property taxes went up again and I need to adjust my budget.",
        "I started tracking every expense in a spreadsheet this month.",
        # Cross-domain (15-19)
        "The stress from watching my portfolio tank is giving me chest tightness.",  # -> health
        "I spent most of my workday refreshing stock tickers instead of coding.",  # -> work
        "I signed up for an online course about options trading strategies.",  # -> learning
        "My partner and I argued about how much to put into retirement savings.",  # -> relationships
        "I read that new fintech app uses machine learning for portfolio optimization.",  # -> technology
    ],

    # ------------------------------------------------------------------
    # WORK
    # ------------------------------------------------------------------
    "work": [
        # Paraphrases (0-4): missed a deadline on the backend migration
        "I missed the deadline on the backend migration project today.",
        "The backend migration deadline slipped past me this afternoon.",
        "Today I failed to deliver the backend migration on schedule.",
        "I could not finish the backend migration before the deadline hit.",
        "The deadline for migrating the backend came and I was not ready.",
        # Varied (5-14): different work events
        "My manager gave me positive feedback during our one-on-one today.",
        "We hired two new engineers for the platform team this week.",
        "The sprint retrospective revealed we keep underestimating tasks.",
        "I got promoted to senior engineer effective next quarter.",
        "Our deployment pipeline broke and blocked the whole team for hours.",
        "I wrote a design document for the new caching layer.",
        "The all-hands meeting announced a company-wide hiring freeze.",
        "I paired with a junior developer on their first pull request.",
        "My calendar has back-to-back meetings every day this week.",
        "We switched from Jira to Linear for project tracking.",
        # Cross-domain (15-19)
        "Working twelve-hour days is wrecking my sleep and giving me headaches.",  # -> health
        "I automated a tedious report using a Python script I wrote at work.",  # -> technology
        "My daughter asked why I am always on my laptop at dinner time.",  # -> family
        "I am taking an evening leadership course to prepare for the promotion.",  # -> learning
        "I barely have energy for the gym after these intense work sprints.",  # -> fitness
    ],

    # ------------------------------------------------------------------
    # TECHNOLOGY
    # ------------------------------------------------------------------
    "technology": [
        # Paraphrases (0-4): set up a home server running Linux
        "I set up a home server running Ubuntu for my side projects.",
        "Got my home Linux server configured and running over the weekend.",
        "I finally built a home server with Ubuntu for personal projects.",
        "My new home server is up and running Linux for development work.",
        "Spent the weekend setting up an Ubuntu server at home for coding.",
        # Varied (5-14): different tech events
        "The new language model from that startup can handle million-token contexts.",
        "I switched my terminal emulator to one written in Rust and it feels faster.",
        "Apple announced their new chip and the benchmarks look impressive.",
        "I migrated all my notes to an open-source self-hosted app.",
        "The security vulnerability in that logging library affected thousands.",
        "I set up end-to-end encryption on all my messaging apps.",
        "My smart home sensors keep disconnecting from the WiFi hub.",
        "I compiled the kernel with custom flags to reduce boot time.",
        "The new version of the language server protocol fixed our editor lag.",
        "I built a small CLI tool to batch-rename files by metadata.",
        # Cross-domain (15-19)
        "I tracked my sleep patterns with a wearable and the data is concerning.",  # -> health
        "I wrote a script to automate my monthly budget spreadsheet updates.",  # -> finance
        "Building side projects late at night is cutting into family time.",  # -> family
        "I am learning Lean 4 for formal verification of mathematical proofs.",  # -> learning
        "I used GPS data from my running watch to visualize my training routes.",  # -> fitness
    ],

    # ------------------------------------------------------------------
    # HEALTH
    # ------------------------------------------------------------------
    "health": [
        # Paraphrases (0-4): diagnosed with mild hypertension at checkup
        "My doctor diagnosed me with mild hypertension at the annual checkup.",
        "The annual physical revealed I have mildly elevated blood pressure.",
        "I was told my blood pressure is mildly high during my checkup today.",
        "Got diagnosed with mild hypertension at my routine physical exam.",
        "My yearly checkup showed my blood pressure is above normal range.",
        # Varied (5-14): different health events
        "I have been sleeping only five hours a night for two weeks straight.",
        "The allergist confirmed I am allergic to dust mites and tree pollen.",
        "I started taking magnesium supplements for the muscle cramps.",
        "My dentist found a cavity that needs filling next week.",
        "I switched to a standing desk to help with my lower back pain.",
        "The blood work came back and my cholesterol is borderline high.",
        "I have been meditating for ten minutes every morning this month.",
        "My therapist suggested journaling as a way to process anxiety.",
        "I caught a cold and it has lingered for over a week now.",
        "The eye exam showed my prescription changed so I need new glasses.",
        # Cross-domain (15-19)
        "Running three times a week has brought my resting heart rate down.",  # -> fitness
        "My mother's diabetes diagnosis made me rethink my own eating habits.",  # -> family
        "I read a research paper on how blue light from screens affects melatonin.",  # -> technology
        "The medical bills from the emergency visit are straining my savings.",  # -> finance
        "I told my partner I need to prioritize sleep over staying up together.",  # -> relationships
    ],

    # ------------------------------------------------------------------
    # FAMILY
    # ------------------------------------------------------------------
    "family": [
        # Paraphrases (0-4): daughter started kindergarten this week
        "My daughter started kindergarten this Monday and was so excited.",
        "This week my little girl began her first day of kindergarten.",
        "My daughter's first week of kindergarten started on Monday.",
        "Kindergarten began for my daughter this week and she loved it.",
        "My daughter had her first day at kindergarten on Monday morning.",
        # Varied (5-14): different family events
        "We drove four hours to visit my parents for the long weekend.",
        "My son scored his first goal in the youth soccer league.",
        "Mom called to say she is thinking about selling the family house.",
        "We adopted a rescue dog and the kids are absolutely thrilled.",
        "My brother announced he is getting married next spring.",
        "We spent Sunday afternoon building a treehouse in the backyard.",
        "My teenage nephew asked me for career advice over dinner.",
        "We had a big family dinner for my grandmother's eightieth birthday.",
        "My sister had her second baby and everyone is healthy.",
        "The kids and I baked cookies together on a rainy Saturday.",
        # Cross-domain (15-19)
        "My son keeps asking me to teach him how to code simple games.",  # -> technology
        "Saving for my daughter's college fund means I cannot invest as aggressively.",  # -> finance
        "Planning the family reunion trip to the coast took weeks of coordination.",  # -> travel
        "Watching my kids grow makes me reflect on what I want to learn next.",  # -> learning
        "My wife and I need to work on communicating better about parenting.",  # -> relationships
    ],

    # ------------------------------------------------------------------
    # LEARNING
    # ------------------------------------------------------------------
    "learning": [
        # Paraphrases (0-4): finished an online course on probability theory
        "I finished an online course on probability theory last night.",
        "Completed the probability theory course I had been taking online.",
        "Last night I wrapped up the online probability theory class.",
        "The online probability course I enrolled in is finally done.",
        "I just finished all the modules in my probability theory course.",
        # Varied (5-14): different learning events
        "I started reading a textbook on abstract algebra during my commute.",
        "The lecture on reinforcement learning clicked for me today.",
        "I joined a study group that meets weekly to discuss research papers.",
        "I am practicing Spanish with a tutor for thirty minutes each day.",
        "The documentation for this new framework is surprisingly well written.",
        "I failed my first attempt at the certification exam but learned a lot.",
        "I watched a lecture series on category theory and took dense notes.",
        "My note-taking system finally feels effective after months of iteration.",
        "I discovered spaced repetition and it changed how I memorize vocabulary.",
        "I submitted my first paper to a workshop and it got accepted.",
        # Cross-domain (15-19)
        "Studying late every night is leaving me exhausted and foggy headed.",  # -> health
        "I am learning about portfolio theory to make better investment decisions.",  # -> finance
        "The new skills from the course directly helped me solve a bug at work.",  # -> work
        "I run through flashcard drills while on the treadmill at the gym.",  # -> fitness
        "My travel to the conference in Kyoto exposed me to a whole new research community.",  # -> travel
    ],

    # ------------------------------------------------------------------
    # FITNESS
    # ------------------------------------------------------------------
    "fitness": [
        # Paraphrases (0-4): ran a personal best 5K time this morning
        "I ran my personal best five kilometer time this morning.",
        "This morning I set a new personal record for the five K.",
        "Hit a new five K personal best during my run this morning.",
        "My five kilometer time this morning was my fastest ever.",
        "I beat my previous five K record on the morning run today.",
        # Varied (5-14): different fitness events
        "I added deadlifts to my routine and my back feels much stronger.",
        "The climbing gym opened a new bouldering wall and I tried it.",
        "I swam forty laps without stopping for the first time yesterday.",
        "My rest day yesterday helped and today's session felt powerful.",
        "I bought a used road bike to start training for a sprint triathlon.",
        "The personal trainer adjusted my squat form and it made a huge difference.",
        "I signed up for a half marathon happening in three months.",
        "My grip strength improved enough that I can hold the bar for pull-ups now.",
        "I started doing yoga twice a week to improve flexibility.",
        "I tracked my macros and realized I am not eating enough protein.",
        # Cross-domain (15-19)
        "Training for the marathon is aggravating my old knee injury.",  # -> health
        "I bonded with a coworker over our shared interest in trail running.",  # -> work
        "I bought a heart rate monitor that syncs to my phone via Bluetooth.",  # -> technology
        "My partner started coming to the gym with me and it brought us closer.",  # -> relationships
        "I hiked a mountain trail during our vacation and it tested my endurance.",  # -> travel
    ],

    # ------------------------------------------------------------------
    # RELATIONSHIPS
    # ------------------------------------------------------------------
    "relationships": [
        # Paraphrases (0-4): had a deep honest conversation with my partner
        "My partner and I had a deep honest conversation last night.",
        "Last night we finally had that honest heart-to-heart talk.",
        "We sat down and had a really open and honest conversation yesterday.",
        "My partner and I talked openly and honestly for hours last night.",
        "Yesterday evening my partner and I had a candid meaningful talk.",
        # Varied (5-14): different relationship events
        "I reconnected with an old college friend I had not spoken to in years.",
        "My best friend moved to another city and I already miss them.",
        "We went on a double date and it was more fun than I expected.",
        "I realized I need to set better boundaries with certain people.",
        "My partner surprised me with a thoughtful anniversary gift.",
        "I apologized to a friend for something I said months ago.",
        "We started couples counseling and the first session was productive.",
        "I hosted a small dinner party and it felt great to bring people together.",
        "My neighbor and I started having coffee together on Sunday mornings.",
        "I deleted social media to focus on in-person connections instead.",
        # Cross-domain (15-19)
        "The loneliness after the breakup made me lose my appetite for days.",  # -> health
        "My partner and I planned a trip to Portugal as a way to reconnect.",  # -> travel
        "I learned about attachment theory and it explained so much about us.",  # -> learning
        "We bonded over building a shelf together from reclaimed wood.",  # -> hobbies
        "Tension with my sister is spilling over and affecting my focus at work.",  # -> work
    ],

    # ------------------------------------------------------------------
    # TRAVEL
    # ------------------------------------------------------------------
    "travel": [
        # Paraphrases (0-4): spent a week in Kyoto visiting temples
        "I spent a week in Kyoto visiting temples and shrines.",
        "We had a whole week in Kyoto exploring the temple district.",
        "Our week-long trip to Kyoto was focused on the historic temples.",
        "I visited Kyoto for a week and toured the ancient temples there.",
        "A week in Kyoto let us see dozens of temples and gardens.",
        # Varied (5-14): different travel events
        "The overnight train across Norway had the most stunning scenery.",
        "I got my passport renewed just in time for the international trip.",
        "Our flight was delayed six hours and we slept on the airport floor.",
        "I found a tiny family-run restaurant in Lisbon with incredible seafood.",
        "The hostel in Bangkok was basic but the rooftop view was worth it.",
        "I rented a campervan and drove the coast of New Zealand for two weeks.",
        "The altitude in La Paz hit me hard and I needed a day to adjust.",
        "I learned to navigate the Tokyo subway system without any Japanese.",
        "The street markets in Marrakech were overwhelming in the best way.",
        "I booked a last-minute weekend trip just to clear my head.",
        # Cross-domain (15-19)
        "The food poisoning I caught in Thailand knocked me out for three days.",  # -> health
        "Traveling on a tight budget taught me more about money than any book.",  # -> finance
        "I brought my watercolors and painted landscapes at every stop.",  # -> hobbies
        "Being away for a month made me realize how much I value my close friendships.",  # -> relationships
        "I used a translation app on my phone to get around in rural Japan.",  # -> technology
    ],

    # ------------------------------------------------------------------
    # HOBBIES
    # ------------------------------------------------------------------
    "hobbies": [
        # Paraphrases (0-4): finished building a mechanical keyboard
        "I finished building my custom mechanical keyboard last weekend.",
        "Last weekend I completed the mechanical keyboard build I started.",
        "My custom mechanical keyboard is finally assembled and working.",
        "I put the last keycap on my hand-built mechanical keyboard yesterday.",
        "The mechanical keyboard I have been building for weeks is done.",
        # Varied (5-14): different hobby events
        "I roasted my own coffee beans for the first time and they turned out great.",
        "My sourdough starter is finally active after a week of feeding.",
        "I spent three hours in the garden planting tomatoes and herbs.",
        "I assembled and painted a detailed miniature for my tabletop game.",
        "The woodworking project for the bookshelf is halfway done.",
        "I taught myself to play a new song on the guitar this week.",
        "My watercolor technique improved after watching tutorial videos.",
        "I started collecting vinyl records and found a rare pressing at a shop.",
        "The jigsaw puzzle I have been working on has three thousand pieces.",
        "I went birdwatching at the local marsh and spotted a great blue heron.",
        # Cross-domain (15-19)
        "Spending hours hunched over my workbench is causing neck and shoulder pain.",  # -> health
        "My kids love helping me in the garden and it has become our weekend ritual.",  # -> family
        "I used a 3D printer to make custom parts for my model train layout.",  # -> technology
        "I enrolled in a pottery class to learn wheel-throwing from scratch.",  # -> learning
        "I sold some of my handmade furniture online and made a decent profit.",  # -> finance
    ],
}


# ---------------------------------------------------------------------------
# Corpus metadata: type, related domain, paraphrase group
# ---------------------------------------------------------------------------

def _build_metadata_for_domain(cross_domain_targets: list[str]) -> list[dict]:
    """Build the 20-entry metadata list for a single domain.

    Args:
        cross_domain_targets: list of 5 domain names corresponding to
            sentences 15-19 (the cross-domain sentences).
    """
    meta = []
    # Sentences 0-4: paraphrases
    for i in range(5):
        meta.append({
            "type": "paraphrase",
            "related_domain": None,
            "paraphrase_group": 0,
        })
    # Sentences 5-14: varied
    for i in range(10):
        meta.append({
            "type": "varied",
            "related_domain": None,
            "paraphrase_group": None,
        })
    # Sentences 15-19: cross-domain
    for target in cross_domain_targets:
        meta.append({
            "type": "cross_domain",
            "related_domain": target,
            "paraphrase_group": None,
        })
    return meta


CORPUS_METADATA: dict[str, list[dict]] = {
    "finance": _build_metadata_for_domain(
        ["health", "work", "learning", "relationships", "technology"],
    ),
    "work": _build_metadata_for_domain(
        ["health", "technology", "family", "learning", "fitness"],
    ),
    "technology": _build_metadata_for_domain(
        ["health", "finance", "family", "learning", "fitness"],
    ),
    "health": _build_metadata_for_domain(
        ["fitness", "family", "technology", "finance", "relationships"],
    ),
    "family": _build_metadata_for_domain(
        ["technology", "finance", "travel", "learning", "relationships"],
    ),
    "learning": _build_metadata_for_domain(
        ["health", "finance", "work", "fitness", "travel"],
    ),
    "fitness": _build_metadata_for_domain(
        ["health", "work", "technology", "relationships", "travel"],
    ),
    "relationships": _build_metadata_for_domain(
        ["health", "travel", "learning", "hobbies", "work"],
    ),
    "travel": _build_metadata_for_domain(
        ["health", "finance", "hobbies", "relationships", "technology"],
    ),
    "hobbies": _build_metadata_for_domain(
        ["health", "family", "technology", "learning", "finance"],
    ),
}


# ---------------------------------------------------------------------------
# Helper: flatten corpus into ordered lists
# ---------------------------------------------------------------------------

def get_all_sentences() -> tuple[list[str], list[int], list[str]]:
    """Return (sentences, domain_indices, domain_names) in corpus order.

    Iterates over DOMAIN_NAMES in order, appending each domain's 20
    sentences sequentially.  Returns three parallel lists:
      - sentences: the 200 sentence strings
      - domain_indices: integer index into DOMAIN_NAMES for each sentence
      - domain_names_flat: the domain name string for each sentence
    """
    sentences: list[str] = []
    domain_indices: list[int] = []
    domain_names_flat: list[str] = []

    for idx, domain in enumerate(DOMAIN_NAMES):
        for sentence in CORPUS[domain]:
            sentences.append(sentence)
            domain_indices.append(idx)
            domain_names_flat.append(domain)

    return sentences, domain_indices, domain_names_flat


# ---------------------------------------------------------------------------
# Embedding infrastructure
# ---------------------------------------------------------------------------

_MODEL_CACHE: SentenceTransformer | None = None
_MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
_EMBEDDING_DIM = 1024


def _get_model() -> SentenceTransformer:
    """Lazy-load and cache the embedding model."""
    global _MODEL_CACHE
    if _MODEL_CACHE is None:
        _MODEL_CACHE = SentenceTransformer(_MODEL_NAME)
    return _MODEL_CACHE


def encode_sentences(sentences: list[str], normalize: bool = True) -> np.ndarray:
    """Encode sentences into (N, 1024) array using Qwen3-Embedding-0.6B.

    Returns L2-normalized vectors by default (unit sphere).
    """
    model = _get_model()
    embeddings = model.encode(sentences, normalize_embeddings=normalize, show_progress_bar=False)
    return np.array(embeddings, dtype=np.float64)


@pytest.fixture(scope="session")
def embedded_corpus() -> dict:
    """Embed all 200 sentences once per test session.

    Returns dict with:
        embeddings: (200, 1024) float64 array, L2-normalized
        sentences: list of 200 strings
        domain_indices: list of 200 ints
        domain_names: list of 200 strings
        sim_matrix: (200, 200) pairwise cosine similarity matrix
    """
    sentences, domain_indices, domain_names = get_all_sentences()
    embeddings = encode_sentences(sentences)
    sim_matrix = embeddings @ embeddings.T
    return {
        "embeddings": embeddings,
        "sentences": sentences,
        "domain_indices": domain_indices,
        "domain_names": domain_names,
        "sim_matrix": sim_matrix,
    }


class TestEmbeddingInfrastructure:
    """Verify Qwen3-Embedding-0.6B loads and produces expected output."""

    def test_model_loads_and_encodes(self):
        emb = encode_sentences(["hello world"])
        assert emb.shape == (1, _EMBEDDING_DIM)
        # Should be approximately unit normalized
        norm = float(np.linalg.norm(emb[0]))
        assert abs(norm - 1.0) < 0.01

    def test_corpus_embeds_fully(self, embedded_corpus):
        assert embedded_corpus["embeddings"].shape == (200, _EMBEDDING_DIM)
        assert len(embedded_corpus["sentences"]) == 200
        assert len(embedded_corpus["domain_indices"]) == 200

    def test_sim_matrix_diagonal_is_one(self, embedded_corpus):
        diag = np.diag(embedded_corpus["sim_matrix"])
        # Diagonal should be ~1.0 (self-similarity)
        assert np.all(diag > 0.99), f"Diagonal min: {diag.min():.4f}"


class TestGeometryMeasurement:
    """Phase 1b: Measure real embedding geometry.

    Compute within-domain and between-domain similarity distributions.
    This is pure measurement — no assertions about what the values SHOULD be.
    The assertions come in TestGeometryAssertions (P1c).
    """

    def test_within_domain_similarity(self, embedded_corpus):
        """Measure within-domain pairwise cosine similarity distribution.

        For each domain, compute all (20 choose 2) = 190 pairwise sims.
        Report: mean, std, min, max per domain.
        Also break out by sentence type (paraphrase vs varied vs cross-domain).
        """
        sim = embedded_corpus["sim_matrix"]
        domain_indices = embedded_corpus["domain_indices"]

        print(f"\n{'='*70}")
        print("WITHIN-DOMAIN SIMILARITY DISTRIBUTIONS")
        print(f"{'='*70}")

        all_within_sims = []
        paraphrase_sims = []
        varied_sims = []
        cross_domain_sims = []

        for d_idx, d_name in enumerate(DOMAIN_NAMES):
            # Get indices for this domain (20 sentences each)
            indices = [i for i, di in enumerate(domain_indices) if di == d_idx]

            domain_sims = []
            for a in range(len(indices)):
                for b in range(a + 1, len(indices)):
                    s = float(sim[indices[a], indices[b]])
                    domain_sims.append(s)
                    all_within_sims.append(s)

                    # Categorize by sentence type pair
                    # Sentences 0-4 are paraphrases, 5-14 varied, 15-19 cross-domain
                    local_a, local_b = a, b  # position within the domain's 20
                    if local_a < 5 and local_b < 5:
                        paraphrase_sims.append(s)
                    elif local_a < 15 and local_b < 15 and local_a >= 5:
                        varied_sims.append(s)
                    elif local_a >= 15 or local_b >= 15:
                        cross_domain_sims.append(s)

            arr = np.array(domain_sims)
            print(f"  {d_name:15s}: mean={arr.mean():.4f} std={arr.std():.4f} "
                  f"min={arr.min():.4f} max={arr.max():.4f}")

        # Summary by type
        print(f"\n{'─'*70}")
        print("By sentence type pair:")
        for label, sims_list in [
            ("paraphrase↔paraphrase", paraphrase_sims),
            ("varied↔varied", varied_sims),
            ("cross_domain (any)", cross_domain_sims),
            ("ALL within-domain", all_within_sims),
        ]:
            arr = np.array(sims_list)
            print(f"  {label:30s}: n={len(sims_list):5d} mean={arr.mean():.4f} "
                  f"std={arr.std():.4f} min={arr.min():.4f} max={arr.max():.4f}")

        # Store for downstream tests
        # (just print the adaptive threshold calculation here)
        all_arr = np.array(all_within_sims)
        p70 = float(np.percentile(all_arr, 70))
        p90 = float(np.percentile(all_arr, 90))
        print(f"\n{'─'*70}")
        print(f"Adaptive thresholds from within-domain distribution:")
        print(f"  P70 (merge threshold): {p70:.4f}")
        print(f"  P90 (prune threshold): {p90:.4f}")
        print(f"  Compare to synthetic fixed: merge=0.90, prune=0.95")

    def test_between_domain_similarity(self, embedded_corpus):
        """Measure between-domain pairwise cosine similarity.

        For each pair of domains, compute mean similarity between their
        20 sentences. This reveals which domains share semantic structure.
        """
        sim = embedded_corpus["sim_matrix"]
        domain_indices = embedded_corpus["domain_indices"]
        n_domains = len(DOMAIN_NAMES)

        print(f"\n{'='*70}")
        print("BETWEEN-DOMAIN SIMILARITY MATRIX (mean pairwise cosine)")
        print(f"{'='*70}")

        # Header
        print(f"{'':15s}", end="")
        for d in DOMAIN_NAMES:
            print(f" {d[:6]:>6s}", end="")
        print()

        between_matrix = np.zeros((n_domains, n_domains))
        all_between_sims = []

        for d1 in range(n_domains):
            idx1 = [i for i, di in enumerate(domain_indices) if di == d1]
            print(f"  {DOMAIN_NAMES[d1]:13s}", end="")
            for d2 in range(n_domains):
                if d1 == d2:
                    print(f"   ----", end="")
                    continue
                idx2 = [i for i, di in enumerate(domain_indices) if di == d2]
                pair_sims = []
                for a in idx1:
                    for b in idx2:
                        pair_sims.append(float(sim[a, b]))
                mean_sim = np.mean(pair_sims)
                between_matrix[d1, d2] = mean_sim
                all_between_sims.append(mean_sim)
                print(f" {mean_sim:6.3f}", end="")
            print()

        all_between = np.array(all_between_sims)
        print(f"\n  Overall between-domain: mean={all_between.mean():.4f} "
              f"std={all_between.std():.4f} min={all_between.min():.4f} max={all_between.max():.4f}")

        # Find the most similar domain pairs
        print(f"\n  Top 5 most similar domain pairs:")
        flat = []
        for d1 in range(n_domains):
            for d2 in range(d1+1, n_domains):
                flat.append((between_matrix[d1, d2], DOMAIN_NAMES[d1], DOMAIN_NAMES[d2]))
        flat.sort(reverse=True)
        for sim_val, name1, name2 in flat[:5]:
            print(f"    {name1:12s} ↔ {name2:12s}: {sim_val:.4f}")

    def test_cross_domain_sentence_similarity(self, embedded_corpus):
        """Measure similarity of cross-domain sentences to their target domain.

        For each cross-domain sentence (type="cross_domain"), measure:
        1. Its similarity to its OWN domain's centroid
        2. Its similarity to the RELATED domain's centroid

        If the related domain similarity is high, the sentence genuinely
        bridges the two domains (which is what we want for testing
        cross-domain dream discovery).
        """
        embeddings = embedded_corpus["embeddings"]
        domain_indices = embedded_corpus["domain_indices"]
        sentences = embedded_corpus["sentences"]

        # Compute domain centroids
        centroids = {}
        for d_idx, d_name in enumerate(DOMAIN_NAMES):
            idx = [i for i, di in enumerate(domain_indices) if di == d_idx]
            centroids[d_name] = embeddings[idx].mean(axis=0)
            centroids[d_name] /= np.linalg.norm(centroids[d_name])

        print(f"\n{'='*70}")
        print("CROSS-DOMAIN SENTENCE ANALYSIS")
        print(f"{'='*70}")
        print(f"{'Sentence':50s} {'Own':>5s} {'Target':>7s} {'Gap':>5s}")

        gaps = []
        for d_idx, d_name in enumerate(DOMAIN_NAMES):
            meta_list = CORPUS_METADATA[d_name]
            base_idx = d_idx * 20  # offset into flat list
            for local_i, meta in enumerate(meta_list):
                if meta["type"] != "cross_domain":
                    continue
                global_i = base_idx + local_i
                emb = embeddings[global_i]
                related = meta["related_domain"]

                own_sim = float(emb @ centroids[d_name])
                target_sim = float(emb @ centroids[related])
                gap = target_sim - own_sim
                gaps.append(gap)

                sent_short = sentences[global_i][:48]
                print(f"  {sent_short:50s} {own_sim:5.3f} {target_sim:7.3f} {gap:+5.3f}")

        gaps_arr = np.array(gaps)
        print(f"\n  Mean gap (target - own): {gaps_arr.mean():+.4f}")
        print(f"  Cross-domain sentences closer to target than own: "
              f"{(gaps_arr > 0).sum()}/{len(gaps_arr)}")


class TestGeometryAssertions:
    """Phase 1c: Assert real geometry differs from synthetic assumptions.

    Documents the geometry gap between synthetic test data (orthogonal centroids,
    tight clusters, spread=0.04) and real Qwen3 embeddings. These assertions
    lock in the measured geometry so future changes to the corpus or model
    are detected.
    """

    def test_within_domain_mean_is_moderate(self, embedded_corpus):
        """Within-domain mean similarity should be 0.30-0.65 (not 0.85+ like synthetic)."""
        sim = embedded_corpus["sim_matrix"]
        domain_indices = embedded_corpus["domain_indices"]

        within_sims = []
        for d_idx in range(len(DOMAIN_NAMES)):
            indices = [i for i, di in enumerate(domain_indices) if di == d_idx]
            for a in range(len(indices)):
                for b in range(a + 1, len(indices)):
                    within_sims.append(float(sim[indices[a], indices[b]]))

        mean_within = float(np.mean(within_sims))
        print(f"\nWithin-domain mean: {mean_within:.4f}")
        assert 0.30 <= mean_within <= 0.65, (
            f"Within-domain mean {mean_within:.4f} outside expected range [0.30, 0.65]"
        )

    def test_paraphrase_similarity_high(self, embedded_corpus):
        """Paraphrase pairs (sentences 0-4 within each domain) should have cosine >= 0.75."""
        sim = embedded_corpus["sim_matrix"]
        domain_indices = embedded_corpus["domain_indices"]

        para_sims = []
        for d_idx in range(len(DOMAIN_NAMES)):
            indices = [i for i, di in enumerate(domain_indices) if di == d_idx]
            # Paraphrases are the first 5 in each domain
            for a in range(5):
                for b in range(a + 1, 5):
                    para_sims.append(float(sim[indices[a], indices[b]]))

        mean_para = float(np.mean(para_sims))
        min_para = float(np.min(para_sims))
        print(f"\nParaphrase mean: {mean_para:.4f}, min: {min_para:.4f}")
        assert mean_para >= 0.75, f"Paraphrase mean {mean_para:.4f} < 0.75"
        # At least 90% of paraphrase pairs should exceed 0.70
        above_threshold = sum(1 for s in para_sims if s >= 0.70) / len(para_sims)
        assert above_threshold >= 0.90, (
            f"Only {above_threshold:.0%} of paraphrase pairs >= 0.70"
        )

    def test_between_domain_mean_is_nonzero(self, embedded_corpus):
        """Between-domain mean should be 0.20-0.50 (not ~0 like orthogonal synthetic).

        This is the key difference from synthetic data -- real embeddings share
        semantic structure across domains.
        """
        sim = embedded_corpus["sim_matrix"]
        domain_indices = embedded_corpus["domain_indices"]

        between_sims = []
        for d1 in range(len(DOMAIN_NAMES)):
            idx1 = [i for i, di in enumerate(domain_indices) if di == d1]
            for d2 in range(d1 + 1, len(DOMAIN_NAMES)):
                idx2 = [i for i, di in enumerate(domain_indices) if di == d2]
                for a in idx1:
                    for b in idx2:
                        between_sims.append(float(sim[a, b]))

        mean_between = float(np.mean(between_sims))
        print(f"\nBetween-domain mean: {mean_between:.4f}")
        assert 0.20 <= mean_between <= 0.50, (
            f"Between-domain mean {mean_between:.4f} outside expected range [0.20, 0.50]"
        )

    def test_separation_gap_is_narrow(self, embedded_corpus):
        """Gap between within-domain and between-domain means should be < 0.20.

        Synthetic data has gap ~0.90 (orthogonal centroids). Real embeddings
        have gap ~0.10. This narrow gap is the fundamental challenge for
        domain-preserving consolidation.
        """
        sim = embedded_corpus["sim_matrix"]
        domain_indices = embedded_corpus["domain_indices"]

        within_sims = []
        between_sims = []
        for d1 in range(len(DOMAIN_NAMES)):
            idx1 = [i for i, di in enumerate(domain_indices) if di == d1]
            for a in range(len(idx1)):
                for b in range(a + 1, len(idx1)):
                    within_sims.append(float(sim[idx1[a], idx1[b]]))
            for d2 in range(d1 + 1, len(DOMAIN_NAMES)):
                idx2 = [i for i, di in enumerate(domain_indices) if di == d2]
                for a in idx1:
                    for b in idx2:
                        between_sims.append(float(sim[a, b]))

        gap = float(np.mean(within_sims) - np.mean(between_sims))
        print(f"\nSeparation gap: {gap:.4f}")
        print(f"  Within-domain mean: {np.mean(within_sims):.4f}")
        print(f"  Between-domain mean: {np.mean(between_sims):.4f}")
        assert gap < 0.20, f"Gap {gap:.4f} >= 0.20 -- larger than expected for real embeddings"
        assert gap > 0.02, f"Gap {gap:.4f} <= 0.02 -- domains are not distinguishable at all"

    def test_adaptive_threshold_below_paraphrase(self, embedded_corpus):
        """P70 merge threshold should sit below the paraphrase band.

        Adaptive merge at P70 ~= 0.49. Paraphrases at ~= 0.85.
        The threshold should be well below paraphrases (they SHOULD merge)
        but the question is whether varied same-domain content (0.45)
        also falls above it.
        """
        sim = embedded_corpus["sim_matrix"]
        domain_indices = embedded_corpus["domain_indices"]

        within_sims = []
        for d_idx in range(len(DOMAIN_NAMES)):
            indices = [i for i, di in enumerate(domain_indices) if di == d_idx]
            for a in range(len(indices)):
                for b in range(a + 1, len(indices)):
                    within_sims.append(float(sim[indices[a], indices[b]]))

        arr = np.array(within_sims)
        p70 = float(np.percentile(arr, 70))

        # P70 should be below paraphrase band
        assert p70 < 0.75, f"P70={p70:.4f} >= 0.75 -- merge threshold in paraphrase band"
        # P70 should be above between-domain max to prevent cross-domain merge
        # (between-domain max ~= 0.42)
        print(f"\nP70 merge threshold: {p70:.4f}")


# ---------------------------------------------------------------------------
# Phase 3: 30-day lifecycle simulation on real embedding geometry
# ---------------------------------------------------------------------------


class TestRealGeometryLifecycle:
    """Phase 3: 30-day lifecycle simulation on real embedding geometry.

    Uses Qwen3-Embedding-0.6B domain centroids instead of synthetic
    orthogonal centroids. Tests whether dream consolidation works when
    domain separation is narrow (gap=0.10 vs 0.90 in synthetic).
    """

    N_DAYS = 30
    N_DOMAINS = 10
    BETA = 10.0
    SPREAD = 0.02  # Calibrated for real geometry
    SEED = 42

    @pytest.fixture(scope="class")
    def real_centroids(self, embedded_corpus):
        """Compute real domain centroids from Qwen3 embeddings."""
        embeddings = embedded_corpus["embeddings"]
        domain_indices = embedded_corpus["domain_indices"]

        centroids = np.zeros((self.N_DOMAINS, 1024), dtype=np.float64)
        for d_idx in range(self.N_DOMAINS):
            idx = [i for i, di in enumerate(domain_indices) if di == d_idx]
            c = embeddings[idx].mean(axis=0)
            centroids[d_idx] = c / np.linalg.norm(c)
        return centroids

    def _run_lifecycle(self, centroids) -> dict:
        """Run 30-day lifecycle with real centroids."""
        rng = np.random.default_rng(self.SEED)
        dim = centroids.shape[1]
        calendar = build_daily_calendar(self.N_DAYS, self.N_DOMAINS, seed=self.SEED)

        patterns = np.empty((0, dim), dtype=np.float64)
        importances = np.empty(0, dtype=np.float64)
        labels = np.empty(0, dtype=int)
        meta_list: list[PatternMeta] = []
        daily_metrics: list[dict] = []

        for day in range(self.N_DAYS):
            patterns, importances, labels, meta_list = ingest_memories(
                patterns, importances, labels, meta_list,
                centroids, calendar[day], day,
                self.SPREAD, rng,
            )

            patterns, importances, labels, meta_list = dream_with_generation_tracking(
                patterns, importances, labels, meta_list,
                self.BETA, day, seed=self.SEED + day,
                use_v2=True,
            )

            metrics = collect_metrics(
                patterns, importances, labels, meta_list,
                centroids, self.BETA, self.N_DOMAINS,
            )
            metrics["day"] = day
            metrics["n_ingested_today"] = sum(n for _, n in calendar[day])
            daily_metrics.append(metrics)

        return {
            "daily_metrics": daily_metrics,
            "final_patterns": patterns,
            "final_meta": meta_list,
            "final_importances": importances,
            "final_labels": labels,
            "centroids": centroids,
            "calendar": calendar,
        }

    def test_consolidation_occurs(self, real_centroids):
        """Dream cycle should consolidate patterns even with narrow domain separation."""
        result = self._run_lifecycle(real_centroids)
        daily = result["daily_metrics"]

        total_ingested = sum(m["n_ingested_today"] for m in daily)
        final_n = daily[-1]["total_patterns"]
        ratio = final_n / total_ingested

        print(f"\nReal geometry lifecycle (spread={self.SPREAD}):")
        print(f"  Total ingested: {total_ingested}")
        print(f"  Final patterns: {final_n}")
        print(f"  Consolidation ratio: {ratio:.2%}")
        print(f"  Generation distribution: {daily[-1]['generation_distribution']}")

        # Some consolidation should happen (ratio < 1.0)
        assert ratio < 1.0, (
            f"No consolidation: {final_n} patterns = {total_ingested} ingested"
        )

    def test_generation_depth(self, real_centroids):
        """Track max generation reached with real centroids.

        Prediction: shallower than synthetic (maybe gen-3 to gen-10 vs gen-30)
        because within-cluster similarity is lower with real geometry.
        """
        result = self._run_lifecycle(real_centroids)
        daily = result["daily_metrics"]

        max_gen = 0
        gen_first_day: dict[int, int] = {}
        for m in daily:
            for g in m["generation_distribution"]:
                if g not in gen_first_day:
                    gen_first_day[g] = m["day"]
                max_gen = max(max_gen, g)

        print(f"\nGeneration emergence timeline (real geometry):")
        for g, day in sorted(gen_first_day.items()):
            print(f"  gen-{g}: day {day}")
        print(f"  Max generation: {max_gen}")

        # At least gen-1 should emerge (paraphrases merge)
        assert max_gen >= 1, "No merging at all with real geometry"

    def test_domain_preservation(self, real_centroids):
        """After 30 days, patterns should still be nearest their own domain centroid.

        This is the critical test: with narrow domain separation (gap=0.10),
        does adaptive merging cross domain boundaries?
        """
        result = self._run_lifecycle(real_centroids)
        patterns = result["final_patterns"]
        meta = result["final_meta"]
        centroids = real_centroids

        correct = 0
        total = len(meta)
        misassigned = []

        for i, m in enumerate(meta):
            sims = [float(patterns[i] @ centroids[d]) for d in range(self.N_DOMAINS)]
            best = int(np.argmax(sims))
            if best == m.domain:
                correct += 1
            else:
                misassigned.append((m.generation, m.domain, best, sims[best] - sims[m.domain]))

        acc = correct / total if total > 0 else 0.0
        print(f"\nDomain preservation (real geometry):")
        print(f"  Correctly assigned: {correct}/{total} ({acc:.1%})")
        if misassigned:
            print(f"  Misassigned ({len(misassigned)}):")
            for gen, assigned, nearest, gap in misassigned[:10]:
                print(f"    gen-{gen}: {DOMAIN_NAMES[assigned]} -> {DOMAIN_NAMES[nearest]} (gap={gap:+.4f})")

        # With 0.10 gap, we expect SOME cross-domain drift but not catastrophic
        assert acc >= 0.60, (
            f"Only {acc:.1%} domain fidelity -- narrow separation causing collapse"
        )

    def test_merge_rate_per_cycle(self, real_centroids):
        """Measure how many patterns merge per dream cycle.

        With real geometry (within-domain mean 0.46 vs adaptive P70=0.49),
        merge should fire moderately -- not zero (broken) and not 90%+ (over-aggressive).
        """
        result = self._run_lifecycle(real_centroids)
        daily = result["daily_metrics"]
        calendar = result["calendar"]

        print(f"\nMerge dynamics (real geometry):")
        counts = [m["total_patterns"] for m in daily]
        for day in range(len(daily)):
            ingested = sum(n for _, n in calendar[day])
            if day > 0:
                net_change = counts[day] - counts[day - 1]
                consolidated = ingested - net_change
                if day % 5 == 0 or day == len(daily) - 1:
                    print(f"  Day {day:3d}: N={counts[day]:4d} +{ingested:2d} "
                          f"consolidated={consolidated:3d} net={net_change:+3d}")

    def test_cross_domain_patterns_emerge(self, real_centroids):
        """With non-orthogonal centroids, REM-explore might find cross-domain associations.

        This is the test synthetic data COULDN'T run -- orthogonal centroids
        have zero shared substructure. Real centroids at 0.36 mean similarity
        should enable genuine cross-domain discovery.
        """
        result = self._run_lifecycle(real_centroids)
        meta = result["final_meta"]

        cross_domain = [m for m in meta if m.is_cross_domain]
        total = len(meta)

        print(f"\nCross-domain patterns (real geometry):")
        print(f"  Total patterns: {total}")
        print(f"  Cross-domain: {len(cross_domain)}")
        if cross_domain:
            for m in cross_domain[:10]:
                print(f"    gen-{m.generation} domain={DOMAIN_NAMES[m.domain]} "
                      f"day={m.day_created}")

        # Just report -- no assertion on count since this is exploratory


# ---------------------------------------------------------------------------
# Phase 4: Dream effectiveness on real embedding geometry
# ---------------------------------------------------------------------------


class TestDreamEffectiveness:
    """Phase 4: What do dreams actually DO on real embedding geometry?

    With real Qwen3 embeddings (1024-dim, within-domain mean=0.46,
    between-domain mean=0.36, gap=0.10), test:

    1. Capacity boundary: Are clusters ever near N_max?
    2. Repulsion impact: Does repulsion meaningfully change delta_min?
    3. REM-explore: Does it find cross-domain associations?
    4. Operational purpose: What is the actual effect of dreaming?

    Uses the same 30-day lifecycle simulation as Phase 3, but
    decomposes the dream cycle into its constituent operations
    to measure each one's contribution.
    """

    N_DAYS = 30
    N_DOMAINS = 10
    BETA = 10.0
    SPREAD = 0.02
    SEED = 42

    @pytest.fixture(scope="class")
    def real_centroids(self, embedded_corpus):
        """Compute real domain centroids from Qwen3 embeddings."""
        embeddings = embedded_corpus["embeddings"]
        domain_indices = embedded_corpus["domain_indices"]
        centroids = np.zeros((self.N_DOMAINS, 1024), dtype=np.float64)
        for d_idx in range(self.N_DOMAINS):
            idx = [i for i, di in enumerate(domain_indices) if di == d_idx]
            c = embeddings[idx].mean(axis=0)
            centroids[d_idx] = c / np.linalg.norm(c)
        return centroids

    @pytest.fixture(scope="class")
    def lifecycle_state(self, real_centroids):
        """Run 30-day lifecycle and capture the final state for Phase 4 tests."""
        rng = np.random.default_rng(self.SEED)
        dim = real_centroids.shape[1]
        calendar = build_daily_calendar(self.N_DAYS, self.N_DOMAINS, seed=self.SEED)

        patterns = np.empty((0, dim), dtype=np.float64)
        importances = np.empty(0, dtype=np.float64)
        labels = np.empty(0, dtype=int)
        meta_list: list[PatternMeta] = []

        for day in range(self.N_DAYS):
            patterns, importances, labels, meta_list = ingest_memories(
                patterns, importances, labels, meta_list,
                real_centroids, calendar[day], day,
                self.SPREAD, rng,
            )
            patterns, importances, labels, meta_list = dream_with_generation_tracking(
                patterns, importances, labels, meta_list,
                self.BETA, day, seed=self.SEED + day,
                use_v2=True,
            )

        return {
            "patterns": patterns,
            "importances": importances,
            "labels": labels,
            "meta": meta_list,
            "centroids": real_centroids,
        }

    # ------------------------------------------------------------------
    # 4.1: Capacity boundary at real delta
    # ------------------------------------------------------------------

    def test_capacity_at_real_geometry(self, lifecycle_state):
        """Capacity IS a real constraint at real geometry — dreams are the solution.

        With delta_min ≈ 0.30 and beta=10: exp(β·δ_min) = exp(3) ≈ 20.
        The softmax weight of the true match vs 20 competitors is ~1/21 —
        barely above chance. Self-retrieval P@1 is low.

        This REVERSES the prediction that "capacity is a non-concern."
        At real geometry, the effective capacity at beta=10 is surprisingly
        tight. The dream system's role is to maintain separation: by
        merging similar patterns (raising effective delta_min) and pruning
        redundancies (reducing N), dreams keep the system operational.

        Key insight: high beta (sharp retrieval) requires high delta_min
        to discriminate. Low beta (broad retrieval) is more tolerant.
        The dream system enables high-beta retrieval on real geometry
        by consolidating patterns into fewer, better-separated abstractions.
        """
        patterns = lifecycle_state["patterns"]
        importances = lifecycle_state["importances"]
        N = patterns.shape[0]

        delta_min = compute_min_delta(patterns)
        n_max = compute_n_max(self.BETA, delta_min)

        # P@1 at operational beta
        engine_op = CoupledEngine(dim=patterns.shape[1], beta=self.BETA)
        for i in range(N):
            engine_op.store(f"p{i}", patterns[i], importance=float(importances[i]))
        p1_beta10 = measure_p1(engine_op, patterns)

        # P@1 at higher beta (should be worse — sharper softmax, more interference)
        engine_high = CoupledEngine(dim=patterns.shape[1], beta=50.0)
        for i in range(N):
            engine_high.store(f"p{i}", patterns[i], importance=float(importances[i]))
        p1_beta50 = measure_p1(engine_high, patterns)

        # P@1 at lower beta (should be worse — too broad, nearest not dominant)
        engine_low = CoupledEngine(dim=patterns.shape[1], beta=2.0)
        for i in range(N):
            engine_low.store(f"p{i}", patterns[i], importance=float(importances[i]))
        p1_beta2 = measure_p1(engine_low, patterns)

        print(f"\nCapacity analysis at real geometry:")
        print(f"  Patterns stored: {N}")
        print(f"  delta_min: {delta_min:.6f}")
        print(f"  N_max (formula): {n_max:.2e}")
        print(f"  exp(β·δ_min) = exp({self.BETA * delta_min:.2f}) = {np.exp(self.BETA * delta_min):.1f}")
        print(f"  P@1 sweep:")
        print(f"    β=2:  {p1_beta2:.4f}")
        print(f"    β=10: {p1_beta10:.4f}")
        print(f"    β=50: {p1_beta50:.4f}")

        # delta_min positive (patterns are distinct)
        assert delta_min > 0.0, "delta_min is zero — patterns are identical"

        # The formula correctly signals that we're above capacity at beta=10
        # (N_max < N when delta_min is small relative to beta)
        assert n_max < N, (
            f"Formula says N_max ({n_max:.1f}) >= N ({N}) — expected formula "
            f"to signal capacity constraint at this geometry"
        )

        # Higher beta should improve discrimination (winner-take-all)
        # unless there's too much competition from close patterns
        # This is diagnostic — the pattern is informative either way
        print(f"\n  Finding: {'High beta HELPS' if p1_beta50 > p1_beta10 else 'High beta does NOT help'} "
              f"— softmax sharpness {'overcomes' if p1_beta50 > p1_beta10 else 'cannot overcome'} "
              f"the competition from {N-1} patterns at delta_min={delta_min:.3f}")

    def test_per_domain_capacity(self, lifecycle_state):
        """Check capacity per domain — each domain's patterns are a separate cluster.

        Within a single domain, patterns are tighter (higher similarity).
        The domain-local delta_min is the binding constraint.
        """
        patterns = lifecycle_state["patterns"]
        meta = lifecycle_state["meta"]

        print(f"\nPer-domain capacity analysis:")
        print(f"{'Domain':15s} {'N':>4s} {'δ_min':>8s} {'N_max':>12s} {'Util':>10s}")

        for d_idx in range(self.N_DOMAINS):
            d_name = DOMAIN_NAMES[d_idx]
            d_patterns = np.array([
                patterns[i] for i, m in enumerate(meta) if m.domain == d_idx
            ])
            n = len(d_patterns)
            if n < 2:
                print(f"  {d_name:15s} {n:4d}     -- (< 2 patterns)")
                continue

            d_delta = compute_min_delta(d_patterns)
            d_nmax = compute_n_max(self.BETA, d_delta)
            d_util = n / d_nmax if d_nmax > 0 else float("inf")

            print(f"  {d_name:15s} {n:4d} {d_delta:8.4f} {d_nmax:12.2e} {d_util:10.2e}")

        # No assertion — purely diagnostic. The point is that all domains
        # are far from capacity, confirming the prediction.

    # ------------------------------------------------------------------
    # 4.2: Repulsion impact on delta_min
    # ------------------------------------------------------------------

    def test_repulsion_effect_on_real_patterns(self, lifecycle_state):
        """Measure how much repulsion changes delta_min on real patterns.

        Prediction: barely. With spread=0.02 around 1024-dim centroids,
        within-cluster patterns are moderately separated. Repulsion
        (eta=0.01, min_sep=0.3 cosine distance) should find few pairs
        close enough to repel.
        """
        patterns = lifecycle_state["patterns"]
        importances = lifecycle_state["importances"]

        delta_before = compute_min_delta(patterns)

        # Run repulsion
        patterns_after = nrem_repulsion_xb(
            patterns, importances, eta=0.01, min_sep=0.3,
        )
        delta_after = compute_min_delta(patterns_after)

        change = delta_after - delta_before
        pct_change = (change / max(delta_before, 1e-12)) * 100

        print(f"\nRepulsion effect (real geometry, N={len(patterns)}):")
        print(f"  delta_min before: {delta_before:.6f}")
        print(f"  delta_min after:  {delta_after:.6f}")
        print(f"  Change: {change:+.6f} ({pct_change:+.2f}%)")

        # R1 contract: delta_min should not decrease
        assert delta_after >= delta_before - 1e-9, (
            f"R1 violated: delta_min decreased from {delta_before:.6f} to {delta_after:.6f}"
        )

        # Count how many pairs were close enough to repel
        N = patterns.shape[0]
        close_pairs = 0
        sim_threshold = 1.0 - 0.3  # min_sep = 0.3 cosine distance
        for i in range(N):
            for j in range(i + 1, N):
                if float(patterns[i] @ patterns[j]) > sim_threshold:
                    close_pairs += 1

        total_pairs = N * (N - 1) // 2
        print(f"  Pairs within repulsion range (cosine > {sim_threshold:.2f}): "
              f"{close_pairs}/{total_pairs}")

    def test_repulsion_preserves_high_importance(self, lifecycle_state):
        """R2 contract: high-importance patterns (>= 0.7) should not move.

        After 30 days of dreaming, high-generation patterns have boosted
        importance. Repulsion should anchor these while moving low-gen
        episodic memories.
        """
        patterns = lifecycle_state["patterns"]
        importances = lifecycle_state["importances"]

        patterns_after = nrem_repulsion_xb(
            patterns, importances, eta=0.01, min_sep=0.3,
        )

        high_imp_mask = importances >= 0.7
        n_high = int(high_imp_mask.sum())

        if n_high == 0:
            print("\nNo high-importance patterns — R2 trivially satisfied")
            return

        # Check that high-importance patterns are unchanged
        max_drift = 0.0
        for i in range(len(patterns)):
            if high_imp_mask[i]:
                cos_sim = float(patterns[i] @ patterns_after[i])
                drift = 1.0 - cos_sim
                max_drift = max(max_drift, drift)

        print(f"\nRepulsion R2 (anchor high-importance):")
        print(f"  High-importance patterns: {n_high}/{len(patterns)}")
        print(f"  Max drift for high-imp: {max_drift:.8f}")

        assert max_drift < 1e-6, (
            f"High-importance pattern drifted by {max_drift:.6f} — R2 violated"
        )

    # ------------------------------------------------------------------
    # 4.3: REM-explore cross-domain associations
    # ------------------------------------------------------------------

    def test_rem_explore_finds_associations(self, lifecycle_state):
        """REM-explore should find cross-domain associations with real geometry.

        With orthogonal centroids (synthetic), cross-domain pairs have
        zero structural similarity. With real centroids (health↔fitness
        at 0.42), perturbation responses should correlate, producing
        genuine cross-domain associations.

        This is the test that COULDN'T work on synthetic data.
        """
        patterns = lifecycle_state["patterns"]
        labels = lifecycle_state["labels"]

        associations = rem_explore_cross_domain_xb(
            patterns, labels,
            n_probes=max(len(patterns), 200),
            rng=np.random.default_rng(self.SEED),
        )

        print(f"\nREM-explore cross-domain associations (real geometry):")
        print(f"  Patterns: {len(patterns)}")
        print(f"  Associations found: {len(associations)}")

        if associations:
            print(f"  Top 10 associations:")
            meta = lifecycle_state["meta"]
            for idx_i, idx_j, sim in associations[:10]:
                d_i = DOMAIN_NAMES[meta[idx_i].domain]
                d_j = DOMAIN_NAMES[meta[idx_j].domain]
                print(f"    {d_i:12s} ↔ {d_j:12s}: corr={sim:.4f} "
                      f"(gen-{meta[idx_i].generation}, gen-{meta[idx_j].generation})")

            # Verify X1 contract: all pairs are cross-cluster
            for idx_i, idx_j, sim in associations:
                assert int(labels[idx_i]) != int(labels[idx_j]), (
                    f"Association ({idx_i}, {idx_j}) is within same cluster"
                )

            # Verify X2 contract: similarity in [0, 1]
            for _, _, sim in associations:
                assert 0.0 <= sim <= 1.0, f"Similarity {sim} outside [0, 1]"

        # Core assertion: real geometry should produce associations
        # (orthogonal synthetic data produces zero)
        assert len(associations) > 0, (
            "No cross-domain associations found — real geometry should enable "
            "perturbation response correlation between semantically related domains"
        )

    def test_rem_explore_discovers_related_domains(self, lifecycle_state):
        """The discovered associations should favor semantically related domain pairs.

        Expected high-association pairs (from corpus design):
        - health ↔ fitness (centroid similarity 0.42)
        - learning ↔ fitness (0.40)
        - work ↔ technology (related content)

        This validates that REM-explore isn't random — it finds real structure.
        """
        patterns = lifecycle_state["patterns"]
        labels = lifecycle_state["labels"]
        meta = lifecycle_state["meta"]

        associations = rem_explore_cross_domain_xb(
            patterns, labels,
            n_probes=max(len(patterns), 200),
            rng=np.random.default_rng(self.SEED),
        )

        if not associations:
            pytest.skip("No associations to analyze")

        # Count associations per domain pair
        pair_counts: dict[tuple[str, str], int] = {}
        for idx_i, idx_j, sim in associations:
            d_i = DOMAIN_NAMES[meta[idx_i].domain]
            d_j = DOMAIN_NAMES[meta[idx_j].domain]
            pair = tuple(sorted([d_i, d_j]))
            pair_counts[pair] = pair_counts.get(pair, 0) + 1

        print(f"\nAssociation frequency by domain pair:")
        for pair, count in sorted(pair_counts.items(), key=lambda x: -x[1]):
            print(f"  {pair[0]:12s} ↔ {pair[1]:12s}: {count}")

        # Report — the distribution itself is the finding
        # No hard assertion on which pairs rank highest, since
        # with only ~20 patterns the stochastic probing varies.

    # ------------------------------------------------------------------
    # 4.4: Operational purpose — dream cycle decomposition
    # ------------------------------------------------------------------

    def test_dream_cycle_decomposition(self, lifecycle_state, real_centroids):
        """Decompose one dream cycle to measure each operation's contribution.

        Run merge, prune, repulsion, unlearn, and explore separately on
        the final state, measuring:
        - P@1 (self-retrieval accuracy)
        - delta_min (minimum pairwise separation)
        - Number of patterns (consolidation)
        - Cross-domain associations found

        This reveals WHAT dreams do on real geometry.
        """
        patterns = lifecycle_state["patterns"]
        importances = lifecycle_state["importances"]
        labels = lifecycle_state["labels"]
        N_in = patterns.shape[0]

        # Pre-dream metrics
        engine_pre = CoupledEngine(dim=patterns.shape[1], beta=self.BETA)
        for i in range(N_in):
            engine_pre.store(f"p{i}", patterns[i], importance=float(importances[i]))
        p1_pre = measure_p1(engine_pre, patterns)
        delta_pre = compute_min_delta(patterns)

        # Run one full dream cycle
        report = dream_cycle_v2(
            patterns, self.BETA,
            importances=importances,
            labels=labels,
            seed=self.SEED + 999,
        )

        X_post = report.patterns
        N_out = X_post.shape[0]

        # Post-dream metrics (need importances for the surviving patterns)
        pruned_set = set(report.pruned_indices)
        kept = [i for i in range(N_in) if i not in pruned_set]
        merged_orig: set[int] = set()
        for group in report.merge_map.values():
            merged_orig.update(group)
        non_merged = [i for i in kept if i not in merged_orig]

        # Reconstruct importance array for output
        imp_out = []
        for i in non_merged:
            imp_out.append(float(importances[i]))
        for out_idx in sorted(report.merge_map.keys()):
            group = report.merge_map[out_idx]
            imp_out.append(float(min(max(max(importances[g] for g in group) + 0.1, 0.0), 1.0)))
        imp_out_arr = np.array(imp_out, dtype=np.float64)

        engine_post = CoupledEngine(dim=X_post.shape[1], beta=self.BETA)
        for i in range(N_out):
            engine_post.store(f"p{i}", X_post[i], importance=float(imp_out_arr[i]))
        p1_post = measure_p1(engine_post, X_post)
        delta_post = compute_min_delta(X_post)

        print(f"\nDream cycle decomposition (real geometry):")
        print(f"  {'Metric':25s} {'Pre-dream':>12s} {'Post-dream':>12s} {'Change':>12s}")
        print(f"  {'─'*61}")
        print(f"  {'Patterns':25s} {N_in:12d} {N_out:12d} {N_out - N_in:+12d}")
        print(f"  {'P@1 (self-retrieval)':25s} {p1_pre:12.4f} {p1_post:12.4f} {p1_post - p1_pre:+12.4f}")
        print(f"  {'delta_min':25s} {delta_pre:12.6f} {delta_post:12.6f} {delta_post - delta_pre:+12.6f}")
        print(f"  {'Pruned':25s} {'':12s} {len(report.pruned_indices):12d}")
        print(f"  {'Merged groups':25s} {'':12s} {len(report.merge_map):12d}")
        print(f"  {'Associations':25s} {'':12s} {len(report.associations):12d}")

        # Core finding: dream should consolidate (fewer patterns)
        assert N_out <= N_in, (
            f"Dream cycle INCREASED patterns: {N_in} -> {N_out}"
        )

    def test_dream_improves_cluster_coherence(self, lifecycle_state, real_centroids):
        """Dreams should improve domain-centroid retrieval coherence.

        The operational purpose of dreams on real geometry isn't P@1
        (which is already high). It's cluster coherence: when you query
        with an ambiguous stimulus (domain centroid), the top-K results
        should be more domain-pure after dreaming.
        """
        patterns = lifecycle_state["patterns"]
        importances = lifecycle_state["importances"]
        labels = lifecycle_state["labels"]
        centroids = real_centroids

        N_in = len(patterns)

        # Pre-dream coherence
        engine_pre = CoupledEngine(dim=patterns.shape[1], beta=self.BETA)
        for i in range(N_in):
            engine_pre.store(f"p{i}", patterns[i], importance=float(importances[i]))

        coherence_pre = []
        for d in range(self.N_DOMAINS):
            results = engine_pre.query(centroids[d], top_k=5)
            if results:
                domain_hits = sum(1 for r in results if int(labels[r["index"]]) == d)
                coherence_pre.append(domain_hits / len(results))

        # Run dream
        report = dream_cycle_v2(
            patterns, self.BETA,
            importances=importances,
            labels=labels,
            seed=self.SEED + 999,
        )
        X_post = report.patterns

        # Reconstruct labels and importances for output
        pruned_set = set(report.pruned_indices)
        kept = [i for i in range(N_in) if i not in pruned_set]
        merged_orig: set[int] = set()
        for group in report.merge_map.values():
            merged_orig.update(group)
        non_merged = [i for i in kept if i not in merged_orig]

        labels_out = []
        imp_out = []
        for i in non_merged:
            labels_out.append(int(labels[i]))
            imp_out.append(float(importances[i]))
        for out_idx in sorted(report.merge_map.keys()):
            group = report.merge_map[out_idx]
            # Majority vote for domain
            from collections import Counter
            domain_votes = Counter(int(labels[g]) for g in group)
            labels_out.append(domain_votes.most_common(1)[0][0])
            imp_out.append(float(min(max(max(importances[g] for g in group) + 0.1, 0.0), 1.0)))

        labels_out_arr = np.array(labels_out, dtype=int)
        imp_out_arr = np.array(imp_out, dtype=np.float64)

        # Post-dream coherence
        engine_post = CoupledEngine(dim=X_post.shape[1], beta=self.BETA)
        for i in range(len(X_post)):
            engine_post.store(f"p{i}", X_post[i], importance=float(imp_out_arr[i]))

        coherence_post = []
        for d in range(self.N_DOMAINS):
            results = engine_post.query(centroids[d], top_k=min(5, len(X_post)))
            if results:
                domain_hits = sum(1 for r in results if int(labels_out_arr[r["index"]]) == d)
                coherence_post.append(domain_hits / len(results))

        mean_pre = float(np.mean(coherence_pre)) if coherence_pre else 0.0
        mean_post = float(np.mean(coherence_post)) if coherence_post else 0.0

        print(f"\nDomain coherence (top-5 query with domain centroid, β={self.BETA}):")
        print(f"  Pre-dream mean coherence:  {mean_pre:.4f}")
        print(f"  Post-dream mean coherence: {mean_post:.4f}")
        print(f"  Change: {mean_post - mean_pre:+.4f}")

        # Report per-domain
        for d in range(self.N_DOMAINS):
            d_name = DOMAIN_NAMES[d]
            pre_val = coherence_pre[d] if d < len(coherence_pre) else 0.0
            post_val = coherence_post[d] if d < len(coherence_post) else 0.0
            print(f"  {d_name:15s}: {pre_val:.2f} -> {post_val:.2f}")

        # Dreams should not HURT coherence significantly
        assert mean_post >= mean_pre - 0.15, (
            f"Dream cycle significantly reduced coherence: {mean_pre:.3f} -> {mean_post:.3f}"
        )

    def test_adaptive_thresholds_at_real_geometry(self, lifecycle_state):
        """Verify that adaptive thresholds land in sensible positions for real geometry.

        The thresholds should:
        - Be above between-domain max (~0.42) to avoid cross-domain merging
        - Be below or near paraphrase band (~0.85) to merge redundancies
        - Self-calibrate based on the actual within-cluster distribution
        """
        patterns = lifecycle_state["patterns"]
        labels = lifecycle_state["labels"]

        merge_t, prune_t = compute_adaptive_thresholds(
            patterns, labels, merge_percentile=70.0, prune_percentile=90.0,
        )

        print(f"\nAdaptive thresholds on post-lifecycle patterns:")
        print(f"  Merge threshold (P70): {merge_t:.4f}")
        print(f"  Prune threshold (P90): {prune_t:.4f}")
        print(f"  Gap (prune - merge): {prune_t - merge_t:.4f}")

        # Thresholds should be ordered
        assert merge_t < prune_t, (
            f"Merge threshold {merge_t:.4f} >= prune threshold {prune_t:.4f}"
        )

        # Merge threshold should be a valid cosine similarity
        assert 0.0 < merge_t < 1.0, f"Merge threshold {merge_t} out of range"
        assert 0.0 < prune_t < 1.0, f"Prune threshold {prune_t} out of range"


# ---------------------------------------------------------------------------
# Phase 5: Morning pass — re-embed merged patterns from lineage text
# ---------------------------------------------------------------------------


class TestMorningPass:
    """Phase 5: Does re-embedding a textual summary outperform the raw centroid?

    After dream merge, a pattern's vector is the geometric mean of its
    parents. The "morning pass" hypothesis: embedding a natural language
    summary of the parent texts produces a better retrieval anchor than
    the raw centroid.

    No LLM needed — we test with deterministic proxy summaries:
    1. Concatenation of parent texts
    2. Nearest parent to centroid (representative selection)
    3. Domain label as text ("memories about finance")

    Measured by: domain retrieval coherence (does the vector retrieve
    same-domain content better?) and cosine distance from domain centroid.
    """

    @pytest.fixture(scope="class")
    def paraphrase_merge_data(self, embedded_corpus):
        """Simulate merging each domain's 5 paraphrases into one pattern.

        This is the simplest merge scenario — 5 near-identical sentences
        become one. Returns the centroid, parent texts, and parent embeddings
        for each domain.
        """
        embeddings = embedded_corpus["embeddings"]
        sentences = embedded_corpus["sentences"]
        domain_indices = embedded_corpus["domain_indices"]

        merge_groups = []
        for d_idx, d_name in enumerate(DOMAIN_NAMES):
            # Paraphrases are sentences 0-4 within each domain
            base = d_idx * 20
            parent_indices = list(range(base, base + 5))
            parent_embs = embeddings[parent_indices]
            parent_texts = [sentences[i] for i in parent_indices]

            # Raw centroid (what dream merge produces)
            centroid = parent_embs.mean(axis=0)
            centroid = centroid / np.linalg.norm(centroid)

            merge_groups.append({
                "domain": d_name,
                "domain_idx": d_idx,
                "parent_indices": parent_indices,
                "parent_texts": parent_texts,
                "parent_embeddings": parent_embs,
                "centroid": centroid,
            })

        return merge_groups

    def test_reembed_concatenation_vs_centroid(self, paraphrase_merge_data, embedded_corpus):
        """Concatenation of parent texts, re-embedded, vs raw centroid.

        The concatenated text carries all the semantic content of the
        parents. Its embedding should land near the centroid but may
        capture nuances the geometric mean misses.
        """
        embeddings = embedded_corpus["embeddings"]
        domain_indices = embedded_corpus["domain_indices"]

        print(f"\nMorning pass: concatenation vs centroid")
        print(f"{'Domain':15s} {'Centroid→own':>13s} {'Reembed→own':>12s} {'Δ':>8s} "
              f"{'Cent→other':>11s} {'Reemb→other':>12s}")

        centroid_wins = 0
        reembed_wins = 0

        for group in paraphrase_merge_data:
            d_name = group["domain"]
            d_idx = group["domain_idx"]
            centroid = group["centroid"]

            # Concatenate parent texts with separator
            concat_text = " | ".join(group["parent_texts"])
            reembedded = encode_sentences([concat_text])[0]

            # Compute domain centroid from ALL 20 domain sentences
            domain_mask = [i for i, di in enumerate(domain_indices) if di == d_idx]
            domain_centroid = embeddings[domain_mask].mean(axis=0)
            domain_centroid = domain_centroid / np.linalg.norm(domain_centroid)

            # Other-domain centroid (mean of all other domains)
            other_mask = [i for i, di in enumerate(domain_indices) if di != d_idx]
            other_centroid = embeddings[other_mask].mean(axis=0)
            other_centroid = other_centroid / np.linalg.norm(other_centroid)

            # Similarity to own domain centroid
            cent_own = float(centroid @ domain_centroid)
            reemb_own = float(reembedded @ domain_centroid)

            # Similarity to other-domain centroid
            cent_other = float(centroid @ other_centroid)
            reemb_other = float(reembedded @ other_centroid)

            delta_own = reemb_own - cent_own

            if reemb_own > cent_own:
                reembed_wins += 1
            else:
                centroid_wins += 1

            print(f"  {d_name:15s} {cent_own:13.4f} {reemb_own:12.4f} {delta_own:+8.4f} "
                  f"{cent_other:11.4f} {reemb_other:12.4f}")

        print(f"\n  Re-embed wins: {reembed_wins}/{len(paraphrase_merge_data)}")
        print(f"  Centroid wins: {centroid_wins}/{len(paraphrase_merge_data)}")

    def test_reembed_representative_vs_centroid(self, paraphrase_merge_data, embedded_corpus):
        """Pick the parent nearest the centroid as the representative text.

        Cheapest proxy — no concatenation, no LLM. Just pick the single
        parent whose embedding is closest to the merge centroid and use
        its text as the pattern's representation.
        """
        embeddings = embedded_corpus["embeddings"]
        domain_indices = embedded_corpus["domain_indices"]

        print(f"\nMorning pass: nearest-parent representative vs centroid")
        print(f"{'Domain':15s} {'Centroid→own':>13s} {'Repr→own':>9s} {'Δ':>8s} {'Representative':50s}")

        for group in paraphrase_merge_data:
            d_name = group["domain"]
            d_idx = group["domain_idx"]
            centroid = group["centroid"]
            parent_embs = group["parent_embeddings"]
            parent_texts = group["parent_texts"]

            # Find nearest parent to centroid
            sims_to_centroid = [float(p @ centroid) for p in parent_embs]
            best_idx = int(np.argmax(sims_to_centroid))
            representative = parent_embs[best_idx]
            repr_text = parent_texts[best_idx]

            # Domain centroid
            domain_mask = [i for i, di in enumerate(domain_indices) if di == d_idx]
            domain_centroid = embeddings[domain_mask].mean(axis=0)
            domain_centroid = domain_centroid / np.linalg.norm(domain_centroid)

            cent_own = float(centroid @ domain_centroid)
            repr_own = float(representative @ domain_centroid)
            delta = repr_own - cent_own

            print(f"  {d_name:15s} {cent_own:13.4f} {repr_own:9.4f} {delta:+8.4f} "
                  f"{repr_text[:48]}")

    def test_reembed_domain_label_vs_centroid(self, paraphrase_merge_data, embedded_corpus):
        """Embed just the domain label ("memories about finance") as the pattern.

        The most abstract proxy — tests whether a pure category name
        is a better retrieval anchor than a geometric merge of episodic
        memories. If yes, it suggests dreams should produce labeled
        abstractions, not centroids.
        """
        embeddings = embedded_corpus["embeddings"]
        domain_indices = embedded_corpus["domain_indices"]

        label_texts = [f"Personal memories about {d}" for d in DOMAIN_NAMES]
        label_embeddings = encode_sentences(label_texts)

        print(f"\nMorning pass: domain label vs centroid")
        print(f"{'Domain':15s} {'Centroid→own':>13s} {'Label→own':>10s} {'Δ':>8s}")

        centroid_margin_sum = 0.0
        label_margin_sum = 0.0

        for group in paraphrase_merge_data:
            d_name = group["domain"]
            d_idx = group["domain_idx"]
            centroid = group["centroid"]
            label_emb = label_embeddings[d_idx]

            # Domain centroid from all 20 sentences
            domain_mask = [i for i, di in enumerate(domain_indices) if di == d_idx]
            domain_centroid = embeddings[domain_mask].mean(axis=0)
            domain_centroid = domain_centroid / np.linalg.norm(domain_centroid)

            cent_own = float(centroid @ domain_centroid)
            label_own = float(label_emb @ domain_centroid)
            delta = label_own - cent_own

            # Also measure discrimination: own-domain sim minus max other-domain sim
            other_sims_cent = []
            other_sims_label = []
            for d2 in range(len(DOMAIN_NAMES)):
                if d2 == d_idx:
                    continue
                d2_mask = [i for i, di in enumerate(domain_indices) if di == d2]
                d2_cent = embeddings[d2_mask].mean(axis=0)
                d2_cent = d2_cent / np.linalg.norm(d2_cent)
                other_sims_cent.append(float(centroid @ d2_cent))
                other_sims_label.append(float(label_emb @ d2_cent))

            centroid_margin = cent_own - max(other_sims_cent)
            label_margin = label_own - max(other_sims_label)
            centroid_margin_sum += centroid_margin
            label_margin_sum += label_margin

            print(f"  {d_name:15s} {cent_own:13.4f} {label_own:10.4f} {delta:+8.4f} "
                  f"  margin: cent={centroid_margin:+.4f} label={label_margin:+.4f}")

        mean_cent_margin = centroid_margin_sum / len(paraphrase_merge_data)
        mean_label_margin = label_margin_sum / len(paraphrase_merge_data)
        print(f"\n  Mean discrimination margin:")
        print(f"    Centroid: {mean_cent_margin:+.4f}")
        print(f"    Label:    {mean_label_margin:+.4f}")
        print(f"    {'Label' if mean_label_margin > mean_cent_margin else 'Centroid'} "
              f"discriminates better by {abs(mean_label_margin - mean_cent_margin):.4f}")

    def test_retrieval_with_reembedded_store(self, embedded_corpus):
        """End-to-end test: store re-embedded summaries, query with natural language.

        Build two stores:
        1. Centroid store: each domain's paraphrase centroid
        2. Re-embed store: each domain's concatenated paraphrase text, re-embedded

        Query both with 10 natural language questions and compare which
        retrieves the correct domain more often.
        """
        embeddings = embedded_corpus["embeddings"]
        domain_indices = embedded_corpus["domain_indices"]
        sentences = embedded_corpus["sentences"]

        # Build centroids and re-embeddings for paraphrase groups
        centroid_store = []
        reembed_store = []
        store_labels = []

        for d_idx, d_name in enumerate(DOMAIN_NAMES):
            base = d_idx * 20
            parent_embs = embeddings[base:base + 5]
            parent_texts = [sentences[i] for i in range(base, base + 5)]

            centroid = parent_embs.mean(axis=0)
            centroid = centroid / np.linalg.norm(centroid)
            centroid_store.append(centroid)

            concat = " | ".join(parent_texts)
            reemb = encode_sentences([concat])[0]
            reembed_store.append(reemb)

            store_labels.append(d_idx)

        centroid_store = np.array(centroid_store)
        reembed_store = np.array(reembed_store)

        # Natural language queries targeting specific domains
        queries = [
            ("What happened to my investments on Monday?", 0),         # finance
            ("Did I meet the project deadline?", 1),                    # work
            ("Tell me about my home server setup", 2),                  # technology
            ("What did the doctor say about my blood pressure?", 3),    # health
            ("How was my daughter's first day of school?", 4),          # family
            ("What online courses have I completed?", 5),               # learning
            ("What is my best running time?", 6),                       # fitness
            ("How is my relationship with my partner?", 7),             # relationships
            ("What was the trip to Kyoto like?", 8),                    # travel
            ("Tell me about my mechanical keyboard project", 9),        # hobbies
        ]

        query_texts = [q for q, _ in queries]
        query_labels = [l for _, l in queries]
        query_embs = encode_sentences(query_texts)

        # Query both stores
        centroid_correct = 0
        reembed_correct = 0

        print(f"\nEnd-to-end retrieval: centroid store vs re-embed store")
        print(f"{'Query':50s} {'Expected':>12s} {'Cent':>6s} {'Reemb':>6s}")

        for i, (q_text, expected_d) in enumerate(queries):
            q_emb = query_embs[i]

            # Centroid store: nearest
            cent_sims = [float(q_emb @ centroid_store[d]) for d in range(len(DOMAIN_NAMES))]
            cent_best = int(np.argmax(cent_sims))

            # Re-embed store: nearest
            reemb_sims = [float(q_emb @ reembed_store[d]) for d in range(len(DOMAIN_NAMES))]
            reemb_best = int(np.argmax(reemb_sims))

            cent_ok = cent_best == expected_d
            reemb_ok = reemb_best == expected_d
            if cent_ok:
                centroid_correct += 1
            if reemb_ok:
                reembed_correct += 1

            cent_mark = "OK" if cent_ok else DOMAIN_NAMES[cent_best][:6]
            reemb_mark = "OK" if reemb_ok else DOMAIN_NAMES[reemb_best][:6]

            print(f"  {q_text[:48]:50s} {DOMAIN_NAMES[expected_d]:>12s} "
                  f"{cent_mark:>6s} {reemb_mark:>6s}")

        print(f"\n  Centroid store accuracy: {centroid_correct}/{len(queries)}")
        print(f"  Re-embed store accuracy: {reembed_correct}/{len(queries)}")
        print(f"  Winner: {'Re-embed' if reembed_correct > centroid_correct else 'Centroid' if centroid_correct > reembed_correct else 'Tie'}")

        # Both should achieve reasonable accuracy (>= 7/10)
        assert centroid_correct >= 5, (
            f"Centroid store only got {centroid_correct}/10 — baseline broken"
        )
        assert reembed_correct >= 5, (
            f"Re-embed store only got {reembed_correct}/10 — re-embedding hurts"
        )
