# Research: Extracting Structure from Conversations

**Date**: 2026-04-14
**Context**: Hermes Foundry ingestion pipeline — multilingual extraction alternatives to VerbNet
**Sources**: 6 papers from Exa search (ACL 2025, ICLR 2026, Stanford, RPI, Renmin U)

---

## 1. SGMem — Sentence Graph Memory (ICLR 2026)

**Paper**: Wu et al., "SGMem: Sentence Graph Memory for Long-Term Conversational Agents"
**URL**: https://arxiv.org/abs/2509.21212
**Venue**: ICLR 2026 (Huawei Technologies)

### Key Insight
Represents dialogue as **sentence-level graphs** within chunked units. No LLM extraction needed — uses only NLTK sentence segmentation. Captures associations across turn-, round-, and session-level contexts.

### Architecture
- **Sentence nodes**: each sentence = graph node
- **Edges**: co-occurrence within chunks + cross-chunk semantic similarity
- **Multi-hop retrieval**: traverses sentence graph to collect coherent evidence
- **Mixed retrieval**: combines raw dialogue + generated memory (summaries, facts, insights)

### Results
- Outperforms baselines on **LongMemEval** and **LoCoMo**
- Key advantage: no LLM-based extraction = no extraction hallucination

### Relevance to Foundry
- **Validates** our raw-chunks-as-primary approach (SGMem also prioritizes raw dialogue)
- **Alternative** to triple extraction: sentence graphs capture structure without SPO extraction
- **Lightweight**: no extraction model needed, just sentence segmentation
- **Language-agnostic**: NLTK sentence splitting works for many languages
- **Risk**: sentence-level granularity may be too coarse for aggregation queries ("how many bikes?")

---

## 2. Associa — Graph-Structured Long-Term Memory (ACL Findings 2025)

**Paper**: Zhang et al., "Bridging Intuitive Associations and Deliberate Recall"
**URL**: https://aclanthology.org/2025.findings-acl.901/
**Venue**: ACL 2025 Findings

### Key Insight
Event-centric memory graph with **two cognitive modes**: Intuitive Association (fast subgraph extraction via Prize-Collecting Steiner Tree) and Deliberating Recall (iterative query refinement).

### Architecture
- **Event-centric graph**: events as central nodes, entities linked via roles
- **Prize-Collecting Steiner Tree (PCST)**: optimization-based subgraph extraction — finds minimum-cost tree connecting prize nodes (relevant entities)
- **Iterative query refinement**: LLM rewrites query based on retrieved context

### Results
- Significantly outperforms existing methods on LongMemEval and LoCoMo
- PCST gives structured, evidence-rich subgraphs vs flat ranked lists

### Relevance to Foundry
- **PCST optimization** is a novel retrieval primitive we don't have — could be a Tier 1 pipeline
- **Event-centric** aligns with our TemporalEvent data model
- **Two-mode** (fast intuitive + slow deliberate) maps to our Tier 1 + Tier 3 architecture
- **Extraction still uses LLM** — doesn't solve multilingual extraction problem directly

---

## 3. PROM — Personal Knowledge Graphs from Conversations (ICAART 2025)

**Paper**: Maoliniyazi et al., "PROM: Personal Knowledge Graph Construction with Large Language Models"
**URL**: https://www.scitepress.org/Papers/2025/138309/138309.pdf
**Venue**: ICAART 2025 (Renmin University)

### Key Insight
Complete pipeline for **personal KG construction from conversations**. Three components: API Proxy Engine, Knowledge Extraction (few-shot LLM), Knowledge Fusion (multi-strategy conflict resolution).

### Architecture
1. **API Proxy Engine**: collect conversation data from apps (ChatBox, Zotero)
2. **Knowledge Extraction**: prompt-engineered LLM with few-shot examples → SPO triples
3. **Knowledge Fusion**:
   - Semantic similarity for entity resolution
   - Rule-based conflict resolution
   - Confidence scores for contradictions
   - Optional user input for ambiguous cases

### Key Design Decisions
- **Few-shot > fine-tuning** for triple extraction (more adaptable, less training data)
- **Context preservation**: each triple links back to source conversation
- **Multi-strategy fusion**: combines heuristics + confidence + user feedback

### Relevance to Foundry
- **Validates** our architecture: triple extraction + entity resolution + conflict resolution
- **Few-shot prompting** is what we already do (Qwen2.5-7B with examples)
- **Knowledge fusion** = our supersession + conflict resolution
- **Language-agnostic** by design: LLM handles any language in extraction
- **Missing**: no VerbNet-like role classification, no temporal reasoning

---

## 4. OKED — Open Knowledge Extraction from Dialogue (ISWC 2025)

**Paper**: Rook, "Open Knowledge Extraction from Dialogue Using In-Context Learning"
**URL**: https://ceur-ws.org/Vol-4085/paper25.pdf
**Venue**: ISWC 2025 Doctoral Consortium (RPI)

### Key Insight
Formalizes **dialogue-specific challenges** for KG construction: cross-turn assertions, anaphora density, dialogue acts (fact vs opinion vs suggestion), and perspectives.

### Dialogue-Specific Challenges Identified
1. **Cross-turn assertions**: facts distributed across multiple turns and speakers
2. **Anaphora/pronoun density**: heavy use of pronouns requiring coreference resolution
3. **Dialogue acts**: not all utterances assert facts (questions, hedging, disagreement)
4. **Perspectives**: multiple speakers may express conflicting views
5. **Pragmatics**: meaning depends on conversational context, not just syntax/semantics

### Proposed Solutions
- **Perspective-aware ontology**: model WHO said WHAT (speaker attribution)
- **ICL with retrieved examples**: dynamically select few-shot examples based on dialogue features
- **Dialogue act classification**: filter non-factual utterances before extraction

### Relevance to Foundry
- **Cross-turn extraction** is our biggest extraction challenge — facts span multiple turns
- **Dialogue act filtering** could reduce extraction noise (we currently extract from everything)
- **Perspective-aware ontology** maps to our role tagging (user vs assistant)
- **Speaker attribution** is critical for "what did I tell you about X?" queries
- **Language-agnostic**: ICL approach works with any LLM in any language

---

## 5. KGGen — Knowledge Graph Extraction with Language Models (Stanford, 2025)

**Paper**: Mo et al., "KGGen: Extracting Knowledge Graphs from Plain Text with Language Models"
**URL**: https://arxiv.org/abs/2502.09956
**Venue**: arXiv 2025 (Stanford)

### Key Insight
Novel **entity resolution via iterative clustering** after extraction. Solves the sparsity problem (OpenIE and GraphRAG produce sparse, disconnected graphs with too many unique relation types).

### Architecture
1. **LM-based extraction**: extract SPO triples from text
2. **Entity clustering**: inspired by crowd-sourcing strategies — iteratively cluster entities that refer to the same real-world entity
3. **Edge resolution**: consolidate edges with equivalent meanings
4. **MINE benchmark**: first benchmark for text-to-KG extraction quality

### Key Innovation: Entity Resolution via Clustering
- **Problem**: raw extraction produces "nearly as many unique relation types as edges"
- **Solution**: post-extraction clustering using LM similarity
- **Result**: denser, more connected graphs with better retrieval accuracy

### Relevance to Foundry
- **Entity clustering** is a better approach than our substring+embedding resolution (H3 falsified)
- **Post-extraction resolution** decouples extraction from resolution — each can be optimized independently
- **Relation normalization** addresses our predicate explosion problem (too many unique predicates)
- **MINE benchmark** could be used to evaluate our extraction quality
- **Open source**: https://github.com/stair-lab/kg-gen/

---

## 6. MASSIE — Multilingual Instruction-Following IE (ACL 2025)

**Paper**: Le et al., "Massively Multilingual Instruction-Following Information Extraction"
**URL**: https://aclanthology.org/2025.findings-acl.182/
**Venue**: ACL 2025 Findings

### Key Insight
First comprehensive multilingual IE benchmark: **215 datasets, 96 languages, 18 language families**. Tests 21 LLMs (0.5B to 72B) on few-shot in-context learning for IE.

### Key Findings
- **Multilingual IE remains very challenging** for existing LLMs, especially relations and events
- **Performance gap is extremely large** between high- and low-performing languages
- **Similar-performing language groups overlap** between different LLMs — shared performance bias
- **LF1 metric**: structure-aware metric that captures partial matches (vs exact match which overpenalizes)
- **Instruction-following** format enables any LLM to do IE without fine-tuning

### Language Performance Tiers (from paper)
- **High**: English, Spanish, French, German, Chinese, Japanese
- **Medium**: Arabic, Hindi, Portuguese, Korean, Turkish
- **Low**: Tibetan, Yoruba, Amharic, many low-resource languages

### Relevance to Foundry
- **Directly answers the VerbNet question**: instruction-following LLM extraction works across 96 languages
- **No VerbNet needed**: the LLM handles role/relation classification natively in any language
- **LF1 metric** should replace exact-match SubEM for extraction quality evaluation
- **Language tiers** tell us where our extraction pipeline will work well and where it won't
- **Instruction templates** are the key — same template works across languages when using instruction-tuned LLMs
- **Model size matters**: 7B+ models needed for reasonable multilingual IE performance

---

## Synthesis: Implications for Hermes Foundry

### The VerbNet Problem is Solved

The answer is clear from MASSIE and the other papers: **use instruction-following LLM extraction instead of lexicon-based classification**. Specifically:

1. **Drop VerbNet as primary**: Keep it as English-only validation/fallback
2. **LLM extraction handles role/sub_role natively**: Add role classification to the extraction prompt
3. **Same extraction call**: We already call Qwen2.5-7B — just add role/sub_role to the output schema
4. **Instruction template is language-agnostic**: "Extract subject, predicate, object, temporal expression, role (event/state/opinion), sub_role (acquisition/maintenance/intent/hypothetical)"

### New Patterns to Consider

| Pattern | Source | Adopt? | Rationale |
|---------|--------|--------|-----------|
| Sentence-level graphs (no LLM extraction) | SGMem | INVESTIGATE | Lightweight, language-agnostic, but may be too coarse |
| Prize-Collecting Steiner Tree for retrieval | Associa | INVESTIGATE | Novel retrieval primitive, could be Tier 1 pipeline |
| Post-extraction entity clustering | KGGen | ADOPT | Better than substring resolution (H3 validated) |
| Dialogue act filtering | OKED | ADOPT | Reduce extraction noise by filtering non-factual utterances |
| Cross-turn assertion detection | OKED | ADOPT | Critical for multi-turn fact extraction |
| Few-shot ICL for extraction | PROM/MASSIE | ALREADY DOING | Validates our Qwen2.5-7B approach |
| Multi-strategy knowledge fusion | PROM | ALREADY DOING | Validates our supersession + CR pipeline |
| LF1 partial-match metric | MASSIE | ADOPT for eval | Better than exact SubEM for extraction quality |

### Architecture Update Recommendation

```
Current (English-only):
  Extraction → VerbNet lookup → role/sub_role classification

Proposed (multilingual):
  Extraction (LLM) → role/sub_role in same prompt → VerbNet validation (English only)
                                                   → no validation needed (other languages)
```

The extraction prompt already asks for structured output. Adding `"role": "event|state|opinion"` and `"sub_role": "acquisition|maintenance|intent|hypothetical"` to the JSON schema costs nothing — the LLM handles it natively in any language.

### Competitive Landscape

| System | Extraction | Multilingual | Entity Resolution | Temporal |
|--------|-----------|-------------|------------------|----------|
| SGMem | None (sentence graphs) | Yes (NLTK) | N/A | No |
| Associa | LLM event extraction | Depends on LLM | PCST subgraph | Partial |
| PROM | Few-shot LLM triples | Depends on LLM | Semantic similarity | No |
| KGGen | LLM triples + clustering | Depends on LLM | Iterative clustering | No |
| **Hermes Foundry** | Few-shot LLM (Qwen2.5-7B) | **Yes (with fix)** | EntityStore BM25 | **Yes (Allen IA)** |

Hermes Foundry's unique advantages remain:
1. **Allen interval algebra** — no competitor has temporal reasoning
2. **3-tier query architecture** — no competitor separates deterministic from agentic
3. **ObjectSet query language** — no competitor has composable lazy queries
4. **Always-on subsystems** — no competitor avoids routing

---

## Source Papers (full text scraped via Firecrawl)

1. SGMem: `arxiv.org/html/2509.21212v1`
2. Associa: `aclanthology.org/2025.findings-acl.901/`
3. PROM: `scitepress.org/Papers/2025/138309/138309.pdf`
4. OKED: `ceur-ws.org/Vol-4085/paper25.pdf`
5. KGGen: `arxiv.org/html/2502.09956v2`
6. MASSIE: `aclanthology.org/2025.findings-acl.182/`
