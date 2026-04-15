# Engram Build — Shared Agent Scratchpad

## Module Dependency Order
1. ontology.py (no deps)
2. storage.py (depends on ontology)
3. query.py (depends on storage, ontology)
4. ingestion.py (depends on storage, ontology)
5. retrieval.py (depends on storage, query)
6. writeback.py (depends on storage, ontology)
7. core.py (depends on all above)
8. dynamics.py (depends on core, storage)

## Findings
(agents append here)
- 2026-04-15: Direct test import contract extracted for eight modules. Test-facing exports are: ontology(7), storage(12), query(4), ingestion(13), retrieval(6), writeback(5), dynamics(5), core(2).
- 2026-04-15: Dependency-ordered build plan written to `foundry/phased-plan.yaml`. Topological order is ontology -> storage -> query -> ingestion -> writeback -> retrieval -> core -> dynamics.
- 2026-04-15: Resolver boundary is the main architectural constraint. Spec section 3.4 governs query-time semantics, but the concrete `EntityResolver` export must live in `foundry.ingestion`; the plan keeps query resolver-agnostic to avoid a query/ingestion cycle.
- 2026-04-15: Fixed four package import bugs in `foundry/tests/*` by switching `from conftest import ...` to package-relative imports.
- 2026-04-15: Implemented `foundry/ontology.py` from spec section 1 with stdlib-only enums/dataclasses, deterministic property inference, `EntityType.normalize()` / `depluralize()` invariants, plus `Interface` / `ValueType` helpers from sections 1.7-1.8. `python3 -m pytest -o addopts='' foundry/tests/test_unit_ontology.py -q --tb=short` passes with 62/62 tests in this environment because repo-local `.venv` lacks `pytest` and the default pytest config requires `xdist`.

## Blockers
(agents note blockers here)

## Completed Modules
(agents mark done here)
