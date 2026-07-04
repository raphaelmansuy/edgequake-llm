# EdgeQuake LLM — Improvement Specification Suite

> **Version**: 1.3.0  
> **Date**: 2026-07-04  
> **Status**: REVISED (Anti-Heuristic + Vertex AI + Azure/Bedrock Pass Complete)  
> **Scope**: Model Discovery API, Provider Conformance, Capability Registry

## Document Index

| # | Document | Lens | Purpose |
|---|----------|------|---------|
| 01 | [5-WHY Analysis](./01-FIVE-WHY-ANALYSIS.md) | Product Owner | Root-cause analysis of current gaps |
| 02 | [Provider Conformance Audit](./02-PROVIDER-CONFORMANCE-AUDIT.md) | AI Engineer | Gap analysis per provider vs official specs (REVISED: Azure+Bedrock) |
| 03 | [Model Discovery API](./03-MODEL-DISCOVERY-API.md) | API/SDK Designer | Core discovery API design |
| 04 | [Provider Discovery Approaches](./04-PROVIDER-DISCOVERY-APPROACHES.md) | Full Stack Developer | Per-provider discovery strategy (REVISED: Azure+Bedrock) |
| 05 | [Architecture & Implementation](./05-ARCHITECTURE-IMPLEMENTATION.md) | Full Stack Developer | DRY/SOLID plan + code-proven roadblocks (REVISED) |
| 06 | [Model Capability Registry](./06-MODEL-CAPABILITY-REGISTRY.md) | AI Engineer | Type system & anti-heuristic design (REVISED: Azure+Bedrock) |
| 07 | [Edge Cases & Migration](./07-EDGE-CASES-MIGRATION.md) | Product Owner | Compatibility, migration, edge cases (REVISED: EC-13 to EC-17 Azure+Bedrock) |
| 08 | [Research Findings July 2026](./08-RESEARCH-FINDINGS-JULY-2026.md) | AI Engineer | Ground-truth corrections from live research |
| 09 | [Implementation Plan (Final)](./09-IMPLEMENTATION-PLAN-FINAL.md) | Full Stack Developer | Phased plan with all roadblocks from code (REVISED: RB-17 to RB-21 Azure+Bedrock) |

## Principles

1. **Code is Law** — existing implementations are ground truth
2. **First Principles** — every design decision traced to WHY
3. **DRY/SOLID** — no duplication, single responsibility, open for extension
4. **Ascending Compatibility** — zero breaking changes to public API
5. **Battle-Tested** — every design validated against real provider quirks
6. **No Heuristics** — capabilities from API responses or cited docs, never from name patterns
