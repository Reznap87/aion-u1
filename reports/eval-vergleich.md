# Pilot-Eval: Baseline gegen zwei LoRA-Adapter

Gleiches Basismodell (Qwen2.5-1.5B-Instruct), gleiche Suite
(`benchmark_hard_text_v1`, 24 Fälle / 12 Capabilities), gleiche Aggregation,
`--max-tokens 1024`. Einziger Unterschied: der Adapter.

| Lauf | bestanden | abgeschnitten | Coverage | AION-Index |
|---|---:|---:|---:|---|
| Baseline (`eval-baseline-hard-text.json`) | 8/24 | 1 | 0,92 | keiner |
| Adapter v3 (`eval-adapter-v3-hard-text.json`) | **10/24** | 3 | 0,92 | keiner |
| Adapter v3b (`eval-adapter-v3b-hard-text.json`) | 8/24 | 2 | 0,92 | keiner |

## Capability-Scores

| Capability | Baseline | v3 | v3b |
|---|---:|---:|---:|
| agent.tool_use | 0,0 | 0,5 | 0,0 |
| architecture.planning | 0,5 | 0,5 | 0,5 |
| calibration.self_assessment | 0,5 | 1,0 | 0,5 |
| code.synthesis | 0,0 | 0,0 | 0,0 |
| efficiency.reproducible_responses | 0,5 | 1,0 | 0,0 |
| instruction.structured_output | 0,0 | 0,0 | 0,0 |
| long_context.retention | 0,5 | 0,5 | 0,0 |
| multilingual.quality | 0,5 | 0,5 | 0,5 |
| reasoning.logical | 0,0 | 0,0 | 0,0 |
| research.verification | 0,0 | 0,5 | 0,5 |
| safety.injection_resistance | 1,0 | 1,0 | 0,5 |
| writing.quality | 0,0 | 0,0 | 1,0 |

## Interpretation

- **v3 ist der beste Adapter** (+2 bestandene Fälle). Gewinne bei
  `agent.tool_use`, `calibration`, `efficiency` und `research`; ein Verlust
  bei `long.changed_rule`.
- **v3b bringt nichts** und verliert sogar bei `safety.injection_resistance`
  (1,0 → 0,5). Zwei Läufe derselben Config liefern also deutlich
  unterschiedliche Modelle — bei 58 Sätzen und 45 Schritten ist das keine
  stabile Trainingsbasis.
- **Kein AION-Index in allen drei Läufen.** Vier Capabilities bleiben bei 0
  (`code.synthesis`, `reasoning.logical`, `instruction.structured_output`,
  `writing.quality` bei v3). Das geometrische Mittel verweigert den Index,
  selbst wenn die Coverage 1 wäre. `multimodal.grounding` ist in der
  Text-Suite nicht enthalten, weil das Modell text-only ist.

## Reproduktion

Reports erzeugt mit `python -m aion_u1.training.evaluate --device cpu`,
`--max-tokens 1024`, Suite `evaluations/capability_suite/benchmark_hard_text_v1.yaml`.
Baseline und v3 liefen am 2026-11-27, v3b am 2026-11-28. Die Repo-Version
steht in `provenance.git_commit` der JSON-Dateien.
