# AION-U1

Infrastruktur für Capability-Audit, Teacher-Verifikation und verifizierten
Multi-Teacher-Datenaustausch als Grundlage eines einzelnen trainierten
Unified-Supermodells.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
cp .env.example .env
```

OPENROUTER_API_KEY in `.env` eintragen.

## Teacher verifizieren

```bash
aion-u1 verify-teachers --output reports/teachers.json
```

Nur live im Provider-Katalog gefundene Modelle werden als `verified` markiert.

## Capability-Audit

```bash
aion-u1 audit \
  --suite evaluations/capability_suite/smoke.yaml \
  --output reports/audit-smoke.json
```

## Pilot-SFT-Daten exportieren

```bash
aion-u1 dataset \
  --report reports/audit-smoke.json \
  --output data/generated/teacher_candidates.jsonl \
  --pending-output data/generated/smoke_sft_pending.jsonl
```

Bestandene Smoke-Cases bleiben `pending_review`. Erst eine explizite Prüfung
und Freigabe macht diese Beispiele trainingsrelevant:

```bash
aion-u1-approve-sft \
  --input data/generated/smoke_sft_pending.jsonl \
  --output data/verified/smoke_sft.jsonl \
  --license "<geprüfte Lizenz>" --operator "<prüfende Person>" \
  --ids-file review/approved_ids.txt
```

`review/approved_ids.txt` enthält je Zeile die ID eines tatsächlich geprüften
Beispiels. Der Freigabebefehl exportiert ausschließlich freigegebene Zeilen.
Die Lizenz und die Betreiberangabe müssen inhaltlich geprüft werden; der
Befehl selbst führt keine Rechtsprüfung durch.

Die Smoke-Suite ist nur ein technischer Vertical Slice. Sie ersetzt nicht das
vollständige, blinde Capability-Benchmark.

## Tests

```bash
pytest -q
```

## Wichtige Grenze

Die Teacher-IDs in `configs/teachers.yaml` sind Kandidaten. Sie gelten erst
nach erfolgreicher Katalogverifikation als verfügbar.
## Commit 2: Unified-SFT

## Commit 3: Multi-Teacher Data Factory

Adds a 12-task capability pilot, live OpenRouter candidate generation, deterministic and Docker-isolated Python verification, deduplication, ranking, preference-pair export, and CI validation. Teacher outputs remain unreviewed until explicitly approved for training.
