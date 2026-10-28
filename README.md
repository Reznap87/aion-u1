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
  --verified-output data/verified/smoke_sft.jsonl
```

Die Smoke-Suite ist nur ein technischer Vertical Slice. Sie ersetzt nicht das
vollständige, blinde Capability-Benchmark.

## Tests

```bash
pytest -q
```

## Wichtige Grenze

Die Teacher-IDs in `configs/teachers.yaml` sind Kandidaten. Sie gelten erst
nach erfolgreicher Katalogverifikation als verfügbar.
