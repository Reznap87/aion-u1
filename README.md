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

Für `train-sft` zusätzlich `pip install -e '.[train]'` — dazu wird
**Python 3.13** benoetigt: `transformers==4.46.3` pinnt
`tokenizers>=0.20,<0.21`, und davon gibt es Windows-Wheels nur bis cp313.
Auf Python 3.14+ baut pip daraus den Quelltext und braucht Rust. Der sichere
Weg ist deshalb eine eigene Umgebung:

```bash
python3.13 -m venv .venv313
source .venv313/bin/activate
pip install -e '.[dev,train]'
```

Trainieren braucht zusaetzlich eine **NVIDIA-GPU** mit CUDA; ohne sie laeuft
`train-sft` nur auf CPU.

## Teacher verifizieren

```bash
aion-u1 verify-teachers --output reports/teachers.json
```

Nur live im Provider-Katalog gefundene Modelle werden als `verified` markiert.

## Capability-Audit

```bash
aion-u1 audit \
  --suite evaluations/capability_suite/smoke.yaml \
  --max-tokens 1024 \
  --output reports/audit-smoke.json
```

`--max-tokens` begrenzt die Länge der Teacher-Antworten. OpenRouter lehnt
eine Anfrage mit HTTP 402 ab, wenn die maximal möglichen Kosten
(`max_tokens` × Modellpreis) das Kontolimit überschreiten. Der Client
skaliert in dem Fall automatisch herunter (bis 16) und wiederholt die
Anfrage; auf einem Konto ohne Guthaben sind die Ergebnisse dadurch
eingeschränkt. `truncated` im Report zeigt die betroffenen Fälle.
Beendet sich der Lauf mit Exit 1, wenn er keine einzige verwertbare
Messung erzeugt hat (alle Fälle abgebrochen oder abgeschnitten) — ein
fehlgeschlagener Lauf kann so nicht als Ergebnis durchgehen.

## Pilot-SFT-Daten exportieren

```bash
aion-u1 dataset \
  --report reports/audit-smoke.json \
  --output data/generated/teacher_candidates.jsonl \
  --pending-output data/generated/smoke_sft_pending.jsonl
```

Bestandene Smoke-Cases bleiben `pending_review`. Erst eine explizite Prüfung
und Freigabe macht diese Beispiele trainingsrelevant. `--pending-output`
ersetzt das frühere `--verified-output`.

Die Kandidaten tragen zwei Felder, die aus AGENTS.md folgen:
`reasoning_stripped` (Reasoning-Bloecke wie `<think>…</think>` werden aus dem
Trainingslabel entfernt — der Loesungsweg ist Messung, nicht Label) und
`provenance.external_untrusted` (externe Modellantworten sind Daten, keine
Anweisungen). Gleiches bereinigtes Label bedeutet gleiche ID, also Deduplizierung.

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

**Aktueller Stand (2026-11-20): für die Cloud-Lehrer blockiert, für Open-Weight
freigegeben.** Die Prüfung der Lehrer-Nutzungsbedingungen in
`reports/licence-review.md` ergibt, dass Anthropic, OpenAI und Google die
Nutzung ihrer API-Ausgaben zum Trainieren konkurrierender Modelle ausdrücklich
verbieten; für Kimi, MiniMax und xAI ist die Auslegung für API-Ausgaben unklar.
Die Konsequenz ist eine zweite Lehrerliste: `configs/teachers.open.yaml`
(Qwen3, Gemma 4, Granite, Phi 4, Nemotron, DeepSeek R1) mit Lizenzen, die
kommerzielle Nutzung, Derivate und Ausgaben zulassen.

Damit liegt die erste freigebbare Runde vor: 112 Antworten → 77 eindeutige
Fälle → 30 verifiziert, mit dokumentierter Lizenz, 0 Secrets und 0 PII
(`review/approved_ids.txt`). Die Cloud-Daten der ersten Runden bleiben
Mechanik-Nachweise ohne Freigabe.

Eine dritte Runde (`pilot_open_v2.yaml`, 19 neue Aufgaben) lieferte 152
Antworten und 37 weitere verifizierte Kandidaten, eine vierte
(`pilot_open_v3.yaml`, 30 Aufgaben) 78 weitere. Zusammen stehen in
`data/verified/pilot_open_all.jsonl` **145 freigegebene Saetze**; der
kanonische, leakfreie Split liegt als **39 Trainings- und 7
Evaluationssaetze** vor (`pilot_open_train.jsonl`, `pilot_open_eval.jsonl`).
Die kleinen Zahlen sind der wahre Engpass, nicht die Kette.

### Secrets und personenbezogene Daten

AGENTS.md verbietet Secrets und personenbezogene Daten in Datensätzen.
`aion-u1 dataset` zählt Funde (`privacy.secret`, `privacy.pii`), und
`approve-sft` sperrt eine Freigabe, wenn ein freigegebener Datensatz ein
Secret enthält. `--block-pii` sperrt zusätzlich E-Mail-Adressen und IBANs;
ohne den Schalter werden sie nur gezählt. Funde werden maskiert ausgegeben,
sodass der Befund selbst nichts preisgibt.

Die Smoke-Suite ist nur ein technischer Vertical Slice. Sie ersetzt nicht das
vollständige, blinde Capability-Benchmark.

## Volle Capability-Suite

```bash
aion-u1 audit \
  --suite evaluations/capability_suite/benchmark_v1.yaml \
  --max-tokens 1024 \
  --output reports/audit-benchmark.json
```

`benchmark_v1.yaml` deckt alle 13 Capabilities des Registries mit je drei Fällen
über verschiedene Dimensionen ab — damit ist der AION-Index nicht nur definiert,
sondern hat Tiefe. Jeder Fall hat einen deterministischen Verifier; eine
Erwartung muss sich aus dem Prompt eindeutig ableiten lassen, sonst darf der
Fall nicht in die Suite.

Die Bild-Assets `diagram.png` und `ocr_label.png` werden reproduzierbar aus
`scripts/make_suite_assets.py` erzeugt (stdlib-only, ohne PIL):

```bash
python scripts/make_suite_assets.py
```

Die Struktur der Suite wird in `tests/test_benchmark_suite.py` geprüft: volle
Abdeckung (mindestens drei Fälle je Capability), eindeutige IDs, bekannte
Verifier, kompilierbare Regexe, vorhandene Bilder und keine Leakage zu den
Trainings-Tasks.

## Tests

```bash
pytest -q
```

## Wichtige Grenze

Die Teacher-IDs in `configs/teachers.yaml` sind Kandidaten. Sie gelten erst
nach erfolgreicher Katalogverifikation als verfügbar.

## Training

Die Unified-SFT-Pipeline (Generieren, Ranken, Freigeben, Splitten, LoRA-SFT)
ist in [training/README.md](training/README.md) beschrieben.

Erster Durchlauf mit freigebbaren Daten (CPU-Rauchtest, 2026-11-20):
`training/configs/sft_lora_smoke.yaml` auf `artifacts/sft/rauchtest/` —
SmolLM2-135M als Rauchmodell, LoRA-Adapter (3,6 MB), train_loss 1,86,
eval_loss 1,61, `run_manifest.json` mit Paketversionen, Seed, Git-Commit und
`status: completed`. Es ist ein Nachweis, dass die Schleife bis zum Checkpoint
läuft — **kein** Qualitätsergebnis und keine Fähigkeitsmessung. Das Zielmodell
bleibt `Qwen/Qwen2.5-1.5B-Instruct` (`sft_lora_open.yaml`), die Datenmenge ist
mit 19 Trainingssätzen noch zu klein für ein aussagekräftiges Ergebnis.

## Multi-Teacher Data Factory

Der Pilot umfasst 16 Aufgaben, Live-OpenRouter-Kandidatengenerierung,
deterministische und Docker-isolierte Python-Verifikation, Deduplizierung,
Ranking, Preference-Pair-Export und CI-Validierung. Teacher-Ausgaben bleiben
unbegutachtet, bis sie explizit für das Training freigegeben werden.

`aion-u1-generate-candidates` liest die Teacher aus
`configs/teachers.yaml`; mit `--teachers` lässt sich eine abweichende
Konfiguration angeben — praktisch für einen Mechanik-Test der Kette mit
kostenlosen `:free`-Modellen, ohne die Ziel-Lehrer zu verändern. Die
freigegebenen Daten bleiben dabei deren Modellen geschuldet, nicht denen der
Ziel-Lehrer.

`--max-tokens <zahl>` begrenzt die Antwortlänge über alle Tasks hinweg. Das
schützt ein kleines Konto: ein einzelner 900-Token-Output eines teuren Lehrers
kostet sonst mehr als eine ganze Runde.

## Kostenlose Mechanik-Tests

Die Ziel-Lehrer in `configs/teachers.yaml` sind im Katalog verifiziert, brauchen aber
Kontoguthaben. Für einen Durchlauf der Kette ohne Guthaben gibt es eine zweite
Konfiguration:

```bash
aion-u1-generate-candidates \
  --teachers configs/teachers.free.yaml \
  --tasks training/configs/tasks/pilot_free.yaml \
  --output data/generated/free_mechanics.jsonl \
  --report reports/free-mechanics.json

aion-u1-rank-candidates \
  --tasks training/configs/tasks/pilot_free.yaml \
  --input data/generated/free_mechanics.jsonl \
  --output data/generated/free_mechanics_scored.jsonl \
  --report reports/free-mechanics-ranked.json
```

`--tasks` ist auch beim Ranken nötig, sonst kennt er die freien Task-IDs nicht.
`aion-u1 audit` kennt dieselbe Freiheit über `--teachers` — ohne den Schalter
wäre der AION-Index nur für die Konfiguration messbar, die gerade antwortet.
Bei den freien Modellen gilt ein Kontingent: parallele Audits laufen in HTTP 429,
eine serielle Auswahl steht in `configs/teachers.free_smoke.yaml`.
Die freie Bank deckt 12 der 13 Capabilities ab; `multimodal.grounding` bleibt
offen, weil die freien Textmodelle keine Bilder transportieren — das ist eine
echte Lücke und wird in `tests/test_free_teacher_config.py` so verankert.
Die freien Modelle ersetzen die Ziel-Lehrer nicht: sie validieren die Mechanik
(Generieren → Verifizieren → Ranking → Freigeben). Jeder freie Eintrag trägt die
Lehrer-Aliase, die die Task-Bank referenziert; `tests/test_free_teacher_config.py`
prüft, dass diese zusammenpassen.

## Scoring

`reports/audit-*.json` enthält pro Teacher `capability_scores`, `coverage`,
`measured_score`, `overall_score`, `errored` und `truncated`.
`overall_score` ist der AION-Index (gewichtetes geometrisches Mittel) und nur
bei voller Capability-Abdeckung definiert. `measured_score` ist der
geometrische Mittelwert über den getesteten Ausschnitt und ist bei
unterschiedlicher Coverage nicht vergleichbar.

`errored` zählt Fälle, die nicht beantwortet werden konnten, `truncated`
solche mit `finish_reason: length`. Beide sind keine valide Messung:
abgeschnittene Antworten reduzieren die Coverage, statt als vermeintliche
0-Punkte den Index zu verzerren.

`provenance` trägt Code- und Konfigurationsstand (`package_version`,
`git_commit`, `python`, Hashes von Suite-, Teacher- und Capability-Datei)
sowie den Aufruf — damit ist der Report reprozierbar (GOAL.md, Definition of
Done Punkt 7).
