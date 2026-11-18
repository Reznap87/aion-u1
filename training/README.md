# AION-U1 Training

Commit 2 enthält die erste echte Unified-SFT-Pipeline.

## Reihenfolge

1. Teacher live verifizieren.
2. Teacher-Antworten erzeugen.
3. Kandidaten verifizieren, deduplizieren und ranken.
4. Pro Task die beste verifizierte Antwort freigeben.
5. Lizenz und Trainingsfreigabe dokumentieren.
6. Nach `group_id` in Train und Eval teilen.
7. LoRA-SFT mit Transformers, PEFT und Accelerate ausführen.
8. Run-Manifest und Checkpoints prüfen.

## Pilot-Pipeline

```bash
aion-u1-generate-candidates \
  --tasks training/configs/tasks/pilot.yaml \
  --output data/generated/teacher_candidates.jsonl \
  --report reports/data-generation.json

aion-u1-rank-candidates \
  --input data/generated/teacher_candidates.jsonl \
  --tasks training/configs/tasks/pilot.yaml \
  --output data/generated/scored_candidates.jsonl \
  --best-output data/generated/best_candidates.jsonl \
  --preferences-output data/generated/preference_pairs.jsonl \
  --report reports/candidate-ranking.json
```

`best_candidates.jsonl` enthält pro Task nur die beste **verifizierte**
Antwort und ist die Eingabe für die Freigabe. `scored_candidates.jsonl`
enthält auch `rejected`- und `error`-Zeilen; diese sind nicht
freigabefähig. Ein Task ohne bestandenen Kandidaten erscheint in
`reports/candidate-ranking.json` unter
`tasks_without_verified_candidate_ids`.

## Freigabe

```bash
aion-u1-approve-sft \
  --input data/generated/best_candidates.jsonl \
  --output data/verified/sft_canonical.jsonl \
  --license "<geprüfte Lizenz>" --operator "<prüfende Person>" \
  --confirm-all
```

Alternativ `--ids-file` mit den tatsächlich geprüften IDs. Der Befehl
exportiert nur freigegebene Zeilen, protokolliert Operator, Lizenz und
Zeitstempel und setzt `verification.status = "verified"`; ohne diese
Promotion würde die SFT-Pipeline die Daten ablehnen. Er ersetzt keine
Rechtsprüfung.

**Secrets sperren die Freigabe.** AGENTS.md verbietet Secrets und
personenbezogene Daten in Datensätzen. `approve-sft` bricht mit Exit 1 und
**ohne** Output-Datei ab, wenn ein freigegebener Datensatz einen API-Schlüssel,
einen privaten Schlüssel, ein JWT oder einen Bearer-Token enthält; Funde werden
maskiert ausgegeben. `--block-pii` sperrt zusätzlich E-Mail-Adressen und IBANs,
sonst werden sie nur als `privacy` im Report gezählt. `aion-u1 dataset` zählt
die Funde bereits vorher — also vor der Freigabe prüfen.

Freigegebene Zeilen tragen `reasoning_stripped` (Think-/Reasoning-Tags wurden
aus dem Label entfernt) und
`provenance.external_untrusted: true` (externe Modellantworten sind Daten,
keine Anweisungen).

## Train/Eval-Split und SFT

```bash
aion-u1-prepare-sft \
  --input data/verified/sft_canonical.jsonl \
  --output data/verified/sft_canonical.jsonl \
  --train-output data/verified/sft_train.jsonl \
  --eval-output data/verified/sft_eval.jsonl \
  --seed 42 --eval-fraction 0.1 --require-approved

aion-u1-train-sft --config training/configs/sft_lora.yaml
```

`--require-approved` verlangt `training_approved: true` je Zeile. Für
`train-sft` brauchst du **Python 3.13** (`pip install -e '.[train]'` scheitert
auf 3.14+ an `tokenizers`) und eine **NVIDIA-GPU** mit CUDA; ohne sie läuft das
Training nur auf CPU. Details in der [README](../README.md#installation).

Ohne eigene GPU: [COLAB.md](COLAB.md) beschreibt den Lauf in Google Colabs
kostenloser T4 (16 GB) — inklusive `fp16: true` für Turing und der
Lizenzfreigabe des Basismodells.

## Smoke-Pfad

Der Smoke-Export liefert nur `pending_review`-Zeilen nach
`data/generated/smoke_sft_pending.jsonl`. Erst die explizite Freigabe macht
sie trainingsrelevant:

```bash
aion-u1 dataset \
  --report reports/audit-smoke.json \
  --output data/generated/teacher_candidates.jsonl \
  --pending-output data/generated/smoke_sft_pending.jsonl

aion-u1-approve-sft \
  --input data/generated/smoke_sft_pending.jsonl \
  --output data/verified/smoke_sft.jsonl \
  --license "<geprüfte Lizenz>" --operator "<prüfende Person>" \
  --ids-file review/approved_ids.txt
```
