# AION-U1 Training

Commit 2 enthält die erste echte Unified-SFT-Pipeline.

## Reihenfolge

1. Teacher-Antworten erzeugen und verifizieren.
2. Pro semantischem Task die beste freigegebene Antwort auswählen.
3. Lizenz und Trainingsfreigabe dokumentieren.
4. Nach `group_id` in Train und Eval teilen.
5. LoRA-SFT mit Transformers, PEFT und Accelerate ausführen.
6. Run-Manifest und Checkpoints prüfen.

## Teacher-Kandidaten vorbereiten

```bash
aion-u1-prepare-sft \
  --input data/generated/teacher_candidates.jsonl \
  --output data/verified/sft_canonical.jsonl
