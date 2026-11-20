# SFT in Google Colab (kostenlos)

Für den Fall, dass keine lokale NVIDIA-GPU zur Verfuegung steht: Colabs Free-Tier
stellt eine **Tesla T4 mit 16 GB VRAM** bereit. Das genuegt fuer den Piloten
(Qwen2.5-1.5B-Instruct, LoRA r=16, `max_seq_length 4096`, Batch 1,
Gradient-Checkpointing).

Voraussetzungen: ein Google-Konto. Die GPU-Auswahl faellt unter
`Laufzeit > Laufzeittyp aendern > T4 GPU`.

## 1. Repo klonen

```python
!git clone https://github.com/Reznap87/aion-u1.git
%cd aion-u1
```

Falls das Repo privat ist, zuerst einen Token in Colab hinterlegen und die URL
damit bauen; den Token nicht hier hineinschreiben.

## 2. Train-Extras installieren

```python
!pip install -e '.[train]'
```

Python 3.10–3.13 verwenden: `transformers==4.46.3` pinnt
`tokenizers>=0.20,<0.21`, und davon gibt es keine Wheels fuer Python 3.14+.

## 3. Trainingsdaten einlegen

`data/generated/` ist in `.gitignore`; die freigegebenen Saetze der Open-Weight-
Piloten sind versioniert und landen mit dem Klon:

- `data/verified/pilot_open_train.jsonl`
- `data/verified/pilot_open_eval.jsonl`

Die Config `training/configs/sft_lora_open.yaml` verweist bereits auf diese
Pfade. Eine Ablehnung durch `require_training_approval` bedeutet: eine Zeile hat
`training_approved: false` — dann keine freigegebenen Daten erzeugen, nicht die
Pruefung abschalten.

## 4. Config: fp16 fuer Turing

Die T4 (Turing, sm_75) unterstuetzt **kein bf16** — die Config hat
`bf16: false, fp16: false`. Fuer Colab:

```python
import pathlib
p = pathlib.Path("training/configs/sft_lora.yaml")
p.write_text(
    p.read_text(encoding="utf-8").replace("fp16: false", "fp16: true"),
    encoding="utf-8",
)
```

`model.license_acknowledged: true` steht in der Committeden Config; die
Grundlage ist Apache-2.0 (Qwen2.5 bis 1.5B) ohne MAU-Schwelle —
<https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/blob/main/LICENSE>. Ohne
diese Freigabe weigert sich `train-sft`.

## 5. Training starten

```python
!aion-u1-train-sft --config training/configs/sft_lora_open.yaml
```

Ergebnis: `artifacts/sft/pilot_open/run_manifest.json` (Datenhash, Revision,
Paketversionen, Seed, Git-Commit, Metriken). `status` muss `completed` sein;
ein Abbruch steht als `failed`.

Die freigegebenen Trainingssaetze stammen aus `configs/teachers.open.yaml`
(Apache-2.0, MIT, NVIDIA Open Model License); der Befund steht in
`reports/licence-review.md`.

## 6. Artefakte sichern

```python
from google.colab import files
!zip -r sft_pilot.zip artifacts/sft/pilot
files.download("sft_pilot.zip")
```

Die Session loescht den Arbeitsspeicher nach der Laufzeit — Manifest und
Checkpoints herunterladen, bevor die Verbindung dropt.
