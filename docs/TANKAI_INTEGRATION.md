# AION-U1 in TankAI

AION-U1 gehört zur eigenen Modellfamilie von TankAI. Dieses Repository entwickelt
Datenerzeugung, Verifikation, Training und den lokalen Inferenzpfad eines
gemeinsamen Checkpoints. TankAI Core übernimmt Commander, Workspace-Zuordnung,
Projektgedächtnis, Werkzeuge, Budgets und Ergebnisprüfung.

Die Ausrichtung folgt dem Nutzerauftrag vom 2. November 2026 und dem
[TankAI-Masterplan](https://github.com/Reznap87/TankAICore/blob/main/TANKAI_MASTERPLAN.md),
Abschnitte 0.11.1, 0.11.3 und 10. Der geprüfte Core-Ausgangspunkt ist
`b9eb349363889efe90e84460258ee19f0892b003`.

## Belegter Entwicklungsstand

Vorhanden sind Capability-Audit, Teacher-Katalogprüfung, eine Datengenerierungs-
und Auswahlpipeline sowie ein LoRA-/SFT-Trainingspfad. Der neue Inferenzdienst
stellt vollständige lokale Modellexporte über Chat Completions bereit.

Das Repository enthält keine trainierten AION-U1-Gewichte und keine bestandene
Modellqualitätsabnahme. Die Fähigkeiten in `configs/capabilities.yaml` sind
Entwicklungsziele. Protokolltests mit synthetischem Backend belegen die technische
Anbindung, keine trainierte Fähigkeit.

## Vom Trainingsrun zum Inferenzmodell

Trainingsdaten benötigen weiterhin dokumentierte Provenienz, Verifikation,
Lizenzinformationen und ausdrückliche Trainingsfreigabe. Die Lizenz des
Basismodells ist separat zu prüfen. Ein Token für einen Teacher erteilt keine
Trainingsdatenrechte.

Beim LoRA-Training muss für einen eigenständig ladbaren Export
`peft.merge_and_save: true` gewählt werden. Ein Adapter in `final_model/` allein
genügt nicht. Vollständiges Fine-Tuning mit `peft.enabled: false` exportiert das
ladbare Modell direkt nach `final_model/`.

Der Trainingslauf schreibt vor `status: completed` die SHA-256-Inventare von
Gewichten, Tokenizer und Konfiguration nach `inference_artifacts` im
`run_manifest.json`. Dadurch bleibt ein Export auch nach dem Umzug des gesamten
Run-Verzeichnisses prüfbar. Ältere Exporte ohne Inventar werden abgewiesen.

```bash
aion-u1 model-check --run-dir artifacts/sft/pilot
```

Die Prüfung verlangt einen abgeschlossenen Run mit dokumentierter Freigabepolicy,
Git-Commit, aufgelöster Basisrevision, Datenhash und unverändertem vollständigem
Export. Hinzugefügte, fehlende oder veränderte Dateien, Symlinks, reine Adapter
und Shards außerhalb des Exports werden abgewiesen. `artifact_verified` belegt
die Integrität gegenüber dem lokalen Manifest; es ist keine Signatur eines
externen Herausgebers und keine Prüfung der Modellqualität.

## Lokal starten

Die Inferenzabhängigkeiten sind optional; Audit und Datenerzeugung benötigen
keine Installation von PyTorch.

```bash
pip install -e ".[inference]"
```

Einen eigenen Schlüssel in `AION_U1_API_KEY` setzen. Er ist unabhängig vom
Teacher-Schlüssel. Den Wert nicht in einen Commit oder Trainingsdatensatz
aufnehmen.

```bash
aion-u1 serve \
  --run-dir artifacts/sft/pilot \
  --model-id aion-u1 \
  --host 127.0.0.1 \
  --port 8091 \
  --context-window 4096 \
  --max-new-tokens 2048 \
  --device cpu
```

Der Dienst lädt ausschließlich den verifizierten lokalen Export mit
`local_files_only=True`, `trust_remote_code=False` und Safetensors. Er lädt keine
Basisgewichte nach und verwendet keine Teacher oder externen Fallbacks.
`--device cuda` wählt bei einem passenden Host explizit die GPU.

Für einen lokalen Versuch mit knapperem Speicher kann `--dtype float16`
explizit gesetzt werden. Die Voreinstellung `auto` bleibt unverändert. Eine
geringere Präzision kann Antworten und Laufzeit verändern; sie ist gesondert
auszuwerten. Die Größe der Quelldateien ist keine Obergrenze für den kurzzeitigen
Speicherbedarf beim Laden: Konvertierung, Tokenizer und Betriebssystem benötigen
zusätzlichen Speicher. Bei unzureichender Reserve den Versuch nicht starten.

Unter Windows prüft `python scripts/inference_memory_preflight.py --model-dir
<Exportverzeichnis>` die Safetensors-Header und den freien physischen sowie
Commit-Speicher, ohne Gewichte zu laden. Exitcode 2 bedeutet, dass die konservativ
geschätzte Reserve nicht erreicht wird. Das ist kein gemessener Mindestbedarf
und keine automatische Sperre anderer Startwege.

`python scripts/measure_synthetic_loading.py` untersucht ausschließlich ein
temporäres kleines Zufallsmodell mit float32 und float16. Es benötigt die
installierten Inferenzabhängigkeiten und psutil, verwendet keine Downloads und
meldet den Speicher der beobachteten Prozessfamilie. Zeit- und Reservegrenzen
brechen den Versuch ab. Die Ergebnisse belegen weder den Speicherbedarf des
echten AION-Exports noch dessen Qualität; bereits verwaiste, nie beobachtete
Nachkommen werden durch diese Messung nicht garantiert erfasst.

`GET /health` antwortet erst nach dem Laden des Modells mit `status: ready`.
`quality_verified` und `production_ready` bleiben `false`, weil ein geladener
Checkpoint weder Qualität noch Produktionsfreigabe beweist.

Die authentifizierten Endpunkte sind:

- `GET /v1/models`
- `POST /v1/chat/completions`

Unterstützt werden Textnachrichten mit Rollen `system`, `user` und `assistant`,
`max_tokens` und `temperature`. Streaming, Bildnachrichten, native Tool-Calls
und unbekannte Parameter werden explizit abgewiesen. Eine Generierung läuft
gleichzeitig; weitere Generierungen erhalten HTTP 503. Anfragen sind auf 1 MiB
begrenzt. Zu lange Prompts werden vor der Generierung abgewiesen und nicht
still gekürzt. Modellfehler enthalten keine Prompts, Tokens oder internen Pfade.

Der Dienst dient der lokalen bzw. privaten Modellanbindung. Für einen gemeinsam
betriebenen Host sind TLS, Netzisolation, Prozessüberwachung und die Hostabnahme
Teil des gesonderten Betriebsauftrags. Der Standard bindet nur an Loopback.

## TankAI Core anschließen

TankAI Cores bestehender `OpenAILLM`-Adapter ist bereits passend. Es ist keine
zweite Queue, Auth-Datenbank oder Memory-Implementierung erforderlich.

Für einen später ausdrücklich gestarteten lokalen Modelltest:

```dotenv
TANKAI_LLM=openai
OPENAI_BASE_URL=http://127.0.0.1:8091/v1
OPENAI_MODEL=aion-u1
OPENAI_API_KEY=<derselbe eigene Wert wie AION_U1_API_KEY>
TANKAI_LLM_MAX_TOKENS=1024
```

Die Providerkennung `openai` bezeichnet hier den vorhandenen Protokolladapter.
Bei diesem lokalen `OPENAI_BASE_URL` entsteht kein Aufruf des OpenAI-Dienstes.
TankAIs bestehende Run-Limits und Ergebnisprüfung bleiben im Core.

`127.0.0.1` setzt voraus, dass beide Prozesse auf demselben Host laufen. In
Containern muss ein ausdrücklich konfigurierter privater Dienstname verwendet
werden. Dieses Inkrement ändert keine laufende TankAI-Konfiguration.

Eine Vertragsprüfung gegen einen tatsächlichen Core-Checkout ist möglich:

```bash
python scripts/check_tankai_adapter.py --tankai-root ../TankAICore
```

Die Prüfung nutzt den echten `OpenAILLM` und `BudgetedLLM` aus diesem Checkout
und ein klar gekennzeichnetes synthetisches Inferenzbackend. Sie prüft Antwort,
Authentifizierung, Modellkennung und den gemeinsamen Aufrufdeckel ohne
Teacher-Anfrage oder Modelltraining.

## Nächster Modellnachweis

Nach freigegebenen Trainingsdaten und einem dokumentierten Trainingslauf muss
der vollständige Checkpoint auf einem geeigneten Host geladen werden. Danach
folgen eingefrorene, unabhängige Capability-Evaluationen und der Vergleich mit
der unveränderten TankAI-Baseline. Erst die dafür gebundenen Ergebnisse können
eine Modellpromotion begründen.
