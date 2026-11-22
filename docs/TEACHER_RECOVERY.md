# Teacher-Daten nach einem Teillauf fortsetzen

Der Generator speichert jede erfolgreich empfangene Kandidatenzeile unmittelbar
in der JSONL-Datei und synchronisiert sie auf den Datenträger. Ein vorhandener
Output wird beim normalen Start nicht überschrieben. Die Dateisperre bleibt bis
zum Abschluss des Reports bestehen und verhindert zwei gleichzeitige Writer.

Bei HTTP 401, 402 oder 403 wird keine weitere wartende Generierung gestartet.
Bereits abgeschickte Anfragen können noch enden und Kosten verursachen.
Erfolgreiche Antworten daraus werden ebenfalls gespeichert. Provider-Antworttexte
und Ausnahmeinhalte werden nicht in die Fehlerdetails kopiert; HTTP-Status
und stabile Fehlercodes bleiben auswertbar.

Der OpenRouter-Client sendet standardmäßig genau einen Versuch. Mehrere
Versuche für vorübergehende HTTP- oder Transportfehler müssen ausdrücklich
über `max_attempts` gewählt werden. Ein Timeout kann auch bei bereits
bearbeiteter Anfrage auftreten; zusätzliche Versuche garantieren deshalb keine
einmalige Abrechnung. Eine erfolgreiche, aber ungültige JSON-Antwort wird
nicht erneut angefordert. HTTP 401, 402 und 403 werden ebenfalls nicht
wiederholt und stoppen weiterhin die noch wartenden Generatoraufträge.

## Fortsetzung

Nach Klärung der externen Abrechnungsursache kann eine Fortsetzung mit denselben
Aufgaben und verifizierten Modellen ausdrücklich gestartet werden:

```bash
aion-u1-generate-candidates \
  --tasks training/configs/tasks/pilot.yaml \
  --output data/generated/teacher_candidates.jsonl \
  --report reports/data-generation.json \
  --concurrency 4 \
  --resume
```

Vorhandene Antworten werden nach Aufgabe, Teacher, konkreter Modell-ID und
Sample-Index erkannt und nicht erneut angefragt. Neue Zeilen besitzen einen
Fingerprint der Aufgabenkonfiguration einschließlich Eingabe und Medieninhalt.
Bei veränderter Aufgabe oder Modellzuordnung stoppt die Fortsetzung vor der
ersten Generierungsanfrage.

Ältere Kandidaten ohne Fingerprint werden anhand der noch vorhandenen Felder,
des Kandidatenhashes, der Eingabe und des aktuell verifizierten Modells geprüft.
Frühere Sampling-Einstellungen sind daraus nicht vollständig rekonstruierbar.
Der Report weist solche Zeilen als `legacy_reused_candidates` aus; er bestätigt
für sie keine vollständige Konfigurationsbindung.

Die Fortsetzung ist auf dauerhaft gespeicherte Antworten begrenzt. Nach einem
Prozessabbruch kann der Ausgang bereits abgeschickter, noch nicht gespeicherter
Anfragen unklar sein. Eine zweite Abrechnung solcher unsicheren Anfragen wird
ohne providerseitige Idempotenz nicht ausgeschlossen.

Eine beschädigte oder unvollständig geschriebene JSONL-Zeile wird erhalten und
abgewiesen. Nach einem harten Prozessabbruch kann eine `.lock`-Datei zurückbleiben.
Vor ihrer manuellen Entfernung muss feststehen, dass kein Writer mehr läuft.

## Report und Exit-Codes

- `candidates` zählt vorhandene und neu gespeicherte Kandidaten.
- `reused_candidates` und `new_candidates` unterscheiden die beiden Anteile.
- `errors` zählt echte Fehler. `skipped_candidates` zählt nach dem Stopp nicht
  gestartete Aufträge. `skipped` erhält die getrennten Listen
  `unbekannte_aliase` und `tasks_ohne_lehrer` aus der Aufgabenplanung.
- `known_cost_usd` summiert bekannte, vom Provider gemeldete Antwortkosten.
  `unknown_cost_candidates` kennzeichnet Antworten ohne Kostenangabe.
  Kosten fehlgeschlagener oder unklar beendeter Anfragen sind darin nicht belegt.
- `status: completed` und Exit-Code 0 setzen voraus, dass alle erwarteten
  Kandidaten vorhanden sind und kein Fehler oder Stopp vorliegt.
- Ein Teillauf mit Kandidaten erhält Exit-Code 2, ein erfolgloser Lauf 1 und
  eine Tastaturunterbrechung 130.

Die Kostenstatistik ersetzt kein hartes Ausgabenlimit. Der vorhandene
`--max-tokens`-Wert begrenzt die Ausgabetoken pro Anfrage; HTTP 402 beendet den
Lauf ohne automatische Tokenverkleinerung oder erneutes Senden. Dieses
Inkrement startet keinen Live-Lauf und verändert keine Guthaben- oder Key-Limits.

Ranking, Verifikation und Trainingsfreigabe bleiben getrennte nachgelagerte
Schritte. Der Generator setzt `training_approved` weiterhin auf false.
