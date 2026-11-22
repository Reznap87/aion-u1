# Projekt AION-U1

## Ziel

AION-U1 ist ein selbstständig trainiertes Unified-Supermodell, das die
messbaren Fähigkeiten der Ziel-Lehrermodelle in einem einzelnen
Modell-Artifact vereinigt.

Die Ziel-Lehrermodelle sind:

- minimax-m3.1-flash
- kimi-k3
- anthropic/claude-fable-5.1
- gpt-6.1-sol
- anthropic/claude-sonnet-5.5
- google/gemini-3.8-flash
- x-ai/grok-4.7
- moonshotai/kimi-k2.7-code

Die tatsächlichen Modell-IDs, Verfügbarkeit, Nutzungsbedingungen und
Lizenzen werden vor jedem Training empirisch verifiziert.

## Produktdefinition

AION-U1 wird als EIN Modell ausgeliefert.

Im Inference-Betrieb gilt:

- keine externen Teacher-Modelle,
- kein Provider-Fallback,
- keine Laufzeitentscheidung zwischen acht fremden APIs,
- keine nachträgliche Prompt-Auswahl anderer Modelle,
- ein gemeinsames Modell-Artifact,
- ein gemeinsames Tokenizer- und Tool-Format.

Ein intern Mixture-of-Experts-Design ist erlaubt, sofern das gesamte
Modell als eine trainierte Einheit geladen, versioniert und evaluiert wird.

## Zu vereinbarende Fähigkeiten

1. Softwareentwicklung und Code-Repair
2. Mathematisches und logisches Reasoning
3. Architektur- und Produktplanung
4. Long-Context-Verarbeitung
5. Multimodalität: Bilder, Dokumente, Diagramme und OCR
6. Agentische Tool-Nutzung
7. Recherche mit optionaler Websuche
8. Hochwertiges Writing und Dokumentation
9. Mehrsprachigkeit, insbesondere Deutsch und Englisch
10. Instruction Following und strukturiertes Output
11. Kalibrierte Selbsteinschätzung
12. Safety, Prompt-Injection-Abwehr und Tool-Governance
13. Effiziente Antworten mit reproduzierbarem Verhalten

## Nicht-Ziele

- Keine Behauptung, geschützte Modellgewichte zu kopieren.
- Kein Training auf Daten ohne geklärte Nutzungsrechte.
- Keine erfundenen Benchmarkergebnisse.
- Kein statisches Modell für Garantien über aktuelle Live-Daten.
- Keine stillen externen Fallbacks.

## Definition of Done

AION-U1 gilt als fertig, wenn:

1. es als ein Modell-Artifact lokal ausgeliefert werden kann;
2. es ohne Zugriff auf die Teacher-Modelle inferiert;
3. sein AION-Index den besten verfügbaren Einzel-Lehrer erreicht
   oder übertrifft;
4. es in keiner Kernfähigkeit unter 95 % des jeweiligen besten Lehrers
   fällt;
5. Tool-Call-Erfolgsrate und Safety-Score die Zielgrenzen erreichen;
6. alle Trainingsdaten, Versionen und Lizenzen dokumentiert sind;
7. ein reproduzierbarer Evaluationsbericht vorliegt;
8. das Modell quantisiert und ohne Qualitätsverlust deployed werden kann.

## Messung

Der AION-Index ist ein gewichteter geometrischer Mittelwert:

I = exp(sum(weight_i * ln(score_i)))

Dadurch kann eine sehr starke Fähigkeit keine schwache andere Fähigkeit
vollständig kompensieren.

## Verbindlicher Ausbau aus der ursprünglichen Projektidee

Die am 2. November 2026 bereitgestellte ursprüngliche Beschreibung präzisiert:

- Ein Modell-Artifact, eine Modell-ID, ein gemeinsamer Tokenizer und ein
  gemeinsamer Transformer bilden die ausgelieferte Einheit.
- Ein intern trainiertes Mixture-of-Experts ist als Teil dieser Einheit möglich.
- Native strukturierte Tool-Aufrufe und der Umgang mit Werkzeugergebnissen
  werden trainiert und eigenständig evaluiert.
- Text, Code, Bilder, Dokumente, Long Context, Safety und Kalibrierung benötigen
  eigene unabhängige Holdouts. Entwicklungsdaten dürfen nicht als Holdout dienen.
- Die 70B-Klasse, acht Experten und 131072 Kontexttoken sind Architekturziele.
  Der bisher konfigurierte kleine textbasierte LoRA-Pilot erfüllt diese Ziele
  nicht und bleibt ein technischer Trainingsprototyp.
- Ein eigener Tokenizer muss gemeinsam mit dem Checkpoint versioniert werden.
  Der aktuelle Pilot übernimmt den Tokenizer des Basismodells; ein neu
  trainierter Tokenizer ist noch nicht implementiert.

Die vorstehenden Ziele sind keine Messergebnisse. Safety und Tool-Erfolg
benötigen zusätzlich vorab festgelegte Grenzen.

Die nächste Entwicklung folgt der Trainingskette: nachvollziehbare
Teacher-Kandidaten, deterministische Verifikation und Auswahl, Rechtefreigabe,
reproduzierbares SFT, Spezialisten-/Merge- und Distillationsverfahren,
Tool-/Vision-/Long-Context-Alignment und unabhängige Regressionsevaluation.
Ein lokaler Inferenzdienst ist der Auslieferungsweg für den Checkpoint.
