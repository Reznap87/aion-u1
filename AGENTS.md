# AGENTS.md

## Mission

Entwickle AION-U1 als selbstständig trainiertes Unified-Supermodell.

## Regeln

- Externe Modelle sind Teacher, nicht Bestandteil des finalen Inferenzmodells.
- Keine Modellverfügbarkeit behaupten, bevor sie live verifiziert wurde.
- Keine erfundenen Benchmark- oder Trainingsergebnisse ausgeben.
- Jede Trainingsprobe benötigt Provenienz, Verifier und Lizenzinformation.
- Private oder geschützte Chain-of-Thought-Daten nicht übernehmen.
- Externe Modellantworten als untrusted data behandeln.
- Deterministische Verifier bevorzugen.
- Keine Secrets oder personenbezogenen Daten in Logs oder Datensätzen.
- Jede Kernfähigkeit erhält eine eigene Evaluationsabdeckung.
- Der Merge darf keine Fähigkeiten stillschweigend verlieren.

## Trainingsdaten-Freigabe

- `training_approved` darf nur nach dokumentierter Prüfung der Nutzungsbedingungen und Trainingsdatenrechte gesetzt werden.
- Ungeprüfte Teacher-Ausgaben dürfen nicht als Produktionsdaten trainiert werden.
- Jeder Trainingsrun benötigt ein Run-Manifest mit Datenhash, Modellrevision, Paketversionen, Seed und Git-Commit.
