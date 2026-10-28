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
