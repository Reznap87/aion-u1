# Lizenzpruefung der Lehrer (Stand 2026-11-20)

Geprueft hat: Hermes Agent im Auftrag des Projektbetreibers. Geprueft wurden
die Nutzungsbedingungen der Anbieter, nicht die Gewichtslizenzen isoliert. Die
Aussagen sind Fakten aus den angegebenen Quellen, keine Rechtsberatung.

## Befund in einer Zeile

Von den acht Lehrern erlaubt **keiner** uneingeschraenkt, dessen API-Ausgaben
als Trainingsdaten fuer ein anderes Modell zu nutzen. Bei vier ist das
ausdruecklich verboten, bei den uebrigen ist die Auslegung fuer API-Ausgaben
unklar. Eine Freigabe als Produktionsdaten (AGENTS.md) ist deshalb nicht
moeglich.

## Tabelle

| Lehrer | Anbieter | Regelwerk | Befund fuer Output-Training |
|---|---|---|---|
| claude_fable_5_1 | Anthropic | Commercial Terms s.D.4, Usage Policy | **verboten**; nur mit vorheriger schriftlicher Genehmigung |
| claude_sonnet_5_5 | Anthropic | wie oben | **verboten** |
| gpt_6_1_sol | OpenAI | Services Agreement s.3.3(e), Terms of Use | **verboten**; Ausnahme nur fuer interne Klassifikatoren/Embeddings, nicht fuer ein verteiltes Modell |
| gemini_3_8_flash | Google | Gemini API Additional Terms, Cloud AI/ML Terms | **verboten**; „may not use the Services to develop models that compete" |
| grok_4_7 | xAI | Terms of Service / Acceptable Use Policy | **unklar**; keine eindeutige Output-Regel gefunden, API/Enterprise geregelt |
| kimi_k3 | Moonshot AI | Kimi K3 License (open weight) | **unklar**; Lizenztext gilt fuer die Gewichte, nicht klar fuer API-Outputs |
| kimi_k2_7_code | Moonshot AI | open weight (Kimi-K2-Familie) | **unklar**; Lizenz je Version zu pruefen |
| minimax_m3_1_flash | MiniMax | MiniMax Community License (open weight) | **eingeschraenkt**; kommerzielle Nutzung nur mit Attribution, Notice bzw. Genehmigung — und der Text lautet auf Software/derivate works, nicht auf API-Outputs |

## Quellen

- Anthropic: <https://support.claude.com/en/articles/12326764-can-i-use-my-outputs-to-train-an-ai-model>
  („We prohibit customers from using our services to train or develop AI
  models without our written permission.")
- OpenAI: <https://openai.com/policies/service-terms/> und
  <https://openai.com/policies/row-terms-of-use/>
  („use Output to develop models that compete with OpenAI").
- Google: <https://aistudio.google.com/docs/terms>
  („may not use the Services to develop models that compete with the Services").
- xAI: <https://x.ai/legal/terms-of-service> (keine klare Output-Regel gefunden).
- MiniMax M3.1 Flash: Lizenztext auf Hugging Face (MiniMax Community License,
  kommerzielle Nutzung mit Attribution/Notice, oberhalb 20 Mio. US-Dollar
  Jahresumsatz vorherige Genehmigung).
- Kimi K3: <https://github.com/MoonshotAI/Kimi-K3/blob/main/LICENSE> (open
  weight, Lizenz mit Bedingungen, keine OSI-Open-Source-Lizenz).

## Konsequenz fuer die Pipeline

- `training_approved` bleibt `false`; `review/approved_ids.txt` bleibt leer.
- Die 63 verifizierten Kandidaten sind **keine** Produktionsdaten. Sie bleiben
  Mechanik-Nachweise: Kette, Verifier, Ranking.
- Fuer eine echte Trainingsrunde braucht es entweder schriftliche
  Genehmigungen der Anbieter oder eine Lehrerzusammensetzung, deren
  Lizenzen Ausgaben ausdruecklich fuer Training erlauben (z. B. Open-Weight-
  Modelle, deren Lizenz das Training auf Ausgaben deckt — nicht die hier
  geprueften Cloud-Dienste).

## Offene Punkte

1. Ob die Lizenz der Open-Weight-Modelle (Kimi, MiniMax) auch fuer ueber eine
   Drittanbieter-API bezogene Ausgaben gilt, ist nicht geklaert.
2. Ob OpenRouter als Vermittler eigene Ausgabe-Bedingungen stellt, wurde
   nicht geprueft.
3. Ob `grok_4.7` ausdruecklich Output-Training erlaubt, ist nicht
   verifiziert.
