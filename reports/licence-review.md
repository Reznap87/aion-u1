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

## Freigebbare Alternative: Open-Weight-Konstellation

`configs/teachers.open.yaml` ersetzt die Cloud-Lehrer durch acht Open-Weight-
Modelle, deren Lizenzen kommerzielle Nutzung, Derivate und Ausgaben zulassen:

| Lehrer | Lizenz | Kernbestandteil |
|---|---|---|
| qwen3-coder-next | Apache 2.0 | „All our open-weight models are licensed under Apache 2.0", Finetuning ausdrücklich erwähnt (QwenLM/Qwen3) |
| qwen3-235b-a22b-2507 | Apache 2.0 | wie oben |
| google/gemma-4-31b-it | Apache 2.0 | Gemma 4 ist von den Gemma Terms zu Apache 2.0 gewechselt |
| ibm-granite/granite-4.2-8b | Apache 2.0 | IBM: Granite 4.0 unter Apache 2.0 |
| microsoft/phi-4 | MIT | Microsoft-Policy für offene Phi-Gewichte |
| nvidia/nemotron-3.5-lightning | NVIDIA Open Model License | „NVIDIA does not claim ownership to any outputs generated using the Works", Derivate frei |
| nvidia/nemotron-3-ultra-550b-a55b | NVIDIA Open Model License | wie oben |
| deepseek/deepseek-r1-0528 | MIT | Gewichte ab DeepSeek-R1 MIT-lizenziert |

Auswirkungen:
- `TeacherConfig` traegt ein `license`-Feld, Default `unreviewed`; die
  Open-Liste nennt die gepruefte Lizenz je Lehrer.
- Der AION-Index und die Datenrunden sind mit dieser Konfiguration messbar und
  freigebaur, ohne Anbieter-Genehmigung einzuholen.
- Der Namenszwang von Llama (verteilte Derivate muessen „Llama" fuehren)
  entfaellt; deshalb ist Llama bewusst nicht in der Liste.

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
