# Local AION pilot review, 2026-11-02

User authorization: the active goal requests an own, demonstrably trained model.
The user selected the current PC for the first training work.

Base: Qwen/Qwen2.5-1.5B-Instruct, revision
989aa7980e4cf806f80c7fef2b1adb7bc71aa306.
The public model repository supplies the Apache License 2.0 in LICENSE:
https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/blob/989aa7980e4cf806f80c7fef2b1adb7bc71aa306/LICENSE

The license permits modification and redistribution subject to its conditions.
The local export retains the base license and adds a modification notice.
The original model identity, revision and downloaded weight hash are recorded.
The CPU pilot configuration acknowledges this reviewed base-model license.

Data: only the project-authored corpus covered by PROCEDURAL_DATA_RIGHTS.md.
The 56 existing Teacher candidates remain pending and unreviewed; their
training rights are not assumed.

Training plan: the already configured 1.5B text-model pilot, LoRA on q_proj/v_proj,
16 optimizer steps, short sequences, periodic checkpoints, separate development
and held-out evaluation. CPU float32 is selected after an actual local matrix
benchmark; bfloat16 was materially slower on this processor.

This is an initial training milestone, not the full AION goal. It does not
establish a newly trained tokenizer, native tool runtime, multimodal MoE,
131072-token context, frontier Teacher parity or the complete capability gates.
Those requirements remain open after this pilot.
