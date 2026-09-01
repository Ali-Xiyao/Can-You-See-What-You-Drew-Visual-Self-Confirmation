# 72-image blind human baseline

The only human-labelled sample in this project, and the evidence every verifier
claim in STATUS 11 and the Proposal 6.2 rests on. It is checked in because it
cannot be regenerated: the labels were written by a person looking at the pixels,
once, before any verifier output on those images was visible.

## What is here

| file | contents |
|---|---|
| `blind_sample_72.jsonl` | the 72 candidates, sampled from the v3 held-out natural pool |
| `human_labels_72.py` | the labels, plus the note on how they were recorded |
| `qwen3vl8b_detections_72.jsonl` | Qwen3-VL-8B-Instruct output on the same 72 |
| `qwen3vl4b_detections_72.jsonl` | Qwen3-VL-4B-Instruct output, the rejected candidate |
| `internvl35_8b_detections_72.jsonl` | InternVL3.5-8B output |

## What it established

Object-level, against the human labels:

| verifier | precision | recall | F1 | whole-image object list exact |
|---|---:|---:|---:|---:|
| geometric (v3, deleted) | 0.750 | 0.757 | 0.754 | 0.486 |
| Qwen3-VL-4B-Instruct | 0.964 | 0.922 | 0.943 | 0.806 |
| Qwen3-VL-8B-Instruct | 0.981 | 0.990 | 0.986 | 0.944 |
| InternVL3.5-8B | 0.985 | 0.971 | 0.978 | 0.903 |

Verdict level, on the 48 with a scoreable verdict: geometric 0.708, 4B 0.792,
Qwen 8B 0.917, InternVL 0.958.

4B was rejected on the shape of its errors rather than their count. Its misses
are missing objects (12, against 8B's 1), and a verifier that misses an object is
indistinguishable from a generator that failed to draw it -- the exact confound
this project measures. Qwen and InternVL fail in opposite directions (Qwen
invents 3 / misses 1; InternVL invents 1 / misses 4) and share only one error,
which is what makes them useful as a pair.

## Its limits, stated rather than buried

Sample of 72, labelled by one person, on the v3 geometric stimulus. Two things
follow. Nothing larger than 8B could be separated at this resolution: 8B's four
residual disagreements are three annotation disputes and one real error. And the
stimulus is not the v4 stimulus, so these numbers transfer as evidence about
these detectors' object reading, not as a calibration of the v4 corpus. STATUS
asks for a 120-image observer audit of Qwen3-VL-8B; this is 72 of it.
