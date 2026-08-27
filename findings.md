# Findings & Decisions

## Requirements
- Implement the approved dual-3090 local pilot and single-A800 formal experiment architecture.
- Keep code in the proposal workspace and all large/cache/runtime data on H:.
- Use native Windows locally with curated dependencies and SDPA.
- Build SelfSight data, deterministic verifier, Gate audits, RFO isolation, LoRA/GDA instrumentation, local E2, Figure 1, and A800 handoff.

## Environment Facts
- Workspace initially contains only the 66,923-byte proposal; it is not yet a Git repository.
- Host has two NVIDIA GeForce RTX 3090 GPUs, each 24,576 MiB, driver 560.94/CUDA compatibility 12.6, and no active NVLink.
- Host has 63.8GB RAM and 20 logical processors.
- H: had 147GB free at inspection time; user will clear sufficient space and wants all large paths on H:.
- C: had about 5.4GB free, so TEMP/cache redirection is mandatory before package/model downloads.
- Python 3.10 is available at `D:\python\python.exe`; Python 3.12 is the current default; Conda 4.12 and git-lfs are installed.
- WSL currently only has a stopped `docker-desktop` distro; native Windows is the selected first path.
- No `AGENTS.md` or existing source tree is present; the implementation starts from a clean proposal-only workspace.
- H: currently has about 127GB free; this is enough for the local observer ladder but remains
  below the 650GB formal-stage target. All generated data, environments, repositories, caches,
  models, and runs remain on H:.

## Research and Model Facts
- Official Show-o supplies 512px pure-discrete and CLIP-ViT checkpoints plus Accelerate training code, but no native PEFT/LoRA integration.
- Official Show-o requirements include Linux-oriented optional packages; a curated Windows environment is safer than installing the full list.
- Main checkpoint weight is about 5.4GB; MAGVIT-v2 about 0.36GB.
- Janus-Pro-1B is about 3.9GB and inference-only in this project.
- Candidate observers form a predeclared capability ladder: SmolVLM-500M, InternVL2-1B, Qwen2-VL-2B; Qwen2.5-VL-7B is an upper-bound observer.
- Main data is programmatically generated; LLaVA/LAION/ImageNet pretraining corpora are not required.
- Tier C external data is deferred until after the primary local/A800 gates.
- Pinned repository heads captured at implementation start: Show-o `45a5a2de01d1ebd10cd5864d29310a76476cdf23`, Janus `1daa72fa409002d40931bd7b36a9280362469ead`, SmolVLM-500M `a7da5b986cb59b408707209984f360a5f4ad7e47`.
- A batched PowerShell REST revision query produced no usable output after 30 seconds; direct `git ls-remote` is the verified revision-resolution method for repositories.

## Technical Decisions
| Decision | Rationale |
|---|---|
| Use short H-drive roots via environment variables | Avoid C-drive exhaustion and Windows path-length/cache problems |
| Separate Show-o, observer, and legacy-eval environments | Show-o, current VLMs, and MMDetection/VQAScore have incompatible dependency generations |
| Isolate Janus in a minimal Torch>=2.6 environment | Current Transformers safely refuses PyTorch `.bin` loading below 2.6; Qwen/Intern remain untouched in their proven environment |
| JSONL subprocess observer protocol | Enforces environment separation and RFO information isolation |
| Program-rendered reference images for Gate -1 | Provides exact scene truth independent of any learned judge |
| Deterministic CV verifier for controlled generated images | Allows external correctness without a drifting VLM judge |
| English prompts/questions | Matches checkpoint training distribution and proposal assumptions |
| Adapter-only checkpoints | Makes 21-checkpoint trajectories practical and portable |
| Publication Figure 1 uses shared x with stacked axes, not dual y-axis | Prevents visually manufactured agreement/divergence and follows SciPilot guidance |
| Disable Hugging Face Xet on this Windows host | The Xet/proxy path stalled at zero useful bytes; direct locked LFS URLs work reliably |
| Use aria2 only for large locked LFS objects | It provides resumable multi-range transfer while expected size and LFS SHA-256 preserve revision integrity |
| Exclude non-PyTorch export trees by default | SmolVLM's ONNX exports alone inflated the plan from about 1GB to 6.9GB without serving any configured backend |
| Separate exact-reference and approximate-generation verifier modes | Exact palette matching is ideal for rendered ground truth but is not a fair parser for antialiased/rotated/hollow Show-o outputs; both remain deterministic and are gated separately |

## Resources
- Proposal: `Can You See What You Drew Visual Self-Confirmation.md`
- Show-o: https://github.com/showlab/show-o
- Janus: https://github.com/deepseek-ai/Janus
- GenEval: https://github.com/djghosh13/geneval
- T2I-CompBench: https://github.com/Karine-Huang/T2I-CompBench
- MME-Unify: https://github.com/MME-Benchmarks/MME-Unify
- VQAScore: https://github.com/linzhiqiu/t2v_metrics

## Issues Encountered
| Issue | Resolution |
|---|---|
| Full plan includes multi-day GPU work and large downloads | Implement gated smallest-scale evidence first and persist resumable commands/artifacts |
## Implementation Findings (2026-08-27)

- A blind observer request must not serialize `AtomicQuestion.expected_answer`; merely removing the prompt is insufficient because the answer itself would otherwise cross the process boundary.
- The first Tier-B audit caught two semantic bugs before any model evaluation: swapping two objects with equal x coordinates does not change left/right truth, and pixel area is not a valid universal proxy for categorical size across different shapes (a large triangle can have less area than a small square). The implementation now requires strict x separation and uses detected size labels.
- Figure 1 uses three vertically stacked panels with a shared optimizer-step axis. Internal/external scores share a proportion axis; gradient cosines and their Gate -1b band use their own panel; SCFR uses the third. This preserves the registered temporal argument without a misleading dual y-axis.
- All formal decision helpers reject fewer than three seeds for paired-bootstrap inference. Local one-seed output is explicitly exploratory.
- PyTorch's official Windows CDN transferred the 2.455GB Torch 2.2.1 wheel at about 85KB/s on this host (an >8 hour download). A pre-existing Python 3.10 CUDA environment contains Torch 2.5.1+cu121 and is safe to clone locally into H:. The local canary may use that clone after pinning Show-o's Transformers stack; the A800 formal environment remains on the exact official 2.2.1 lock, and the migration canary must quantify any difference.
- The complete corrected reference audit is exact: 3,200/3,200 images parse and answer correctly across existence, count, color, size, spatial, and binding; all train/probe/outcome overlaps are zero. The 400 Tier-B counterfactual pairs also pass exactly with the registered 100/100/50/50/100 composition. Manual stratified agreement remains explicitly pending and must not be inferred from these programmatic results.
- Show-o's CLIP-ViT checkpoint implicitly asks Transformers for `microsoft/phi-1_5`; locking only the advertised Show-o/MAGVIT/CLIP repositories is therefore insufficient for a truly offline, revision-fixed canary.
- Model acquisition on this host needs transport separation: Hugging Face Hub is retained for small metadata/config files, while large LFS objects are downloaded directly into the locked snapshot with aria2 and verified against both declared length and LFS SHA-256.
- Restarting aria2 from its control file safely retained 3.8GiB of completed Show-o ranges and restored 15 parallel connections; segmented state is therefore the operational recovery path for Windows proxy slowdowns.
- The manual verifier audit is now explicitly blinded: review sheets expose only pixels, an atomic question, and an audit ID. Prompt, expected answer, and verifier answer live only in a separate digest-protected key, preventing intention leakage into the human check.
- Registered manifests are written in family blocks. Any `first N` pilot silently destroys the
  six-family design (the original local probe was all existence questions). Every limited path
  now uses a stable hash/seed stratified sample; the balanced counts are 107/107/107/107/106/106
  for 640 train prompts, 6/6/5/5/5/5 for 32 probes, and 20 per family for 120 outcomes.
- The real Show-o adapter is not merely a mocked shell: generation, MMU RGB reload, fixed-target
  PEFT attachment, combined T2I/replay loss, backward, AdamW update, adapter digest restoration,
  and optimizer/scheduler/RNG resume all ran on an RTX 3090. Engineering stability therefore
  passed independently of scientific competence.
- Show-o's controlled-generation weakness is currently the main blocker. Four independent prompt
  styles still fused/nested objects or invented geometry. On a balanced train-only 12-image
  canary, the fixed 25-step configuration gave 50% primary-answer coverage and 16.7% correctness;
  the official README's 50 steps only reached 66.7%/25%. This rules out generation-step count as
  the main explanation.
- A deterministic HSV/contour verifier can recover rotated, hollow, and nested foreground shapes
  without using prompt/scene/answer information. Its development v2 improved fixed-image answer
  coverage to 83.3% and correctness to 41.7%, but two genuinely ambiguous generations contained
  multiple target shapes with conflicting colors/sizes. Forcing a choice would manufacture
  coverage; Gate 0 remains red until the 60-image dev audit and a blinded generated-image human
  audit justify a frozen rule.
- The larger balanced 60-image train-only audit is decisive: contour answer coverage falls to
  55% (binding 0%, spatial 30%, color/size 50%) and intended-scene exact recovery is 0%. The
  contact sheet shows real duplicate target shapes and conflicting attributes, so selecting the
  highest-confidence contour would conceal ambiguity rather than improve measurement. Verifier
  threshold search is therefore stopped; the project needs a material domain-adaptation or
  measurement-design decision before self-training.
- SmolVLM is repeatable and largely unbiased but not capable enough for the registered detector:
  120 reference images yielded 100/95/100% for color/existence/binding, but 65/70/70% for
  spatial/size/count. Only 3/6 families clear 80%, so it fails Gate -1 despite 83.3% macro accuracy.
- InternVL2-1B is also not eligible: the balanced 120-reference audit reached 70.8% macro open
  accuracy, only color/existence/binding cleared 80%, and its 30pt absolute yes-bias exceeded the
  registered 10pt limit. Its smaller size does not supply the missing capability-matched detector.
- Show-o itself fails the registered observation floor on program-perfect references: its 65.0%
  macro accuracy hides only 2/6 passing families (color and binding). The failure is not yes-bias
  (5pt) and cannot be repaired by selecting a stronger heterogeneous observer; Gate -1 is red by
  the preregistered Show-o condition alone.
- Qwen2-VL-2B demonstrates that the test is solvable: it reaches 90.0% macro accuracy, passes five
  families, and stays within the bias limit. It cannot serve as the matched control because its
  macro accuracy is 25pt above Show-o instead of within 3pt. The ladder therefore distinguishes
  observer incapacity from an intrinsically impossible benchmark while still leaving Gate -1 red.
- The finalized Gate decision hashes every input report. This prevents a later observer result,
  threshold change, or regenerated reference audit from silently altering which evidence justified
  the stop decision.
- The former InternVL2-1B SHA was a transcription error and did not resolve. The corrected lock
  `0d75ccd166b1d0b79446ae6c5d1a4a667f1e6187` is the exact value independently returned by the
  official Hub and the repository's `refs/heads/main` at lock time; mutable fallback was never used.
- Third-party observer stdout is quarantined to stderr so remote-code informational prints cannot
  enter or desynchronize the blind JSONL evidence channel.
- Every real phenomenon entry point now fails closed on a missing, malformed, inconsistent, red,
  or threshold-lowered prerequisite report before allocating a model or creating an experiment run.
- Gate identity now includes the exact selected detector model, revision, and capability-audit
  SHA-256. This prevents an operator from accidentally substituting Qwen (or another stronger,
  unmatched observer) after a future green decision while keeping copied evidence portable across
  Windows and Linux paths.
- The registered Tier-D subset is now concrete rather than a config-only count: six Tier-A
  families contribute 50 images each, and five Tier-B intervention categories contribute 30
  complete pairs each. All 600 images retain questions, atoms, provenance, pair links, and hashes.
- Absolute H-drive image paths cannot be consumed on A800. The migration tool now writes a new
  platform-local manifest view, verifies every original file/RGB hash, and leaves the Windows
  source manifests immutable; a full same-host dry run reproduced every manifest byte-for-byte.
- Pure-discrete Show-o is repeatable but does not solve the ability-floor confound. Its 80.0% macro
  average hides only 2/6 passing families (color and binding); existence/count/spatial/size remain
  75/70/70/65%. This mirrors the CLIP variant's family pattern despite a higher macro score.
- Show-o's official pure-discrete `mmu_generate` grows an FP32 attention bias after every token.
  Forcing that audit path to BF16 causes a genuine query/bias mismatch; leaving the inference-only
  audit in FP32 fits one 3090 and does not alter the BF16 trainable backbone definition.
- Janus-Pro-1B passes the standalone capability and bias checks on the same balanced 120-reference
  audit: 87.5% macro, five families at or above 80%, and 7.5pt absolute yes-bias. Its spatial score
  remains 60%, it abstains on 15.8% of open questions, and its macro score is 22.5pt above Show-o;
  it is therefore another strong diagnostic upper row, not a capability-matched detector.
- The six-backbone Figure 2 expansion passed both programmatic layout QA and direct color/grayscale
  review. Its subtitle says the original Gate -1 decision is frozen, so diagnostic additions cannot
  be mistaken for a post-hoc gate redecision.
- The locked Qwen2.5-VL-7B upper-bound snapshot is fully materialized (16,595,981,281 registered
  bytes). On GPU1 it loads in 20.46 seconds with 16,636,220,928 peak allocated bytes, gives 6/6
  correct first answers and 100% agreement across 12 deterministic requests. This establishes that
  the 7B inference ceiling fits one local 3090, but it is not a new Gate candidate after the decision
  freeze and receives no post-hoc 120-image audit.
