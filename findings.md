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

## 2026-08-28 — v2.2 backbone revision facts

- The official Show Lab repository now contains the `show-o2/` implementation and lists
  `showlab/show-o2-1.5B`, `showlab/show-o2-1.5B-HQ`, and `showlab/show-o2-7B`. The local sparse
  checkout is already pinned to the current official `main` commit
  `45a5a2de01d1ebd10cd5864d29310a76476cdf23`, but its sparse specification does not yet include
  `show-o2/`; expanding the sparse checkout is sufficient and does not require following a new
  mutable source revision.
- The official 1.5B Hub repository is approximately 5.66 GB and exposes a Diffusers loading path.
  A historical full revision `2fef922658dacf15ec1a962faf4a3ab19aa21643` is visible, but the
  currently displayed Hub head must be resolved through the Hub API before it is locked in code.
- Show-o2 is structurally different from Show-o v1: the official 1.5B training config uses a
  Qwen2.5-1.5B language base, continuous image latents, and a Wan 2.1 VAE. The v1 MAGVIT adapter
  cannot be relabeled as a Show-o2 adapter; generation, observation, and LoRA target discovery need
  separate implementation and readiness evidence.
- H: currently has about 86.3 GB free. That is enough for the first 1.5B candidate, source and a
  dedicated environment, but not enough to download every fallback model at once. Downloads must
  remain gate-ordered: 1.5B first, HQ only after failure, 7B only after both 1.5B checkpoints fail.
- Immutable Hub heads resolved on 2026-08-28 are: base 1.5B
  `07ec16589d4fc5422a74dddbbc4b2cd11e551039`, 1.5B-HQ
  `d3a220ec55feaacbdfcb053847edee14edd4e69a`, and 7B
  `3012b1d6aee8b57829b23d02cba9190ef5cc3361`. These are candidates for the model lock; only the
  base 1.5B is authorized for the first download/readiness attempt.
- The official inference code needs an additional Wan2.1 VAE checkpoint plus
  `google/siglip-so400m-patch14-384`, the Qwen2.5 tokenizer/base configuration, flow-transport
  utilities, and PyTorch flex-attention. The top-level Hub checkpoint alone is not a self-contained
  two-method Diffusers pipeline despite the generic Hub widget shown on the model page.
- The official base 1.5B demo is native 432x432, while the released 512x512 configuration points to
  `showlab/show-o2-1.5B-HQ`. Gate -2 must therefore record native resolution per checkpoint and must
  not silently evaluate base and HQ under mismatched latent geometry. The first base audit should
  use the official 432x432 config; 512x512 remains the HQ fallback and eventual paper setting if it
  passes joint readiness.
- The v1 schema currently puts `larger_than` inside the `spatial` family while `size` means a unary
  small/large attribute question. v2.2 must not mutate those old manifests in place. A versioned
  readiness-family layer should map predicates to `horizontal_spatial` and `larger_than` while
  retaining the original six-family schema for frozen v1 evidence and backwards-compatible tests.
- The existing `ModelAdapter` already exposes the four scientific operations needed by v2.2, but
  it has no declared capability/resource metadata or LoRA target discovery. A new backbone contract
  can extend rather than replace it, allowing `ShowoV1Adapter` to wrap the proven implementation and
  `Showo2Adapter` to provide independent loading/generation/observation logic.
- Existing evidence stamps hash artifact files but do not bind an ordered set of input reports,
  model/source/dependency revisions, native resolution, or eligible families. Gate -2 needs a
  dedicated decision schema with those fields and fail-closed validation before E1/E2.
- The user-approved amendment is stricter than a one-family exploratory gate. Gate -2B requires at
  least four families with open accuracy >=80%, yes-bias <=10pt, repeat agreement >=90%, and
  abstention <=20%. Gate -2C is precision-first: blind-manual verifier precision >=95%, overall
  primary-answer coverage >=80%, per-retained-family coverage >=70%, Oracle@K=4 >=70%, and fixed-seed
  coverage swing <=10pt. Gate -2D requires at least four families in the B/C intersection before E1
  or E2; the active plan and code must enforce this exact minimum.
- The v2.2 benchmark keeps six main families (existence, color, absolute size, horizontal/vertical
  spatial, count, and two-object binding). `larger_than` is removed from main spatial and relegated
  to an independent/appendix relative-size family, rather than replacing absolute size as a main
  family.
- Frozen v2.1 evidence hashes recorded for the new index include Gate -1 decision
  `968aa3c315bb039238bd6e101414ac755e9541fe5e39c0c69b338e2a825d988b` and decisive generated-domain
  report `cda066e699e71e31ef738016002a6d8b62944b5724000223adb0b539c964fd92`.
- The Show-o2 sparse subtree finished materializing at the locked source commit after the initial
  fetch outlived the command window. No stale lock was removed; the original live Git process
  completed and added `show-o2` while leaving the v1 sparse paths intact.
- The official environment script is an 8-GPU research environment, not a suitable Windows runtime
  lock: it includes flash-attn, DeepSpeed, TensorFlow, ONNX, video packages, and online W&B. The
  image-only inference path uses naive masks/SDPA in the relevant sections, but imports PyTorch
  flex-attention types unconditionally. Native Windows feasibility therefore depends first on
  import/load tests under Torch 2.5.1; the project should omit flash-attn/DeepSpeed and keep W&B
  disabled for the local canary.
- Exact dependency heads resolved for the first candidate are Wan2.1-T2V-14B
  `a064a6c71f5be440641209c07bf2a5ce7a2ff5e4`, SigLIP SO400M
  `9fdffc58afc957d1a03a25b10dba0329ab15c2a3`, Qwen2.5-1.5B-Instruct
  `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`, and the optional official safety checker
  `cb41f3a270d63d454d385fc2e4f571c487c253c5`.
- The official base 432x432 config uses 50 flow-matching inference steps (not 20), guidance 5.0,
  729 image tokens, and BF16. The candidate configuration must mirror those values for readiness;
  any later 20-step speed ablation is a separate registered condition.
- The official first-candidate plan is 12,775,937,051 bytes before the optional 1.22 GB safety
  checker: Show-o2 5.66 GB, Wan VAE 0.51 GB, SigLIP 3.51 GB, and Qwen2.5-1.5B weights/config/tokenizer.
  The Wan repository itself contains tens of gigabytes of unrelated T2V shards, so its
  lock must allow only `Wan2.1_VAE.pth` (SHA-256
  `38071ab59bd94681c686fa51d75a1968f64e470262043be31f7a094e442fd981`).
- Although the demo YAML says `load_from_showo: true`, the immutable Hub `config.json` embedded in
  the base 1.5B snapshot says `load_from_showo: false`, and the official demo calls
  `from_pretrained()` without overriding it. The reproduced official path therefore **does** load
  the standalone 3.09 GB Qwen weights before applying the unified state. Candidate 1 keeps those
  weights rather than introducing an unvalidated constructor optimization. SigLIP also loads its
  3.51 GB vision weight (SHA-256
  `ea2abad2b7f8a9c1aa5e49a244d5d57ffa71c56f720c94bc5d240ef4d6e1d94a`).
- The existing downloader rejects unregistered group names and the repository sync script's sparse
  set omitted `show-o2`. Both control surfaces must be extended before a reproducible fresh-host run;
  otherwise the manually expanded local checkout would work while the documented bootstrap would
  silently recreate only v1 source paths.
- HQ is the same 5.66 GB checkpoint footprint as base 1.5B, whereas Show-o2-7B is about 17.86 GB in
  two `.bin` shards before its Qwen2.5-7B construction dependency. These remain locked but excluded
  from the candidate-1 download group.
- The official T2I path uses the dense `omni_attn_mask_naive` path despite importing flex-attention,
  so an SDPA-first Windows adapter can reproduce the actual 432 demo without invoking the compiled
  flex kernel. MMU constructs image latents with the same Wan VAE, fuses semantic and generation
  embeddings, and autoregressively answers from RGB; the adapter can lift this logic without W&B or
  the safety checker for the benign geometric benchmark.
- The tokenizer adds only a pad token, `<image>`, `<|vid_start|>`, and `<|vid_end|>` on top of the
  Qwen tokenizer; all image boundary/pad tokens already exist in Qwen2.5. The 432 profile increases
  both image-token counts by one time token, yielding 730 T2I/MMU tokens and `max_text_len=290` in a
  1024-token sequence.
- Show-o v1 and Show-o2 both publish a top-level Python package named `models`. They cannot be safely
  imported into the same interpreter after either package is cached in `sys.modules`. The v2.2
  adapter must fail closed on a package-origin collision and readiness runs must use a dedicated
  Show-o2 process/environment; cross-backbone comparisons happen through artifact/JSONL boundaries.
- Gate -2 can be finalized without weakening v1 prerequisites: a separate decision schema validates
  A1-A4 identities, exact registered thresholds, per-family intersection, candidate rank, and every
  input SHA-256. Synthetic green/red/tamper/fallback tests confirm that four eligible families pass,
  three fail, modified evidence is rejected, and HQ cannot run without a failed predecessor that
  explicitly authorizes its model ID.
- The current v1 scene dataclasses and observer-audit wire format can safely carry v2.2 main-family
  records if new manifests retain the six existing `QuestionFamily` values. The minimal generator
  should therefore reuse `SceneSpec`, `Atom`, and the blind JSONL protocol while adding
  `schema_version: 2`, `benchmark_version: 2.2`, and a separate relative-size appendix manifest.
  This avoids a schema migration that could change frozen v1 parsing.
- Existing reference manifest rows already contain open plus both forced-choice orders, file/RGB
  hashes, and exact scene/atom data. A versioned readiness manifest builder can reuse that structure,
  and the existing observer auditor can consume it unchanged; Gate -2 aggregation must add the
  stricter repeatability/abstention checks because Gate -1 currently only gates capability and bias.
- A v2.2 generator must construct minimal scenes directly rather than calling the v1 `_make_objects`:
  v1 always creates two objects (three for count/binding), decorates every object with size/color/
  position, and still samples `larger_than` inside spatial. Reusing it would preserve precisely the
  compound-prompt confound that v2.2 is intended to remove.
- The isolated minimal generator can reuse the proven renderer/verifier without changing their v1
  semantics: all 186 A1/A2/A3 program references (6 + 120 + 60) verify exactly in the new tests,
  all six families are balanced, and semantic signatures are disjoint across canary/reference/
  generated splits. Materialization refuses any root not ending in `selfsight-v2.2` and fails closed
  if an existing RGB, manifest, or registry differs.
- A3 must not compute fixed-seed swing across the full K=1 set plus a smaller K=4 subset: those seed
  columns would have unequal family/prompt denominators. The implemented summary keeps overall and
  per-family coverage on all first candidates, while Oracle@4 and seed swing use only the A2-retained
  prompts that have all four registered candidates. It validates equal denominators before emitting
  the metrics.
- Verifier precision is conditional on cases where the deterministic verifier answered. With only
  60 first candidates, sampling a smaller audit would add avoidable selection variance, so the v2.2
  blind packet exports every answered K=1 case. The visible packet contains only an audit ID, RGB,
  atomic question, and empty human fields; all intended content and verifier output remain in the
  separate hash-bound key.
- PEFT receives the inner Qwen module namespace after `self.model.showo` is wrapped. The A1 audit,
  however, discovers names from the outer unified model (`showo.model.layers...`). The adapter must
  validate those full names against the audited outer tree and strip only the leading `showo.` when
  passing exact targets into PEFT; passing the outer names unchanged would select no modules.
- The HQ official profile is not a 50-step 432 configuration: it is 512px, uses a 32x32 latent grid,
  a 1,280-token sequence, and 20 Euler steps. Its same-resolution RGB observation path carries 1,024
  spatial image tokens plus the time token. Candidate profiles now record these values explicitly so
  base/HQ comparisons cannot silently reuse base geometry.
- Show-o2-7B construction depends on the separate `Qwen/Qwen2.5-7B-Instruct` snapshot when the Hub
  config follows the same official constructor path. Its current immutable revision is
  `a09a35458c702b33eeacc393d103063234e8bc28`, with four model shards totaling about 15.23 GB. It is
  locked in the rank-3 fallback group but remains excluded from the candidate-1 download.
- A readiness figure is fundamentally matrix data, not a distribution or trend. The truthful main
  visualization is therefore an exact annotated candidate-by-gate matrix plus a family-by-metric
  threshold matrix for the latest candidate; averaging families into bars would hide the joint
  intersection that defines Gate -2. Categorical pass/fail cells require no continuous colorbar.
- Matplotlib `imshow` rasterizes even a categorical matrix inside SVG, which defeats the vector
  export requirement. Drawing each cell as a `Rectangle` preserves vector structure and also allows
  redundant fail hatching. In the first preview, family labels at x=-0.18 intruded into the first
  cell; moving them to x=-0.55 cleared the overlap without shrinking the final-size typography.
- The SciPilot PDF checker treats top-level Type-0 fonts without a direct FontDescriptor as possibly
  unembedded and warns even when Matplotlib uses `pdf.fonttype=42`; it does not recurse into the
  descendant CID font. This is a conservative WARN, not a Type-3 FAIL. SVG and 600-DPI PNG checks
  pass, and the PDF remains generated with TrueType/fonttype-42 settings.
- Cloning a known-good Windows Conda CUDA environment can preserve a broken combination of pip
  package files and distribution metadata even when `python -m pip --version` worked in the source
  environment. `ensurepip --upgrade` is insufficient when the stale metadata already advertises a
  newer version; a force reinstall from the known-good environment into the clone's site-packages
  repairs the clone deterministically before dependency installation.
- The minimal native-Windows Show-o2 import path is viable with Torch 2.5.1+cu121 and the official
  source commit: `models`, `transport`, and flex-attention types import successfully, CUDA sees both
  3090s, and no Triton, DeepSpeed, flash-attn, or xFormers package was needed for the import canary.
- Publication renderers must select Matplotlib's non-interactive `Agg` backend before importing
  `pyplot` on native Windows. Relying on ambient backend selection makes the suite order-dependent:
  Tk may initialize successfully in earlier tests and then fail later with an invalid Tcl library
  command even though no figure needs a window.
- On Torch 2.5.1+cu121 for native Windows, `reset_peak_memory_stats` fails with `Invalid device
  argument` before the target CUDA context exists, even with integer device 0. Selecting the GPU
  and materializing an empty CUDA tensor first makes the same integer-indexed call deterministic;
  the shared helper now applies this to readiness, LoRA, and migration runners.
- Official Show-o2's `WanVAE` is a lightweight wrapper whose internal `.model` is already evaluated,
  frozen, assigned from meta, and moved to CUDA in its constructor. Treating the wrapper as an
  `nn.Module` is incorrect. Auditing both the unified model and `WanVAE.model` for residual meta
  parameters/buffers gives a stronger post-load invariant than accepting the constructor warnings.
- A green A1 can coexist with an apparently incorrect rendered count: the model produced a scene
  that visually may contain four green quadrilaterals while answering the registered count as
  three twice. A1 therefore remains strictly an engineering/functionality gate; A3 deterministic
  coverage and blinded verifier precision are necessary to distinguish seeing from expectation.
- Show-o2-1.5B clears the registered reference observation floor for five of six families with
  perfect repeatability and no response/bias pathology. Absolute size is the isolated failure:
  every small reference is read correctly but 9/10 large references collapse to small. Count's only
  failures are 2/10 four-object references collapsing to three. This supports family-conditioned
  retention rather than weakening the common 80% threshold.
- Gate -1b cannot safely reuse Show-o v1's `ShowoSFTBatch` for Show-o2. The latter's registered
  generation gradient is velocity/flow-matching over Wan latents, so the shared orchestration must
  dispatch `Showo2GenerationBatch` while preserving identical prompt IDs, candidate pools, and
  microbatch boundaries. Treating only the adapter surface as common avoids silently changing the
  training objective.
- Public-observer prerequisite checks must reject NaN explicitly: ordinary comparisons such as
  `nan < 0.8` and `nan > 0.1` are both false and would otherwise let malformed accuracy/bias fields
  bypass a threshold. All family, yes-bias, and abstention values are now finite fractions in
  `[0, 1]` before any Gate comparison.
- The RFO-Self selector cannot reuse the existing `showo` observer service after the backbone
  revision: that backend reconstructs Show-o v1. A distinct `showo2` service backend is required so
  the frozen observer is the exact step-0 member of the same unified model family. Keeping it behind
  the blind JSONL protocol preserves RGB/question-only isolation even though it uses a different
  Python environment and GPU.
- A base experiment-config digest is insufficient for Show-o2 resume because the exact LoRA target
  list is selected only after A1's real module-tree audit. The checkpoint contract must also hash
  the green Gate -2 decision, backbone config, and expanded target-module list; otherwise two runs
  with the same YAML but different audited targets could load each other's adapters.
- Checkpoint evaluation is part of the same scientific contract as training. It must filter
  outcome/probe families before stratified sampling and reconstruct the same objective-specific
  gradient batches; filtering only reported metrics after generating images would spend samples on
  ineligible claims and change the effective denominators.
- Scene identity cannot be reconstructed from prompt text alone. A3 deliberately includes cases
  where the same drawing instruction is paired with different atomic questions (for example a
  positive and negative existence query). Candidate filenames based only on prompt hash and seed
  therefore collide across scientifically distinct cases. The checkpoint/case namespace must
  include `scene_id`, and Gate evidence must assert one candidate ID and one image path per row even
  when repeated prompts legitimately produce byte-identical RGBs.
- Filtering the original six-family 2400/200/600 manifests after Gate -2 is sufficient for the
  640-prompt local pilot but not for formal E2: with one failed family it leaves only 2000 training
  and 500 outcome cases, below the registered 2400/600 unique denominators. Formal data must be
  regenerated deterministically after the green decision, redistributing the fixed totals across
  eligible families while excluding readiness signatures; it must not cycle or duplicate rows.
- A migration boolean is not adequate formal authorization. The A800 handoff must bind local and
  remote row/summary hashes, full backbone/source/dependency identity, and the exact eligible-family
  list. Formal orchestration should reject modified seed YAMLs, insufficient family-conditioned
  manifests, or missing migration evidence before creating a run directory.
- Blind-human precision and A4 test different necessary conditions, but neither can rescue an A3
  automatic failure in answer coverage, Oracle@4, retained-family coverage, or seed stability.
  Continuing into those stages after an automatic red wastes compute and risks representing absent
  measurements as negative results. A valid fail-fast decision must therefore preserve `null`
  evidence plus an explicit skip reason, while remaining permanently ineligible for E1/E2.
- Missing measurements require a third visual state. Encoding them with the failure color/hatch
  would silently turn a preregistered stop into apparent negative evidence. The readiness matrix
  therefore uses explicit `measured` metadata and redundant gray + dotted + `N/T` encoding; orange
  + diagonal hatch is reserved for checks that were actually evaluated and failed.
- A documented candidate ladder is not an execution control. If the downloader accepts a fallback
  group or direct model ID without validating the predecessor, an accidental command can consume
  storage and destroy the preregistered audit order before finalization has a chance to reject it.
  Authorization therefore belongs at the first mutating boundary—the download itself—and its
  predecessor SHA must be carried into the model registry.
- Authorization must also be revalidated at the final scientific decision boundary. A correct
  download registry does not prove that the predecessor evidence remained unchanged during the
  next audit, so fallback finalization independently checks every predecessor evidence hash and the
  adjacent candidate rank before binding the red decision SHA.
