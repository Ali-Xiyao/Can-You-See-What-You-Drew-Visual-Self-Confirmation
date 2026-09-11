# Findings

## Conditional local-runtime estimate at 91d8bbc, 2026-09-05 America/Los_Angeles

- User asks how long finding behavioral D* and a gradient signal would take if entrusted to the assistant on local 3090 hardware. This is an estimate request, not authorization to launch training, create a goal or schedule a background task.
- Read-only nvidia-smi confirms two RTX 3090 cards, each 24,576 MiB, both 0 MiB used at inspection. Estimate assumes both remain available; current training requires distinct generator/ladder devices and explicitly rejects one-card placement, so single-card execution first needs stage scheduling changes rather than simply doubling runtime.
- STATUS32/33 measured generation at 18.66 s/image on GPU1 under a specific concurrent workload; dual detection estimate 10.8 s/image and observe 1.5 s/image. These are historical workload-specific rates, not a fresh benchmark. The 18h preview figure is a forecast, not a completed run.
- Current preview has 10 rounds, 7 training prompts/round, K=8, two arms, 120 outcome images/arm/checkpoint. Each round entails 112 candidate images + 240 outcome images =352 generated images, around1.8245h generation; gold-arm training candidates56 + outcomes240 =296 dual-detected images, around0.888h detection. Current run_l3_preview.sh runs train first, then sequential per-arm/checkpoint generation and adjudication; merely assigning two GPUs does not overlap this work automatically. Rough serial generation+detection core is27.1h/10 rounds, before training, crops, loading, factual observations, gradients and any human adjudication. A pilot execution budget of30–40h or more is defensible; eighteen hours is not an observed completion time.
- The stopped preview never reached optimizer updates; STOPPED.txt records empty checkpoints. Its 30-update configuration is a small engineering preview, not evidence of sufficient training to expose a research effect. Current stage_generate still reports only cycle_scores; outcome-set s_select and the proposed prospective/equivalence inference are not yet wired. Thus runtime estimates must include engineering completion and a short learnability/timing check.
- Gate B gradients.meta timestamps span44m28s for stored232 prompts/3 criteria, but resume/incremental work and loading/IO must be checked before treating that as a fresh full-checkpoint benchmark. Repeating large51GB gradient artifacts per checkpoint would exceed current practical storage; trajectory storage/summary design is part of the needed engineering work.
- Planning estimate, explicitly conditional on existing1.5B/LoRA scope, both cards continuously available and no major new annotation need: roughly3–7 calendar days to a first useful pilot trajectory after fixes,7–14days for an initial evidence-based assessment of candidate decoupling and gradient trends,3–6weeks for multiple-run confirmation if the event exists and is measurable. These are engineering budgets, not probability estimates or promises that D* or positive predictive lead exists. A gradient trend alone is not a validated early-warning signal; meaningful confirmation requires observing an outcome event and prospective added value beyond outcome history.

## Review of oracle proposal at 419aff8, 2026-09-05 America/Los_Angeles

- This follows the previous 2026-09-06 Asia/Shanghai audit; the environment timezone/date changed. User asks whether the pasted work and next-step proposal are correct, not to launch more models or write a full proposal. Source/STATUS edits currently belong to the user's ongoing work.
- d835dd9 uses color-qualified questions and sums duplicate (color, canonical noun) spec entries. This addresses the previously confirmed scope bug on the current corpus. 419aff8 adds factual counting to v4_resolution_verdict.py. New untracked v4_residual_diagnosis.py analyzes the saved question bank; its reported 68/69 and 12/12 come from the OLD defective openct run, not the pending openct2 revision.
- Residual reachability uses any(correct/wrong pair with unequal projections). That does not establish full separation, score sensitivity, or correct top-ranked selection. Oracle needs actual expected-answer grading plus fixed tie-breaking, and should report wrong candidates tied at the best score, not only mean separation.
- Newly identified truth-completeness issue: verifier.py:222-240 explicitly preserves an agreed image-level verdict while dropping ALL detections of unresolved disputed keys from the retained core. These absent entries are not confirmed zero counts. A finalized image_correct therefore does not guarantee complete per-question image facts. resolved_by_crop's disputed field is historical and must not be confused with still-unresolved agreed_verdict disputes.
- New counting_accuracy treats unnameable and dropped disputed keys as absent/zero. My preceding factual-accuracy audit excluded unnameable but also failed to account for unresolved agreed_verdict keys; the earlier 76.4%/68.9% point estimates must therefore be superseded by a query-aware availability analysis. This does not invalidate the independently confirmed count-scope bug or its agreed-image examples.
- The within-arm two-gold comparison is valid as paired agreement description only after both targets are semantically valid and image facts defined. It is not a causal effect of prompting; distinct models can have distinct priors/biases, and the old request targets still contain the known scope defect. On conflicting target/image facts, report matching image, matching request, matching neither and missing rather than calling every deviation an observer error.

### Completed query-aware checks and new 40d4625 ceiling script

- Independent factual-count recheck: old openct has 297 candidates with agreed_verdict disputes. The old noun-total count queries intersect unresolved keys on 321 answers/arm; unnameable contributes another 40. The remaining 2,417/2,778 count facts/arm are usable, retaining resolved_by_crop facts. On that explicitly selected subset, naive errors 551/2417=22.80% and RFO errors 348/2417=14.40%. These replace neither the full-run target-mismatch numbers nor a full-population accuracy estimate. The prior nameable-only 76.4%/68.9% accuracy figures are superseded because they treated some unknown counts as zero.
- Corrected-question CPU oracle (current spec_questions, original 232 pools, original spec expected values, equal question weights and fixed seed/id tie-break): 858/1116 candidates have fully defined answers; 258 have unknown scores (242 touch unresolved disputed keys and 16 unnameable). All 415 correct images are fully defined and score 1. There are 79 entirely evaluable pools over 67 specs and 153 mixed-availability pools.
- On the 79 complete pools (378 candidates), oracle gives greater/equal/reverse 77/2/0; all-tied 2; full-score candidates 178. Fixed tie-break selects a correct image in 77/79. Highest-score ties occur in 55 pools, but only 8 contain an incorrect top-scoring candidate; ties among multiple correct candidates are expected and should not be called failure. Nine incorrect candidates in this complete subset score 1, proving residual question coverage gaps despite good aggregate resolution.
- Concrete coverage example: conf-2plus1:v4a-0081 requests a white mug and two black spoons, with two extra black lids in the image. Every requested existence/count fact can be correct while exact image correctness is false. main-1plus1plus1:v4b-0050 likewise has all requested objects and only extra-category disputes. These are the two fixed-tie-break selection errors in the complete subset. The two all-tied pools are different IDs: main-1plus1plus1:v4b-0033, conf-1plus1plus1:v4b-0049.
- For all 232 pools, retaining known question scores and assigning undefined question scores only their legal [0,1] range yields conservative outer bounds: greater pools 218–230, equal/all-tied 2–14, full-score candidates 429–552/1116, selected-correct pools 182–227. Reverse mean ranking is impossible in this corrected ideal-score construction because all correct images are known to score 1. These bounds fall within the four old resolution cutoffs but are a missing-score sensitivity analysis, not a defined factual oracle on unnameable images, and cannot rejudge the real observer experiment.
- User added a new pasted report during the audit; HEAD is now 40d4625, committing v4_observation_ceiling.py and v4_residual_diagnosis.py. STATUS remains user-modified. The new old-run oracle table reports 92.2% positive and 1.3% equal, implicitly leaving about 6.5% reverse. Passing four aggregate cutoffs does not certify semantic validity; the known defective question bank itself can pass them.
- v4_observation_ceiling.py:52–60 discards resolution/disputed and treats any non-None detections list as complete; :71–76 maps absent entries to zero/no. Thus it shares the factual-count report's unknown-as-zero bug. Additionally, on OLD openct, existence atom_id stores only a noun although its question text includes color; using the slug for existence truth silently drops the actual color condition. The new color-slug question format does not repair that historical mapping retroactively.
- The script's automatic conclusion at :125–129 ('shortfall is observation accuracy', 'rewriting questions buys nothing') overstates even a correctly implemented oracle result. Ideal factual answering under one fixed question/scoring map can establish available discrimination on that dataset; it cannot isolate question wording from answerability, certify every incorrect candidate is excluded, prove an effect of prompting, or validate longitudinal D*/D_g. This need not be a strict upper bound on every possible real/model-assisted scoring policy.
- Ran targeted CPU tests at the current source: envs/core/python.exe -m pytest -q tests/test_v4_probe.py, CUDA_VISIBLE_DEVICES empty, exit 0 (27 tests). Did not rerun the user's full 354-test suite or interrupt/launch GPU jobs. Recommendation: continue the bounded corrected-question answer collection, repair query-aware factual truth and the oracle report in CPU, then compare actual and ideal scores on the same evaluable pools before selecting any observer intervention.
- Final direct reproduction of 40d4625's old-run table: greater/equal/reverse 214/3/15, all-tied 2/232, full-score 398/1116=35.6631%. At main-2plus1:v4a-0089, the script scores the correct 3-book image 1/2 while three incorrect images each score 3/4; old incompatible count targets remain in use.
- Keeping the script's detections fixed, correcting ONLY old existence-query interpretation from bare noun to its actual color+noun changes 56/2778 generated existence answers from yes to no, affecting 54 candidates in 35 pools. This quantifies a semantic mapping error, not 56 independently established factual errors because some facts remain unresolved. Example main-2plus1:v4a-0000 candidate 2: question asks white plate, old atom_id merely says exists:plate, and script answers yes for a plate of another color.
- Oracle comparisons must use the identical question bank and identical evaluable candidate pools for actual and ideal responses. Do not subtract an actual rate over all 232 pools from the oracle's complete-subset rate over 79 pools. Preserve the full real-run preregistered verdict separately, with unknown coverage and bounds explicitly reported.

## Completed open-count experiment audit at 10f5dda, 2026-09-06

- Scope: verify the user's pasted report against code and complete artifacts before recommending the next step. CPU-only, no source/STATUS/preregistration/run changes, no model run. Two independent sub-audits covered exact selection statistics and full data/question integrity; the parent audited image-gold counting accuracy and synthesis. Previous 346-test pass belongs to the readiness review; this audit used artifact checks and targeted computations rather than claiming a new full-suite run.
- Confirmed registered verdict: naive correct-greater/equal/reverse pools are old 81/142/9 and new 143/69/20. New all-tied pools 60/232; full-score candidates 668/1116. Thus two of four registered criteria pass and the overall result remains NOT PASSED. RFO counts are old 167/63/2 and new 197/12/23, with 8 all-tied pools and 364 full-score candidates. Exact Fraction arithmetic reproduces these categories; floating-point ties did not change this run's result.
- Integrity: each arm has exactly 1,116 unique candidates and 5,556 answers (2,778 existence and 2,778 count), matching the complete question sets/order. All 11,112 answers have zero errors and zero abstentions; all 5,556 count answers are raw digits. All 1,116 current RGB hashes match old and new records, and re-normalizing all answers produces zero discrepancies. Pools/candidates/seeds/correctness/order are identical across versions except questions. Revisions match: RFO Qwen/Qwen2-VL-2B-Instruct@895c3a49bc3fa70a340399125c650a463535e71c; naive showlab/show-o2-1.5B@07ec16589d4fc5422a74dddbbc4b2cd11e551039. Fresh run isolation avoided the prior stale-cache bug in this run; zero abstention avoided the denominator issue, without fixing either future code risk.

### P1: Count question scope and expected-answer scope contradict each other

- probe.py:118 canonicalizes the noun, :136 asks the total number of that noun without its color, while :138 grades against obj.count for one color-qualified spec entry. Synonyms include notebook -> book and cup -> mug. A spec can legitimately have two colors of the same canonical noun. The old design already had this defect in a few multiple-count cases; adding singleton count questions expanded its reach.
- Affected: 24/232 pool rows, 18 unique specs, 112/1116 candidates; 48/580 pool-level count questions, expanding to 224/2778 count answers per arm. Two pools ask identical text with mutually incompatible expected values 1 and 2 (main-2plus1:v4a-0089, conf-2plus1:v4a-0043). The other 22 ask the same noun-total question twice and expect 1 despite requesting two in total.
- Direct example: runs/v4/gate-b-openct/pools.jsonl:72, main-1plus1plus1:v4b-0018, requests a red book, a blue notebook, and a yellow candle. Candidate :0 is adjudicated correct (source verified.jsonl:65). RFO observations.rfo.jsonl:285 answers both 'How many books' questions with 2, candle with 1, and all existence questions correctly, yet scores 2/3 because both book targets are 1. Parent visually inspected the corresponding PNG and confirmed the two book-category objects plus candle. Candidate :2 has the same issue.
- Second example: pools.jsonl:104, main-1plus1plus1:v4b-0060, requests blue mug + green cup + white plate. Candidate :0 has agreed image gold and image_correct=true (source verified.jsonl:221). RFO observations.rfo.jsonl:413 truthfully answers both mug-total questions with 2 and the other facts correctly, yet scores 2/3.
- Nine of the reported 12 RFO positive-to-negative pools contain this bug. A CPU diagnostic counterfactual retaining original text, answers, weights, candidate ordering and tie-breaking, changing ONLY expected counts to the spec's total canonical-noun count, restores EIGHT of those 12 to positive. The four remaining negative pools are main-1plus1plus1:v4b-0024, v4b-0034, conf-2plus1:v4a-0109, conf-1plus1plus1:v4b-0032. Both naive old-positive/new-negative cases restore to positive. Therefore the report cannot interpret all 12 as an inherent cost of improved resolution.
- Diagnostic counterfactual (not a new preregistered result): naive greater/equal/reverse 143/69/20 -> 152/68/12, all-tied 60->59, full-score candidates 668->679; RFO 197/12/23 -> 212/8/12, all-tied 8->5, full-score 364->410. Naive selected hits K4 67->70/138, K6 39->44/94; RFO K4 91->99/138, K6 59->63/94. Naive has 11 wrong-to-correct and 3 correct-to-wrong selection changes; RFO has 12 and 0. Even mechanically applying old numerical thresholds to this diagnostic still leaves naive two criteria below threshold (68>58 tied, 59>58 all-tied). DO NOT replace registered results or rejudge this run with these numbers.
- The popular gold=1/said=2 example is contaminated: 113/343 naive and 140/460 RFO such answers concern a noun whose total requested count is actually 2. They cannot all be described as wanting one and visually observing an incorrectly drawn second object.

### Corrections to the report's factual and causal descriptions

- Full-count target mismatch is naive 457/2778=16.45%, RFO 950/2778=34.20%, not the interim 17.4%/32.8%. Outside hypothetical {obj.count,obj.count-1} answers are naive 419/2778=15.08%, RFO 640/2778=23.04%, not 16.1%/21.6%. For singleton items the old design never asked a count, so 'outside old binary' combines added question coverage and answer-format expansion. The old 3.3% anchor also used the naive arm and old multi-count-only subset; it is not a matched comparison to all new RFO questions.
- These are mismatches with requested/spec targets, NOT observer errors relative to image facts. RFO and naive here are also different models, so their gap cannot identify the causal effect of showing a prompt. A prompt-leakage estimate requires a matched same-model contrast; existing earlier same-model studies can motivate the hypothesis but do not remove confounding in this comparison.
- Image-gold diagnostic excludes count facts for the 16 unnameable images (40 count answers/arm) while retaining those images as generation failures. On the remaining 2,738 answers/arm, factual counting accuracy is naive 1887/2738=68.92%, RFO 2091/2738=76.37%. Among answers differing from the saved expected value, only 221/449=49.22% naive and 527/939=56.12% RFO match actual image counts. Thus a target disagreement does not by itself establish correct visual observation. This is based on the existing verifier's image gold, not new human re-adjudication.
- Of the 218 auditable bad-scope count answers/arm, 106 concern entirely correctly generated images; perfect image counting is nevertheless penalized for all 106. On 2,520 valid-scope count answers/arm, 846 have actual image count differing from requested count. Naive answers image/request/neither on 117/589/140 of these; RFO on 399/230/217. This descriptive contrast is compatible with prompting/model differences but is not a same-model causal estimate.
- Existence was not completely unchanged: removal of count option randomization alters shared RNG consumption. Seven of 580 pool-level existence definitions swap A/B, affecting 32 candidate questions/arm. Full normalized-answer agreement is RFO 2776/2778=99.928%, naive 2768/2778=99.640%. For truly unchanged text/choices (2746 comparisons), it is RFO 2746/2746=100%, naive 2738/2746=99.709%. The original 99.89/99.60 figures are interim. High reproducibility is supported, but neither exact identity of the instrument nor exclusive attribution of every score change to count questions follows.
- STATUS:1838's claim that both naive selection CIs include the uniform baseline is arithmetically false. K4 Wilson interval is [40.3648%,56.8152%] and the per-pool uniform expectation is 40.2174%, just below its lower endpoint. K6 includes its baseline. Do not promote the tiny K4 distinction to a robust finding, especially given instrument defects and the dependent sample structure.
- 'The gate was attached to the wrong arm' is an unresolved design argument, not implied by another arm passing. The project's own Naive-score divergence remains relevant to its training claim. Future gates must be chosen by measurement purpose, not by which column clears existing thresholds.

### Pairing and bounded next-step recommendation

- The 232 pool rows represent 166 unique specs (66 appear twice, 100 once); K4 and K6 subsets are not independent. An exploratory spec-cluster paired bootstrap (100,000 replicates, seed 20260905, no multiplicity adjustment, original defective instrument) gives pooled new-minus-old selection gains naive +8.62 pp [3.48,13.78] and RFO +11.21 pp [5.04,17.45]. These describe current saved measurements; they do not validate the instrument or replace the registered gate.
- Immediate recommendation: keep the current architecture, original failed verdict and immutable artifacts; first repair the meaning of the count questions. Prefer color-qualified per-entry questions if retaining per-entry obj.count and the original color/count target. Alternatively use a documented canonical-noun aggregation with summed targets and deliberate handling of duplicate-question weighting. A factual image-gold audit is already possible and useful, but replacing intended spec answers with image facts in the selection reward changes the research target.
- Add semantic tests for same noun in two colors and aliases (book/notebook, mug/cup); a perfect factual observer on a spec-correct image must receive full score, and identical questions must not have conflicting targets. Stabilize existence option randomization independently of counting changes and freeze/version the full question/normalizer/scoring definitions before the bounded retest. Test outputs need not improve every metric; the target is valid measurement.
- Correct STATUS's causal and interim-number claims, add the discovered instrument limitation, then run one clearly versioned, bounded validation. Existing answered data can support a marked diagnostic for corrected labels; changed question wording needs fresh observations. This audit does not authorize a reclassification of the present run as passed or a full training/gradient launch.

## Readiness review at bc726a1, 2026-09-05

- Final decision: do not relaunch the existing gate-b shell chain or formal L3 training as-is. A bounded new-selector GPU diagnostic can follow fresh/versioned run isolation and explicit missingness/acceptance reporting. Longitudinal training additionally needs actual outcome-set selection-score measurement, appropriate D*/online-alarm implementation, and fail-fast handling of absent nonzero-round checkpoints. This is a targeted completion of the current approach, not a new architecture.
- Scope: audit the five commits after 7f8f597 and current artifacts; do not edit experimental code, revise the preregistration, or launch GPU experiments. Eleven tracked files changed. New proposal 3.5 clarifies evidence status and registers missing s_select measurement; new question builder asks an open count for every spec object; round-0 restoration adds a cloned adapter/RNG snapshot.
- Validation: ran `envs/core/python.exe -m pytest -q -m "not gpu and not slow"` with CUDA_VISIBLE_DEVICES empty. Exit 0; separately confirmed 346 collected tests under the same marker selection. Two existing Pillow deprecation warnings. Both `v4_selector_resolution.py --expect` and `v4_question_coverage.py` reproduced their published old-data and coverage numbers. Targeted Ruff found ten style/unused-import issues; none is the central execution blocker.
- Existing Gate B observation files are still September 3 artifacts and the current L3 directory still has September 4 STOPPED.txt, no new updates/evaluation outputs. No new open-count observation run is present in runs/v4. Current evidence establishes code/coverage changes, not improved model discrimination.
- P1 before remeasurement: question-version isolation is absent. `stage_observe` (scripts/v4_gate_b_probe.py:199) resumes on prompt_id/candidate_id only; `stage_select` (274) reconstructs current questions while accepting the old stored observations, and observation_score does not require a complete question set. The launcher still fixes OUT=runs/v4/gate-b (run_gate_b_gpu1.sh:20) and proceeds directly to overwrite selection and run gradients. CPU-only read/re-score reproduced 0 pending candidates, all 1,116 old answer sets incomplete under the new 4/6-question schema, and 551 changed scores without a single new answer. First candidate old=1.0, old answers scored against new expected values=.6666667. Require a fresh versioned output, preserved old artifacts, and mismatch rejection using a question/normalizer/scoring fingerprint plus exact answer-set validation.
- Scoring concern before training: observation_score (rfo/selection.py:29) drops abstentions from the denominator. CPU fixture with four current questions gives .75 when one count is wrong and 1.0 when only that same answer becomes an abstention. Increasing the vocabulary fixes some parse failures but leaves uncertainty as a reward. Current STATUS 40.6 acknowledges it; prereg has no coverage acceptance/denominator rule for validating the new score. For the bounded diagnostic, freeze and report coverage and missingness policy before new responses; retain frozen old scoring as its own baseline. Before training, use an explicit missing-answer policy that does not silently promote uncertain images, or explicitly justify studying that reward.
- P2 resume safety: restore_arm_state falls back to base for ANY absent previous path. A CPU fixture requesting nonexistent round-004 restored a weight from 10 to 1, returned base, and raised no error. Base fallback should be explicitly allowed only for round 0; missing later checkpoints must fail rather than restart an arm with stale/fresh optimizer state. Normal round-0 isolation itself is fixed and its eight tests pass in the suite.
- New-method reporting is not wired: scripts/v4_selector_resolution.py fixes GATE_B=runs/v4/gate-b and only exposes --expect. It verifies old published numbers, not acceptance thresholds for a new run. selection_table expects every pool to have a selected candidate and resolution has no explicit NaN/abstention accounting. Add a separate new-run report input, fixed prereg criteria, full coverage/denominators and explicit non-evaluable handling; keep --expect as historical verification.
- CPU report edge-case reproduction: a selector_abstained row raises KeyError('selected'); a mixed pool with one NaN candidate reports zero for all three separation categories while retaining n_pools=1 and returning mean_gap=NaN. Do not silently treat unavailable observations as valid discrimination or discard their denominator.
- Training/outcome readiness remains incomplete: scripts/v4_train.py:497 and CheckpointMetrics still only record cycle log-probability and external correctness; s_select is documented as missing but has not been implemented on the outcome set. analysis/breakpoints.py is unchanged and lacks the new equivalence/online-alarm semantics described in proposal 3.5. These do not block a bounded selector diagnostic, but they prevent treating current training output as the proposed reward-decoupling/early-warning result.
- Inference-unit caution remains: retained pools are 232 run/spec rows over 166 unique specs, with 66 shared across the K=4 and K=6 subsets. New prereg/script prose calling them independent subsets and quoting a simple n=232 SE overstates independence. Descriptive fixed-pool proportions are reproducible; generalization intervals should preserve spec clustering and pairing.
- Additional proposal correction before formal warning evaluation: lines 242-244 say no true positives implies no observed decoupling. This is false when FN>0 (all observed events were missed); event presence must be decided from TP+FN. The current four-way table also needs a declared finite forecast horizon/censoring treatment before interpreting uneventful end-of-budget alarms.
- Read-only final hardware snapshot: both RTX 3090 cards reported memory.used=0 MiB and memory.free=24,326 MiB. This is current availability, not a model-load or training canary. No GPU context/model evaluation was started by the review. Final HEAD remains bc726a1; no source, preregistration, or runs artifacts changed.

## Review of user's targeted measurement-refactor proposal, 2026-09-05

- User asks whether the pasted proposal is sound. Treat its closing offer to begin as part of the proposal being reviewed, not authorization to rewrite code or resume GPUs.
- Agree with retaining infrastructure, measuring decoupling without presuming a geometric kink, and testing prospective gradient information beyond outcome-history baselines. Do not promise an order-of-magnitude sample saving: event timing, near-zero external change, and repeated monitoring still require precision.
- Code correction: estimate_d_star intersects the admissible knot grids (usually the same grid), then searches for a shared knot minimizing internal_sse + external_sse subject to post-slope signs. It does not require intersection of two independently estimated/confident break locations.
- Statistical corrections: a CI crossing zero is inconclusive, not evidence of a plateau or proof of random noise. Specify a meaningful improvement/equivalence margin before data inspection. Raw internal-minus-external slopes have different units; a rising gap is not equivalent to internal improvement plus external stagnation, even after scaling. Paired difference variance is Var(R)+Var(Q)-2Cov(R,Q), not automatically less than either marginal variance. Sequential monitoring needs an appropriate error-control rule; no universal sample-size claim is supported.
- Verified primary methodological sources: Lakens (2017), https://pmc.ncbi.nlm.nih.gov/articles/PMC5502906/ (equivalence/non-significance and paired variance); Howard et al. (2021), https://arxiv.org/abs/1810.08240 (time-uniform confidence sequences). These motivate design constraints, not a plug-and-play test for these dependent training trajectories.
- Crucial selector distinction: image-specific factual answers are correct gold for auditing observer accuracy. Replacing the spec's desired answers with those factual answers in the generation-selection reward changes the task to recognizing an image, potentially rewarding an accurately recognized but incorrectly generated image. Keep intended facts and observed facts distinct; keep per-spec questions comparable across candidate images. More questions alone does not guarantee discrimination.
- CPU recomputation confirms the user's 83.2% full-score figure for all retained mixed-K probe pools: 928/1116 candidate scores = 83.1541%, 232 pools. Main K=4 retained subset: 453/552=82.0652%, 138 pools. These are candidate-level full-score rates, not pool-level tie rates or representative natural-pool estimates.
- Gate B correction: GDA-gold=.980621 has CI width .088431 (below .10); Gate B's failing statistic is GDA-free=.969911 with width .128493. GDA-gold requires verifier labels; the proposal explicitly reserves it for a mechanism reference and defines GDA-free as the unlabeled main signal. Neither cosine establishes that a decline precedes an outcome event.
- Do not collapse all absent/late warnings to k=0. Zero is simultaneous detection; a late alarm has negative lead under the current convention; no alarm is a miss; no outcome event leaves event lead undefined/censored. Prospective validation needs fixed thresholds, held-out training runs, and false-alarm/miss reporting.
- The old +4.17pt number is the Aug28 exploratory Naive-vs-base gain (.5625-.5208) in the earlier setting. Current Naive ties do not establish that historical gain was caused by tie-breaking. It is unsuitable as a precise planning effect without making that causal claim.
- Keep the training framework, but round-0 arm restoration and selection/evaluation score mismatch remain necessary changes. CPU can cover definitions, coverage checks, existing-data audits and statistical simulation. New model answers and new selected-sample gradients generally require GPU work; changed selection invalidates reuse of old gradient values as measurements of the new method.

## Sunk-cost-free assessment of breakpoint work, 2026-09-05

- User asks whether completed work is methodologically right for finding a training divergence, without giving past effort extra weight. This remains an assessment request, not authorization to resume training or rewrite experimental code.
- Keep the RGB/fresh-context manipulation, independent verification and adjudicated static evidence as measurement foundations. Treat static prompt effects as motivation, not evidence of a temporal breakpoint. Park additional Tier-B/error-prior work and expansion of step-0 gradient matrices until a useful training trajectory exists.
- The target should be a well-defined external/proxy trajectory relationship. A gradient feature is a candidate predictor; requiring a second abrupt gradient change in advance unnecessarily narrows the hypothesis. A stationary mismatch, gradual drift, and no observed event must remain valid outcomes.
- Current Naive score ties and conditional random-baseline comparison call for diagnosis, not modifying the proxy until the desired failure curve appears. Fix the definition/implementation mismatch while preserving a realistically imperfect proxy as the object of study.
- Shared LoRA targets are inside showo.model.layers; observe_atoms uses the same current model. Freezing the vision tower does not freeze the evolving self-judge. The coupled generator/judge setting is legitimate, but proxy drift cannot automatically be attributed to generator quality or intention leakage. Fixed reference images and/or a step-0 observer provide simple controls for the changing judge.
- pair_decisions (train.py:227) retains only prompts selected by every arm, so Gold abstention also removes the prompt from Naive. This is a conditional matched-selector comparison, not an unfiltered Naive training trajectory; interpret and report its scope accordingly.
- Important verification correction: evaluate.py:178 lists legacy resolution 'unnameable' as excluded, but the current verifier has no such Resolution enum value. Human unnameable images are emitted with resolution='human' and image_correct=False (verifier.py:182-195), and no saved v4 JSONL resolution='unnameable' rows were found. Do NOT report that current adjudicated unnameable failures are removed by external_correctness. Pending-human exclusion still needs coverage reporting and completion/appropriate uncertainty before interpreting a changing external curve.

## Bounded proposal revision assessment, 2026-09-05

- User requests a limited assessment of what in the existing proposal should change and the immediate next step; explicitly does not want a full replacement proposal or complete future plan. No proposal/source edits or GPU launches are implied.
- Read exact current abstract, claims §1, static-result interpretation §3.4, metrics §4, method §6, arms §7, statistics §8, and gates §9.
- Immediate editorial corrections: change promised conclusions to testable hypotheses; remove the untested assertion that static intention leakage is amplified into training divergence; align selection proxy/internal metric/gradient criterion; distinguish retrospective D_g from online alarm and censor unobserved D*; make Gold-vs-Naive a setting-specific positive control conditional on learning and adequate precision, not universal rejection of the project.
- CPU-only join of current gate-b/pools.jsonl and selection.jsonl (no re-observation, no memmap reads): main K=4 informative subset n=138 unique scenes, uniform-random expected selected-image accuracy 0.4021739, Naive actual 54/138=0.3913043, heterogeneous blind observer 76/138=0.5507246, Gold 138/138 by construction. Naive all four scores tied in 89/138 pools, highest score tied in 131/138; blind all tied 41/138, highest tied 128/138. Naive/blind select same candidate in 96/138.
- Full n=232 selected pool rows (166 unique scenes, mixed candidate counts): random expectation 0.3778736; Naive 86/232=0.3706897; heterogeneous blind 124/232=0.5344828; Naive all tied 140/232, highest tied 224/232. These are conditional descriptive statistics on the existing filtered probe set, not natural-pool results or a new significance claim.
- spec_questions checks existence for every requested noun/color, but counts only objects with requested count >1 and offers requested versus requested-minus-one. Thus the feedback checklist does not fully express the exact whole-image multiset gold criterion (extra objects/duplicated singletons may be unqueried). Recommend diagnose fact coverage and score ties before training; any new scoring variant must remain separate from frozen old evidence.
- Immediate next experiment recommendation: a bounded selection audit on natural K=4 pools, with fixed image-independent questions per spec, same-model prompted versus same-model blind conditions, analytic random baseline, and verifier reference. Report selected-image correctness, discrimination between correct and incorrect candidates, top-score ties, and excluded/abstained pool coverage. Existing heterogeneous observer result is a reference, not evidence that removing the prompt alone caused the selection gain. Choose next training step only after this audit.

## Method assessment follow-up, 2026-09-05

- Pool audit adds a material inference-unit correction: 232 retained Gate B pools cover only 166 distinct spec IDs/prompt texts (66 specs appear in two generation batches; 100 appear once). Current paired_bootstrap_cosine resamples rows, not spec clusters. Recommended new analysis should resample scene/spec clusters while keeping arms/checkpoints paired; do not treat current 232 as 232 independent scenes. No gradient memmaps were re-read or large report rerun.
- Recommend keep the research direction but revise protocol before more training: align selection proxy, internal reported score, and gradient construction; establish selector-level ranking/value on natural pools and actual learnability; keep D_g candidate signal rather than presumed lead; separate retrospective change point from prospective alarm time and measure false positives, missed events, and intervention value on held-out runs.
- Static prompt effects at step 0 do not imply a later training transition. Allow start-of-run mismatch, gradual drift, and no event. If no D* within budget, record right censoring; never substitute T_max as an observed event and claim a positive measured lead.
- Pre/post classification should distinguish proxy/gold decoupling from actual quality degradation, and normal convergence from detector false alarms. Compare no-change versus change models and estimate paired slope/change uncertainty rather than accepting any least-squares knot.
- Global gradient cosine may carry dominant shared task gradients and omit subtle error directions; its numerical level alone neither validates nor refutes it. Keep simple observation/selection disagreement and answer entropy as competing low-cost warnings; test added predictive value and reliable stopping rather than only two fitted kinks.

- User asks whether method is right, whether to revise or simply continue, and whether central idea involves an observable training divergence plus an internal gradient breakpoint. This is advice/discussion, not authorization to restart GPUs or rewrite preregistration.
- Conceptually D* is within a training trajectory: proxy score increases while external task correctness stops improving/declines; it is neither line crossing nor just Naive-vs-RFO arm separation.
- Current v4 select_by_observation/observe_naive and Gate B use prompted atomic-question correctness for naive selection. v4 evaluate.cycle_scores uses backbone.cycle_consistency_score = log p(prompt|image) under neutral captioning. These are distinct proxies and should be aligned or explicitly tested separately before describing a single self-reward being overoptimized.
- Current breakpoints.fit_segmented forces a one-break model by minimum SSE over candidates, without comparison to a no-break alternative. estimate_d_star gates slopes but does not require a positive external pre-slope or quantify external-flat equivalence. internal_noise_slope is a heuristic one-SEM-over-span floor, not formal slope significance.
- Current estimate_d_g fits the whole time series before selecting a persistent crossing and timestamps the beginning of the persistence window. This can be a retrospective estimate but is not a causally available online alarm. Proposal also maps unobserved D* to T_max, whereas current code returns None; recommend censoring, not calling the end of a run an observed event.
- Primary sources verified online: Gao et al. arXiv:2210.10760 reward overoptimization; GRIFT arXiv:2604.16242 (v2 Aug 23 2026) uses gradient fingerprints for reasoning-trace hacking detection; PRIME arXiv:2606.09711 studies learned precursors before visible coding-RL hacking. These motivate testing internal signals but do not establish that this project's aggregate gradient cosine leads its external correctness decline.

## 2026-09-05 refresh (newer evidence supersedes stale audit notes below)

- Independently recomputed paired question metrics from raw JSONL without loading models: 3623 pairs, 3581 scored; conflicts 436/667 correct image-only versus 224/667 prompted, with flips 221 vs 9; matching facts 2234/2281 versus 2260/2281. Note that gold_source describes the fact being asked, not necessarily correctness of the entire scene.
- Visually checked a concrete example v4a-0005, candidate 0, image v4-main-7035a5cb9d5a43d977a6-20260921-0.png. Request: one yellow banana, two green apples. One apple is visibly present; identical counting question is answered A (one) image-only and B (two) prompted. This is an illustrative case, selected to explain the aggregate result, not an additional statistical test.
- Current code inspection confirms no reset inside train_arm either: it operates on the shared model and calls optimizer.step. Round-0 second arm therefore inherits first arm's updates if execution reaches that code. Current stopped run had not reached those updates, so this did not generate the existing static results.
- Both later generation batches still contain 117+111 unique specs. Extra images/seed draws are not expansion to the roughly 1470 independent specs proposed for training/outcome/probe separation.
- Direct current L3 artifact counts confirm naive 56, gold 56, Qwen rows 56, no InternVL file, no checkpoint directory, no metrics or TERMINAL file.
- No full test suite or GPU experiment was rerun in this refresh; prior 2026-09-04 test counts are historical and will not be presented as current validation.

- Direct latest Gate B artifact: n=232, cosine=0.9699108259681765, 95% CI [0.8632055759529197,0.9916983721185697], width=0.12849279616565001 > 0.10, passed=false. Gold reference width=0.08843076718733045. This supersedes all n=138/queued descriptions below.
- L3 latest STOPPED.txt: stopped by user's request at 2026-09-04 21:32, not an experimental terminal condition. Round 0 naive/gold each generated 56 images; Qwen detected 56/56; InternVL never completed. No optimizer steps or checkpoints. Current process query finds no matching project experiment processes.
- STOPPED.txt records unresolved InternVL CUDA allocation/load failure. Its allocator-cap explanation is only a hypothesis, not a verified diagnosis. Several earlier capacity/parking/split/resource-race failures are in archived runs; these are instrument failures and do not reject the scientific hypothesis.
- Confirmatory third-batch raw conf_fork.txt: prone 21/173=0.121, never 15/149=0.101, one-sided Fisher p=0.3419. Second batch p=0.062. STATUS §35 locks the error-prior split as exploratory; no fourth batch. These results do not invalidate the main static prompt-effect result.
- Main final verified reports: 468+444=912 images, 103+139=242 correct (26.53% combined); 63+75=138 balanced pools (60.53% combined); 70 human decisions, 23 unnameable, zero pending. Question-level correctness and whole-image exact-match correctness use different denominators and must be explained separately.
- Main pooled raw comparison: 3623 paired questions, 3581 scored in both conditions, 42 image-only abstentions, prompted abstentions zero. Conflict subset n=667: 0.6537 to 0.3358; matched-fact subset n=2281: 0.9794 to 0.9908. Remaining scored questions n=633 have no diagnostic spec contrast. McNemar inference is question-level; avoid implying independent images/seeds or broad-model generality.
- Current scripts/v4_train.py still restores weights only when previous checkpoint exists before each train_arm; round 0 has no previous checkpoint. The old audit's initial-arm contamination concern remains visible in current code, and must be treated as a pre-relaunch correctness issue, not an already-observed training result.

- User asks for a detailed explanation of project goals, completed work, and results, not code changes or experiment launches.
- Git worktree initially clean; HEAD 7f8f597 (2026-09-04). No AGENTS.md found in project or checked ancestors.
- Reuse this existing isolated audit record; repository asks not to create additional planning documents.
- STATUS has reached §37c. Its top running-state text and some unresolved items predate its later sections, so latest artifacts/process state must be checked.
- Proposal v3.4 includes measured static main result: 228 specs, 912 images, 3623 paired questions; 667 image/spec-conflict questions fall from 0.654 to 0.336 accuracy when the original prompt is supplied. This is evidence of a static prompt effect, not proof of training-time divergence or advance warning.
- Prior audit notes about n=138 Gate B, queued reports, active processes, test failures, and cross-arm lifecycle bug are historical and must not be repeated as current facts without checking newer code/artifacts.

## Initial State

- 项目根目录原先没有 `task_plan.md`、`findings.md` 或 `progress.md`。
- 当前任务为只读项目审计；规划文件存放在隐藏的 `.planning/2026-09-04-project-audit/` 中。

## Evidence Log

### Repository identity and shape

- Package name is `selfsight` v0.1.0, Python 3.10–3.12, using NumPy/Pillow/SciPy/Pandas/Matplotlib for core analysis, PyTorch-adjacent stacks via optional `training`, `observer`, and `showo2` dependency groups.
- The project studies **visual self-confirmation in unified multimodal models**: whether a model that generates an incorrect image later accepts the image as correct because it matches its own generation intent.
- The active branch is `experiment/v2.3-rfo-gold`, 62 commits ahead of `origin`; latest local commit is `257a9c5` (`Give the Gate B waiter a liveness check instead of a wall clock`).
- There is one pre-existing modified product file: `scripts/v4_train.py`; it belongs to the user and must remain untouched during this read-only audit.
- The repository has evolved through v2.x, v3, and now v4. Current code includes dedicated `src/selfsight/v4/` modules and 11 v4 test files.

### Documentation contract

- `README.md` says the four authoritative roles are: proposal/specification, `STATUS.md` current state, `docs/EVIDENCE_LOG.md` measurements, and `docs/RUNBOOK.md` operations.
- `STATUS.md` is explicitly the single current-state source and reaches at least section 34, including results through 2026-09-04.
- The proposal defines Gates A–D, while current STATUS also introduces D* / L3 preview work.
- `docs/RUNBOOK.md` is partly historical: it explicitly says key v3 scripts were not yet implemented. Current v4 code appears to have superseded portions of that runbook, so implementation and STATUS must be checked before treating it as current.

### Scientific invariants from README

- Observer sees only RGB plus an atomic question—never prompt, latent/image tokens, KV cache, or source label.
- Training selector and gradient detector must be decoupled.
- Balanced candidate banks are for gradient probes; training/outcome curves use natural pools.
- SCFR and gradient-cosine claims require denominators/coverage and paired uncertainty intervals.
- Single-seed paths and mock results cannot support significance claims; failed/incomplete evidence is preserved.

### Execution setup

- Intended hardware: local Windows with 2×RTX 3090 plus Linux server with 2×A800 80GB; each GPU is an independent worker.
- Paths are rooted under the checkout except model weights (`H:\selfsight-models`).
- The new `configs/v4_l3_preview.yaml` openly labels the current L3 preview as underpowered: 228 available specs versus roughly 1,470 required; 30 optimizer updates can validate/falsify the pipeline but cannot falsify the scientific hypothesis.

### Errors / caveats

- `rg ... docs/prereg/*.md` failed because PowerShell/Windows passed the wildcard literally to ripgrep. Use `-g '*.md' docs/prereg` or explicit paths next time.

### Current scientific picture (STATUS/proposal, first pass)

- Current backbone is **standard `showlab/show-o2-1.5B` rev `07ec1658`, native 432px**, not the HQ 512px variant. Earlier v2/v3 audit results on HQ remain historical evidence but are not the current backbone result.
- v4 replaces synthetic colored geometry with verifiable natural scenes containing three everyday objects. Correctness is a single family-independent exact multiset match over `(object, color)` pairs.
- Question families are 2AFC: existence, counting, binding, spatial, absence. Atomic one-token answers avoid a second model for grading answers.
- The central failure is defined as: the rendered pixels violate the scene spec, but the generator answers a question according to the original prompt rather than the actual pixels.
- Primary metrics: `GDA-free = cos(g_naive, g_rfo)`, its breakpoint `D_g`, performance divergence point `D*`, `Lead = D* - D_g`, SCFR, POE, and CSR.
- Gradient meaning is operational: SFT-loss gradients on samples selected from the same prompts/candidate pools by three criteria—gold verifier, naïve cycle score, and isolated blind observation.

### Strong result already obtained before training

- v4 has completed both corpus halves through `generate -> detect x2 -> crop -> verify -> observe(image_only + prompted) -> analyze`; 70 disputed images were manually adjudicated and no gold labels remain pending.
- On conflicting image/spec trials, adding the original generation description drops answer accuracy from roughly **0.654 to 0.336** in the consolidated proposal result; self-confirmation rises from **0.346 to 0.664**. On matching trials the description slightly helps, so generic prompt length/distraction cannot explain the effect.
- The strongest subtype is “asked for 2, drew 1”: image-only accuracy 0.804 versus prompted 0.049, meaning the model usually asserts the missing object when its original description is restored.
- This establishes **static intention leakage at step 0**, but it does not yet establish the training-time breakpoint, gradient lead, generality across models, or a causal training intervention.

### Corpus/gold-standard lessons

- Synthetic geometry previously produced highly correlated candidates for the same prompt: even with p near 0.5, pools were usually all-right or all-wrong. Natural scenes make candidates nearly independent enough for selection, raising balanced-pool supply.
- Scene complexity is a cliff: five or more objects are essentially always wrong; v4 fixes three total objects and at most two instances per item.
- The previous geometry verifier was invalid (object-level F1 0.754 against blind human labels); Qwen3-VL achieved about 0.986 F1 on that benchmark. This forced the redesign of the gold-standard ladder.

### SelfSight-Bench v4 and Tier B

- The main pool planned 240 specs but currently contains **228**: 117 `2+1` scenes and 111 `1+1+1` scenes. Each spec has K=4 generated images, totaling 912 candidates.
- Prompt/spec authoring is performed by Qwen3-VL-8B and then subjected to structural/color whitelist checks; pass rate is only about 16%. The prompt and machine-readable spec are emitted together.
- Main-pool diagnostic-trial supply is uneven: absence and counting are plentiful, binding and existence sparse, spatial zero because natural specs intentionally omit spatial constraints to preserve generation correctness supply.
- Tier B creates controlled pixel counterfactual pairs from externally verified correct images: horizontal flip (spatial sensitivity), recolor (binding), object deletion (existence), and sham inpainting (artifact control).
- Actual Tier B output is 306 pairs, but only 153 are diagnostic; the proposal target of 300 diagnostic pairs is **not met**. This shortfall is explicitly reported rather than hidden.
- Recolor produces almost no prompted obedience (accuracy 1.000 -> 0.964), while deletion produces a larger drop (0.979 -> 0.835). A sham arm stays flat, ruling out inpainting artifacts as the source of the deletion effect.
- A same-spec origin comparison refutes the simple claim that obedience follows who caused the error. A post-hoc split suggests obedience follows whether the generator commonly makes that error, but the preregistered replication missed the one-sided 0.05 threshold (p=0.062); it remains exploratory.
- The deletion editor uses a bounding box minus protected-neighbor masks, LaMa inpainting, an edited-area ceiling of 20%, and the same two-detector verifier gate used elsewhere. Several idempotence/resume bugs were found and fixed (detector keying, whole-list gate, question RNG).

### Current corpus-level Gate A / adjudication

- Both v4 halves pass balanced-pool Gate A (0.5385 and 0.6757 versus threshold 0.40).
- Dual-detector disagreement still requires human review: 70/912 images (7.68%) were adjudicated, within the locked ≤10% ceiling. Labels: Qwen wins 38, InternVL wins 9, unnameable/invalid image `X` 23.
- `X` images count as generation failures in p but do not generate questions because no defensible pixel-grounded answer exists.

### Formal experiment gates and completion implications

- Gate A requires ≥40% balanced-pool rate and ≥128 informative pools. v4 passes this with 138 pools.
- Gate B requires external-verifier precision/cascade coverage, identical-selection cosine ≥0.999, paired-bootstrap CI width ≤0.10 for GDA-free, and a non-degenerate LoRA-space cosine.
- Current Gate B result on 138 pools: GDA-free cosine 0.9678, 95% CI [0.8199, 0.9927], width **0.1728**. The cosine is non-degenerate, but CI precision fails, so Gate B is **RED/incomplete**. Roughly 412 pools are projected to reach width 0.10.
- High cosine despite 30% selection disagreement suggests image choice moves gradients only slightly relative to prompt/loss effects. It is not yet interpretable due to the wide interval.
- Gate C is the life-or-death positive-control test: RFO-Gold must beat Naive at matched candidates/updates. It has not yet been established.
- Gate D contains the actual headline claims: positive `Lead = D* - D_g`, mechanistic ablations, and RFO-Self delaying D* or improving external correctness by ≥3 points. It has not been run/established.

### Training-loop reconstruction and data bottleneck

- Deleting the obsolete v3 geometry stack also deleted the old paired training/evaluation loop. By 2026-09-03, v4 code had to rebuild L1 training and L2 checkpoint evaluation against the new natural-scene/gold-standard instrument.
- The statistical breakpoint functions survived, but originally there was no code producing their training-curve inputs. Current `src/selfsight/v4/train.py`, `evaluate.py`, and `scripts/v4_train.py` indicate this loop has since been reconstructed and must be inspected/tested.
- Split on the 228-spec corpus yields 76 train / 120 outcome / 32 probe. The registered full design wants 640 unique training prompts and roughly 800 outcome prompts, so a meaningful D* experiment needs about **1,470 distinct specs**, 6–7× the current corpus.
- Reusing 76 train prompts over 4–9 epochs would confound memorization with learning and bend the external curve in the same direction that masks the target divergence.
- K=2 leaves the perfect selector without a correct candidate 61% of the time; K=8 cuts abstention to about 14% and is used for the L3 preview.

### L3 preview scope

- Preview config uses 76 train / 120 outcome / 32 probe, 10 rounds × 7 prompts, K=8, one epoch, and 3 optimizer steps per round (30 total; first 3 warmup).
- The preregistered preview can find a descriptive D* or expose pipeline failure, but flat curves mean “insufficient updates to move the backbone,” not evidence against divergence.
- Full project budget is estimated at 700–1,140 GPU-hours for one main seed, with an MVP at 160–270 GPU-hours. The full E2/E3/E4 program is far from complete even if the preview pipeline works.

### Live state observed at 2026-09-04 ~13:14 CST

- `STATUS.md` top is stale relative to the process table: the third-batch pipeline has progressed beyond its initial detection stage into `v4_tierb_build.py plan` for `runs/v4/tierb-conf2`.
- Active orchestration includes `scripts/run_conf2_pipeline.sh`, `scripts/run_gate_b_report_after.sh`, and `scripts/run_l3_preview.sh`; the latter has been alive since 00:46 and appears queued behind upstream work rather than having already completed.
- GPU snapshot: GPU0 idle (0 MiB) and GPU1 at 76% / 2,975 MiB. This likely reflects the current Tier-B planning/inpainting stage or another active process; logs must be checked before attributing it.
- The current third corpus exists on disk: `conf2-2plus1` (~314 MB, 1,647 files), `conf2-1plus1plus1` (~291 MB, 1,563 files), and a growing `tierb-conf2` (~46.5 MB, 262 files).
- `l3-preview` contains only 2 small files (~4 KB), so despite the commit title “Run the L3 preview to a terminal state,” the actual preview outputs are not yet complete. The commit adds orchestration and a terminal verdict mechanism; it is not evidence that the run itself finished.
- Latest Git history confirms v4 L1/L2 reconstruction: paired training loop (`3c76054`), per-checkpoint evaluation/D* runner (`4a16685`), paired arm corrections (`219e9c0`), L3 config/runner/verdict, and queue/liveness handling.

### Live pipeline details

- `conf2` generation and both full-image detections are finished. Verification currently reports:
  - `2+1`: n=1,633, p=0.2492, 95 pending human, balanced-pool rate 0.8291.
  - `1+1+1`: n=1,549, p=0.2866, 163 pending human, balanced-pool rate 0.8559.
- Third-batch protocol deliberately excludes pending-human images rather than adjudicating them, but reports exclusion-rate balance; therefore these 258 pending decisions do not imply the pipeline is blocked on user review.
- The pipeline is actively creating Tier-B deletion/inpainting images in its `plan` stage. `l3-preview/orchestrator.log` still says it is waiting for batch 3, and `gate-b/report.queued.log` is empty and queued.

### Implemented code architecture

- v4 core is ~3.3k source lines; supporting training ~399, observers ~765, backbones ~1.1k, analysis ~1.6k; scripts ~4.8k and tests ~3.4k lines.
- `scripts/v4_run_pipeline.py`: generate -> dual detect -> disputed-object crop -> verifier ladder -> observe.
- `spec.py`: canonical objects/specs, detected multiset, exact image correctness.
- `questions.py`: five 2AFC families and deterministic grading.
- `detectors.py` + `verifier.py`: Qwen/InternVL parsing and three-level adjudication.
- `tierb.py` + `inpaint.py`: counterfactual image edits and acceptance gates.
- `observe.py`: prompted versus fresh image-only observations.
- `probe.py` + `v4_gate_b_probe.py`: balanced pools, three selectors, LoRA gradients, bootstrap report.
- `train.py` + `scripts/v4_train.py`: frozen split, paired schedules, candidate selection, resumable round checkpoints, outcome generation, metric folding, and D* report.
- `analysis/breakpoints.py` and `v4/evaluate.py`: segmented fits, noise-slope guard, D*, D_g, and Lead plumbing.

### Worktree state / code under active repair

- `scripts/v4_train.py` has a user-owned uncommitted edit that avoids loading an unused observer during Naive-vs-RFO-Gold pass 1. This confirms the L3 runner is still being tuned immediately before launch and the tree is not a frozen release state.
- **Confirmed high-risk implementation defect:** `stage_train` swaps one shared backbone between arms by loading each arm's previous checkpoint, but in round 0 neither previous checkpoint exists. It trains Naive first, mutating the shared parameters, then trains RFO-Gold without restoring the base weights. Therefore RFO-Gold round 0 starts from Naive's updated weights, contradicting the module contract that both arms start from the same base weights.
- The existing v4 training tests cover schedules, pairing, abstention, selection, manifests, and resume sentinels, but do not execute/test the full `stage_train` cross-arm weight lifecycle. The defect is therefore not caught by current unit coverage.
- This matters immediately: `run_l3_preview.sh` calls this `stage_train` after batch 3 and Gate B reporting. Unless fixed before launch, the preview's paired comparison would be invalid from its first checkpoint.
- The current uncommitted observer-loading edit does not address this weight-state issue.

### Orchestration chain

- `run_conf2_pipeline.sh` completes crop/verify, Tier-B deletion planning/check/accept/questions, two observation conditions, then writes the preregistered `conf_fork.txt` sentinel.
- `run_gate_b_report_after.sh` waits for that sentinel, reruns the pure-read Gate B report over three large gradient memmaps, then writes `REPORT_DONE`.
- `run_l3_preview.sh` waits for both upstream sentinels, chooses GPU placement, runs the paired training, adjudicates every arm/checkpoint with the same dual-detector ladder, folds checkpoint metrics, estimates divergence, applies a mechanical preregistered terminal rule, and writes `TERMINAL.txt`.

### Gate B has already been expanded, but not re-reported

- `gate_b.json` is the older n=138 report (timestamp 19:02), while `gradients.meta.json` now records **232 prompts** and the run list includes both main and first confirmatory batches. Three float32 memmaps are 17.1 GB each, consistent with 232 × 18,464,768 gradients.
- The queued report will recompute Gate B at n=232; it is intentionally waiting for the disk-intensive third-batch pipeline because the report performs roughly 51 GB of near-random reads.
- A simple 1/sqrt(n) projection from width 0.1728 at n=138 gives about **0.133 at n=232**, so Gate B is likely still above the ≤0.10 threshold unless the added prompts materially reduce variance. This is an estimate, not a result; the queued report has not run.
- The third confirmatory batch is not part of those gradient files. Adding it to Gate B would require pool/observation/selection/gradient stages, not merely rerunning the report.

### Additional implementation details

- Outcome evaluation seeds intentionally ignore the `arm` parameter, so both arms draw the same latent for each `(step, prompt)`; this part of pairing is implemented correctly.
- External correctness excludes unresolved `pending_human`/`unnameable` cases rather than counting them wrong, and reports the unresolved denominator separately to avoid fabricating a D* from increasing detector disagreement.
- D* reporting uses a noise-derived minimum slope (median internal SEM divided by step span), preventing floating-point drift on a flat internal curve from being misclassified as “still rising.”

### Additional tooling error

- A second PowerShell `foreach (...) { ... } | Format-Table` command hit the same parser limitation while counting Gate B rows; use an intermediate `$rows` variable if the count is needed.

### Verification status

- The repository collects **300 tests**.
- Full CPU/file-level suite result on the current worktree: **299 passed, 1 failed** (inferred from one failure out of 300; pytest's double-quiet config suppresses the aggregate pass line).
- The single failure is in `tests/test_gradients_checkpoint.py`: the test calls `pytest.importorskip` without importing `pytest`; it then also references `_Model`, which is not defined in that file. This is a broken test implementation, not a runtime traceback from product code.
- Two warnings come from Matplotlib PDF backend using Pillow's soon-to-be-removed `mode` parameter; they do not fail output generation.
- Passing unit tests do not exonerate the round-0 arm-state bug because no test invokes `scripts/v4_train.py::stage_train` end to end.

### Concrete data example

- First `2+1` spec: prompt `two blue mugs, one white plate`; machine spec contains `(mug, blue, count=2)` and `(plate, white, count=1)` on a marble counter. The generator creates K images, detectors reconstruct the pixel-side multiset, and questions compare the observed image facts to this spec/prompt under image-only and prompted conditions.
- Main `2+1` run preserves extensive immutable artifacts: manifests, two detector outputs, crop resolutions, human labels/review codes, pre/post-human verified files, both observation-condition answers, paired comparison, and per-condition analyses.

### End-to-end completion gaps confirmed

- The v4 runner hardcodes `ARMS = ("naive", "rfo_gold")`; `RFO_SELF` exists only as a selectable helper and is explicitly excluded from the default pairing. There is no CLI argument that launches a Pass-2 training arm. Therefore **Gate D / RFO-Self training is not end-to-end implemented**.
- Legacy configs still list `arms: [naive, rfo_self]`, but `v4_train.py` ignores them and uses the constant. This is an acknowledged v3-era leftover, another documentation/config divergence.
- README/RUNBOOK commands reference missing files: `src/selfsight/cli.py`, `scripts/download_models.py`, `scripts/run_main_seed_e2.py`, `scripts/evaluate_checkpoints.py`, and `scripts/build_v3_scenes.py`. The live v4 flow uses different scripts, so the documented quick start and server runbook cannot be followed literally from the current checkout.

### Gold-standard main results verified from artifacts

- `main-2plus1`: 468 images, p=0.2201, 63 balanced pools, human rate 5.98%, 12 unnameable.
- `main-1plus1plus1`: 444 images, p=0.3131, 75 balanced pools, human rate 9.46%, 11 unnameable.
- Pooled paired result: 3,623 trials (3,581 scored); conflict trials n=667 drop from 0.6537 image-only to 0.3358 prompted, with 221 right→wrong vs 9 wrong→right and p≈5.29e-44.
- Matching trials n=2,281 move in the opposite direction, 0.9794→0.9908. This is why generic prompt distraction is implausible as the main explanation.
- Family conflict results are all same-direction: counting 0.8245→0.2500, binding 0.7971→0.4928, absence 0.5574→0.3552, existence 0.5000→0.2955.
- A real example (`v4a-0000`) requests two blue mugs and a white plate; both detectors agree the rendered image contains exactly that multiset, and the generated existence question (`plate` vs `banana`) is graded deterministically from image-derived gold.

### Further tooling error

- The combined `RFO_SELF` search + missing-file check failed to parse because of the same direct-foreach-pipe pattern; rerun with intermediate variables.

### Static-quality checks

- `ruff check src tests scripts` reports **43 issues**, 19 automatically fixable. Most are style/import hygiene, but actionable correctness signals include the two undefined names in the broken checkpoint test and an unresolved `AtomicQuestion` type name in `v4/questions.py` (safe at ordinary runtime under postponed annotations, but problematic for type-hint introspection).
- Full repository history has 92 commits; 62 are local-only relative to the tracked remote branch. The current worktree has the user-owned `scripts/v4_train.py` modification plus the audit's temporary `.planning/` directory.
- `git diff --check` reports only a CRLF→LF warning for the modified training script, no whitespace-error failure.

### More evidence of current maturity

- The scientific artifact trail is considerably stronger than the release/maintenance polish: main results, preregistrations, immutable failed evidence, and per-stage run outputs are meticulous; tests/docs/entrypoint consistency and the brand-new training orchestrator are not yet at the same level.
- This repository is an active research workbench rather than a packaged application or a reproducible public release. Its internal measurement pipeline is advanced; its public setup/run instructions are stale.

### Repeated tooling error

- The latest sentinel-status command again failed to parse due to direct piping after `foreach`; no state was changed. The final live-state check must use an intermediate array.

### Logged tooling errors

- Function inventory again used a Windows-incompatible wildcard (`scripts/v4_*.py`); corrected with ripgrep `-g 'v4_*.py'`.
- First LOC PowerShell pipeline had a parser error due to piping directly after a `foreach` statement; corrected by assigning rows first, then piping.
