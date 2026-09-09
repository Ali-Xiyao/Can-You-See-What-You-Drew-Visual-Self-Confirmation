**Use `resume_with_original_env.py` for the reviewed recovery.** The older
`resume_after_step24.py` and `expected.json` below are preserved preparation
artifacts, superseded after supervisor 26896 failed with a missing
`SELFSIGHT_MODEL_ROOT`. They must not be launched. No successful hold is claimed.

The replacement preserves copies of the failed state/logs here, checks the
frozen hashes, verifies no active owners, and runs a CPU-only check inside the
actual Show-o2 Python environment. It explicitly restores the non-secret path,
cache, temp and offline variables recorded in `recovery-expected.json`.

```powershell
& .\envs\core\python.exe .\review-packets\factual-diagnostics-20260906\continuation\resume_with_original_env.py --check-only
```

After review, the same command with `--execute` launches the existing supervisor
with `--through-round 10` and no `--accept-code-update`. Existing stage sentinels
make it first complete Naive step24 gradient and Gold step24 measurements, then
advance to round3. It does not change any existing source, parameter, threshold,
checkpoint or the original deadline. It returns after confirming the new PID
and original time origin; root remains responsible for monitoring the pilot.
No process is killed, no failed stage is marked successful, and no automatic
retry occurs. Launch hidden and keep stdout/stderr in this directory.

The prior preparation notes follow for provenance only:

This controller was prepared for the authorized resumption of the original
10-round / 80-update pilot after the current step24 hold completes.

The bound owners are helper **18492** and supervisor **26896**. Their creation
times, executables and commands are frozen in `expected.json`. The observed
supervisor stage at preparation was `naive.step-00024.gradient`.

The original 60-hour deadline remains **1788847464.2176192 Unix seconds**.
No `--accept-code-update` is passed. Historical diagnostic scripts outside the
frozen source list can be repaired independently; this controller does not edit
them, training sources, the protocol, configuration, prior helper or run data.

From the repository root, inspect without controlling or starting processes:

```powershell
& .\envs\core\python.exe .\review-packets\factual-diagnostics-20260906\continuation\resume_after_step24.py --check-only
```

After review, launch the waiting controller with `--execute` in a hidden process,
redirecting its output to this continuation directory. It must bind the original
process handles before those process objects become unavailable. A successful
check-only run does not keep those handles open for a later execute run.

Execution never stops any process. It waits for both real exit codes to be zero,
then checks `hold-complete.json`, all step24 stage records and frozen hashes. An
exclusive local lock, a one-attempt record, owner checks and the supervisor's
existing OS lock prevent overlapping execution. Missing identity, nonzero exit,
hash mismatch or exhausted budget causes diagnosis without automatic retry.

The launched command is the existing supervisor with the original output/config
and `--through-round 10`. It resumes completed-stage sentinels and uses exact
round indices; it does not repeat completed training. The controller also waits
for the resumed supervisor and records final step80 completion, independently of
whether any scientific divergence event is detected.
