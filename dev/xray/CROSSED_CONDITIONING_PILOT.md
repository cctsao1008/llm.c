# Crossed Conditioning Pilot

## Question

The crossed exact-state x suffix-execution pilot found three qualitatively different direct patterns:

- `42 / attproj`: execution disagreement is created on the alternate state at every sampled depth (`CCCCCCCCCCCC`).
- `1186 / fcproj`: disagreement is created on the alternate state at L0-L1 and disappears from L2 onward (`CC..........`).
- `9766 / fc`: disagreement exists on the reference state at L0 and is removed by the alternate state (`R...........`).

The next falsification question is deliberately narrower:

> Are these C/R patterns merely terminal decision-boundary conditioning, or does changing the complete suffix execution alter downstream state before the decision changes?

This pilot does **not** claim that terminal dot-product conditioning is a condition number for the full Transformer suffix. It only asks whether the final tied classifier is already sufficient to reproduce the complete CPU64-suffix winner from the exact GPU final-normalized state.

## Nested direct measurements

For each selected exact checkpoint state `h`:

1. `GPU suffix -> GPU terminal`
   - Original complete GPU suffix from the exact checkpoint.
   - Must bitwise reconstruct the corresponding natural execution target logits.

2. `GPU suffix -> CPU64 classifier on exact GPU ln_f`
   - Hold the exact GPU `ln_f` target row fixed.
   - Change only the tied vocabulary-projection arithmetic to CPU double.

3. `CPU64 complete suffix -> CPU64 terminal`
   - Evaluate the same exact checkpoint through the validated CPU64 complete suffix.

The direct winner pattern is classified as:

- `exec-agree`: original GPU suffix and complete CPU64 suffix choose the same winner.
- `terminal-classifier-sufficient`: changing only classifier arithmetic on the exact GPU `ln_f` is sufficient to reach the complete CPU64-suffix winner.
- `suffix-state-change-required`: classifier arithmetic alone preserves the GPU winner; the complete CPU64 suffix must alter downstream state before the winner changes.
- `mixed-third-winner`: a third-winner case; report without forcing a binary story.

These are nested sufficiency statements under the tested executions. They are not additive operator attributions.

## Terminal pair-dot conditioning

For the exact final-normalized state `u` and the fixed reference/competitor token pair, report

```
kappa_dot = sum_i |u_i * (w_ref_i - w_comp_i)|
            / |sum_i u_i * (w_ref_i - w_comp_i)|
```

The numerator and double-precision pair dot are also reported directly.

`kappa_dot` diagnoses cancellation in the **terminal linear pair readout only**. It is not a formal error bound and it is not a condition number for the complete suffix.

## Pre-registered specimens

The runner fixes five checkpoints before inspecting these conditioning results:

1. `42 / attproj / L11`
   - terminal-sensitive positive control from the earlier terminal-readout audit.

2. `1186 / fcproj / L0`
   - created execution disagreement.

3. `1186 / fcproj / L1`
   - created execution disagreement.

4. `1186 / fcproj / L2`
   - first depth where both tested complete suffix executions choose the competitor.

5. `9766 / fc / L0`
   - disagreement removed by the alternate state.
   - especially informative because the reference and alternate original-GPU pair magnitudes are already close (`~2.06e-4` vs `~2.29e-4`) in the same context, depth, and token pair.

## Validity gates

Interpret a case only when all are true:

- `ref_repeat_exact=1`
- `alt_repeat_exact=1`
- `ref_gpu_replay_exact=1`
- `alt_gpu_replay_exact=1`
- `case_valid=1`

## Falsification logic

### Outcome A: terminal explanation dominates

If the C/R cases are predominantly `terminal-classifier-sufficient`, and the disagreeing states are distinguished by substantially worse terminal pair-dot conditioning, then the current state-conditioned-execution story should be reduced to ordinary near-boundary / terminal-readout conditioning. Do not expand the population yet.

### Outcome B: full-suffix state change is required

If important C/R cases are `suffix-state-change-required`, especially where a same-context near-|margin| state does not show the same behavior, then terminal classifier conditioning alone is insufficient. This keeps alive the narrower hypothesis that numerical state displacement changes susceptibility to **downstream execution dynamics**, not merely to the final linear readout.

This still does not establish novelty or a new mechanism. The next step would be matched controls for complete-suffix conditioning/sensitivity, not an operator-localization probe.

## Stop rule

If every informative C/R checkpoint in this pilot is adequately explained by terminal classifier sufficiency plus terminal pair-dot conditioning, stop this paper line before population expansion.
