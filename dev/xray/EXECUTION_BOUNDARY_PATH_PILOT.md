# Execution-Boundary Path Pilot

## Question

The crossed-state × suffix-execution pilot showed that numerical state displacement can create (`C`) or remove (`R`) GPU-vs-CPU64 decision disagreement. The conditioning pilot then showed two distinct mechanisms:

- scan 42 / attproj L11: terminal classifier arithmetic alone is sufficient;
- scan 1186 / fcproj L0-L1 and scan 9766 / fc L0: the complete CPU64 suffix must change downstream state before the winner changes.

This pilot asks a narrower geometric question before any population expansion:

> Along the straight FP32 state segment between the exact natural reference and alternate checkpoint states, where do GPU and CPU64 suffix decisions agree or disagree, and is disagreement terminal-classifier dominated or full-suffix-state dominated?

## State path

For one fixed residual3 checkpoint layer,

\[
h(\alpha)=h_{ref}+\alpha(h_{alt}-h_{ref}),\qquad \alpha\in[0,1].
\]

`alpha=0` and `alpha=1` are copied bit-exact from naturally executed reference/alternate checkpoints. Interior points are counterfactual FP32 interpolated states; they are not claimed to be naturally executed states.

The primary alpha grid is pre-registered as:

```text
0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1
```

No monotonicity, single-crossing, or linear-response assumption is made.

## Three direct downstream evaluations per alpha

For the same FP32 checkpoint state:

1. original GPU complete suffix;
2. the exact GPU `ln_f` activation from (1), with only the tied classifier reevaluated in CPU64;
3. validated CPU64 complete suffix.

The point is classified as:

- `.` — GPU and CPU64 complete suffix agree;
- `T` — they disagree and CPU64 classifier on the exact GPU `ln_f` is already sufficient to reach the CPU64-suffix winner;
- `S` — they disagree but classifier-only switching is insufficient; the complete CPU64 suffix must change downstream state;
- `M` — mixed/third-winner case.

## Conditioning observable

For the terminal reference-vs-competitor dot product, report

\[
\kappa_{dot}=\frac{\sum_i |u_i(w_{r,i}-w_{c,i})|}{|\sum_i u_i(w_{r,i}-w_{c,i})|}.
\]

This is only a diagnostic of terminal pair-dot cancellation. It is not a complete-suffix condition number and not a formal error bound.

## State-displacement audit

Report both causal-prefix and target-row displacement:

- reference L2 norm;
- `||h_alt-h_ref||_2`;
- relative L2;
- max absolute component delta.

This quantifies how small the natural state displacement is before interpreting any disagreement geometry.

## Pre-registered cases

1. scan 42 / attproj / L11 — terminal-classifier dominated positive control;
2. scan 1186 / fcproj / L0 — created disagreement requiring full suffix state change;
3. scan 1186 / fcproj / L2 — first tested-execution agreement competitor state;
4. scan 9766 / fc / L0 — disagreement removed by the alternate state, with similar endpoint absolute GPU margins.

## Validity gates

- natural reference and alternate target logits must repeat bit-exactly;
- alpha=0 GPU suffix must reconstruct the reference logits bit-exactly;
- alpha=1 GPU suffix must reconstruct the alternate logits bit-exactly.

Interior alpha states are counterfactual and therefore have no natural replay target.

## Interpretation / stopping rule

This pilot is descriptive and falsificatory. It does not identify an operator-level mechanism.

Evidence favoring a purely terminal-conditioning account would be:

- disagreement confined to `T` points;
- `T` points coinciding with extreme terminal `kappa_dot` / tiny pair margins;
- no `S` interval along the natural state segment.

Evidence that warrants a stronger matched-control phase is:

- a reproducible `S` interval along the natural displacement segment, especially when terminal classifier conditioning is not exceptional;
- or a state segment that creates/removes such an `S` region in a way not reducible to endpoint margin alone.

Even then, the next step is matched controls / direction controls, not a novelty claim.
