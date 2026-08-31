# Numerical Fate Atlas

## Research question

The atlas is not a search for a privileged operator such as QK, attention, or an MLP projection. Its primary question is:

> How does an execution-level numerical perturbation propagate through a Transformer, change downstream decision geometry, and eventually become persistent fate?

The working object is an **operational model** rather than an architecture/weight pair alone:

\[
F_{\mathrm{exec}}(x; H, P, K, C),
\]

where hardware, precision, kernel/algorithm, and compiler/runtime semantics may change the realized computation.

## Core observable

For a natural perturbed execution, preserve the exact realized residual checkpoint \(h_l\). From that same checkpoint, reevaluate the remaining suffix using the independently validated CPU64 reference semantics. For a fixed reference winner \(a\) and competitor \(b\), define

\[
m_l = z_a\bigl(G_l^{64}(h_l)\bigr)-z_b\bigl(G_l^{64}(h_l)\bigr).
\]

The **pair topology** is the ordered sign sequence

\[
\tau = (\operatorname{sign} m_0,\ldots,\operatorname{sign} m_{L-1}).
\]

Examples such as `++----------` and `+++--+++++--` are therefore experimental objects, not debug decoration.

A separate **winner topology** records whether the CPU64 suffix winner at each depth is the reference token (`R`), selected competitor (`C`), or another token (`O`). This prevents a two-token projection from silently hiding third-token behavior.

For final flips, the competitor is the natural perturbed winner. For non-flip controls, the competitor is the reference runner-up. This is the same pair-selection convention as the population landscape survey.

## Stable fate

First crossing and persistent fate are distinct.

For a flipped case, the stable competitor layer is the earliest \(l\) such that the competitor remains the CPU64 suffix winner at every later checkpoint. Pair-sign stability is reported independently because winner agreement and pair-margin agreement are different observables.

A stable fate is always relative to the validated CPU64 suffix semantics. It is **not** claimed to be exact real-arithmetic irreversibility.

## Validity gates

Every atlas trajectory must satisfy all applicable gates before interpretation:

1. Reference target logits repeat exactly.
2. Perturbed target logits repeat exactly.
3. Every exact GPU checkpoint reconstructs the natural perturbed target logits bit-for-bit through the original GPU suffix.
4. CPU64 suffix semantics remain the previously validated high-precision reference; they are not labeled \(F_{\mathbb R}\).
5. No subtraction-heavy interaction statistic is treated as a model mechanism without a conditioning analysis.
6. Adjacent depth or stage margin changes are transport/localization observables, not standalone operator attributions.

## Competing explanations

The atlas keeps at least three explanations alive until population evidence separates them.

### H1 — Shared computational channel

Different numerical sources propagate through different paths or depths but converge onto a recurring computation class before decision fate becomes persistent.

### H2 — Decision-geometry transport

No privileged computation class is required. Transformer dynamics rotate, attenuate, amplify, or redirect a disturbance relative to the downstream decision direction; apparent channels are state-dependent transition sites.

### H3 — Boundary-conditioned coincidence

Shared-looking topology is primarily a consequence of extremely small baseline decision margin. Matched non-flip controls with comparable margins should remove or strongly weaken the pattern.

These hypotheses are not mutually exhaustive.

## Experimental sequence

### Phase 1 — Replicate before localizing further

Use the existing population landscape to select an independent shared-vulnerability case and matched near-margin controls. The immediate target is scan index `9766`, because `10374` has already been depth- and stage-localized and should not be rerun merely to reconfirm known facts.

For each flipping perturbation-family membership at 9766:

- run the full depth trajectory;
- run one globally clean near-margin non-flip control from the same perturbation family;
- compare pair topology, winner topology, number of reversals, stable-fate depth, and minimum absolute CPU64 pair margin.

Control matching is explicit and auditable: same reference token pair is preferred, then nearby local sequence position, while nearest log baseline margin is used inside the first available matching scope.

### Phase 2 — Build topology classes

If the second shared case survives validity and shows structured behavior, expand to all validated flip memberships plus matched controls. Cluster or enumerate topology classes before performing more operator-specific localization.

Questions include:

- Are a small number of sign/winner topologies recurrent?
- Does flip multiplicity predict topology class after controlling for baseline margin?
- Are high-multiplicity vulnerabilities topologically different from single-family vulnerabilities?
- Do matched non-flip controls approach the same internal boundary and recover, or stay on one side throughout depth?

### Phase 3 — Study transport geometry

Only after recurring topology classes exist should the project ask how perturbation direction is transported through residual space. Candidate measurements include directional projections, local linearization/Jacobian tests, and low-dimensional structure, with direct recomputation retained as ground truth.

### Phase 4 — Execution invariants

Repeat the atlas under different numerical semantics and then a second minimal Transformer architecture. The target is not identical layer numbers or operator names; the target is any fate structure that remains stable across changes in precision, kernel realization, hardware, or architecture.

## Current evidence boundary

`10374` is a validated first specimen: four heterogeneous L00 component perturbations reach the same final token boundary, acquire stable fate at different depths, and all four depth-specific transitions fall between the exact GPU QKV and ATTY checkpoints. This supports a case-level shared attention-computation transition site. It does not establish a QK mechanism, an attention-general law, or an architecture invariant.

`1186` remains a validated path-specific specimen whose L00-FCProj trajectory was localized to the L02 QK numerical realization. It is a contrast case, not a template for interpreting other specimens.

The atlas exists to determine whether these specimens belong to recurring dynamical classes or are isolated boundary events.
