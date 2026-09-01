# Crossed Exact-State × Execution Pilot

## Question

Does a naturally realized numerical state displacement change a Transformer's susceptibility to downstream numerical execution semantics?

This pilot explicitly separates two axes at every residual3 depth checkpoint:

- **state axis**: exact reference state `h_ref` vs exact natural alternate state `h_alt`;
- **execution axis**: original GPU complete suffix `E_gpu` vs validated CPU64 complete suffix `E_cpu64`.

For each layer `l`, the primary measurement is the direct 2×2 counterfactual square:

| exact state | `E_gpu` complete suffix | `E_cpu64` complete suffix |
|---|---:|---:|
| `h_ref,l` | `M(h_ref,l, E_gpu)` | `M(h_ref,l, E_cpu64)` |
| `h_alt,l` | `M(h_alt,l, E_gpu)` | `M(h_alt,l, E_cpu64)` |

`M` is the reference-vs-natural-alternate-winner logit margin. Full top-1 winners are recorded separately so pairwise ties or third-token winners are not hidden.

## Primary observable

The raw four margins and four winners are the primary evidence. No state×execution second-difference scalar is formed in this pilot.

For a state, execution agreement means the two complete suffix realizations produce the same top-1 winner.

The per-layer cross-state execution pattern is encoded as:

- `.` — reference and alternate states both execution-agree;
- `C` — execution disagreement is **created** on the alternate state: reference state agrees across executions, alternate state disagrees;
- `R` — execution disagreement is removed on the alternate state;
- `B` — both states execution-disagree.

`C` is the pre-registered positive signal for **state-conditioned execution susceptibility**. It is only descriptive at this stage and must later survive matched margin / numerical-conditioning controls.

## Validity gates

Every case must satisfy:

1. repeated natural reference target logits are bit-exact;
2. repeated natural alternate target logits are bit-exact;
3. every exact reference residual3 checkpoint replayed through the original GPU suffix reconstructs the repeated natural reference target logits bit-for-bit;
4. every exact alternate residual3 checkpoint replayed through the original GPU suffix reconstructs the repeated natural alternate target logits bit-for-bit.

A case is interpretable only when all four conditions hold.

## Pre-registered pilot specimens

1. `scan_index=42`, `attproj` — prior terminal-sensitive specimen;
2. `scan_index=9766`, `fc` — prior CPU64 persistent vs GPU-terminal reversible specimen;
3. `scan_index=1186`, `fcproj` — prior monotone specimen with dual-readout topology agreement.

The pilot intentionally spans qualitatively different prior fate behavior rather than selecting new cases after seeing the crossed result.

## Interpretation / stop rule

- If no specimen contains a valid `C` layer, the hypothesis that the natural state displacement creates downstream execution susceptibility is weakened substantially; do not expand population on this hypothesis without another independent reason.
- If valid `C` layers appear, do **not** infer a new mechanism yet. The next required step is to test whether the effect is explained by ordinary decision margin and numerical conditioning using matched controls.
- If `C` disappears after conditioning-matched controls, treat the phenomenon as ordinary near-boundary numerical sensitivity rather than a distinct transport mechanism.

The purpose of this pilot is falsification, not mechanism localization.
