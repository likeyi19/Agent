# Constrained regulatory sequence optimization

`optimize_regulatory_sequences` edits existing human DNA using the installed,
unmodified CREsted DeepPBMC in silico evolution (ISE) method. It is an independent
scientific tool: DHS64 prediction is not a prerequisite and its scores are never
used as DeepPBMC objectives. The Registry contains 30 tools. De novo generation
and scATAC-guided design are outside this capability.

## Researcher workflow

Attach an ordinary FASTA, paste a DNA sequence, upload BED3–6 with an explicit
qualified human assembly, or continue from a result with explicitly selected
candidate DNA. Each starting sequence must contain **2,114 real A/C/G/T bases**.
Lowercase and wrapped FASTA are supported; repeated headers retain distinct
ordinal IDs. No sequence is truncated, padded or assigned invented genomic flanks.
BED extraction reuses ScientificResourceLibrary's exact genome, interval and
strand semantics. Supply the desired 2,114-base interval; shorter coordinates
are not silently expanded. Minus-strand coordinates are reverse complemented.
Editing positions refer to this oriented sequence, not chromosome positions.
These are DeepPBMC requirements, not restrictions on other sequence tools.

For example:

> Optimize this DNA for B_cell accessibility with DeepPBMC. Change at most two
> bases. Edit only positions 1047 to 1067 (0-based, end exclusive), protecting
> 1052:1057. Penalize increases in other PBMC conditions. Keep the candidate DNA
> and design result, explain the original and candidate scores, and show the
> native trajectory figure.

The seven exact model conditions are `B_cell`, `CD14_monocyte`, `CD16_monocyte`,
`CD4_T_cell`, `Cytotoxic_T_cell`, `Dendritic_cell`, and `Natural_killer_cell`.
A named unsupported condition requires clarification; no nearest cell type or
other predictor is substituted. These outputs concern normalized human PBMC
peak accessibility, not arbitrary enhancer activity or gene expression.

Use **Continue from this result** to choose a historical accepted candidate as
the starting DNA. A subsequent request may change the target, comparison conditions
or edit budget. An additional edit budget is relative to that selected starting
sequence; it is not automatically a cumulative budget against a grandparent.
Previewing a historical result does not change scientific context. An artifact
not selected by its historical Revision can be explicitly downloaded and uploaded
as new FASTA; hidden sibling artifacts do not acquire execution authority.

## Native objective and constraint semantics

Native weighted difference chooses the best single substitution in each iteration:

```text
objective = target_score - sum(comparison_weight * comparison_score) / 7
```

The denominator is always the model's seven output classes, including when only
one comparison condition is selected. The default compares all six non-target
conditions with weight 1. Explicit comma-separated comparison names select a
subset; weight 0 maximizes the target alone. This objective penalizes comparison
**increases**, rewards decreases, and does not constrain absolute changes or
promise unchanged comparison cells. Hard bounds on comparison-score changes are
not implemented. The Agent must not reinterpret such a request as this objective.

`max_changes` defaults to 5; `iterations` defaults to that budget. Both have a
1–100 execution bound, and iterations cannot exceed the budget. Each native step
substitutes one nucleotide and can revisit a position or revert an earlier change.
The owner independently measures the returned candidate's actual Hamming distance
against its starting sequence and verifies every protected position. Iteration
count alone is not proof of final edit distance.

`edit_start`/`edit_end` delimit one allowed 0-based half-open sequence interval;
defaults allow the full context. `protected_regions` adds any explicitly supplied
comma-separated `start:end` intervals. Their complement can represent multiple
editable windows. Overlapping protection is harmless; no remaining mutable
position is an input error. These controls are passed to native flank/protection
arguments, without a replacement search algorithm.

ISE excludes the unchanged sequence from each step's candidates and has no
improvement-based early stopping in this pinned API. Consequently even the best
available substitution can worsen the objective. Results expose actual objective
change, improvement status and non-improving steps. Successful execution does not
mean successful biological design. One final candidate is returned per starting
sequence, with no uniqueness or requested-alternative-count guarantee. The local
execution bound is 32 sequences/32 MiB; records run independently to bound native
mutagenesis allocation. The native mutation prediction batch remains 32.

## Results and evidence

The accepted publication retains original and candidate FASTA, actual substitutions,
all seven original/candidate scores, the native intermediate trajectory, exact
sequence hashes and source/reference/model identity. Downloads include candidate
and original FASTA, `mutations.tsv`, `design-scores.tsv`, `trajectory.json`, model
metadata and a PNG objective trajectory. Positions in mutation tables are 0-based.
Public FASTA IDs correspond to score-table IDs; unsafe original labels are redacted
only in downloaded copies. Full private originals retain their exact identities.

The compact result includes bounded candidate examples. New examples contain
substitution letters and exact comparison-cell scores so explanations do not have
to infer scores from penalties. Older accepted examples remain readable with their
original contents. Native figures use the existing pinned figure inventory and
scientific download path. Saved prose is immutable, including earlier incorrect
explanations documented in acceptance evidence. Model-generated explanations can
still fail or contain errors; the tables and exact scientific evidence remain
available independently.

No second predictor, optimization replay or biological ground truth is required
for normal publication. Verification binds native completion, checked actual
constraints, source/resource bytes and exact output integrity. Accepted reads,
downloads, historical navigation and presentation recovery do not rerun ISE.
A failed or interrupted execution does not grant automatic retry authority.

## Operator runtime and license

CREsted source revision `85ee5a730f918afe00ff5cc775b935774957ce0d` (1.11.0), Keras
3.15.1 and Torch 2.7.1+cu128 were reused from RSD-0 outside Git. The fixed Agent
launcher calls native `crested.tl.design.in_silico_evolution`; it does not vendor,
modify or reimplement CREsted. Qualified DeepPBMC checkpoint and ordered class
metadata are pinned in the existing `sequence_predictor` resource kind, schema
`deeppbmc-sequence-optimizer.v1`. The shared library owns installation and discovery.

The runtime uses a separately installed venv overlay of the existing Agent Torch
environment. It is pinned to that deployment, not a self-contained portable
package. Its complete installed dependency inventory and native qualification
are retained in the acceptance record. Provision an equivalent isolated runtime
and qualify it before using a different host; do not downgrade Agent or DHS64.
Runtime settings explicitly select device, threads and seed, with float32,
TF32 disabled and no autocast. There is no automatic hardware/precision/model
fallback. The 4090 is the primary qualification device, not a scientific input requirement.
An explicit CPU run returned the same two-edit candidate, with maximum prediction
difference 2.61e-7 versus GPU. Its two narrow iterations took about 728 seconds;
this is bounded portability evidence, not full-window CPU throughput qualification.

The actual [CREsted academic license](https://github.com/aertslab/CREsted/blob/85ee5a730f918afe00ff5cc775b935774957ce0d/LICENSE)
permits internal academic research and restricts redistribution, modification and
commercial use. Agent's adapter confers no rights to redistribute CREsted or offer
third-party hosted access. Such deployments need separate rights review. Preserve
VIB/Aerts lab notices and the required Kempynck et al. attribution. Model context
and training scope are described in the [DeepPBMC model card](https://crested.readthedocs.io/en/latest/models/deeppbmc.html).

The authorized local academic deployment now installs the exact qualified
DeepPBMC entry in `agent-data/resources/library`, the library selected by the
unchanged `agent-data/config/operator-uat.json` and normal `launch-uat.sh` launcher.
Installation uses `ScientificResourceLibrary.install` with existing files in
place; all ten earlier entries, including DHS64, remain unchanged. The historical
`library-rsd-2` index is acceptance evidence, not an operational dependency.
Researchers use ordinary Web requests and supported DNA inputs without switching
configuration, supplying model paths or constructing resource manifests. Other
authorized deployments must provision and qualify their resources through the
same library owner. See `normal-deployment-closeout/install.py` in the acceptance
record for this installation's exact identity and mechanism.

## Scientific status and evolution

DeepPBMC optimization and DHS64 prediction have equal architectural status through
the ordinary Registry, planning, scientific owners, accepted results and Web
delivery. Their assay meanings, supported conditions, sequence contexts and scores
remain distinct. Equal availability does not establish model interchangeability.

Scientific methods are selected for suitability and maturity, not permanent
architectural preference. Future qualified predictors, optimizers or generators,
including EpiZooSeq when its relevant sequence capability is qualified, may extend
or replace current methods. Such changes must retain exact historical model,
objective, assay, context and score meanings. Generalize existing owners only for
concrete scientific consumers; no speculative universal model framework is needed.

See the [RSD-2 report](../../../agent-data/acceptance/regulatory-sequence-design/rsd-2/REPORT.md)
for genuine native/browser evidence, regressions, preservation and bounded readiness.
