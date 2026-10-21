# UA3.6.4c MAESTRO peak input contract review

Assessment: **2026-10-14, Asia/Shanghai**. Scientific boundary review complete;
ordinary uploaded peak-H5AD acceptance remains **unqualified**. Overall
UA3.6.4c remains **PARTIAL**. No commit or push.

**An exact Agent cCRE vocabulary and a historical M11 Run are not intrinsic
requirements of MAESTRO Enhanced RP.** The old M13 integration deliberately
qualified a narrower input and preprocessing profile. That historical scope
must not be presented as a universal limitation of MAESTRO.

There is a separate substantive distinction: the accepted M13 path converts
fragment counts to a private binary support matrix before RP, whereas the
documented upstream count-matrix path passes counts into RP. Cell Ranger ATAC
peak matrices also represent cut-site counts, which are not necessarily
fragment-overlap counts. Removing the cCRE requirement does not by itself settle
which scientific profile a new upload should execute.

The user's latest instruction takes priority over the earlier cCRE-first choice.
Further cCRE-only implementation and acceptance preparation are stopped. The
existing draft is preserved. Under item 8 of that instruction, a switch to
count-weighted RP is held for a scientific-profile decision before implementation.
This is not a claim that ordinary peaks are scientifically unusable.

## Actual scientific call path

The unchanged, historically pinned
[`marker_annotation.py`](../../src/agent/tools/analysis/marker_annotation.py)
calls the original functions in MAESTRO revision
`74f10babf4125b98ef005df4186b14ab02e13bfd`, source SHA-256
`9925cd2c6a24ae23da3bb018ff8735b537a38bd213adedc22f6948871eca62fe`.
Its private sparse working matrix replaces positive stored counts with float64
ones. It reversibly encodes each supplied genomic interval, transposes cells by
regions to regions by cells, and calls `calculate_RP_score` with `Enhanced` and
decay 10,000. No cCRE lookup or historical execution identity is passed to that
function.

In the pinned upstream implementation, `ExtractGeneInfo` reads transcript,
strand, TSS, promoter, exon and gene-symbol information.
`RP_AddExonRemovePromoter` constructs weights from supplied peak coordinates and
gene annotations. `calculate_RP_score` multiplies those weights by the supplied
matrix, selects transcript-derived gene rows using total RP, and emits ordered
gene scores. The input feature count and coordinates are supplied by the caller;
there is no fixed cCRE vocabulary in this computation. The CLI `genescore`
passes its loaded count matrix into this calculation. These findings come from
the [pinned upstream Python source](https://raw.githubusercontent.com/liulab-dfci/MAESTRO/74f10babf4125b98ef005df4186b14ab02e13bfd/MAESTRO/scATAC_Genescore.py).

The upstream tutorial demonstrates genomic peak matrices from Cell Ranger and
MAESTRO. It documents explicit peak coordinates, matched barcode order, organism,
gene-distance setting and RP model, without requiring an Agent reference feature
set. The tutorial therefore supports coordinate-defined peak input, not a
requirement to project peaks into cCREs. See
[MAESTRO gene activity modeling](https://liulab-dfci.github.io/MAESTRO/example/Gene_activity_modelling/Gene_activity_modelling.html).
The pinned R wrapper likewise documents unnormalized counts and forwards its
input matrix to RP; it is a separate wrapper from the one M13 invokes. See
[`ATACCalculateGenescore.R`](https://raw.githubusercontent.com/liulab-dfci/MAESTRO/74f10babf4125b98ef005df4186b14ab02e13bfd/R/ATACCalculateGenescore.R).

M13 then applies its repository-controlled sparse
`log1p(RP / cell_RP_sum * 10000)` normalization. The pinned
`FindAllMarkersPresto` and Presto one-versus-rest statistics consume gene scores
and explicit groups. Existing marker filters and signature scoring produce
primary labels, unresolved/ambiguous outcomes, and `validation_state=not_assessed`.
These stages do not require a cCRE reference either. Their resource, grouping
and evidence requirements remain necessary.

## Requirements and inherited restrictions

| Property | Actual requirement and responsibility |
| --- | --- |
| Genomic features | Unambiguous interval coordinates aligned exactly to matrix columns; positive interval length and consistent coordinate convention. The native algorithm uses interval centers and gene geometry. A fixed cCRE vocabulary is unnecessary. |
| Species and assembly | Explicit source declarations compatible with the gene resource. Coordinates alone do not prove assembly or species. Neither filenames nor matching chromosome strings establish compatibility. |
| Contig scope | The project requires an explicitly qualified primary nuclear scope. Contig names must not be silently aliased or classified by spelling. This need does not imply a full canonical cCRE feature set. |
| Matrix representation | A declared supported measurement, valid sparse storage, finite nonnegative values, exact axes, and bounded resource use. M13's signed int64 CSR requirement comes from its canonical matrix contract; upstream mathematics does not require that specific integer width. Supporting other storage requires exact representation checks, not conversion of measurement semantics. |
| Count meaning | Raw fragment-overlap counts, cut-site counts, and already-binary accessibility must remain distinct declarations. A numeric matrix cannot independently prove its counting or normalization history. |
| Grouping | Every cell appears once in exact matrix order, with explicit opaque group labels and at least two groups. Group derivation must remain separate from annotation/signature inspection. There is no requirement that groups come from EpiZoo or an internal Run. Numeric group labels convey no cell identity. |
| Gene scores | Correct resource geometry and namespace; finite nonnegative sparse RP and positive totals for each cell before normalization. RP is regulatory-potential evidence, not measured RNA expression. |
| Signatures and tissue | Explicit appropriate context and compatible gene namespace; pinned resources, disclosed coverage and duplicates. No automatic tissue or cell-type inference from file names. |
| Source identity | Immutable upload identity, source digest, exact axis/value conservation and truthful external provenance are necessary. Historical M11 authority is one supported source contract, not a biological prerequisite. |
| Biological validation | Replaying an annotation backend verifies its computation; it does not independently validate the assigned cell identities. The current primary annotation policy remains `not_assessed`. |

M13.2 and [M13.3](m13.3-generic-marker-annotation.md) explicitly qualified only
canonical fragment counts with private binary preprocessing.
[M13.5](m13.5-agent-annotation-integration.md) then required an exact accepted
`build_scATAC_cell_by_ccre` publication and trusted RunStore binding. These checks
protect that historical profile. Applying them as mandatory prerequisites to
every future MAESTRO input is unnecessarily narrow.

The existing exact external cCRE adoption owner checks a complete reference axis,
source/output conservation and declared matrix semantics. It cannot establish
eligibility for arbitrary peaks because its own artifact contract requires that
exact axis. Forcing an ordinary peak matrix through it would add an unnecessary
dependency or require an unjustified transformation.

## Count semantics are a separate decision

Cell Ranger ATAC documents a peak-barcode matrix of fragment ends, or cut sites,
inside each peak. This is not synonymous with counting all fragments whose
interval overlaps the peak. For example, a fragment spanning `[90,120)` overlaps
peak `[100,110)` while neither endpoint lies in the peak. The two measurements
can therefore differ even after binarization. See the
[official Cell Ranger ATAC algorithms description](https://www.10xgenomics.com/support/software/cell-ranger-atac/latest/algorithms-overview/algorithms-overview).

A fresh synthetic mechanism probe ran the exact pinned upstream functions on
four cells and two ordinary coordinate-defined regions, without a cCRE reference,
source Run or annotation authority. Source arrays remained unchanged. Reordering
the two regions together with their matrix columns preserved the output.

| Input counts for one cell | Native count-weighted Enhanced RP | Current M13 binary Enhanced RP |
| --- | --- | --- |
| `[1, 1]` | `[1, 1]` | `[1, 1]` |
| `[9, 1]` | `[9, 1]` | `[1, 1]` |
| `[1, 8]` | `[1, 8]` | `[1, 1]` |

The synthetic annotation placed each region inside a different gene exon with
unit weight. After the same M13 normalization, `[9,1]` becomes approximately
`[9.10509,6.90875]` in the weighted route and `[8.51739,8.51739]` in the binary
route. Thus changing the working representation changes scientific values before
marker testing; normalization does not generally remove the difference.

This is mechanism evidence, not biological acceptance or a full MAESTRO CLI
equivalence claim. Script, exact inputs, hashes and results are outside Git in
`/home/likeyi/program/ua364c-upload-zlby99wq/probes/peak_contract_review.py`,
`peak-contract-review.log`, and `peak-contract-review/result.json`.

## Concrete implementation decision

The direct-entry design should consume uploaded H5AD and exact groups at the
annotation owner, with reviewed context/resources and an explicit measurement
profile. It should not require external cCRE adoption, fabricate a source matrix
authority, or require users to author internal manifests or Run identifiers.
Existing upload registration, file roles, pending continuation, compilation,
publication, replay verification and safe export remain the intended machinery.

Two scientifically different implementations are possible:

1. **Extend the M13 binary profile to coordinate-defined fragment-count peaks.**
   Preserve the existing binary working matrix, Enhanced RP, normalization,
   marker statistics and assignment policy. Qualify the new source/coordinate
   contract separately. This removes the fixed cCRE restriction without changing
   those formulas. It does not establish that Cell Ranger cut-site counts or
   arbitrary binary uploads satisfy that fragment-count input profile.
2. **Add a versioned native count-weighted peak profile.** Preserve input counts
   in the private float64 sparse working matrix, call the same pinned Enhanced
   RP functions, and explicitly admit reviewed raw-count measurement classes,
   including the intended Cell Ranger cut-site class. Keep the old M13 binary
   profile intact for historical results. This is the recommended direction if
   the product target is the upstream documented Cell Ranger workflow. It changes
   preprocessing semantics and needs separate genuine-data, replay and user-flow
   acceptance; it cannot silently replace the old profile.

Neither option needs peak-to-cCRE projection. Exact cCRE coordinates can remain
a valid feature set within the appropriate declared profile. A generic “peak
counts” label is insufficient to choose between these scientific meanings.
No new statistical threshold, annotation algorithm, validator requirement or
independent workflow engine is proposed.

The pending decision is which profile the ordinary uploaded peak workflow should
promise. This report does **not** treat every new feature axis as a material
algorithm change. The stop concerns silently changing binary preprocessing to
count weighting, or admitting another measurement class as though its semantics
were already qualified. The user's latest item 8 requires reporting this
distinction before implementing that change.

## Preserved draft and validation status

Starting HEAD, local main and origin/main were
`88cd19f11c0dd76d406746779a1b8092a238bef4`, the reviewed UA3.6.4c integration.
The initial index and tracked worktree were clean. The current index remains
empty; all follow-up changes are unstaged. The earlier acceptance report remains
unchanged.

Before the peak-priority update, the draft added an external fragment-count
annotation profile using existing exact cCRE adoption, reusable resource
specifications, group path/digest ports, group TSV upload and shared file-role
bindings. It also extended annotation delivery/profile handling. A synthetic
explicit adoption/annotation chain reached production plus backend-stub replay;
insertion and binary declarations were rejected. These are development results,
not acceptance of the unfinished draft.

Known incomplete checks are preserved rather than represented as a PASS:

- The earlier unchanged-baseline affected regression passed **143 tests in
  222.80 seconds**.
- The first affected draft batch finished **71 passed, 1 failed in 113.29
  seconds**. The v3 planner fixture does not supply null entries for the newly
  optional group arguments, as required by that wire format.
- The new HTTP test reaches upload, a durable missing-group question, group-file
  upload and role continuation, then fails with `PLANNER_OUTPUT_INVALID` in the
  scripted plan. Its upstream-source wire needs correction and revalidation.
- A previous assertion incorrectly accessed a nonexistent verification-result
  property; it was corrected to use `VerifiedArtifactAuthority`. No combined
  clean new-draft acceptance run is claimed.

No genuine external-data annotation, native Firefox workflow, continuous live
provider execution or provider expenditure occurred in this follow-up. No
upstream reconstruction or peak projection was performed. The older scripted
real-data and separate live-result observations remain historical and are not
combined with these development checks.

The immutable historical `marker_annotation.py`, original MAESTRO/R sources,
resources and old acceptance files remain unchanged. The bounded preservation
audit checked **120,226 protected files**, including **119,570 SHA-256 checks**,
and found zero unexpected changes; it excluded the 15 intentionally modified
tracked implementation files. Large files without baseline hashes were checked
by size, mode and nanosecond mtime, not freshly byte-hashed. All **981** pre-existing
Git-listed untracked evaluation paths remain exact, and their baseline content
checks passed. Evidence is in `review-preservation.json` under the isolated
workspace above. `git diff --check` passed.

| Scope | Status |
| --- | --- |
| Scientific necessity of exact cCRE | Not necessary for native Enhanced RP; an inherited integration restriction |
| Ordinary coordinate peaks at native RP function boundary | Demonstrated by upstream documentation/source and a bounded synthetic execution |
| Equivalence of native weighted and M13 binary profiles | Rejected by an explicit numerical counterexample |
| Current external cCRE implementation draft | Incomplete; preserved for reuse/review |
| Ordinary uploaded peak-H5AD user workflow | Not implemented or accepted |
| Overall UA3.6.4c | **PARTIAL**, unchanged |

Stop at this scientific-profile decision. No commit, push, UA3.6.4d or UA3.6.5.
