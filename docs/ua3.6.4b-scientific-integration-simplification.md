# Work Unit B — Raw scATAC-seq integration simplification

2026-10-14 (Asia/Shanghai). **Targeted review and compatibility verification;
no production-code change.** The existing qualified routes can produce and deliver
H5AD through ordinary application execution. The broader experience of supplying
arbitrary new raw files and only requesting H5AD is **not complete**. The remaining
gaps require explicit application/resource integration, not removal of scientific
validation. All UA3.6.4a/b/c/d product scenarios remain **PARTIAL**.

This is the new simplification work unit, distinct from the earlier “Work Unit B”
in [raw preprocessing acceptance](ua3.6.4b-raw-preprocessing-acceptance.md).
That record's latest FASTQ/BAM and real-fragments results supersede its initial
blocked-stage observations; none are newly executed biological results here.

## Baseline and preservation

Starting HEAD, branch `main`, and local `origin/main` are exactly
`bc9963840394a5df802857d7ce09794d3e990379`, **Simplify EpiZoo input and decouple
UMAP from Leiden**. A/A.1 are committed. Initial index and tracked worktree are
clean; only pre-existing `evals/` is untracked. No remote synchronization is
claimed: no fetch, pull, commit or push was performed.

The task evidence directory is `/tmp/agent-raw-simplification-1bw2x8_l`.
`before.json` records SHA-256 for 676 tracked files and names, sizes and nanosecond
mtimes for all 1,274 eval files. Eval preservation is a metadata comparison,
not a new full-content audit. Operator configuration is separately hash-pinned.
Native EpiZoo is clean at `029cd631d0f9806a646c4a3b42ce10b958f2b67f`.
Historical scientific payloads and acceptance workspaces are not replayed or edited.
Test resources and outputs are isolated under the task directory; no downloads,
checkpoint loads, GPU work or full-matrix densification occur.

## Actual entry and admission matrix

All three registered raw producers retain human/hg38 and mouse/mm10 scope.
The table describes successful execution prerequisites, not requirements for
merely uploading a file or performing bounded FASTQ/BAM inspection.

| Entry | Researcher information genuinely needed | Work already performed by Agent/owners | Deployment or source-bound prerequisite | Ordinary execution and result |
| --- | --- | --- | --- | --- |
| FASTQ | Supported 10x ATAC assay/read layout; exact file roles, library membership, lane/chunk and compression at upload; species/target reference; barcode policy and applicable whitelist; namespace; explicit selection criteria | Streams and hashes uploads; groups explicit members into one immutable collection; validates role completeness; inspects sources; checks full read synchronization; runs qualified Chromap and packaging; publishes fragments, provenance and authority | Qualified FASTA/FAI/reference, matching Chromap index/executable, whitelist, QC/TSS resources and runtimes. The existing library-context builder must bind declarations to the exact uploaded intake/groups before producer admission | `inspect_raw_scATAC → prepare_scATAC_fragments → compute_scATAC_qc → select_scATAC_cells → build_scATAC_cell_by_ccre`; verified downloadable H5AD when prerequisites are supplied |
| BAM | Species/source assembly and supported assay where not established by intake evidence; one library/namespace; corrected `CB` identity declaration; unshifted paired ATAC coordinates and retained duplicate pairs; explicit selection criteria | Hashes source; inspects header/tags/readiness; fully decodes and checks primary pairs, CIGAR, mate consistency, barcode agreement and reference; applies the qualified fragment policy and independently reconstructs results | Exact BAM-bound intake/library context, compatible reference and QC/TSS resources; qualified decoder/packaging/BEDTools. No input index, coordinate sorting or original FASTQ requirement | `inspect_raw_scATAC → prepare_scATAC_bam_fragments → QC → selection → canonical matrix`; same H5AD delivery |
| External fragments | Explicit compatible species/assembly/reference; reviewed `10x-atac-fragments.v1` semantics; namespace; explicit selection criteria. Export coverage may remain `unknown` | Detects plain/gzip/BGZF from bytes; hashes source; checks complete records, coordinates, barcode identifiers, support and duplicate keys; sorts/packages/indexes; independently verifies exact source conservation | Compatible reference and independently qualified QC/TSS resources/runtimes. Optional supplied TBI is pinned and checked. No raw intake or library-processing context is fabricated | `import_scATAC_fragments → QC → selection → canonical matrix`; no FASTQ/BAM pathway or historical producer recreation |

FASTQ layout A uses R1/R3 genomic reads and R2 barcode; layout B uses R1/R2
genomic reads and I2 barcode. I1 is optional. File roles are not assay proof.
BAM `CB` presence alone is not proof of correction; upstream processing history
remains declared. External adoption proves contents and conservation, not previous
alignment, Tn5 adjustment, barcode correction or original sequencing completeness.
These distinctions survive in producer-specific records and evidence.

Both required selection operands—minimum QC fragment-record depth and minimum
TSS enrichment—must be explicit. Optional flank evidence, maximum depth and
nucleosome criteria retain existing semantics. Historical acceptance values are
not universal defaults. Results describe **selected candidates**;
`cell_call_method=none` / `cell_call_state=not_assessed` remain unchanged.

Evidence: `application/uploads.py`, `local_resources.py` (`compose_*`, collection
and exact-consumption checks); `orchestration/{fragments,bam_fragments,
external_fragments,barcode_qc,cell_selection,matrix}_registry.py`;
`tools/data/_fragments_binding.py`, `_bam_fragment_io.py`,
`external_fragments.py`, `_barcode_qc_binding.py`, `_cell_by_ccre_io.py`.
These are existing owner and integration paths, not a proposed alternate engine.

## Internal work, planning and continuation

Upload/registration already generates file digests, opaque identities and exact
source bindings. The Planner receives typed input descriptions and semantic ports;
the compiler resolves explicit edges. Intake can be a same-plan output. Fragment,
QC, selection and matrix manifests, managed output locations, producer records,
RunStore records and scientific authority are generated internally. Users do not
need to upload downstream fragments/QC/selection artifacts in advance.

The significant remaining internal prerequisite is **library context generation
for a newly located FASTQ collection or BAM**. The data-layer
`scatac_library_context.build_scatac_library_processing_context` already derives
group/barcode bindings and the context identity from exact intake plus explicit
declarations. However, both producer Registry `library_context` ports are
request-only; no registered step produces that context. Current Web acceptance
helpers `qualified_companion` explicitly inspect uploaded locators, call the owner,
publish the context, and supply an operator `ScientificInputSet`. A same-plan
fresh intake can then match its exact identity. This is working operator-assisted
integration, not evidence that chat already builds the context for arbitrary files.

The internal context document is not a biological fact users should author.
Its membership/barcode declarations and exact intake binding remain necessary.
A safe future handoff must collect those declarations, invoke the existing owner
against the actual accepted intake, and persist an exact resumable binding.
Deleting the producer's context requirement or reusing an old source-bound context
would not implement that handoff.

`resource_continuation.py` already captures the objective, source identities,
arguments and offered InputSets. `request_resources` / `answer_resources` can
resume it after an explicit choice; parameter and namespace clarification also
exist. Changed sources, changed captured catalogs and conflicting values fail
closed. No-catalog requests preserve the objective with a provisioning explanation,
but cannot provision or refresh the catalog themselves. This is not a general
solver for arbitrary interleavings of missing resources, roles and declarations.
Incomplete FASTQ members remain registered; missing roles can be uploaded and the
collection completed. A chat objective submitted before any selectable collection
exists is not qualified by that mechanism.

## Defaults, custom peaks and delivery

**A configured default can avoid a peak upload, but selection is narrower than
automatic compatibility discovery.** `web/app.py:_raw_selection` chooses an explicit
family companion or the sole companion marked as that family's default. Zero
defaults leave inputs unresolved; multiple defaults remain ambiguous. It does not
filter multiple defaults by species/assembly, infer assembly from coordinates,
search disk or independently qualify an InputSet. Owner checks subsequently enforce
the exact reference/source/QC relationships. An incompatible explicit choice cannot
fall back to another reference.

Existing M11.6 resources supply qualified human/hg38 and mouse/mm10 canonical
references, QC/TSS resources and matching alignment resources. Their availability
does not itself wire them into each uploaded source. Read-only inspection of
`/home/likeyi/program/agent-web-config/operator-web.json` finds two InputSets and
**no FASTQ, BAM or fragments companions/defaults**. The configured per-file limit
is 256 MiB. Current code supports an operator-configured maximum of **8 GiB**;
the older 1 GiB statement in early raw acceptance is historical. Neither deployment
configuration nor upload limits are changed in this unit.

**Custom peaks are supported by the neutral scientific owner, not by a complete
raw-upload-plus-BED Web route.** `build_regulatory_feature_reference` validates an
explicit species/assembly, exact FASTA/FAI and ordered BED with `peak_set` or other
supported category. `build_scATAC_cell_by_features` uses the existing counting
engine and independent verifier; a different feature vocabulary on the exact
same genome is supported and tested. Exact axes and zero columns are preserved.

Concrete boundaries prevent advertising that as universal raw attachment support:

- `uploads.py` does not accept a BED input type. Current reference supplementation
  requires an H5AD source and derives its ordered feature axis; it is not raw BED
  registration.
- Raw companion declaration allowlists cover canonical matrix construction.
  The neutral matrix's fragments port is request-only. Neutral external adoption
  and neutral BAM production exist at the data layer, not as registered raw
  producers. New neutral workflows also require the primary-nuclear scope and
  preparation lineage; historical unscoped objects are not reclassified.
- `_fragment_feature_binding.bind` requires neutral species/reference identity,
  qualified producer lineage, and matching genome bytes/ordered contigs. Its
  QC-selected variant requires neutral QC; it does not accept arbitrary legacy
  FASTQ/BAM output by changing a reference pointer. The canonical matrix binder
  pins the original fragment/QC/reference identity, including its feature set.

The supported developer/operator route prepares/adopts qualified neutral fragments
and reference through existing owners, then supplies exact artifacts to the
registered downstream tools. QC-selected cells and explicitly caller-ordered cells
are distinct accepted contracts; the latter is an explicit override, not inferred
QC or a shortcut used here. No new peak caller or reference migration is proposed.

`matrix_delivery.py` already admits canonical and both neutral created-matrix
contracts under its closed layout/privacy policy. The interactive facade delivers
pinned original H5AD bytes; HTTP tests check size and SHA-256 and all five canonical
downloads. A valid generic feature H5AD does **not** establish pretrained EpiZoo
compatibility. Exact model vocabulary/order and resource requirements remain intact.

## Implementation gate

| Finding | Classification and decision | Smallest safe correction / compatibility implication |
| --- | --- | --- |
| Suspected FASTQ dependency for BAM, or raw-history dependency for external fragments | **KEEP**: absent in actual registered paths; positive execution tests exist | No correction needed. Preserve distinct producer provenance and shared downstream v2 consumption |
| Source hashes, generated scientific manifests, explicit same-plan edges | **KEEP**: already generated/bound internally | Removing identity or authority checks would weaken accepted-result reuse; no such change |
| Exact source/reference, read/barcode/profile and QC-selection checks | **KEEP**: supported transformations need these facts; declarations and verification are separate | No invented correction history, assembly, thresholds or cell purity |
| Manual source-bound library context provisioning | **DEFER**: real user-facing integration barrier; context creation is not exposed through the typed planning/continuation boundary | Reuse the existing builder with a reviewed declaration-to-intake handoff and recovery contract. This is not a local validation removal; accepted contexts and authorities must stay byte-preserved |
| Choosing among resources from declared species/assembly | **KEEP** existing unique configured default; **DEFER** automatic compatibility-based selection across bundled InputSets | Separate source declarations from reusable reference applicability using existing owners and choices. A list-length or filename heuristic cannot establish compatibility; no new catalog is warranted |
| Raw custom BED and neutral producer composition | **DEFER**: missing upload/role handoff plus explicit public producer/reference scope | Coordinate typed attachment admission, existing reference qualification and neutral producer ports. Broadening only a Web allowlist would still fail scientific binding; do not reinterpret legacy authorities |
| Complete raw-file conversational acquisition and live quality | **DEFER**: incomplete collection/objective and catalog-refresh boundaries; complete live planning remains unqualified | Unified attachment UX and UA3.6.5, with exact-source continuation and ordinary compiler execution |

**No SIMPLIFY candidate passes the local implementation gate.** The historical
couplings actually alleged in B1 are already absent. The evidenced remaining
barriers span typed declaration, source-dependent resource publication and
continuation or public neutral-producer contracts. They deserve explicit follow-up
scope; disabling checks would not remove the underlying requirements. No source,
test, Registry, algorithm, authority schema or scientific semantics change.
The only repository addition is this report.

## Focused verification

Both invocations use `/home/likeyi/anaconda3/envs/agent/bin/python -B -m pytest
-q -p no:cacheprovider`, `PYTHONPATH=src:.`, disabled bytecode and isolated
basetemps/caches. `run_checks.py`, the exact argv JSON, logs and exit records are
retained in the task directory.

| Newly executed checks | Result and scope |
| --- | --- |
| Twelve focused modules: application FASTQ/BAM/fragments composition; Web selection and execution for all three families; Web resource continuation; neutral fragment boundary and feature-matrix owner | **321 passed, 6 skipped in 172.55 s**, exit 0. All `RUN_*` gates removed. Six existing Chromap opt-in cases skipped |
| `tests/web/test_fastq_execution.py::test_fastq_provider_downstream_semantic_handoff_preserves_five_downloads` | **1 passed in 32.39 s**, exit 0. Only `RUN_CHROMAP_QUALIFICATION=1` enabled, with existing `/tmp/m116c-test-build-identity.json`; fixture checks pinned executable/source/patch identities and builds only a tiny index |

The full-chain tests use **scripted provider decisions and simulated scientific
inputs**, with actual production owners, compiler, verification, publication,
evidence/reporting and HTTP delivery. The FASTQ fixture uses seed 112 and a
12,000-base artificial contig; its human/hg38 fields are test contract values,
not real hg38 qualification. QC synthetic admission is explicit and process-local.
BAM/fragments fixtures likewise do not prove biological correction/alignment history.

Fresh sparse inspection of the three complete-chain outputs records:

| Simulated source | H5AD shape | dtype / nnz / count sum | Delivered size |
| --- | --- | --- | --- |
| FASTQ, actual qualified Chromap | 1 × 1 | int64 / 1 / 1 | 31,240 bytes |
| BAM, actual paired-record producer | 1 × 2 | int64 / 1 / 2 | 31,240 bytes |
| External fragments, actual source-conserving importer | 1 × 2 | int64 / 1 / 2 | 31,240 bytes |

`matrix-observations.json` records exact paths and SHA-256. Full-chain assertions
check manifest edges, accepted schema-2 authority, five artifact downloads and
exact size/digest, then retry without additional production. BAM and FASTQ each
record one intake, one producer and one producer verification (FASTQ also one
alignment); each downstream QC/selection/matrix production and independent owner
runs once. External import/QC/selection/matrix each runs once with one independent
owner verification. These counts concern the named instrumented functions,
not absence of other integrity IO.

The focused cases also cover missing/ambiguous defaults, conflicting explicit
choices, missing declarations, source substitution, malformed fragments, incorrect
barcode policy/reference, exact library membership, clarification/restart,
accepted retry and historical reads after source deletion. Neutral-owner tests
verify alternative BED order/values on the same genome, reject genome/assembly
mismatches and forged/mutated artifacts, and distinguish authority reuse from
deep standalone recovery. Those neutral fixtures exercise the preserved unscoped
external/explicit-cell contract; they do not newly qualify primary-scoped biology
or a Web custom-peak workflow. Detailed BAM pairing fault coverage remains the
unchanged M11.3c acceptance; this unit did not rerun its whole fault suite.

Historical biological evidence remains separately attributed: M11.7/M11.8 real
FASTQ PBMC preprocessing; M14.8/M14.11 real neutral macaque processing; and the
later UA3.6.4b genuine PBMC-fragments scripted-provider workflow (568 then 567
candidates, full 1,355,445-feature matrices, browser downloads and zero-upstream
continuation). None was rerun. No new biological-data computation, live-provider
call, native-browser run or full repository regression was performed. HTTP
TestClient success does not establish browser or live-planning product acceptance.

Final preservation compares every tracked file with its initial SHA-256 and all
1,274 eval names/sizes/mtimes, plus the operator configuration hash; all remain
unchanged. HEAD, local `origin/main`, empty index and native EpiZoo remain at the
recorded baseline. `preservation-final.json` retains the check. `git diff --check`
and a separate whitespace check of this new untracked report pass. No historical
authority/result migration or source/test edit occurs; Registry remains 23.

## Closeout and next scope

Work Unit B can close as a **bounded no-code simplification review**, with the
concrete gaps above retained. It does not close the upload-only product objective
or promote any PARTIAL scenario. The existing chain needs scientifically adequate
declarations/resources and explicit selection policy; the live model still owns
planning and explanation, without deterministic workflow completion.

Work Unit C may begin as its own scoped task; its scope was not supplied here and
no C work is started. These findings do not require speculative expansion of B.
If the immediate priority is a researcher completing arbitrary new raw uploads
without operator-generated contexts, prioritize the explicit context-generation
handoff before claiming that experience in UA3.6.5. Unified attachments, raw BED
admission, real FASTQ/BAM qualification and complete live-provider/browser acceptance
remain separately reviewable work.
