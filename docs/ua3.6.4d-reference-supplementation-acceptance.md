# UA3.6.4d Work Unit A — Reference supplementation and persistent reuse

Date: 2026-10-14. Implementation and bounded acceptance are ready for review;
all changes remain unstaged. No commit, push, or Work Unit B training occurred.

The reference interaction passes the scripted, genuine-resource acceptance up
to normal compilation and whole-plan preflight. Authorized reuse and restart
recovery pass. Live-provider qualification remains blocked by a connection
failure. The selected regression has two reproduced inherited failures; it is
not an all-green regression claim.

## Baseline and preservation

Agent started and remains on `main`, HEAD
`ce5b53d7f716b777fc5333ea84e4d868ccf0da71`. The inherited UA3.6.4d work was
inventoried before edits in `baseline.json` and `inherited.patch` beneath:

`/home/likeyi/program/agent-acceptance/ua3.6.4d-reference-a`

This directory, abbreviated **A** below, contains the isolated deployment,
configuration, browser harness, screenshots, provider budget ledger, and logs.
It is outside Git. The prior [UA3.6.4d workflow record](ua3.6.4d-epizoo-cross-species-user-workflow.md)
is unchanged.

Of 1,641 inventoried repository files, 14 existing files changed for Work A;
the remaining files, including all 981 inventoried files beneath `evals/`, retain
their original SHA-256. Work A adds the reference coordinator, its tests, and
this report. All 436 files in the preceding external UA3.6.4d acceptance tree
retain their original size and modification time. Fourteen selected M14.5
resource/preparation/provenance files retain their byte hashes. Broader historical
RunStores and checkpoints were not written or re-audited.

EpiZoo remains clean at `029cd631d0f9806a646c4a3b42ce10b958f2b67f`.
`src/agent/tools/models/epizoo.py`, scientific reference owners, adaptation
algorithms, inherited target-model inference, result delivery and analysis
implementations are unchanged by Work A. No production operator configuration
was changed. The Git index remains empty. The isolated Web server and Firefox
processes were stopped after acceptance; `A/final-preservation.json` records the
final checks.

## Production changes and existing owners

- `application/uploads.py`, `web/config.py`, and the existing upload route accept
  a plain FASTA role and an optional paired FAI through bounded streaming,
  content-bound registration, existing leases, and atomic completed files.
  The configurable ceiling is now 8 GiB; the default remains 256 MiB. Only A's
  isolated configuration raises its limit to 2 GiB for the genuine genome.
- `application/reference_supplementation.py` coordinates the existing pending
  resource request, exact uploaded H5AD registration, native EpiZoo interval
  parsing, M14 reference owner, and existing local resource catalog. It adds no
  scientific execution tool, task store, resource database, or FASTA parser.
- `local_resources.py` retains the existing registration format and digest for
  old records. New optional metadata records a qualified reference, its exact
  resource/code byte bindings, original task/source, and private or deployment
  scope. Registration remains immutable and bounded by existing catalog rules.
- `turn_decisions.py`, `resource_continuation.py`, `sessions.py`, `turns.py`, and
  `responses.py` extend the existing typed resource request with an explicit
  `reference_category`. The interpreter chooses the resource request and answer;
  Agent validates identities and preserves the original objective/declarations.
  No language parser, semantic retry, first-match selection, or workflow repair
  was introduced.
- `species_adaptation_registry.py` adds the explicit H5AD `input_path` /
  `input_sha256` request binding to the existing neutral adoption source port.
  Local admission checks exact source, reference and matrix-semantics consumption
  for that adoption. The registered tool inventory remains 23.
- `interactive.py` exposes reference-prerequisite projection and qualification.
  HTTP calls these facade operations. The existing browser file-type selector
  and index control gain FASTA/FAI labels and upload status. Uploading a reference
  retains the selected H5AD and scientific InputSet. No layout redesign occurred.

The facade routes are GET/POST
`/api/v1/sessions/{session_id}/reference-prerequisite`. Qualification takes only
an uploaded resource ID, expected generation and captured origin-turn ID.
There is no browser promotion endpoint and no browser-supplied resource path,
manifest, digest, scientific qualification claim, or executable code.

## Qualification, indexing and scientific boundaries

A FASTA without an FAI uses installed `pysam.faidx` / htslib to write a separate
managed index; the uploaded FASTA is never edited. A supplied FAI is preserved
and verified, not silently regenerated. The owner checks complete FASTA/FAI
geometry and correspondence. Compressed FASTA is explicitly unsupported in
this upload role; gzip is rejected rather than automatically decompressed.

The bounded existing H5AD reader obtains the ordered `var` axis without loading
or densifying X. Native `epizoo.data.ccre.parse_region` adapts exact
`chr:start-end` identities to ordered BED3. No aliases, feature alignment,
reordering, dropping, liftOver or curated-cCRE inference occurs. Supported
categories here are explicit `peak_set` and `regulatory_regions`.

`build_regulatory_feature_reference`, `verify_fasta_fai` and
`publish_regulatory_feature_reference` establish FASTA structure, exact FAI
correspondence, contig lengths and valid ordered feature coordinates under the
existing [M14.1 contract](m14.1-regulatory-feature-adoption.md). Publication
records the owner's qualified identity. Content must still match that identity
when registration completes and before later consumption.

Species/scientific name, taxonomy and assembly remain **explicit declarations**.
A structurally compatible sequence cannot independently prove its publisher,
true biological assembly identity, or redistribution rights. Uploaded provenance
is labelled caller-supplied. Simulated tests establish software behavior only.

Reference continuation, promotion and reuse check exact pinned bytes/code and
manifest identity without repeating the FASTA scientific verifier. These are
resource-provisioning registrations, not imported scientific Run authority.
Existing independent matrix/adaptation verifiers are unchanged. Their behavior
during actual Work B execution has not been qualified here; this record does
not claim that their runtime checks are skipped.

## Original-task continuation

The first typed resource request captures the original source, species, assembly,
scientific declarations, target and objective in the existing Session interaction.
A successful upload qualifies against that captured task. It does not replace
its H5AD or original InputSet choice. A normal message selecting the offered
qualified reference uses `answer_resources`, then the ordinary Planner,
semantic compiler and preflight. The user need not reupload the H5AD or restate
the analysis request.

Qualification is idempotent for the exact task/upload identity. Stale generation,
changed pending origin, altered source bytes, substituted catalog identities and
conflicting declarations fail closed. Historical interaction replay uses captured
metadata; it does not reconstruct reference science.

The demonstrated entry uses an existing operator InputSet for explicit neutral
species/assembly, binary semantics, strategy and the M14.5 adaptation specification.
It does not establish automatic acquisition of all other model/SEAM/training
resources or arbitrary new-species declaration handling from unrestricted prose.
The qualification handles exact interval-named H5AD axes within existing bounded
H5AD/feature limits. Deployment reuse requires the exact ordered feature
vocabulary as well as species, assembly and category. A different vocabulary
requires a new reference binding; it is never silently aligned.

## Private and deployment resources

The default registration is a **private task resource**, offered only to its
captured Session/origin request. It is not automatically offered to another
Session. This is task-selection scope within the existing operator-managed
application, not a claim of filesystem isolation between authenticated tenants.

An operator can explicitly call
`agent.application.reference_supplementation.promote(resources, resource_id,
operator=..., provenance=..., reuse_basis=...)`. The function checks qualification
and immutable content, then writes another immutable entry in the same existing
catalog. It reuses the same manifest/FASTA/FAI/BED paths and copies no genome.
Missing review declarations fail. New versions cannot overwrite existing entries.

For this acceptance, the review records the user's Work Unit A authorization for
reuse within this isolated, same-operator deployment and the original M14.5
operator-supplied provenance. Original publisher/redistribution rights remain
unknown. This authorizes the tested deployment reuse, not global publication.
Publicly sourced resources can use the same operator entry point with their
reviewed provenance and reuse basis; parsing itself never supplies those rights.
No production default was installed or proposed automatically.

## Genuine resources

The exact real resources come from the preserved
[M14.5 acceptance](m14.5-real-zebrafish-de-novo-acceptance.md), not a new download.
They were reused on 2026-10-14. Original publisher URL/release/acquisition date
and redistribution permission for the operator-supplied genome are not
independently established in this work.

| Resource | Exact size | SHA-256 |
| --- | ---: | --- |
| `danRer11.fa` | 1,712,826,111 bytes | `9d45886b4c686f5fef63e53f14a02277c0c4467d46982dd8dee742420f84661a` |
| Historical and generated FAI | 79,526 bytes | `8be439d7b9c529203a0adf64ca9183b6f72b66d3f2c0bd8adc0924e8e56818aa` |
| M14.5 `matrix.prepared.h5ad` | 27,188,939 bytes | `9a23e66ae06dcd335070481b649db447992daae18841b448bb513d9f3d91fcac` |

Species: **Danio rerio**, taxonomy **7955**, assembly **danRer11**.
The genuine matrix is the existing 2,000 × 466,307 int64 CSR preparation with
2,970,369 nonzeros and explicit binary-accessibility semantics. No new matrix
conversion, biological filtering or subsampling was performed in Work A.

The uploaded reference has 1,923 contigs and all 466,307 ordered regulatory
features. Its reference identity is exactly the M14.5 identity:
`6d60b56a0c11a65548a2fa1fa070cb2944021b4139a19b143245cf998a6c43b4`.
Ordered feature identity:
`da73e2d187bf2c00063c115824ed29500f677ff821e181a8f3832a054b131069`.
The newly published manifest SHA is
`557f596ce82e7682331bab98888d664e9d894b61ddb6f1f778b04e02c6887cd1`;
manifest paths/provenance differ while the scientific reference identity agrees.

Free space was approximately 15 TiB before the test. Uploads use 1 MiB bounded
chunks. The managed upload is the necessary copy of the user-supplied file;
promotion and subsequent Sessions add no genome copies. Installed versions were
Python 3.11.16, pysam 0.24.1, bundled samtools 1.24, AnnData 0.12.19 and Firefox
157.0.1. No dependency was installed or upgraded.

## Native Firefox acceptance

Firefox used the actual existing page, file input, upload button, composer,
Session controls and same-origin read endpoints on isolated port 8898.
Screenshots and `browser-timeline.jsonl` are under A.

| Case | Result and evidence |
| --- | --- |
| A: genuinely missing reference | PASS. Session `26aa5f6b-8986-4f16-b94f-ccac17ecd079` began with no qualified reference catalog entry; the pending choices were empty. The UI asked for plain FASTA and explained optional/generated FAI. |
| A: genuine upload/qualification | PASS. The browser uploaded the entire real FASTA without an FAI. The generated FAI matches the preserved FAI exactly. Qualification covered the complete genome and feature vocabulary. |
| A: original-task admission | PASS, bounded. The original H5AD SHA and original objective reached a compiled `adopt_scATAC_cell_by_features → adapt_epizoo_species` plan. Whole-plan preflight passed. |
| B: authorized separate Session | PASS. Session `85cdfe8b-02fb-4111-a81a-edbb7ecefa77` discovered the promoted reference without another FASTA upload and reached the same normal admission boundary. The reference verifier count did not increase. |
| B: restart | PASS. Fresh-server Firefox reopening restored six captured conversation texts and the pending reference offer. All 35 workspace files retained identical hashes. Provider constructions and scientific reference qualification calls were zero. The check was repeated on final production code. |

The acceptance-only executor records normal `PlanExecutor.preflight` and then
raises an explicit Work A stop before any scientific plan step. Consequently the
browser truthfully displays **Analysis failed** for these deliberately stopped
Runs. There is no accepted new matrix/model Revision or completed post-training
result. `A/admissions.jsonl` contains the two passing preflight reports and exact
compiled arguments. This harness boundary is not a production execution mode.

One initial acceptance logger mistakenly passed a reference dataclass to a
serializer for orchestration models. It failed after scientific verification but
before publication. The harness was corrected, and the **same upload identity**
was retried; no duplicate genome or source substitution occurred. Across those
two attempts the deep owner verification ran twice; the accepted provisioning
attempt ran it once (30.943 seconds). Subsequent continuation, promotion, reuse
and restart did not repeat it. Index generation occurred on the first attempt
and the retained index was verified on retry. Browser harness restart assertions
were also corrected to match the existing six-text Session and server readiness.
These harness failures and retries remain in A; they are not omitted successes.

## Live OpenRouter attempt

One bounded attempt used the configured `agent` environment and
`openai/gpt-5.6-sol`. Credential availability was checked only as a boolean.
The existing durable budget guard, exact model, maximum-price policy, zero SDK
retries and no-fallback behavior were retained.

The interpreter dispatch failed with `PROVIDER_CONNECTION_FAILED`; no model
response was returned. The ledger halted with `PROVIDER_FAILURE_REQUIRES_REVIEW`.
There was no semantic retry, prompt repair, favorable rerun or provider fallback.
The original live Session was `661ff708-faaa-4b44-9b11-d2b436e6fa6c`.

Reserved upper-bound cost: **$0.158690**. Returned token usage and billed cost are
unavailable; this is not a claim of zero charge. `A/provider/ledger.json`, the
captured request and `native-live-start.json` preserve the evidence. Live missing
reference/continuation quality is **BLOCKED by transport**, not demonstrated
provider-authored semantic failure or success. Scripted acceptance is separate.

## Tests and negative cases

The final affected selection ran with all `RUN_*` gates removed, no `-k`
exclusions, and no source/test/static edits during execution:

**869 passed, 3 skipped, 2 failed, 3 warnings in 181.99 seconds** (exit 1).
All 508 frozen source/test/static files remained unchanged during that run.
The exact command and file list are in `A/focused-final-result.json`.

It covers resource/upload configuration and continuation, H5AD composition,
Session and interactive boundaries, real tiny neighbors/Leiden/UMAP, scientific
downloads, M14 adaptation contracts/lifecycle/preprocessing/verifiers, reference
owners, human/mouse EpiZoo inference, inherited target inference and bindings.
The two failures reproduce on a byte-exact isolated reconstruction of the
inherited starting tree:

- `test_h5ad_composition.py::test_companions_cannot_override_source_or_add_unsupported_inputs[assembly]`
  expects assembly to be unsupported although inherited target-model inputs allow it.
- `test_turn_decisions.py::test_parameter_schema_is_narrow_nullable_and_not_offered_on_read_only_or_qc_decisions`
  expects old `start`/`end` fields instead of the inherited `evidence` declaration.

These unrelated inherited assertions were not weakened or silently excluded.
The upload-limit test was updated for the intentional new 8 GiB ceiling.

After adding two explicit different-matrix reuse cases, the final resource,
target-resource and generic-resource continuation selection passes
**29 tests in 47.92 seconds**, including **12 new reference-supplementation cases**.
The two rich guidance/branching checks pass independently on both inherited and
current code (current: 92.44 seconds).

Negative coverage includes invalid/gzip FASTA, inconsistent supplied FAI,
automatic missing-FAI generation, exact supplied-FAI preservation, contig and
coordinate mismatch, incompatible assembly, changed qualified bytes, private
reference selection from another Session, stale generation/origin, conflicting
assembly declarations, catalog mutation, absent public promotion endpoint,
missing operator review, and a separately registered H5AD with different ordered
features. A separately registered matrix with new cell identities and the same
ordered features can reuse the deployment reference.

An earlier broader exploratory test selection was interrupted after failures;
pytest teardown also errored during interruption. It is not acceptance evidence
of a completed suite. Its old upload-limit expectation is resolved above. Two
additional inherited failures (old Leiden literal offsets and rejection of a
single-member role selection) reproduce in the reconstructed baseline. Two rich
checks that failed in the exploratory run pass in isolated baseline and current
runs; their exploratory failure cause is unestablished. Logs are retained, and
no all-suite PASS is claimed. No full repository regression was run.

## Scope conclusions

| Scope | Conclusion |
| --- | --- |
| FASTA/optional FAI upload, M14 qualification and original-task admission | **PASS**, scripted interpreter/Planner with genuine full resources; deliberate stop after preflight |
| Private task scope, authorized deployment promotion, later compatible Session and restart | **PASS** within this operator-managed deployment |
| New negative/reference continuation tests | **PASS** |
| Existing affected workflows | **PARTIAL regression record**: 869 pass, 3 skip, 2 reproduced inherited failures; no new persistent failure found in the selected final run |
| Live provider interaction | **BLOCKED**, one transport failure; no semantic qualification |
| Fresh post-training / biological performance / runtime verifier reuse during training | **NOT RUN**, Work Unit B |
| Global cross-user publication or independently proven redistribution rights | **NOT CLAIMED** |

The mechanism is reviewable without another infrastructure layer. Remaining
limits are exact existing feature-format/vocabulary bounds, operator-supplied
scientific declarations and training resources, explicit sharing review, the
unchanged Work B scientific-verification boundary, and the unqualified live
provider flow. Work stops here for review; no training, commit or push follows.
