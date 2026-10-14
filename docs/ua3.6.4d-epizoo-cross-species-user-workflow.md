# UA3.6.4d — EpiZoo cross-species scientific user workflow

Implementation and bounded acceptance record, 2026-10-14. **Ready for review; no commit or push.**

The primary new edge works: an accepted target-species EpiZooX model feeds the existing embedding tool, neighbors, Leiden, UMAP, evidence, safe downloads and durable Sessions. Real zebrafish data passed both scripted-language and live OpenRouter execution. The first live request failed planning; an explicit same-session instruction to reuse the selected model succeeded, followed by an evidence-grounded live answer. This is not an unattended first-request PASS.

**Overall assessment: PARTIAL.** Target-model reuse and downstream interaction pass within the exact scope below. Automatic new-reference acquisition/FAI supplementation, shared reference publication, and a newly trained model through the complete browser flow are not established. Biological convergence and performance remain unqualified.

All execution evidence is outside Git at `/home/likeyi/program/agent-acceptance/ua3.6.4d` (abbreviated **R** below). This report distinguishes final accepted runs from retained development attempts.

## 1. Baselines and preservation

- Agent `main` and `origin/main`: `ce5b53d7f716b777fc5333ea84e4d868ccf0da71`, the supplied UA3.6.4c baseline. Initial status contained only pre-existing untracked `evals/`.
- EpiZoo: `/home/likeyi/program/EpiZoo`, `029cd631d0f9806a646c4a3b42ce10b958f2b67f`; clean before and after. Its algorithms and native qualification code were not edited.
- `R/baseline.json` captured size, mtime and SHA-256 for 1,926 existing files. `R/preservation.json` identifies the 20 intentionally changed tracked files and 1,906 unchanged files, including all **1,274 pre-existing eval files**. New implementation/tests/report are separately visible in Git status.
- Fifteen consumed historical model/data/reference resources, **6,986,201,744 bytes**, were checked against their recorded hashes. The original H5AD, target publication, reference manifest, FASTA and FAI remain unchanged. This is a bounded consumed-resource audit, not a claim to have rehashed every historical directory.
- Existing web workspaces and operator configuration were not modified. All new Sessions, uploads, provider records and configuration are under R. No biological dataset or checkpoint was added to Git. The index remains empty; HEAD is unchanged.

The detailed architecture contract remains [AGENTS.md](../AGENTS.md). Relevant reviewed records were M14.1–M14.5, M14.7, M14.11, [UA3.5.5](ua3.5.5-human-epizoo-user-workflow.md), [UA3.6.4a](ua3.6.4a-scientific-dialogue-closeout.md), and the accepted UA3.6.4c input/workflow records.

## 2. Existing scientific capabilities reused

The Registry still contains **23 tools**. No new scientific tool, Planner, trainer, resource database or task-state mechanism was introduced.

`adapt_epizoo_species` retains its existing mapped-reference and de-novo EpiZooX species post-training routes. M14.1 reference qualification, M14.2 adaptation publication, schema-2 accepted authority, operation-local verification contexts, semantic ports, InputSets, pending resource questions, upload registration, Session/Revision persistence and artifact delivery are reused.

Native EpiZoo owns TF-IDF, cell sentences, target datasets/collation, strict target model loading and embedding extraction. Existing Agent tools own neighbors, fixed-seed weighted Leiden and UMAP. Generic fine-tuning and `EpiZooFinetuneTrainer` were not added. Mapping for parameter inheritance remains distinct from matrix projection.

## 3. Integration and restrictions removed

`epizoo_embed_cells` now accepts either the unchanged legacy human/mouse branch or an exact target-model binding. A neutral species object contains `scientific_name` and `taxonomy_id`; assembly and measurement semantics are explicit. Legacy normalized human/mouse conversational choices remain available.

The existing semantic compiler can bind the atomic target manifest/SHA pair from an input or the existing `epizoo_target_model.v1` output. Source Run/step coordinates are an atomic operator binding, not an end-user question. No step-name inference or automatic workflow completion was added.

Target inference uses the input's complete compatible vocabulary, without requiring historical human/mouse cCRE collections or M14.5's prepared int64 H5AD layout. Raw integral float32 sparse binary input is supported. Exact sparse column permutation is allowed; row order is preserved. No count projection, feature invention, alias repair, feature dropping, or dataset subsampling occurs inside the tool.

A narrow accepted-target consumer avoids requiring unchanged *training adapter* code for downstream inference. It uses the existing accepted execution and verification-context mechanisms; it does not migrate, renew or weaken historical adaptation authority. The original adaptation recovery still enforces its original full compatibility closure.

## 4. Exact target inference contract

Implementation: `tools/models/epizoo_target_inference.py`, called by the existing `tools/analysis/epizoo_embedding.py`; accepted historical binding: `orchestration/target_model_binding.py`.

The operator configures `AGENT_EPIZOO_SOURCE_STORE` to the trusted historical RunStore and supplies an exact accepted Run/step in the existing InputSet. Providers receive typed input handles, not RunStore access or executable functions. The consumer checks:

1. Accepted successful `adapt_epizoo_species` execution, schema-2 scientific correctness authority, publication/manifest/profile/checkpoint identity and accepted execution anchor.
2. Every published model sidecar and receipt, and the exact complete native EpiZoo Python implementation recorded by that authority. Missing, changed or extra native modules fail closed.
3. Qualified numerical dependency/runtime compatibility. CPU/inter-op worker counts are captured for inference but are not immutable checkpoint compatibility constraints for later read-only presentation.
4. Exact species/taxonomy, assembly, profile measurement semantics, all target feature identities, unique axes, canonical sparse CSR/CSC X, finite nonnegative integral measurements and binary 0/1 values where required. Conflicting dataset declarations fail closed.
5. Native EpiZoo TF-IDF using the accepted **training-corpus document frequencies and training cell count**, followed by native target cell sentences with token offset 4, `InferenceCellDatasetX`, native collation, `strict_target`/`EpiZooXConfig`, and `extract_cell_embeddings`.
6. Strict checkpoint architecture loading, accepted sequence/identity resources, fixed invocation configuration, file/authority checks around inference, and exact source/output hashes and cell correspondence.

Method: `epizoo-target-inference.v1`. Fixed configuration: `cuda:0`, batch 1, sequence length 8192, native random sampling enabled, seed 0, workers 0, float32 model/output, native AMP enabled. No automatic device, batch, dtype or truncation fallback exists. Published vocabulary/resources remain unchanged. The bounded source admission is at most 1 GiB with the existing decoded H5AD layout bound; the full sparse scATAC matrix is never densified.

Result verification checks the accepted model binding, provenance, species/assembly/semantics, complete feature correspondence and whether reordering occurred, input/output hashes, exact ordered cell IDs, 512-dimensional float32 output and finite values. It does not replay inference to assert numerical equivalence or biological accuracy. Without accepted source coordinates, the original adaptation owner remains responsible for target qualification; same-plan owner-proof reuse retains its existing behavior.

A development evidence failure exposed a real integration defect: Scanpy changed the worker's CPU thread count from 4 to 32, and downstream model consumption incorrectly required the training-time count. The fix retains dependency/model integrity checks and records actual inference worker counts separately. Both final runs pass current result verifiers in a fresh process with 32 CPU threads and inference/training/SEAM/mapping prohibited (`R/final-verification.json`).

## 5. Resource discovery, continuation and reference reuse

The existing operator InputSet catalog now offers target-model H5AD companions through the existing typed `request_resources` / `answer_resources` continuation. The offered handle/label is safe; its captured definition binds the exact model, species, assembly and semantics. The original registered H5AD and objective cannot be replaced by a resource reply. Changed source registrations, conflicting declarations, changed catalog identities and stale pending state fail closed.

The isolated `zebrafish-reviewed` InputSet admits the historical qualification model. Its embedded reference identity is the genuine existing danRer11 resource, not a species-name match. Native scripted execution confirms the selected resource through a pending question; software HTTP tests separately cover starting with no selected target companion, adding it later, and rejecting replacement H5ADs. The native first clarification already had the companion selected and is therefore **resource confirmation**, not evidence of missing-FASTA supplementation.

A fresh live Session reused the same operator model/reference binding without uploading or acquiring the genome again. Restart retains this catalog through the isolated operator configuration. Inference consumes the accepted sequence/model sidecars; it does not need to reopen FASTA, regenerate sequence embeddings or reconstruct mapping.

For a genuinely new post-training reference, the existing owner APIs remain `build_regulatory_feature_reference`, `publish_regulatory_feature_reference` and `verify_fasta_fai`. Their full sequence/index/coordinate qualification was retained and regression-tested. **No automatic arbitrary-species public acquisition route, FASTA upload role, generated-FAI browser continuation, or new reference provisioning API was added here.** A missing new reference still requires operator provisioning using those owners and a new immutable InputSet. The original typed pending objective can be retained, but an end-to-end missing-FASTA supplementation route is not qualified by this work.

## 6. Privacy and shared-publication boundary

The existing application is an operator-managed deployment, not a new multi-tenant resource authorization service. Model/source-store coordinates are operator configuration. Uploaded H5ADs remain within the isolated workspace; the catalog does not automatically promote uploads, training corpora or derived checkpoints.

The historical danRer11 FASTA/FAI were operator supplied. Their scientific qualification does **not** establish redistribution permission. No global shared publication or cross-user access was performed or claimed. This part stops at the task's trust/security boundary: building tenant authorization and promotion approvals would require new architecture.

The smallest deployment alternative is an operator-reviewed, immutable existing-catalog entry, scoped to a deployment whose users are authorized for that resource. Before a wider deployment, establish source/version, acquisition provenance and sharing rights, qualify the exact resources with existing owners, and explicitly add the reviewed binding to that deployment's configuration. Keep uncertain private resources private. No production default was changed.

## 7. Genuine datasets and provenance

Target: **Danio rerio**, taxonomy **7955**, assembly **danRer11**. Source: the preserved EpiZoo demo `data/zebrafish_downsampled_2000_cells.h5ad`, 2,000 cells × 466,307 features, float32 CSR, 2,970,369 binary nonzeros. Its matrix-derived 500 bp chr1–chr25 vocabulary is not claimed to be a curated cCRE collection. Original processing/provenance limits remain those in [M14.5](m14.5-real-zebrafish-de-novo-acceptance.md).

| Identity | SHA-256 |
| --- | --- |
| Original 2,000-cell H5AD | `1e1d61970cca6225fb5c39e122a6fcea8e2e86c8c66dafbca8594ef6d1f96205` |
| Final real first-16-cell fixture | `3a065a48a58aa7d0370a5b62637ba8ecf0e9fb932d86401982812814fd82c04d` |
| Original danRer11 FASTA | `9d45886b4c686f5fef63e53f14a02277c0c4467d46982dd8dee742420f84661a` |
| Original FAI | `8be439d7b9c529203a0adf64ca9183b6f72b66d3f2c0bd8adc0924e8e56818aa` |
| Reference manifest | `9dc34c071b009da4f2ed5512daa391b452906cc6d25610610675d7c69f41ceb4` |
| Reference identity | `6d60b56a0c11a65548a2fa1fa070cb2944021b4139a19b143245cf998a6c43b4` |
| Ordered feature identity | `da73e2d187bf2c00063c115824ed29500f677ff821e181a8f3832a054b131069` |

The final acceptance fixture explicitly copies the first 16 original cells and **all 466,307 features**, preserving their values/order and original metadata. It is a real-data subset prepared outside the product, not a simulated matrix or an automatic inference truncation. Synthetic unit fixtures establish software behavior only. No new public data download or dependency installation occurred; current acquisition URLs/rights were not invented for historical operator-supplied files.

## 8. Accepted-checkpoint reuse and adaptation history

Source RunStore: `/home/likeyi/program/agent-acceptance/m14.5/application-r2/application/run_state`; accepted Run/step: `m14.5-real:run` / `adapt`.

- Manifest SHA: `fcb309b7d4e78bd8c0fc9ace8f7eb4e3db6b57653c0e200788ff43bb98517caa`.
- Checkpoint SHA: `1de0a9d7c391a8eba55b307f3318af0a96b65e7d2b01038af23d73f150f1d257`.
- Accepted authority identity: `99c3ef79cd2333f1886a4baec7376ca548aa168ecefe215c18e03bedbfa346b4`.
- Historical strategy: de-novo species post-training; purpose: qualification; completed optimizer steps: **10**.

Historical adaptation recovery rejects the current M14.7 `preprocessing.py` adapter hash under its original full closure. That restriction was not bypassed or rewritten. The narrower downstream consumer checks accepted publication and native inference compatibility without importing stale training authority.

Before that consumer was completed, an isolated attempt requalified existing bundles and launched the unchanged 10-step route under an explicit 1,800-second/24-GiB gate (`R/resource-gate.json`). SEAM progressed to 16,000 sequences in approximately 112 seconds, implying excessive complete-vocabulary cost for this unnecessary rerun. It was explicitly interrupted before training. Its original adoption/owner work and partial SEAM files remain under R; it is **not** successful adaptation evidence. No 500,000-step training was launched.

Final scripted and live model reuse ran with hard guards against adaptation training, mapping, sequence/SEAM production and deep target reconstruction. Existing mapped-reference/de-novo synthetic qualification/regression remains covered. No new real macaque inference was claimed.

## 9. Actual scientific executions

| Scope | Session | Accepted Revision |
| --- | --- | --- |
| Scripted language, real science | `0617ef0a-6692-4932-8fec-6ac973003fda` | `2a8e76ad709a4c68bc48f191b56e20e274a0ecb65cc132b32c17e58987a2f386` |
| Live OpenRouter after explicit reuse follow-up | `60439384-50a1-4053-8796-b80e7022f2de` | `9830dce6280b4be36fee5e71aa00e379ebe4c41e0b8f1ed828b185270edb2fda` |

Both final accepted runs execute `epizoo_embed_cells → build_cell_neighbors → cluster_cells → compute_cell_umap`, each step once with passing normal verification. Their exact run IDs and accepted results are in `R/download-audit.json` and `R/live-download-audit.json`. Both preserve all 16 input cell IDs, produce finite 16 × 512 float32 embeddings and finite 16 × 2 UMAP coordinates, and yield one Leiden cluster at resolution 0.5. Their embedding SHA is identical. This is technical reproducibility for this bounded input, not a biological clustering benchmark.

Final server instrumentation records two target inference calls total, one per accepted analysis. Peak allocated/reserved CUDA memory: **4,666,875,904 / 4,718,592,000 bytes**. No hidden batch/device/dtype fallback was used. Existing downstream defaults and explicit resolution are recorded in accepted evidence.

A development prototype ran the actual full 2,000-cell input through all four tools, with peak reserved memory 5,047,844,864 bytes, but evidence publication failed on the subsequently fixed thread-count check. It created no accepted Revision and is **not** a final end-to-end acceptance. Earlier 16-cell diagnostic runs and direct smoke inference are similarly kept separate from the two final accepted workflows.

## 10. Computational verification and scientific limits

Checks cover species/assembly/semantics mismatch; missing/changed/duplicate features and cells; lossless column reorder; dense, noncanonical, negative, fractional, nonfinite and excessive integer measurements; binary/count distinction; incomplete/conflicting model bindings; changed checkpoint, receipt and native code; changed accepted anchors/runtime; persisted feature/cell/output identity; unsafe or tampered downloads; and historical reference/FASTA/FAI qualification.

The native strict target loader and preserved accepted model sidecars establish technical architecture/vocabulary compatibility. No expensive SEAM, mapping or post-training reconstruction is required for downstream presentation. Finite embeddings, valid graphs and UMAP do not establish meaningful biological clusters, cell types, conserved states, convergence or cross-species generalization. The 10-step checkpoint and 16-cell subset cannot qualify those outcomes.

The result/report projection includes method, assembly, semantics, checkpoint and vocabulary hashes, reorder status, training purpose/steps and `target_biological_performance = not_assessed`. The deterministic original report explicitly states the biological limits. Existing web report delivery remains a labelled safe accepted-evidence projection, not a byte-for-byte download of private-path-bearing Markdown.

## 11. Native Firefox, actual downloads and recovery

Native `/usr/bin/firefox --headless --marionette` used the actual web UI at isolated loopback port 8897. Upload used the native file input; submissions used the normal composer. No HTTP test client replaced browser acceptance. The browser displayed accepted evidence, reports, the real UMAP PNG and download links for both scientific artifacts.

The new closed `analysis.h5ad` export contains ordered cells, `obs.leiden`, `obsm.X_epizoo`, `obsm.X_umap` and minimal export provenance. It omits input X, private metadata, internal paths and unrelated objects. Existing `neighbors.h5ad` export supplies the exact accepted graph. The same pinned-source snapshot/transfer reservation/digest machinery is reused. UMAP results now record an optional producer digest; old results lacking it do not acquire new download authority.

Actual downloaded H5ADs were opened and checked against all accepted embeddings, labels, coordinates, cell IDs and graph sparse values. The input cell order was independently compared. PNG bytes match their accepted SHA. Source and delivered-content digests remain distinct.

| Final scripted download | Bytes | Delivered SHA-256 |
| --- | ---: | --- |
| `analysis.h5ad` | 53,712 | `bb92ebfcb35bcb2aa5a23e6ee9de3360403c07e383edb170fa068ba3f1fe23b9` |
| `neighbors.h5ad` | 83,440 | `3343c4fe8370b8f321d9f96fa029095d167bf138919a214ebffdf18ea7da2a8c` |

The live downloads have their own source-linked identities in `R/live-download-audit.json`; they are not substituted scripted artifacts. Both Sessions were reopened through Firefox after a fresh server with provider construction, inference, Scanpy neighbors/Leiden/UMAP, training and SEAM/mapping guarded against invocation. Reports, evidence, PNG and scientific downloads matched; refresh retained exact history and active Revision. **All 55 workspace files remained byte-identical**, with zero provider construction and zero observed science (`R/native-reopen.json`).

Browser driver adjustments for the current chat-first UI (opening “View result details”, selecting the report rather than the first PNG) were acceptance-harness corrections, not product semantic repair. The product UI change only adds the reviewed `analysis.h5ad` descriptor to its existing download allowlist.

## 12. Continuous real OpenRouter attempt and spending

The independent server was launched after activating the existing `agent` Conda environment. `OPENROUTER_API_KEY` presence was recorded as a boolean only. No secret value was printed or copied. Exact model: **`openai/gpt-5.6-sol`**, profile `openrouter-gpt-5-6-sol`; no silent fallback. Existing acceptance limits: 12 total completions, 6 per turn, $3 conservative reservation ceiling, 4,096 output tokens, 60-second request timeout, SDK retries zero, and provider routing checks.

The same live Session contains:

1. Initial natural-language adaptation/reuse request: interpreter chose `adapt_epizoo_species`; capability selection and two existing Planner attempts proposed unsupported adoption/new adaptation despite the offered target model. Compilation rejected `UNAUTHORIZED_REQUEST_INPUT`. No science or Revision resulted. Requests/responses 1–4 are retained exactly, including the ordinary bounded compiler-feedback Planner retry.
2. Explicit follow-up: “The reviewed compatible target model is already selected. Reuse that model without new training. Extract embeddings and run cell neighbors, Leiden clustering at resolution 0.5, and UMAP for the same uploaded cells.” Interpreter, capability selection, Planner and actual-plan output selection were all real provider calls (5–8). The plan explicitly bound the existing target-model and accepted-source handles and completed all four scientific tools.
3. Question about embedded cell count, convergence and biological performance: real interpreter chose the offered cooutput group `g0`; real answerer (calls 9–10) cited 16 cells, method, qualification purpose, 10 training steps and `not_assessed`, with `insufficient_evidence` for convergence/biological conclusions. No science reran.

**Provider-reported total spending: USD 0.1938485**, all ten calls returned usage/cost. Conservative reserved total: **USD 0.904396**; no unknown-cost request or fallback. Initial failed turn cost $0.1041625; the successful follow-up plus answer cost $0.089686. Full per-call usage, model/routing metadata, request/response hashes and exact content are retained in `R/provider/ledger.json`, `request-01..10.json` and `response-01..10.json`.

This establishes a continuous **live reuse-follow-up → real science → accepted evidence answer** in one Session. It does not turn the failed initial request into a first-request PASS or combine scripted semantic decisions with the live chain. The wording clarification is explicitly disclosed; no species/provider-specific parser, automatic intent override or post-hoc response repair was added.

## 13. Focused checks and environment

Environment retained: `/home/likeyi/anaconda3/envs/agent`, Python 3.11.16, PyTorch 2.7.1+cu128, CUDA 12.8, RTX 4090 24 GiB. Captured target runtime also records NumPy 2.4.6, SciPy 1.17.1, AnnData 0.12.19, pandas 2.3.3, transformers 4.57.6, tokenizers 0.22.2, flash-attn 2.8.3, triton 3.3.1 and pyfaidx 0.9.0.4. No pinned component was upgraded.

Development checkpoints: 555 relevant checks passed, 3 skipped; final target integrity checks passed 41; the updated download/history suite passed 16. Two earlier delivery assertions expected the previous graph-only inventory; they were updated to require the child's own new compact analysis while continuing to reject unauthorized historical graph attachments. No integrity test was relaxed to accept changed scientific bytes.

**Final combined affected-domain regression: 620 passed, 3 skipped, 3 warnings in 190.92 seconds, exit 0** (`R/final-affected-regression.log`). All 517 frozen source/test/static files remained unchanged (`R/final-source-invariance.json`); `git diff --check` passes. Its scope is adaptation, neutral references/adoption, target inference/authority, legacy human/mouse inference, embedding/downstream analysis, resource continuation, exact declarations, evidence/reports, existing artifact privacy/history/recovery and static UI integration. It is **not a full repository regression**. `R/final-source-freeze.json` records 517 source/test/static files for final invariance checking. Three tests remain environment-gated; no live-training gate was enabled for pytest.

## 14. Remaining product limitations

- The broad initial live adaptation request required an explicit reuse follow-up; autonomous intent accuracy is not generally qualified.
- Arbitrary new H5AD feature sets cannot use this checkpoint. Exact target vocabulary correspondence and measurement semantics are mandatory. Coordinate renaming, missing-feature repair and peak-to-cCRE projection are not supplied.
- New adaptation still uses the existing fully specified scientific route. A complete browser upload → new reference supplementation → newly trained target → inference workflow was not accepted here. No converged training run was attempted.
- Missing FASTA acquisition/upload and deterministic FAI creation are not integrated as browser resource roles in this patch. Existing owner qualification APIs remain the provisioning boundary.
- Cross-user/global publication and redistribution rights are not established. The demonstrated catalog is the existing isolated operator deployment catalog.
- Target inference is bounded to the recorded CUDA/native runtime and input limits. CPU target inference, alternative execution settings, arbitrary architectures and broad cross-species biological validation are outside this qualification.
- Embedding results retain their existing accepted-result verification/evidence contract; the UI truthfully does not claim a new reusable schema-2 embedding authority. Safe downloads bind accepted execution plus pinned result bytes.

## 15. Scope assessment

| Scope | Assessment | Evidence / boundary |
| --- | --- | --- |
| Exact accepted target → native EpiZooX embeddings | **PASS** | Real danRer11 input; exact M14.5 model and vocabulary; unchanged native backend |
| Existing neighbors, Leiden, UMAP | **PASS** | Both final 16-cell runs; all four steps verified once |
| Safe scientific downloads, evidence, PNG/report | **PASS** | Native browser reads and full downloaded-content comparisons |
| Session/restart/refresh without replay | **PASS** | Two Sessions; 55 immutable workspace files; zero providers/science |
| Existing InputSet target selection and continuation | **PASS, bounded** | Native resource confirmation; HTTP missing-target supplementation and source-substitution negatives |
| Same model/reference reuse in another Session | **PASS, local operator scope** | Fresh live Session and restart; no genome reupload or scientific reconstruction |
| Continuous real-model reuse follow-up and evidence answer | **PASS, assisted scope** | Calls 5–10; no scripted decisions in this live chain |
| Initial broad live request | **PARTIAL / failed initial turn** | Wrong new-adaptation plan; compiler rejection; explicit follow-up required |
| New post-training browser chain | **PARTIAL** | Existing routes preserved; no fresh completed adaptation acceptance |
| Missing FASTA/public acquisition/generated-FAI continuation | **PARTIAL** | Existing qualification retained; browser supplementation edge not implemented |
| Global/cross-user reference publication | **BLOCKED within this scope** | Sharing basis and tenant authorization absent; private operator alternative documented |
| Biological convergence / meaningful clusters / generalization | **UNQUALIFIED** | Short qualification checkpoint and bounded technical subset only |
| Overall UA3.6.4d | **PARTIAL; primary reuse/inference workflow demonstrated** | Reviewable implementation and preserved evidence; no commit/push |
