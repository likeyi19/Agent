# UA3.5.4a — Focused live semantic qualification

UA3.5.4a is complete within its bounded diagnostic scope. Real-provider Leiden
requests pass through the existing binding and verified execution path. Species
and Cell Selection clarification chains remain **unqualified**: the detailed
Planner returns `unsupported` for missing explicit inputs before the compiler
can produce the existing prerequisite diagnosis. No production changes, commit
or push were made. This is not UA3.6 acceptance.

The 14-case disposition is **5 PASS, 4 FAIL, 5 BLOCKED**, with the two negative
safety passes limited as described below.

## Environment and reproducibility

On 2026-10-10, branch `main`, HEAD, local `main` and local `origin/main` all equal
`8bcad5b42cb812c89dfc232e3ceadea2fc5e44a7`. The index is clean. Entry state
contains 28 modified tracked files and 12 new UA3.5.1–UA3.5.3 implementation,
test and documentation files, plus existing untracked evaluations.

The accepted development-primary profile is `f1-openrouter-gpt-5-6-sol`, provider
`openrouter`, exact model `openai/gpt-5.6-sol`, strict Chat Completions adapter,
60-second timeout. One real model-catalog lookup confirms that exact model and
the existing public prices: $2/million prompt tokens and $10/million completion
tokens. All 26 returned responses identify the exact requested model; routing
reports no pipeline modification and one upstream attempt. The operator's
Groq default remains unchanged; no substitute profile is tested.

There are **26 model-interface calls and 26 actual provider completions**:
9 Interpreter, 7 capability selection, 7 detailed Planner and 3 output selection.
Provider failures, transport retries, semantic repairs and profile failovers
are all zero. The pre-dispatch reservation is **$1.802666** against a fixed $5
ceiling; provider-reported usage cost totals **$0.185001**, with 58,732 prompt and
3,999 completion tokens. No balance or remaining-balance claim is made. Bounds
are 48 completions total, 6 per turn, 4,096 output tokens, and 150,000 conservative
input byte/token units. SDK retries are zero, routing fallback is disabled and
strict parameters are required. A provider-neutral one-dispatch-per-turn/schema
guard prevents extra repair or retry dispatches; it did not fire in this run.

Every dispatched semantic response is live. Fixtures and expectations are
deterministic, but no Interpreter, Planner or output-selection response is
scripted or repaired. The ordinary `InteractiveAgentApplication`, immutable
Registry, scoped-v4 Planner, compiler, binding, Session and scientific owners
remain the mechanism under test. Credentials are read only from the existing
Conda environment, never printed or copied. Saved prompts, schemas, responses
and diagnostics are screened for configured secret values; SDK headers/client
representations are not saved.

The [runner](../benchmarks/interactive/run_ua354a.py) and
[fixture/boundary helper](../benchmarks/interactive/ua354a_fixtures.py) are the only
new benchmark code. Use a fresh output/workspace, the `agent` Conda environment
and `PYTHONPATH=src:.`:

```bash
conda run --no-capture-output -n agent python -m benchmarks.interactive.run_ua354a \
  --output /path/to/new/evaluation \
  --workspace /path/to/new/disposable-workspace \
  --source-h5ad /path/to/the/retained/registered-source.h5ad \
  --operator-config /path/to/unchanged/operator-web.json
```

Optional `--fixtures-json` reuses the already generated tiny owner fixtures;
`--discovery-json` reuses this evaluation's exact catalog evidence. The current
records are under
[evals/ua3.5.4a_live_semantic_2026-10-10](../evals/ua3.5.4a_live_semantic_2026-10-10/).
[live/results.json](../evals/ua3.5.4a_live_semantic_2026-10-10/live/results.json)
links every raw call artifact. Per-case Session/run/evidence records preserve
validation separately from the raw LLM choice. Private disposable scientific
workspaces are outside Git; no datasets or checkpoints are added to Git.

Two preparatory attempts made zero semantic calls. First, the evaluation fixture
incorrectly supplied unpublished `batch_size`; registered-input composition
correctly rejected it. The evaluation now supplies only the public `device`
setting. Second, explicit recovery-policy metadata exposed the independently
reproduced defect below. Both preparatory records remain intact. The successful
matrix uses the ordinary default-`None` recovery configuration, without a
production workaround or patch, and reuses the one catalog lookup.

## Cases and outcomes

The registered H5AD is the retained UA3.2.2 sparse 32-cell mouse fixture with its
complete 1,341,077-feature axis. Its current bytes match the registered source
digest. Species is deliberately absent from the declaration; `device=cuda` and
the exact input remain captured. Only the existing qualified mouse resource is
available. A human answer would require an approved compatible resource; no
human execution qualification is implied by this mouse fixture.

Leiden uses 32 ordered synthetic 512-dimensional embeddings and a genuine
neighbor-owner H5AD. It qualifies downstream software behavior, not EpiZoo model
quality. Selection reuses genuine imported-fragment/QC artifacts from the
accepted UA3.5.3 synthetic fixture, including BEDTools production and independent
QC verification. Only its QC manifest and digest are supplied. Exactly
`min_qc_fragment_records` and `min_tss_enrichment` are required and missing;
other selection thresholds retain their existing optional semantics. Candidate
values 1000/4/20 are never introduced. Planned answers 1, then correction 2,
and TSS 0.5 are fixture operands, not biological policies.

PASS below means the stated bounded check passed, not universal language or
scientific qualification. BLOCKED follow-ups are not sent without a live-created
matching prerequisite. Categories: **1** LLM semantics; **2** context/contract;
**3** deterministic application/binding/Session; **4** correct owner/resource
rejection; **5** unsupported capability/invalid fixture; **6** infrastructure.

| Case / user text | Expected semantics | Actual live decision | Agent outcome | Result / owner |
| --- | --- | --- | --- | --- |
| A1: “Use this dataset to run the supported EpiZoo analysis.” | EpiZoo candidate; compiler-authored missing-species clarification | Interpreter `execute_plan/epizoo_embed_cells`; Planner `unsupported`, species required | `UNSUPPORTED_REQUEST`; no candidate, pending, preflight, steps or revision | **FAIL — 2**, Planner prerequisite contract |
| A2: “These cells are from human” | Answer the exact pending task; preserve input/device; enforce resources | Not dispatched: A1 created no pending task | No state fabrication or science | **BLOCKED — A1**; human resource restriction not reached |
| A3: “Analyze this dataset with the available EpiZoo model.” | Same prerequisite handoff under a paraphrase | Interpreter EpiZoo; Planner `unsupported`, species required | Same failed admission as A1 | **FAIL — 2** |
| A4: “The cells came from mice.” | Interpret mouse; resume original input and qualified resource | Not dispatched: A3 created no pending task | No state fabrication or science | **BLOCKED — A3** |
| B1: “Cluster these cells using Leiden with resolution 0.7.” | Exact `cluster_cells.resolution` binding | Execute with literal `0.7`, span `[49,52)`; one clustering plan | Verified resolution **0.7**; explicit-user evidence | **PASS** |
| B2: “Cluster these cells using Leiden.” | Omit resolution; existing owner default | Execute with no argument; plan omits resolution | Verified resolution **1.0**; `existing_owner_default`; no resolution clarification | **PASS** |
| B3: “Group these cells with Leiden, setting its granularity to 0.7.” | Interpret granularity as resolution without exact parameter wording | Execute with `resolution`, literal `0.7`, span `[58,61)` | Verified resolution **0.7**; explicit-user evidence | **PASS** |
| C1: “Select cells from these barcode QC results.” | Select owner; diagnose exactly two missing thresholds | Interpreter `select_scATAC_cells`; Planner `unsupported`, names both required thresholds | `UNSUPPORTED_REQUEST`; no candidate or pending task; zero selection execution | **FAIL — 2** |
| C2: “Keep barcodes with at least 1 QC fragment record.” | Partial depth binding; keep TSS outstanding | Not dispatched: C1 created no prerequisite | No invented pending state or thresholds | **BLOCKED — C1** |
| C3: “Actually, require at least 2 QC fragment records instead.” | Correct pending depth; preserve original task/input | Not dispatched | Correction mechanism not live-qualified | **BLOCKED — C1** |
| C4: “Use a minimum TSS enrichment of 0.5.” | Complete the exact pending request; run selection | Not dispatched | Original-task continuation not live-qualified | **BLOCKED — C1** |
| N1: “Cluster these cells using Leiden with resolution -0.7.” | No substitution or invalid execution | Interpreter `clarify/invalid_parameter_value` | Zero Planner/science; no accepted values or pending state | **PASS**, conservative rejection only; owner invalid-value path unexercised |
| N2: “Run two independent Leiden clusterings on these cells; use resolution 0.7.” | No guessed parameter scope or execution | Interpreter `clarify/unsupported_intent` | Zero Planner/science | **PASS**, conservative refusal only; repeated-consumer binding unexercised |
| N3: “Use EpiAgent to embed the cells in this dataset.” | Reject unavailable backend without substituting EpiZoo | Interpreter wrongly selects EpiZoo; Planner explicitly rejects absent EpiAgent | `UNSUPPORTED_REQUEST`; no plan, authority or science | **FAIL — 1** Interpreter choice; final capability boundary passes (**5**) |

B1/B3 use the accepted singular `argument` representation and the shared UA3.5.3
binder. They prove literal/span admission, exact actual-consumer scope and accepted
`explicit_user_parameters` provenance. Evidence also records
`clustering_resolution_origin=explicit_execution_argument`. B2 contains no
resolution in admitted inputs, compiled arguments or resolved arguments; the
scientific owner supplies its default and evidence distinguishes that origin.
The general plural/multiple-parameter and pending continuation paths are not
live-qualified by this matrix.

Only three live scientific plans reach the evaluation executor, each containing
one `cluster_cells`; all pass normal verification and preserve cell order. The
executor would explicitly withhold other scientific plans after ordinary
compilation/admission/preflight, but that boundary was not reached by A/C. No
EpiZoo inference, GPU workload, selection production, upstream reconstruction,
full browser acceptance or heavy scientific rerun occurs in the live matrix.
Fixture construction and one offline clustering smoke are recorded separately.

## Diagnosis and next work

The missing-prerequisite failure is primarily **category 2**, supported by the
actual prompts, not an inferred language-model misunderstanding. A1/A3 expose
species as required with `none_available`, no source, and choices human/mouse.
C1 exposes both exact required thresholds as unavailable. The detailed prompt
requires a valid DAG, forbids invented values, and defines `unsupported` to
include genuine **input insufficiency**. It does not explain the existing
compiler-authored clarification handoff. The provider identifies the operation
and missing values correctly, then returns a decision consistent with that
offered branch. The v4 schema
already permits an incomplete supported candidate with required sources omitted;
none was returned. Thus the existing compiler/prerequisite binder is never
invoked. No automatic conversion of provider prose into a plan is justified.

UA3.5.4b should narrowly clarify the generic Planner contract for supported
operations awaiting explicit user declarations, using existing Registry and
application clarification eligibility. Retain authoritative missing-input
diagnosis, genuine incompatibility/unsupported rejection and all current binding
restrictions. Reproduce these exact contexts in focused tests and then repeat
bounded live prerequisite qualification. No per-tool parser, guessed default,
new workflow or new Interpreter is justified.

An independent **category 3** defect exists in
`InteractiveAgentApplication._submission`: `asdict(planning_recovery_policy)`
leaves `retryable_provider_codes` as `frozenset`, which the configuration digest
cannot JSON-serialize. Both `submit_turn` and `validate_submission` raise
`TypeError` before factory/Interpreter invocation. Default-`None` validation
succeeds; all initial state files remain exact. The
[offline reproduction](../evals/ua3.5.4a_live_semantic_2026-10-10/diagnostics/reproduce_explicit_recovery_policy.py)
and [diagnostic](../evals/ua3.5.4a_live_semantic_2026-10-10/diagnostics/explicit-recovery-policy-reproduction.json)
record zero provider/factory calls. The smallest UA3.5.4b fix uses the existing
policy's `to_dict()` for that digest and tests submission/validation while
preserving the default-`None` digest. This is metadata admission, not a scientific
binding or language-model defect.

N3 establishes one Interpreter backend substitution error; the real Planner
correctly prevents execution. Further focused backend-identity qualification is
warranted. A provider-specific rule or architectural fix is not justified by
this one observation.

Current facade presentation has two bounded wording defects: N1 says “Previously
accepted values are retained for the pending request” in a fresh Session with
neither accepted values nor pending state; N2 renders a biological/marker-answer
limitation for a repeated clustering command. `responses.py` owns those generic
reason strings. UA3.5.4b can remove the unsupported state assumption and use
wording applicable to generic unsupported intent, without changing typed reasons
or state. These are current response mismatches, not stale binding state.

Repeated actual planned consumers still require exact scope and fail closed
through the accepted UA3.5.3 `AMBIGUOUS_PARAMETER_SCOPE` path; N2 never reaches
that path. Historical rejected-plan messages remain intentional immutable
history. This run neither produces a repeated rejected plan nor inspects the
native browser, so it establishes no stale-browser defect and authorizes no
first-match resolution or workflow redesign.

UA3.6 retains complete live-provider + real-science + browser qualification,
foundation execution, resource consumption/feature compatibility, species and
multi-threshold continuation, biological interpretation and broader reliability.
UA3.5.3 deterministic binding passes the admitted live Leiden cases and preserved
regression; the live missing-prerequisite chain remains a real integration gap.

## Preservation and validation

The five existing generic binding/safety/continuation/selection-evidence/
clustering-evidence suites pass **120 tests in 74.23 seconds**, exit 0, with
`RUN_*` gates disabled and zero providers. Offline fixture smoke and exact
transport reservation checks also pass. No full regression or browser run is
claimed. [Validation records](../evals/ua3.5.4a_live_semantic_2026-10-10/diagnostics/)
include the exact command, log, independent semantic review and reproduction.

All **576 pre-existing repository files** remain exact in SHA-256, size, mode
and modification time: 202 source files, 259 tests, 23 benchmark files, 77 docs
and 15 other files. This includes every prior UA3.5.1–UA3.5.3 change. The original
705 evaluation files, 198 scientific output files, 208 real-state files and one
operator configuration are unchanged. Checkpoint size/mtime/mode and its
previously pinned identity remain unchanged; its large hash is reused rather
than recomputed. Reused tiny fixture files retain their pinned bytes. The Git
index is byte-identical, and HEAD/main/origin/main remain aligned. Credentials
and operator configuration are not edited. Only this report, two benchmark files
and the new evaluation directory are added; no production algorithm, scientific
default, unrelated output, commit or push changes occur.
