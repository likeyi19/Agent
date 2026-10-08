# UA3.1 — First Web upload and multi-turn scientific interaction

Baseline: `main` at `8e9cacbe8adb4bb7deb8742537819b7c2fcde6a4`
(`Complete UA2.4 FASTQ multi-file resource admission`), matching local `main`
and `origin/main`. The initial tracked tree and index were clean. The preserved
baseline contained 705 evaluation files and 198 scientific/output files.
The final HEAD and local branch references remain at that commit. Changes are
unstaged; no commit, push, Git history rewrite or Git network operation was performed. All 705
evaluation entries and 198 output entries retain their names, sizes, modification
times, symlink identities and repeated content digests. Existing scientific
artifacts and the EpiZoo repository were not modified.

## Scope and ownership

```text
Browser H5AD attachment → controlled completed source → existing UA1 registration
→ per-turn registered input → ordinary Interpreter/Planner/compiler/preflight
→ existing inspect_scATAC/verification → RunStore/Session/Revision
→ saved conversation, accepted evidence and existing safe report presentation
```

Upload transfers bytes and creates no Session, plan, scientific result or tool
invocation. The first binding supplies the existing inspection-compatible input;
it does not select an operation. The LLM retains semantic intent, referent and
workflow decisions. Unsupported requests retain ordinary clarification or
failure behavior; there is no upload-specific planner or keyword routing.

`H5ADUploadAdmission` owns bounded transfer and completed-source membership.
`LocalResourceAdmission` remains the sole durable resource catalog and exact
source/binding owner. The HTTP layer invokes these owners and the unchanged
interactive facade. No second catalog, Session store or background queue exists.
Scientific tools, Registry contracts, independent verification, authority,
Interpreter/Planner, compiler, RunStore and Session/Revision schemas are unchanged.
The existing FASTQ, BAM and external-fragments Python admission paths remain.

## Configuration and deployment

Add these optional fields to the existing operator Web configuration:

```json
{
  "upload_root": "uploads",
  "upload_max_bytes": 268435456,
  "upload_max_concurrent": 2
}
```

This is a configuration fragment; the existing workspace/model fields are still
required. `upload_root` is disabled by default (absent or null). A relative root
resolves beneath `workspace_root`; an absolute root must also be a strict
workspace descendant. Parent traversal, symlink aliases and unsafe directory
states fail closed. `ManagedWorkspace` initializes the controlled directory
before the facade approves it for UA1 registration. Browsers supply no source
or destination paths.

The default file limit is 256 MiB, configurable from one byte through 1 GiB.
Concurrent transfers default to two, configurable from one through four per
upload owner. Existing turn workers remain separate bounded facade callers,
with their existing receipt, polling and cancellation behavior.

Launch remains `PYTHONPATH=src python -m agent.web --config /absolute/operator.json`
with default host `127.0.0.1`, port 8000 and one server process. This is the
existing trusted local/lab-host deployment, not an account or tenant service.
Session IDs do not provide authorization. Existing same-origin protections,
no-store policy and security headers remain; no public-hosting or account system
is introduced.

## HTTP and completed transfer

All paths below use the `/api/v1` prefix.

| Operation | Contract |
| --- | --- |
| `PUT /uploads/{upload_id}?filename=label.h5ad` | Raw `application/octet-stream`; successful completion returns `201` with safe resource metadata. |
| `GET /resources` | `{enabled, choices}`; only durable completed browser H5AD registrations, with bounded public ID/type/label/status metadata. |
| `POST /sessions/{session_id}/turns` | Existing turn schema plus optional opaque `resource_id`; mutually exclusive with `input_set_id`. |

An upload ID contains 1–64 ASCII letters, digits, underscores or hyphens.
The display filename is at most 160 printable characters and cannot contain
path separators or colons, or be `.`/`..`. It is a label; storage names are
server-generated hashes of the bounded upload identity. Declaring H5AD does
not parse the file or establish scientific compatibility.

Incoming bytes are streamed and hashed, with writes/hash updates bounded to
one MiB. Actual byte count enforces the configured limit, including chunked
requests without `Content-Length`. A supplied length is validated and must
match the complete transfer; empty or interrupted transfers are rejected.
Ordinary turn/session/navigation JSON retains its separate 32 KiB limit.
File bytes never enter Session state, prompts or JSON submission mappings.

Incomplete files live in private `staging`; completed files live in `completed`
under the approved root. Completion flushes/fsyncs data, publishes an exclusive
hard link atomically without replacing an existing destination, and fsyncs the
directories. UA1 registration then independently hashes the completed source
and checks the expected transfer SHA-256 and size **before publishing its
registration record**. A transport hash does not confer scientific authority.

Completed plus staging files share a 128-file admission bound. A short admission
lock covers capacity checking/staging creation; sixteen fixed upload lock stripes
bound lock-file growth and serialize conflicting identities across owner
instances. These are filesystem leases, not work records. Normal failure or
interruption removes private staging. Process-crash leftovers stay undiscoverable
and may consume the file bound; operator cleanup is manual. There is no resumable
upload protocol or automatic deletion of scientific inputs.

Discovery scans at most 1,024 existing registration records and returns at most
128 choices. Exceeding these bounds fails explicitly; records are never silently
truncated or ordered to resolve scientific ambiguity. Unrelated operator sources
outside the completed-upload namespace are not exposed.

Exact retries retain the same immutable resource identity. Different filename,
bytes or size conflict and never overwrite a registered source. In particular,
different bytes cannot repopulate a deleted registered source. An exact retry may
restore the same original bytes at their pinned location. A crash after completed
publication but before registration leaves no discoverable resource; an exact
retry can complete the existing UA1 handoff.

## Selection, conversation and recovery

The browser attachment control, upload progress/status and registered-source
selector sit near the existing composer. Upload is optional. Filenames and
errors render as text. Existing model, Session, InputSet, Guidance, navigation,
branch and report controls remain available. Browser selection is a convenience,
not a Session-wide scientific resource pointer.

A scientific turn carries only `resource_id`, never a path, source digest or
serialized binding. The server checks controlled completed-resource membership
and resolves the existing `RegisteredInput`. `_LocalWorkers` forwards that same
object through `validate_submission` and `submit_turn`. Generation, predecessor,
profile, immutable submission and duplicate-turn checks retain their contracts.
Each new consumption is checked by the facade against the original source bytes.

Completed historical reads/retries use captured attribution without reopening
source bytes. Missing or replaced bytes prevent new consumption. A later
accepted-result question can omit the registered source and use the existing
evidence-grounded dialogue path. Reusing the data requires explicit selection;
neither historical filenames nor upload labels become scientific authority.

Resource records and accepted conversation survive fresh application/server
construction. Refresh/reopen reads saved state; it does not replay model calls
or science. New follow-up answers may invoke the configured model, while existing
read-only evidence questions and navigation perform no scientific execution.
Scientific result acceptance, failed/nonaccepted states and Revision branching
remain existing application behavior.

Corrupt H5AD bytes can register successfully and then fail in the unchanged
inspection owner. Readability does not imply eligibility for adoption,
embedding, annotation or pseudobulk. The first slice adds no broader historical
embedding/graph/clustering/UMAP authority or generic multi-resource composition.

Existing downloads remain accepted PNG and labelled client-safe report TXT
projections. No scientific H5AD, uploaded source, raw manifest, table or generic
export route is added. Inspection creates lightweight findings, not a new dataset.

## Validation and review status

Final focused transport/API/multi-turn acceptance: **63 passed in 5.65 seconds**.
The five multi-turn/negative tests include reconstructed-server and independent
Python-process recovery with source-deleted historical reads/retries and forbidden
provider/scientific callbacks.

Final affected-domain regression: **1,101 passed, 3 skipped in 265.63 seconds**,
exit 0. Command, using the `agent` environment, with inherited `RUN_*` gates
removed before collection:

```text
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 python -m pytest -q \
  tests/web tests/application tests/tools/data/test_scatac.py
```

The three skips are existing optional FASTQ production qualification gates. No
full-repository regression, expensive biological pipeline or paid/live provider
was run. All 422 source/test/static/resource files in the final regression freeze
remained byte-identical while it ran. Freeze SHA-256:
`1f0ffbcc43059b7fdf5de18ed0fa46f6d02548de24879b16e1ce90d34151c80d`.

**22 native Firefox 157.0 checks** pass the actual local HTTP/browser path:
upload/registration without model/science, ordinary inspection, grounded
same-Session follow-up, explicit source reuse, refresh, safe evidence/report
presentation, generation-checked navigation and sibling branching, then a real
server restart with all provider/scientific callbacks forbidden. Restart restores
conversation, three Revisions, generation 4 and source selection; exact completed
retry and report reads preserve every workspace file byte. Three explicitly
requested inspections run, with no extra science for questions/navigation/reads.
Native evidence is retained outside Git under
`/home/likeyi/agent-ua31-browser-8vumh_q1/browser-checks.json`.

The existing and added static UI suites pass 26 checks; the UI task additionally
validated 15 DOM/XHR behavior checks in a V8 harness. `git diff --check` passes.
An earlier in-progress subset collected a test before its error-location
expectation was corrected (565 passed, 1 failed, 3 skipped); the final frozen
affected-domain run above replaces that preliminary result.

Acceptance uses small deterministic H5AD inputs, the actual existing inspection
owner and scripted provider-neutral model interfaces. Negative coverage includes
stream/length limits, interruption and staging invisibility, exact/conflicting
retry, missing-source conflict, unsafe filename/destination, source mutation
before registration, corrupt records, source/binding injection, stale generation,
conflicting InputSet/resource selections and corrupt scientific H5AD. It does
not establish real-data biology, live-provider quality, multi-format browser
upload or universal scientific continuation.

**UA3.1 READY FOR FINAL ACCEPTANCE REVIEW.** No blocking defect remains in the
implemented scope. This document does not mark UA3.1 accepted or authorize commit/push.
No subsequent UA3 stage is started.
