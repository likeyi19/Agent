# Native Functions

Agent has two interaction modes. **AI Research** retains natural-language planning,
structured scientific clarifications, result continuation, result-scoped parameter
editing, grounded answers and existing scientific workflows. **Native Functions**
is a structured package/function runner. It never creates a conversation or calls
an Interpreter, LLM Planner, provider, model reviewer or language generator.

Choose **Native Functions**, then a package and an installed public function.
Provide the required inputs, enable only parameters you want to supply, and select
**Run function**. Defaults shown in the form describe the installed implementation;
unchecked arguments are omitted. Explicit null is different from omission. Native
errors are retained without substituting methods, changing parameters or retrying.

The Web operator configuration may omit `model_profiles` and `default_profile_id`.
Native execution then works without an OpenAI key or any other provider credential.
Existing scientific runtime/resource configuration and the
[native process boundary](native-execution.md) still apply.

## Installed API metadata

Python discovery reads public functions in the configured package's root, `pp`,
`tl`, `pl`, `metrics`, `get`, `utils` and `logging` namespaces. Signatures, defaults
and reliable native choices come from installed Python implementations. Imported
EpiScanpy aliases to Scanpy retain their actual implementation. This is broader
than the former hand-maintained matrix function map; it is not a claim that every
function's required objects or return values have adapters.

R discovery reads installed namespace exports, formals and available S3 default
method formals, plus reliably inspectable S4 Matrix method formals. Calls use the exported function with native dispatch; R defaults
that are expressions remain **Native default** and are never evaluated by the UI.
R S4-specific objects are not interchangeable with matrices or data frames.
MACS3 discovery reads its installed argparse definitions and invokes the selected
subcommand as argv, without a shell. A CLI option without a null representation
cannot accept explicit null. No user-provided Python/R source, expressions,
executable path or imported module is accepted.

Runtime locations are operator-owned. Python matrix runtimes retain MatrixPackage
configuration; shared R/CLI locations use `native_runtimes.py`. A discoverable API
can still require an unavailable object adapter or return an unsupported type.
Those are representation limitations, not scientific judgments about the data.

## Inputs and output preservation

Supported Python construction includes H5AD → AnnData, explicit CSV/TSV → pandas
DataFrame, a selected table column → list, inline JSON/list/array, NumPy NPY,
SciPy NPZ or Matrix Market sparse data, and managed native file copies. Parsing and
orientation choices are structural declarations; Agent infers no organism, assay,
genome, normalization or coordinate system. Original files remain unchanged.

R accepts sparse matrices (including explicitly oriented H5AD matrices), native
CSV/TSV data frames, vectors and supported accepted R objects. Native R returns
use RDS for supported atomic, tabular, sparse and simple list data. Arbitrary
uploaded RDS/Pickle objects are not accepted. Complex unsupported native objects
produce an explicit representation failure after native completion is recorded.

Python direct returns preserve scalar values, strings, booleans, null, ordered
lists/tuples, dictionary keys/order, NumPy arrays/scalars, sparse matrices, pandas
tables, Polars tables and AnnData. Typed files and `result.json` retain full values;
browser previews are bounded separately. Unsupported object arrays and specialized
objects fail representation honestly. In-place changes are detected from actual
object state and saved separately from returns. Native filtering may change axes.
A successful null return with no mutation or file output is a valid no-data result.

Actual generated files are preserved in the managed invocation context. Downloads
include individual native data files and a complete output archive. Existing
accepted-file identities and transfer budgets apply. Saved scientific output is
not truncated to fit a browser table, and no scientific result is manufactured
for presentation. Image previews use actual generated image files.

Choose **Accepted result** on a later input to reuse an exact earlier output.
The server resolves Session/Revision/step authority and pinned files, never a
filename or browser cache. Calls remain independent and explicit. Compatible
AnnData outputs can cross Python packages; supported R outputs can cross R
packages. An unavailable cross-language object representation is reported rather
than silently converted. No workflow or extra preprocessing is inferred.

## Shared owners and retained research adapters

The structured boundary binds one existing AgentPlan directly to AgentRuntime's
prepared execution entry. PlanExecutor, RunStore, process isolation, cancellation,
NativePublication, verification authority, SessionTurn, Revision and scientific
download preparation are shared with AI Research. There is no second scheduler,
result database or scientific authority. A Session in this interface stores calls
and accepted Revisions without conversational Interactions or model profiles.

`run_scanpy`, `run_snapatac2`, `run_episcanpy`, `run_signac` and the existing R/CLI
research ToolSpecs remain AI-facing scientific adapters: their workflow ports,
raw/reference construction and specialized accepted scientific contracts are still
required. They are not user-facing function choices in Native Functions. The
repeated Python callable map is removed; both routes use installed callable
resolution. Signature/default introspection is shared. Historical publication
profiles remain readable with their exact meanings.

Research method constraints and Stage 2 result-scoped typed parameter bindings
remain in their existing owners. Direct-call-specific conversational examples and
prompt directions are retired. AI Research does not depend on native form state,
and the form does not pass through research interpretation or planning.

The implementation and bounded verification record is the
[dual-mode report](../../../agent-data/acceptance/dual-mode-2026-11-10/REPORT.md).
Scientific methods and earlier conditional acceptance conclusions are unchanged.
