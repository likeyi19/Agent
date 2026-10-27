# Raw scATAC processing

Agent reuses its registered FASTQ/BAM, fragments, QC, cell-selection and matrix
owners. Upload files through **Attach files**, describe the library and requested
analysis, and answer any missing scientific prerequisites in the same conversation.
Uploading registers exact bytes; it does not certify assay or processing history.
The [SWC-1 evidence](../../../agent-data/acceptance/scientific-workflow-completion/swc-1/REPORT.md)
separates native software checks from real Firefox/OpenRouter qualification.
The genuine full FASTQ journey produced a 1,120 × 1,355,445 cCRE H5AD and a later
PCA/neighbors/Leiden/UMAP result, with actual downloads and exact count/axis checks.
One-turn combined execution is verified with simulated inputs and scripted language;
the [SWC-1B evidence](../../../agent-data/acceptance/scientific-workflow-completion/swc-1b/REPORT.md)
now records genuine external 10x PBMC500 BAM processing through Firefox/OpenRouter
to a 569 × 1,355,445 cCRE H5AD, independently checked and downloaded unchanged.
This qualifies the recorded paired corrected-CB BAM scope, not arbitrary BAMs.

## Input and reference scope

The FASTQ path supports explicitly declared human/mouse 10x ATAC libraries with
R1/R3 genomic reads and R2 barcodes, or R1/R2 genomic reads and I2 barcodes.
I1 is an optional sample index. State which lanes/chunks belong to one processing
library, and that cellular barcodes are raw sequences requiring whitelist correction.
Names can identify an explicitly declared role; they do not establish chemistry.
Multiple lanes of one library are not multiple-sample integration.

The existing operator `ScientificInputSet` provisions a reusable FASTQ companion:
`fastq_companion=true`, an owner-inspected `fastq_whitelist`, exact reference bundle
and QC reference pins, and the same parent reference under the matrix owner's
`reference_manifest_path`/`reference_manifest_sha256` inputs. Do not include a
source-specific library context. The application builds that context from the
researcher's uploaded sources and declarations. An explicit compatible resource
choice wins; otherwise the existing unique/default selection rules apply.

The shared ScientificResourceLibrary exposes qualified QC parent references to
BAM/fragments consumers. It does not automatically associate an arbitrary barcode
whitelist with a genome. FASTQ companion configuration is the existing supported
provisioning route, not a second resource catalog. The SWC-1 deployment recipe is
retained in its evidence helpers; original UAT configuration is not replaced.

BAM processing requires a single paired scATAC library, known source assembly,
already-corrected agreeing CB identifiers, unshifted alignments and retained
duplicate pairs. The owner checks the complete reference dictionary and actual
pair structure, applies its MAPQ/geometry policy and +4/-5 shift, and aggregates
identical fragment keys. Coordinate sorting or a BAI alone does not establish
cell identity or processing history. No new alignment is needed for a qualified
BAM. This is the Agent policy, not a claim of reproducing Cell Ranger output.

Current BAM publications use `agent-cb-paired-atac.v2` (or the existing neutral
reference entrance's `agent-cb-paired-atac.primary-neutral.v2`). Fragment endpoints
come from the inward-facing five-prime POS/CIGAR coordinates, including overlapping
and contained pairs. Reciprocal TLEN is checked but its magnitude is not required
to equal the outer aligned span: standard aligners use different TLEN endpoint
conventions. Three-prime overhangs do not change the chosen fragment endpoints.
The old v1 profiles retain their exact bytes and strict verification semantics
for historical publications; new v1 execution is explicitly read-only.

## Large-file deployment

The existing uploader streams bounded chunks into controlled staging, hashes the
whole encoded source, fsyncs and atomically publishes without overwriting an
existing file. An interrupted transfer is not a selectable source. Explicit retry
retransmits the file; resumable byte-range upload is not implemented. Scientific
role bindings share the uploaded bytes rather than copying each FASTQ again.

The default remains 256 MiB per file. Operators can set `upload_max_bytes` up to
8 GiB in the existing Web configuration. SWC-1 uses 2 GiB in an isolated deployment
and actually uploads eight full PBMC FASTQs, including six above 256 MiB. Check
available disk for uploads, native intermediate files and final results before
admitting long runs. This test does not qualify 8 GiB transfers or arbitrary
sample sizes. Existing concurrency and file-count bounds remain enabled.

## Processing and results

The planner selects operations and explicit dependency bindings. Raw-task
admission also accepts requested PCA, neighbors, Leiden and UMAP parameters;
it does not add downstream operations. Qualified matrix producers expose their
H5AD path directly to PCA through Registry metadata, including accepted outputs
from an earlier turn. The same published dataset port can feed `inspect_scATAC`
for a small follow-up without preprocessing or embedding. Branch-specific
reference/query interfaces remain distinct.
No scientific algorithm or accepted result identity is changed by this handoff.

QC and cell selection retain their owner semantics. Omitted selection settings
use the existing starting defaults (1,000 QC fragment records and TSS enrichment
4); an explicit value takes precedence and is recorded. These are thresholds,
not automatic statistical cell calling or universal biological optima. Required
TSS resources cannot be replaced by invented measurements. Empty selections are
reported; thresholds are not lowered to force cells through.

The cCRE H5AD contains sparse integer **fragment-record overlap counts** in X,
ordered namespace/barcode cells, the full ordered feature axis, genomic intervals,
species/assembly and measurement/profile identities. Each overlapping canonical
fragment record contributes once per feature; its read-pair support is not a
count weight. Different overlapping features may each receive that record.
QC/selection TSVs, the report and accepted manifest preserve the related QC,
threshold and provenance evidence. The original H5AD is delivered unchanged;
not every provenance field is embedded in its `uns`.

Request downstream analysis in the same turn or a later turn. Explicit TF-IDF/L2
before centered PCA uses the [existing PCA method](pca-analysis.md); original X
counts are preserved. This is not Signac LSI. Do not relabel cCRE fragment counts
as cut-site counts to satisfy an annotation method.

## Progress and recovery

The browser reports the current registered scientific step. A long fragment
step includes source qualification, alignment, packaging and verification; it
has no fabricated percentage or ETA. Refresh reads the same Session and Run.
Private role aliases are not additional uploads and do not appear in upload
choices. Accepted science and downloads remain separate from optional prose.

Existing recovery can finish saved presentation or close an interrupted response.
It does not restart alignment or resume an arbitrary native computation mid-step.
Use accepted compatible outputs for a later task; do not resubmit an expensive
preprocessing request merely to obtain another explanation.

For the isolated SWC-1 deployment, activate the `agent` environment and run:

```bash
python -m agent.web \
  --config /home/likeyi/program/agent-data/config/operator-swc1.json \
  --host 127.0.0.1 --port 8022
```

The private configuration retains existing model profiles and uses the dedicated
SWC-1 workspace. It is not committed to Git. The original UAT configuration and
its workspace remain separate.

SWC-1B uses a separate workspace and a 4 GiB upload limit for the complete
2,654,966,758-byte public BAM. Its official GRCh38-2020-A reference preserves
all 194 original BAM contig names/lengths; the earlier UCSC scaffold dictionary
is not silently substituted. Existing cCRE and GENCODE source bytes are reused.
The exact resource pins and acquisition limitations are recorded in the report.
To reopen this isolated deployment from the repository in the `agent` environment:

```bash
AGENT_QC_RESOURCE_CATALOG=/home/likeyi/program/agent-data/config/swc1b-qc-catalog.json \
  python -m agent.web \
  --config /home/likeyi/program/agent-data/config/operator-swc1b.json \
  --host 127.0.0.1 --port 8024
```

Choose downloads sequentially; the existing busy guard can reject overlapping
download preparation. A link marked “Requested” does not establish completed
browser transfer. Ask explicitly to run a fresh file inspection when that is
intended; a question about dimensions may instead be answered from accepted facts.
