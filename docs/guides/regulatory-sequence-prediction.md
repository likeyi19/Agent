# Regulatory sequence prediction

RSD-1 adds `predict_regulatory_sequences` and independent
`evaluate_regulatory_predictions` to the existing scientific Registry (29 tools).
The qualified backend is human DHS64 from native enhancer-design. Prediction
does not require observed measurements. DHS733 and DHS64-MPRA are outside this
capability. Independent [scATAC region preparation](scatac-regulatory-sequences.md)
can supply exact compatible genomic DNA; observed ATAC is not DNase ground truth.
[Generation](regulatory-sequence-generation.md) remains a separate operation.
[Constrained optimization](regulatory-sequence-optimization.md) is an independent
DeepPBMC tool with its own context and objective requirements; it does not consume
DHS64 scores.

## Researcher workflow

Attach ordinary FASTA and request, for example, “Predict DNase accessibility for
these sequences in K562 and provide the scores and sequences.” Wrapped sequences,
lowercase bases and repeated headers are supported. Alternatively write one DNA
sequence directly in the message: the typed parameter owner binds the exact
quoted string, without an uploaded file. No assembly is needed for sequence bytes.

Omitting a condition predicts all 64 ordered biosamples. An explicit condition
must exactly match the installed model metadata; similar cell names are not a
scientific match. Unavailable conditions are clarified without substitution.
The model-specific input policy is 1–500 A/C/G/T bases per record, normalized to
uppercase. Ambiguous IUPAC bases, including N, and overlength sequences are rejected
before inference. This prevents the native encoder's silent unknown-base zeros
and truncation. Native `padding='left'` places bases first and adds trailing zero
vectors to 500 positions. It does not supply genomic flanks. The execution bound
is 10,000 records and 32 MiB source bytes per request.

For BED, upload tab-separated BED3–6 and state its assembly. The owner resolves
the exact qualified human genome through ScientificResourceLibrary and extracts
zero-based half-open intervals using indexed pysam access. BED6 minus strand
produces reverse complement; absent or `.` strand uses the reference plus strand.
There is no liftOver, clipping, flank expansion or BED12 block concatenation.
Unknown assembly and invalid/out-of-bounds intervals cannot execute successfully.

The normal result view shows a bounded score preview and a heatmap of up to 30
sequences. Downloads include the full selected score table, FASTA, PNG and JSON
model/measurement metadata. Sequence IDs are ordinal (`sequence-1`, etc.) in source
order; repeated original headers remain distinct. Download FASTA prefixes these
IDs for unambiguous table correspondence. Private path-like labels are redacted
only at download time; exact private publications and delivered checksums remain
separate. Full native arrays for both heads/all conditions and every fold remain
in the accepted private publication. Metadata contains ordered labels, checkpoint
hashes, exact sequence digests, padding, runtime, source and reference identities.

Ask explicitly to retain sequences and predictions for later use, then request
“Reuse the accepted DNA sequences to predict their binary peak-call scores for
K562.” Exact sequence and prediction ports are independent semantic artifacts;
they are not cells, embeddings or annotation groups. Result selection exposes
these ports to the existing selection model. Later evidence answers, downloads,
historical previews and saved presentation recovery do not rerun prediction.

## Scientific meanings

The default is the arithmetic mean of the continuous head across released folds
0, 1 and 3. Qualification compared this with the native ensemble helper using
native Keras single-head projections. Each fold value and population standard
deviation are retained. The continuous task predicts log10 quantile-normalized
DNase signal with the training sample's minimum-nonzero offset; it is not a
probability and negative values are valid. No conversion back to an absolute
experimental signal is claimed.

`head=binary` selects the sigmoid DNase peak-call task. Its output is not a
probability of enhancer functionality. Neither head predicts arbitrary MPRA,
ATAC or expression measurements. Fold spread is model disagreement, not calibrated
uncertainty or measured error. Averaging folds does not independently validate
sequences optimized against the same folds. Biological functionality, training
overlap and new-domain calibration are not established by numerical execution.

## Optional compatible-observation evaluation

Select an accepted prediction and upload a TSV with `sequence_id`, `condition`,
`measurement`, `value`, `provenance`. IDs refer explicitly to that selected
prediction; labels must match exactly. Optional `sequence_sha256`, when supplied,
must also match. Source hashes and internal manifests are not user prerequisites.
Use measurement `log10_quantile_normalized_DNase_signal` for compatible continuous
observations on the actual native transform, or `DNase_peak_call` with 0/1 labels
for the binary task. Stating the measurement name alone does not establish that
an experiment used it: researchers remain responsible for a truthful measurement
definition and provenance. No assay conversion or approximate matching occurs.

Matched subsets are allowed; duplicate sequence/condition pairs are not. Continuous
metrics are RMSE, MAE, Pearson and Spearman; binary metrics are AUROC, average
precision, Brier and log loss. Undefined correlations and single-class discrimination
metrics are null. Evaluation does not repeat model inference or prove experimental
independence, lack of training overlap, biological accuracy or calibration. RSD-1
acceptance used explicitly simulated observations to test this interface; no
compatible experimental truth dataset was qualified.

## Operator provisioning

Install Agent's optional [sequence dependencies](../../requirements/sequence.txt)
in its existing environment. The isolated CPU environment must be a fresh venv
without system-site-packages, using Python 3.11 and the complete
[DHS64 lock](../../requirements/dhs64-cpu.lock):

```bash
python3.11 -m venv /path/to/scientific-runtimes/dhs64-cpu
/path/to/scientific-runtimes/dhs64-cpu/bin/python -m pip install -r requirements/dhs64-cpu.lock
/path/to/scientific-runtimes/dhs64-cpu/bin/python -m pip check
```

Use the existing ScientificResourceLibrary `install` API with kind
`sequence_predictor`. The qualified `dhs64-sequence-predictor.v1` resource describes
fold checkpoint pins, the XLSX ordered biosample metadata, native measurements,
context/padding/training domain, release/source/license and native source revision.
Its runtime identifies the venv Python and explicit device, batch size, thread
counts and seed. Installation checks metadata and file identities; native
scientific parity remains a separate provisioning gate. The retained RSD-1
`provision.py` and `installed-resource.json` document the actual deployment.

The production library has the new entry alongside its nine preserved earlier
entries. It reuses RSD-0 checkpoints and source at revision
`332f9682b7b10d3fd8a7973eb17848289ee1534f`; weights remain outside Git. Scientific
software provenance is recorded without per-execution source-code hash coupling.
The existing scoped library is shared by discovery and execution. Researchers
never activate a runtime or specify TensorFlow/device/checkpoint paths.

The owner invokes a fixed subprocess inside the ordinary execution lifecycle,
with cancellation and no device/model fallback. CPU TensorFlow 2.14, batch 2,
four intra-op/two inter-op threads and seed 20261029 were qualified. Explicit
batch 1 parity was separately tested; production settings were not changed.
GPU execution is unqualified. No GPU model or memory capacity is part of the
scientific contract. Main Agent NumPy/PyTorch and EpiZoo algorithms were preserved.

See the [RSD-1 acceptance report](../../../agent-data/acceptance/regulatory-sequence-design/rsd-1/REPORT.md)
for genuine native/Firefox/OpenRouter evidence, exact tolerances, failures and
bounded readiness. Earlier saved text and publications retain their meanings.
