# Chromap support-preserving qualification (M11.2b)

This is a repository-controlled representation correction, not an aligner fork
or a registered Agent tool. The only qualified mapping route is paired-end,
cell-level barcode BED output under `chromap-atac-agent-support-v1` in
`src/agent/tools/data/_chromap.py`. Stock Chromap is not accepted for this route.

## Exact source and build

Upstream: https://github.com/haowenz/chromap, tag `v0.3.2`, commit
`5bd17e1f1c50805e76904efd021603cb6a1b6e23`, reported version `0.3.2-r518`.
The version string is unchanged by the patch. `qualification-build.json` records
both separate executable hashes, source archive and patch hashes, GCC/Make,
compile/link flags, zlib version, shared-library hashes and platform. It is an
acceptance record for this build, not an instruction to trust any executable
reporting that version. Agent checks the accepted candidate executable hash and
linked runtime. Another build needs explicit qualification before allowlisting.

The source archive used here is the exact uncompressed output of:

```sh
git -C /path/to/upstream archive --format=tar 5bd17e1f1c50805e76904efd021603cb6a1b6e23 > upstream.tar
```

Its SHA-256 is `0234cc870ef10367cde5b1510555ba4c06498461b69a3960b9697507ab4aa0b7`.
This is a Git archive hash, not the GitHub release tar.gz hash.

Build stock and candidate in separate private trees outside Agent. Verify the
stock archive with `_chromap.verify_patch_source` before extraction. Prepare the
candidate with `_chromap.prepare_support_source(source_tar, patch, output_dir)`,
which hashes the consumed bytes, extracts a fresh tree, and invokes GNU
`patch --batch --fuzz=0 -p1`. It rejects foreign source/patch identities before
application. Build each separate tree with `make -j4`. Record outputs independently; never overwrite
one binary with the other. The patch uses zero context to avoid preserving
upstream trailing whitespace in the repository; exact source and patch hash
checks are mandatory, not optional fuzzy applicability checks. No network,
build, installation, or environment mutation happens during ordinary pytest.

The upstream MIT license is retained in `LICENSE`. No executable, full upstream
source tree, native index, or biological input is stored in this repository.

## Audited source correction

- `src/bed_mapping.h`: only `PairedEndMappingWithBarcode.num_dups_` and its
  constructor input change from uint8 to uint64.
- `src/duplicate_support.h`: checked uint64 increment; an exact assignment
  overload for that record type; the original 255 cap for other record types;
  checked conversion at the optional upstream signed-int summary boundary.
- `src/mapping_processor.h`: uint64 duplicate-group accumulator and exact setter
  in the in-memory path, including the final group.
- `src/mapping_writer.h`: the same changes in the low-memory merge path,
  including the final group, plus refusal of narrowing into summary fields.

Mapping generation initializes support to one. Copies and moves use the widened
member. Native BED temporary records are written/read with `sizeof(MappingRecord)`;
the probe verifies that the full width survives this path. Decimal output already
uses `std::to_string` and needs no change. A uint64 increment at its maximum exits
nonzero with `AGENT_DUPLICATE_SUPPORT_OVERFLOW`. Optional summary narrowing above
INT_MAX exits with `AGENT_DUPLICATE_SUMMARY_OVERFLOW`.

No duplicate comparison, sort comparator, representative-selection branch,
mapping/candidate/MAPQ calculation, barcode correction/ranking, trimming, or Tn5
code changes. Widening changes native temporary-record size and therefore spill
capacity, but native temporary files are private to one process and are never
reused between stock/candidate builds or resumed. Single-end, bulk, SAM, PAF,
pairs, and arbitrary combinations of flags are not newly qualified by this patch.

## Exact scientific meaning and limitations

Support means the total number of Chromap-generated paired mappings assigned to
the accepted corrected-barcode and contig/start/length duplicate group, including
the representative. Strand is absent from that key. Filtering uses the highest
MAPQ representative, so support can include mappings below the individual MAPQ
threshold. It is not a count of distinct molecules or necessarily of individually
MAPQ-passing read pairs. Corrected barcodes participate in deduplication.

The frozen adapter uses whole forward barcode reads, an explicit verified
whitelist of 1..32 bases, one canonical group-ID ordering for all three role
lists, one invocation per processing library, and one thread. Four-thread tiny
probes matched, but arbitrary multithread execution is not qualified. The adapter
accepts trusted M10-derived group facts; it does not implement production context
resolution, full-stream pre-scan, immutable raw-source execution, or fragment
publication. Those remain M11.2c scope. M11.1b still accepts whitelist resources
of 1..256 bases; backend compatibility is a separate gate.

Optional upstream summaries are disabled in the qualified policy. v0.3.2 has a
known non-whitelist accounting defect and signed-int summary accumulators; the
small narrowing guard does not claim to fix cumulative summary overflow. No
summary-derived FRiP/FRIC or cell-calling fact is introduced.

Stock and candidate reject the tested contig-edge pairs: the existing
`DraftMappingGenerator::IsValidCandidate` requires error-threshold flanks. This
is retained behavior, not coordinate clipping. Interior 30-base, overlapping,
ordinary, and swapped-mate fixtures retain the expected Tn5-adjusted coordinates.
Successful backend execution does not establish read-name synchronization.

## Tiny derived index

`chromap_reference_index.py` defines `agent.chromap-reference-index`, schema 1,
contract `chromap-reference-index.v1`. It binds the exact reference manifest,
portable reference identity, FASTA/FAI/ordered contig identities, accepted backend
and build record, explicit k=17/w=7, actual native index hash and byte size.
The bundle identity transitively binds cCRE identity, but cCRE bytes are not an
index input and the index helper does not read them. M11.1a is unchanged.

The index-domain gate streams FASTA and FAI, requiring exact ordered names and
lengths, and matching offsets/line geometry. No chromosome aliases or sequence
rewrites are allowed. Index prepare/reuse is separate from mapping. A cooperating
writer lease, private sibling staging, file/directory fsync and atomic directory
rename protect publication; existing matching artifacts are verified/reused and
conflicts fail without replacement. The lock file persists to avoid inode races.
Manifest loading is lightweight; explicit verification reopens reference/index
sources. An actual index hash is never derived from input identities. Tiny
rebuild equality does not promise deterministic full-genome index bytes.

## Qualification replay

Pure tests:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src conda run -n agent python -m pytest -q tests/chromap/test_contracts.py
```

Guarded tests require `RUN_CHROMAP_QUALIFICATION=1` and an explicit
`AGENT_CHROMAP_QUALIFICATION_RECORD` JSON path. That private build record has
`upstream_commit`, `patch_sha256`, and `stock`/`candidate` objects with absolute
`path` and exact `sha256` fields. Both executables must have passed the isolated
build audit. The wide representation probe also requires the candidate's audited
`src/` and `objs/` beside its executable and the recorded compiler. Only this
explicitly gated probe compiles a tiny local C++ test; it never downloads code.
The source-application replay also requires the exact `upstream.tar` beside the
candidate build directory (one directory above the candidate executable).

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src RUN_CHROMAP_QUALIFICATION=1 \
AGENT_CHROMAP_QUALIFICATION_RECORD=/private/qualification/build-identity.json \
conda run -n agent python -m pytest -q tests/chromap/test_backend_qualification.py -s
```

Generated FASTQs, raw BED output, and logs remain in private pytest/qualification
storage. `QUALIFICATION.md` records observed outcomes and the bounded acceptance.
