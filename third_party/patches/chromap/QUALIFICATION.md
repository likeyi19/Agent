# M11.2b qualification record

Starting Agent HEAD: `78941712bb6011f7140176cea8939fe4b7b30f46`, clean `main`,
matching local and live remote `origin/main`. No commit/push is part of this slice.
Source/build identities are in `qualification-build.json`; fixed execution
settings and accepted executable/runtime identity are in `_chromap.py`.

| True duplicate-group support | Stock v0.3.2 | Corrected candidate |
| ---: | ---: | ---: |
| 1 | 1 | 1 |
| 2 | 2 | 2 |
| 254 | 254 | 254 |
| 255 | 255 | 255 |
| 256 | 255 | 256 |
| 300 | 255 | 300 |
| 65,536 | 255 | 65,536 |

The stock cap was reproduced before the correction was designed/applied. Both
executables remained separate throughout qualification. For unsaturated records,
coordinates, membership, barcode, support, and relevant raw bytes matched.

A separate C++ representation probe preserved 255, 256, 65,536, 4,294,967,303 and
18,446,744,073,709,551,615 through constructor/copy, native spill round-trip, and
decimal writing. Those last two values are representation tests, not claims to
have aligned billions of input pairs. UINT64_MAX increment and INT_MAX summary
narrowing failed closed. Independent 300-mapping probes exercised in-memory
and two-spill merge deduplication, retaining the high-MAPQ representative while
including its low-MAPQ duplicate support. End-to-end in-memory FASTQ probes also
preserved exact support and stock parity below saturation.

Barcode tests covered exact matching, a single one-mismatch candidate with zero
sampled abundance, multiple candidates with a 20:1 prior, equal-prior ambiguity,
one N, excess Ns, uncorrectable sequences, and adapter rejection without a
whitelist. Both stock and corrected executables handled 31/32-base sequences
correctly and rejected 33. All correction outcomes matched below saturation.

| Barcode case | Observed outcome in both builds |
| --- | --- |
| Exact whitelist token | Retained unchanged |
| Single one-mismatch candidate, zero sampled abundance | Corrected to the sole candidate |
| Multiple candidates, 20:1 sampled abundance | Corrected to the dominant candidate |
| Equal-abundance ambiguous candidates | No output mapping |
| One N, resolvable candidates | Corrected to the dominant candidate |
| More than one N | No output mapping |
| No eligible whitelist candidate | No output mapping |

The raw stock reproduction remains in the isolated qualification directory
`/tmp/agent-m112b-qualification/saturation/stock.bed`, alongside candidate output,
synthetic sources, and invocation logs. The automated guarded tests regenerate
their own inputs and retain outputs in pytest temporary storage.

Both reviewed FASTQ layouts produced identical raw fragments without read-format
or barcode translation flags. I1 was excluded. Same endpoint/barcode across two
lanes produced exact support 300 in the candidate; another barcode stayed
separate. Reversing both lane lists in the tiny probe gave identical logical
output; the fixed policy still uses canonical group-ID order because abundance
sampling is sequential and capped. Independent namespaces were executed
separately. Pure tests also exercised actual M10 intake and M11.1b library
binding on generated files.

The ordinary [100,200) fixture yielded [104,195). Interior 30-base, overlapping,
and swapped-mate examples retained expected shifted intervals. Both builds
rejected tested pairs at the extreme contig boundaries due to the existing
candidate-flank rule; neither clipped them into a different fragment.

Unequal barcode/genomic record counts and truncated barcode FASTQ failed in
both builds. Same-count mismatched names succeeded in both. A production
full-stream Agent synchronization check is still required.

Repeated raw streams matched for both layouts/builds at threads 1 (three runs)
and 4 (one probe). The accepted policy remains threads=1. These are synthetic
repeatability results, not proof for arbitrary data/thread profiles.

Two candidate indexes of the same synthetic 12,000-base reference both had size
66,592 bytes and SHA-256
`f5d2e76789ebfb6d4d5c30ee2393c9514db0a407fc1e15487052dd6c2c0a8b99`.
Mapping output using each was identical. Native index serialization can include
unused hash-table storage; no general byte-determinism guarantee follows.

Only synthetic resources were used. No hg38/mm10 indexes, biological FASTQ,
public preprocessing tool, Planner registration, full fragments manifest, BGZF
publication, or canonical production fragments were implemented.

## Final acceptance

All runs used the existing `agent` environment with `PYTHONDONTWRITEBYTECODE=1`
and `PYTHONPATH=src`. Full regression also set the existing temporary Numba and
Matplotlib cache directories. GPU/model acceptance gates were not enabled.

| Suite | Result |
| --- | --- |
| M11.2b pure contracts | 55 passed |
| Explicitly guarded stock/candidate qualification | 14 passed |
| M11.1 reference and library context regression | 278 passed |
| M10/data and pseudobulk regression, excluding M11.1 suites | 709 passed |
| Full lightweight regression | 2,505 passed, 68 skipped, 7 warnings |
| `git diff --check` and separate whitespace checks for every untracked addition | Passed |

The full-suite skips include all 14 guarded backend cases; normal regression
does not provision or execute Chromap. The isolated log directory is
`/tmp/agent-m112b-qualification/` (`pure-final.log`, `backend-final.log`,
`m11-regression.log`, `m10-regression.log`, and `full-regression.log`).
All 11 repository additions remain untracked for review; existing tracked files,
HEAD, and `origin/main` are unchanged. No commit or push was performed.

Disposition: `PATCHED_CHROMAP_ACCEPTED_FOR_M11_2C`.

The small audited correction preserves exact support without changing mapping,
correction, duplicate keys, representative selection, or coordinate algorithms.
Every requested qualification gate passed under the fixed paired-end cell-level
BED policy. Acceptance applies to the pinned candidate executable and runtime,
one thread, verified explicit whitelist (1..32 bases), canonical group ordering,
and separate processing-library executions. It does not accept stock saturation,
arbitrary builds/flags, optional upstream summaries, or claim full-data validation.
The observed synchronization and contig-edge limitations remain explicit.

Smallest recommended M11.2c: bind freshly verified M10 intake, selected M11.1b
library context, and a separately verified derived index; implement bounded-memory
full-stream FASTQ structure/count/name synchronization; define the fragments
manifest with this exact support/backend policy; and exercise private per-library
execution plus independently validated atomic fragment publication on synthetic
fixtures. Keep Planner registration, full human/mouse index builds, biological
acceptance, cell calling/QC, and cCRE matrices in separately authorized slices.
None of that M11.2c implementation is included here.
