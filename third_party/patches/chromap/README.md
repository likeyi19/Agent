# Historical Chromap support-preserving qualification

The patch, [qualification record](QUALIFICATION.md), license and
`qualification-build.json` preserve the identity of a previously accepted native
build. Current `run_chromap` records the actual installed binary; it has no
host-specific build allowlist. The configured patched binary remains available.
See the [current raw guide](../../../docs/guides/raw-scatac-processing.md).

Upstream is MIT-licensed Chromap, tag `v0.3.2`, commit
`5bd17e1f1c50805e76904efd021603cb6a1b6e23`, version `0.3.2-r518`.
The stored patch widens barcode BED duplicate support from uint8 to uint64, including
native temporary/spill records and checked summary conversion. It does not change
alignment, barcode correction, duplicate grouping, representative choice or Tn5
mathematics. The executable version string alone does not distinguish the builds.

The bounded replacement comparison uses 300 simulated paired reads across two
lanes, including one corrected barcode mismatch. Stock and patched builds produce
identical tested interval/barcode keys; their support values are 255 and 300.
Support means mappings within the native duplicate group, including the
representative. It is not distinct molecules, independently valid cells or
necessarily individually MAPQ-passing read pairs. SnapATAC2 region matrices do not
weight this column, while duplicate diagnostics remain sensitive to its precision.
These observations do not qualify arbitrary flags, inputs, scales or platforms.

To reproduce independently, retrieve the pinned upstream revision, review/apply
`support-v1.patch` and run `make -j4` in a private source tree. Preserve independent
binary hashes. No build, patch application or package installation runs during an
Agent scientific call. The former qualification/execution helpers are removed;
old build and acceptance records remain scientific history.
