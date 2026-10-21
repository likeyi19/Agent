# Scientific backend installation

Agent manages code-only snapshots of both scientific backends under
[`third_party/epizoo/`](../../third_party/epizoo/) and
[`third_party/epiagent/`](../../third_party/epiagent/). Agent scientific owners call
these packages; their algorithms remain upstream code. The existing Registry has
24 tools. EpiZoo supports current accepted capabilities. EpiAgent is an installed
backend for future tools, with no current public Agent execution tool or automatic
substitution for EpiZoo or PCA.

Each snapshot's `SOURCE.json` records its upstream revision, original and included
file hashes, exclusions and packaging/test amendments. EpiZoo is pinned to
`029cd631d0f9806a646c4a3b42ce10b958f2b67f`; EpiAgent is pinned to
`9c4b7902537b14871bdb13d2fad43bff9974310a`. Scientific Python modules are unchanged.
Original MIT licenses and attribution remain; EpiZoo's SEAM configuration also
retains its Apache-2.0 notice and DNABERT-2/MosaicBERT attribution. See
[EpiZoo NOTICE](../../third_party/epizoo/NOTICE) and
[EpiAgent provenance](../../third_party/epiagent/UPSTREAM.md).

## Install in the qualified environment

Use the existing Linux x86_64 / Python 3.11 numerical and CUDA environment. The
[constraints](../../requirements/backend-constraints.txt) record the tested stack,
including PyTorch 2.7.1+cu128 and FlashAttention 2.8.3. They constrain installation;
they do not provision CUDA, drivers, R or unrelated scientific tools. Provision
those dependencies explicitly before installing the model backends on a new host.

From the Agent repository root:

```bash
conda activate agent
python -m pip install -c requirements/backend-constraints.txt \
  -r requirements/backend-runtime.txt
python -m pip install --no-deps --no-build-isolation -r requirements/backends.txt
python -m pip check
python -I -B -c 'import epizoo, epiagent; print(epizoo.__file__); print(epiagent.__file__)'
```

Both origins must be inside this repository's `third_party/` tree. Editable
installation is deliberate: changing the repository location requires reinstalling
these two packages. The original independent research checkouts are unnecessary
for Agent deployment and must not be added to `PYTHONPATH`. `PYTHONPATH=src`
continues to expose Agent itself for its existing module entry points.

The optional [backend runtime group](../../requirements/backend-runtime.txt) pins
PyTorch Geometric, CPU Faiss, POT and their previously missing HTTP dependencies.
It is not a prerequisite for unrelated H5AD/Web/PCA use. EpiAgent's package metadata
now declares its existing graph import and the installable `faiss-cpu` distribution.
EpiZoo's NumPy/AnnData declarations match the already qualified 2.4.6/0.12.19
environment; its upstream 2.1.2/0.12.10 declarations remain recorded in the source
manifest. FS-3C changed no previously installed dependency version to resolve this
metadata mismatch. Future numerical-stack changes require appropriate qualification.

## Scientific resources and explicit use

Weights, references and datasets remain outside Git under the configured
`agent-data/` layout. The [scientific storage guide](scientific-storage.md) owns
current deployment paths and existing configuration. EpiZoo consumes the exact
configured checkpoint and species vocabulary/frequency files; SEAM's code and
tokenizer configuration are in `third_party/epizoo/config/SEAM_config/`, while its
checkpoint stays under `resources/models/epizoo/`.

The existing adaptation owner validates the vendored source manifest before
recording the EpiZoo upstream revision. It does not mistake Agent's Git HEAD for
the backend revision. New source bundles bind their actual new paths and code;
historical accepted bundles and outcomes are never rewritten or silently rebound.
The opt-in real-weight qualification helper can use `--epizoo third_party/epizoo`.
It is not invoked by installation and its native training is outside FS-3C.

Optional EpiAgent resources are described by
`resources/models/epiagent/resources.json`: the exact NT checkpoint and class
vocabulary, its separate document-frequency array, and the full ordered cCRE BED.
The BED is shared physically only because complete source-byte equality was
verified. EpiAgent's frequency array differs from EpiZoo's; equal dimensions do not
make them interchangeable. No new ScientificResourceLibrary entry was created.

Native EpiAgent callers explicitly instantiate its original model and supply the
checkpoint to PyTorch's `load_state_dict`; there is no Agent-side replacement
loader or preprocessing implementation. Importing Agent or either package never
requires loading a checkpoint. Do not infer an accepted scientific capability from
the installation or a successful small native forward.

## Updating the snapshots

Update intentionally from an identified upstream source, retain licenses, compare
all scientific module/configuration bytes, and record any packaging amendments.
Run affected scientific/owner tests and native compatibility checks with explicit
resources. Keep generated build metadata, caches, wheels and test output out of
the source snapshot. Use the existing disposable-output conventions in the
[development guide](operating-and-development.md).
