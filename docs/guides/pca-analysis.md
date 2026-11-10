# Historical PCA representations and explicit native PCA

Ordinary compatible scATAC analysis now uses [native feature selection and
Spectral](native-scatac-analysis.md). The dedicated `pca_embed_cells` tool and
its optional TF-IDF/L2 execution path are retired.

Accepted historical results remain readable with their exact meanings:

- v1: centered Scanpy PCA of supplied X unchanged.
- v2: scikit-learn smoothed document-frequency TF-IDF, L2 cell normalization,
  then centered Scanpy PCA. This was exploratory PCA, not Signac LSI.

Historical descriptors, method parameters, reports, downloads and accepted source
lineage are retained. Read-only reuse does not refit PCA, reopen the source matrix,
or reinterpret old evidence under a current method. Unaccepted old publications
cannot acquire new scientific authority.

For explicit new PCA requests use the existing native `run_scanpy` / `pp.pca`
or `run_episcanpy` / `tl.pca` interface. Researcher parameters pass through its
native kwargs; omitted values retain installed defaults. EpiZoo remains a
separate model-based capability with its own reference/vocabulary requirements.
