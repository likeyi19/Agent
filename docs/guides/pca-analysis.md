# Explicit native PCA

AI Research can select `run_scanpy` / `pp.pca` or `run_episcanpy` / `tl.pca`.
Both are supported native APIs. Select methods by scientific purpose and researcher
intent, with no package priority. Parameters pass through native kwargs; omitted
values retain installed defaults. See [native scATAC analysis](native-scatac-analysis.md)
for graph prerequisites and representation handling.

The dedicated `pca_embed_cells` tool, its Agent TF-IDF/L2 execution path, historical
publication readers and tests are removed. Development Sessions and generated
results impose no compatibility requirement. Independent acceptance reports retain
their evidence of earlier testing.

Native Functions retains independent installed public-function discovery and uses
no provider. EpiZoo remains a separate model capability with its own reference and
vocabulary requirements; its compact embedding H5AD is not a count matrix.
