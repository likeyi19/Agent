# EpiAgent backend snapshot

This code-only snapshot is EpiAgent 0.0.3 at upstream commit
`9c4b7902537b14871bdb13d2fad43bff9974310a` from
https://github.com/xy-chen16/EpiAgent. The independent research checkout was clean.
`SOURCE.json` records the exact upstream Git blobs and source/vendored SHA-256
identities. All eight scientific Python modules are byte-for-byte upstream source.
The original MIT license, README and build/package metadata are retained.

Only `setup.py` dependency declarations differ: `faiss-cpu` supplies the existing
`faiss` import, and `torch-geometric>=2.6.1` declares the existing model import.
No model, preprocessing, tokenization, training, inference or utility algorithm
changed. PyTorch, FlashAttention, Transformers and the numerical environment need
compatible explicit installation; installing this snapshot does not qualify a new
Agent scientific capability. Use the Agent requirements and deployment guide to
install both managed backends without upgrading an already qualified stack.

Notebooks, examples, research scripts, Git internals, data, model weights and
generated results are excluded. Optional EpiAgent-NT weights, its exact feature
frequency resource and class vocabulary are outside Git under
`agent-data/resources/`; their descriptors retain exact source identities.
The full cCRE BED is explicitly byte-identical to the existing canonical human
hg38 cCRE reference; the EpiAgent frequency array is distinct from EpiZoo's and
must not be substituted. No resource is searched through the old research tree.

This backend is installed for future scientific tools. The current Agent Registry
provides no EpiAgent tool or automatic model substitution. Native users supply the
checkpoint and resources explicitly to the unchanged upstream APIs. No checkpoint
is needed merely to import Agent, inspect H5AD or run generic PCA.
