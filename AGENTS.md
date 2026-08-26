# Agent

Agent is an AI system for autonomous single-cell epigenomic analysis.

## Completed milestone

### Milestone 1 — EpiZoo cell embedding backend

The first validated vertical slice is complete:

raw scATAC-seq AnnData
→ validated sparse preprocessing
→ EpiZoo
→ reproducible cell embeddings

Validated on Fang2021:
- 2,000 mouse scATAC-seq cells
- output shape: `(2000, 512)`
- exact scientific parity with the manual EpiZoo pipeline
- deterministic inference with fixed truncation seed
- RTX 4090 peak GPU memory: ~10.9 GiB
- validated default batch size: 4

The validated implementation is:

`src/agent/tools/models/epizoo.py`

Do not modify this backend unless required to fix a verified bug or to support a clearly defined new capability.

## Current milestone

### Milestone 2 — Standard scientific tool layer

The current goal is to expose validated scientific capabilities through clean, structured Agent tools.

Initial tools:

1. `inspect_scATAC`
2. `epizoo_embed_cells`

The goal of this milestone is:

user/file input
→ structured scientific tool
→ validated backend
→ structured lightweight result

Do not connect the LLM/planner yet.

Do not implement annotation, clustering, UMAP, RAG, literature retrieval, reports, or multi-agent orchestration in this milestone.

## Development environment

- Linux server
- NVIDIA RTX 4090
- 24 GB VRAM
- VS Code Remote SSH
- Python
- PyTorch
- Scanpy / AnnData

## Scientific backends

EpiAgent and EpiZoo are existing scientific foundation models.

They should be treated as scientific backends and reusable tools,
rather than being reimplemented as part of the agent.

Existing validated scientific logic should be reused whenever possible.

## Engineering rules

- Never densify a complete scATAC-seq matrix.
- Never assume more than 24 GB GPU memory.
- Always test on small datasets before full-scale execution.
- Do not modify validated EpiZoo scientific logic unless necessary.
- Every scientific capability should be exposed through a clean reusable tool.
- Every tool should have explicit inputs and outputs.
- Every tool should validate its inputs and provide informative errors.
- Keep analyses reproducible.
- Record model checkpoints and execution parameters.
- Do not commit biological datasets or model checkpoints to Git.

## Current development task

Build the first standard Agent tool layer around validated scientific backends.

Primary targets:
- inspect a scATAC-seq `.h5ad` file safely
- expose EpiZoo embedding through a file/path-based tool interface
- return structured results suitable for later LLM tool calling

## Do not implement yet

- multi-agent architecture
- literature retrieval
- ENCODE retrieval
- RAG
- cCRE perturbation
- variant interpretation
- web UI
- automatic scientific reports