# Agent

Agent is an AI system for autonomous single-cell epigenomic analysis.

## Current milestone

The first milestone is:

scATAC-seq data
→ standardized input
→ EpiZoo
→ cell embedding

Do not build the complete autonomous agent yet.

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

Wrap the existing EpiZoo cell-embedding inference pipeline
as the first reusable model tool.

Initial capabilities:

- load_model()
- embed_cells()

Target:

src/agent/tools/models/epizoo.py

## Do not implement yet

- multi-agent architecture
- literature retrieval
- ENCODE retrieval
- RAG
- cCRE perturbation
- variant interpretation
- web UI
- automatic scientific reports