# Documentation

Start with the [operating and development guide](guides/operating-and-development.md),
the repository [README](../README.md), and the detailed [engineering contract](../AGENTS.md).
The [scientific storage guide](guides/scientific-storage.md) covers the current
consolidated deployment and fresh UAT environment.
The [Alibaba Qwen provider guide](guides/alibaba-qwen-provider.md) covers ordinary
Web selection and secure persistent credentials for the existing local launcher.
The [CWC-1 readiness report](../../agent-data/acceptance/cross-capability-workflow-completeness/cwc-1/REPORT.md)
assesses every current scientific capability, records new statistical/evaluation
acceptance and shared corrections. Its [blocker closeout](../../agent-data/acceptance/cross-capability-workflow-completeness/cwc-1/blocker-closeout/REPORT.md)
resolves repeated source-scoped operations and fixed annotation evaluation, with a
CONDITIONAL GO recommendation for the documented controlled research scopes.
The [backend installation guide](guides/backends.md) covers the managed EpiZoo and
EpiAgent source snapshots, optional dependencies and exact import origins.
The [PCA representation guide](guides/pca-analysis.md) explains explicit count
preprocessing, original-measurement preservation and interpretation limits.
The [raw scATAC guide](guides/raw-scatac-processing.md) covers FASTQ/BAM prerequisites,
large-file deployment, matrix semantics, downstream handoff and the separately
recorded genuine FASTQ and BAM acceptance scopes.
The [cell-type annotation guide](guides/cell-type-annotation.md) covers native
MAESTRO region inputs, independent groups, optional CellMarker corroboration and
the distinction between candidate predictions and reference evidence. The
[SWC-2 finalization](../../agent-data/acceptance/scientific-workflow-completion/swc-2/FINALIZATION-REPORT.md)
records automatic grouping prerequisites and genuine unclustered-H5AD acceptance.

The [EpiZoo lifecycle guide](guides/epizoo-model-lifecycle.md) covers automatic
transfer assessment, initialized and post-trained checkpoints, native zero-shot
inference, resource acquisition and model reuse.

The [regulatory sequence guide](guides/regulatory-sequence-prediction.md) covers
native DHS64 FASTA/direct DNA/BED prediction, exact biosample conditions, accepted
sequence reuse, independent matched-observation evaluation and the isolated runtime.

The [sequence optimization guide](guides/regulatory-sequence-optimization.md) covers
independent native DeepPBMC editing, exact constraints, comparison outputs and
the academic research runtime/license scope.

The [sequence generation guide](guides/regulatory-sequence-generation.md) covers
native pretrained DNA-Diffusion, exact supported conditions, stochastic sampling,
sequence descriptors, reusable candidates and ordinary resource deployment.

The [motif accessibility guide](guides/motif-accessibility.md) covers native chromVAR, ordinary H5AD counts, versioned motifs, reference requirements and scientifically bounded interpretation.

The [scATAC regulatory sequence guide](guides/scatac-regulatory-sequences.md) covers
descriptive or accepted-DA region selection, exact genomic extraction, explicit
model windows and accepted-candidate continuation.

The documentation has three locations:

| Location | Purpose |
| --- | --- |
| [guides/](guides/) | Current setup, repository navigation and developer entry points |
| [architecture/](architecture/) | Current R4–R7 owner records and the scoped EpiZoo backend interface audit |
| [history/](history/) | Milestone implementation, qualification, development and acceptance records |

Current architecture is owned by these records, in conjunction with README and AGENTS:

| Record | Responsibility |
| --- | --- |
| [R4](architecture/ua3.6.5-r4-unified-scientific-resource-library.md) | ScientificResourceLibrary, installation, discovery and explicit sharing |
| [R5](architecture/ua3.6.5-r5-llm-agent-interface-scientific-interaction-redesign.md) | Registry planning, accepted facts, cited answers and semantic review |
| [R6](architecture/ua3.6.5-r6-agent-scientific-tools-integration-redesign.md) | Scientific inputs, native execution and versioned result verification |
| [R7](architecture/ua3.6.5-r7-agent-internal-state-lifecycle-simplification.md) | Session, Interaction, SessionTurn, RunStore and Revision lifecycle |

The [EpiZoo interface audit](architecture/epizoo_interface.md) records the original
backend investigation and proposed first-milestone interfaces. Its stated scope
and historical proposals remain intact; current contracts above take precedence.

Historical records remain useful evidence and may define scientific contracts
that current owners explicitly retain. Their location does not change their
acceptance outcome or compatibility status. Useful entry points include:

| Area | Records |
| --- | --- |
| Resources and raw preprocessing | [M11.6 operator specification](history/m11.6a-production-resource-specification.md), [M11.8 biological closeout](history/m11.8-biological-closeout.md) |
| Feature matrices and cross-species analysis | [M14.1 feature adoption](history/m14.1-regulatory-feature-adoption.md), [M14.8 primary nuclear scope](history/m14.8-real-macaque-fragment-matrix.md), [M14.11 real macaque QC](history/m14.11-real-macaque-qc.md) |
| Sessions and scientific dialogue | [M15.6 multi-turn closeout](history/m15.6-real-multiturn-closeout.md), [M16 closeout](history/m16-regression-closeout.md), [M17.2 guidance selection](history/m17.2-guidance-selection-handoff.md) |
| Web application | [M18.3 configuration](history/m18.3-basic-web-application.md), [M18.5 browser acceptance](history/m18.5-real-interactive-acceptance.md), [USI-7 Web closeout](history/usi-7-web-ux-reports-download-closeout.md) |
| Interaction and product qualification | [Interaction/model closeout](history/interaction-model-layer-closeout.md), [UA3.6.5 exploration](history/ua3.6.5-round1-real-world-product-exploration.md) |

FS-1 moved the original flat documents into these directories and updated active
Markdown links. Literal commands, recorded repository paths, absolute evidence
locations, hashes and file inventories describe their original checkpoints and
remain unchanged. Run current commands from the repository root unless stated
otherwise. Use the guide for current dependency paths; historical paths are not
an instruction to recreate an older repository layout.
