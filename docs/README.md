# Documentation

Start with the [operating and development guide](guides/operating-and-development.md),
the repository [README](../README.md), and the detailed [engineering contract](../AGENTS.md).
The [scientific storage guide](guides/scientific-storage.md) covers the current
consolidated deployment and fresh UAT environment.
The [backend installation guide](guides/backends.md) covers the managed EpiZoo and
EpiAgent source snapshots, optional dependencies and exact import origins.

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
