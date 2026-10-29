# De novo regulatory DNA generation

`generate_regulatory_sequences` samples **synthetic regulatory DNA candidates**
from the released pretrained DNA-Diffusion model. It is an independent Registry
operation (31 tools), alongside DHS64 prediction and DeepPBMC optimization.
It needs no uploaded DNA, scATAC dataset, genomic reference or researcher-supplied
checkpoint paths.

## Researcher workflow

> Generate 20 candidate regulatory DNA sequences conditioned on K562. Compare
> their basic sequence characteristics, show the composition figure, and let me
> download them. Keep the sequences and generation result for later work.

Supported conditions are exactly **GM12878, HepG2, K562 and hESCT0**. These names
bind the following released sample labels; there is no approximate tissue match:

| Condition | Checkpoint class index | Released training label |
| --- | --- | --- |
| GM12878 | 1 | GM12878_ENCLB441ZZZ |
| HepG2 | 2 | HepG2_ENCLB029COU |
| K562 | 3 | K562_ENCLB843GMH |
| hESCT0 | 4 | hESCT0_ENCLB449ZZZ |

Index 0 is native unconditional masking, not a fifth supported researcher
condition. A request for another tissue requires clarification without replacing
it with one of these four. The released UNet's ten embedding slots do not imply
ten qualified cell types.

Every returned sequence has exactly 200 A/C/G/T bases. The default count is 20;
the bounded execution range is 1–1000. Exact duplicates are retained and reported.
A new batch or changed supported condition is a new generation request with its
own accepted identity. The existing result selector and **Continue from this
result** retain exact prior sequence identity for later compatible work.

Downloads contain generated FASTA, a candidate-level TSV, model/sampling metadata
and a composition PNG. Full records remain accepted artifacts; language context
contains aggregate facts and at most 12 candidate examples. Results and downloads
remain available independently of successful generated prose. Reopening a saved
result, comparing its recorded descriptors or downloading does not repeat sampling.
Generated DNA has no established genomic coordinates and is not exported as BED
or an scATAC matrix.

## Native method and reproducibility

Agent calls the unmodified native `PretrainedUNet.from_pretrained` loader on the
local released safetensors/config, then native `Diffusion` and `create_sample`.
Missing, unexpected or mismatched checkpoint tensors prevent execution. Provisioning
separately checks every loaded tensor against the released checkpoint. The known
upstream local sampling loader that fails to apply safetensors weights is not used.
No diffusion mathematics or alternative search is implemented in Agent.

The qualified native linear beta schedule is **50 steps, 0.0001 to 0.2**. Default
`guidance_scale=1` is the native classifier-free guidance weight. The native noise
estimate is `(1 + weight) × conditional − weight × unconditional`; zero still
uses the conditional estimate. It is not an activity score or a probability.
Explicit finite nonnegative weights are supported, without a performance guarantee.

An explicit seed may be any integer from 0 through 4294967295. If omitted, the
owner chooses and records a fresh seed. Seeding occurs after pretrained loading,
immediately before sampling. Seed, condition/index, schedule, guidance, actual
batch calls, checkpoint hash, runtime versions, device and precision are recorded.
Seed alone does not guarantee identical sequences across batching, devices or
software versions.

Native `create_sample` drops a remainder through integer division. The adapter
calls it for full batches and then a correctly sized final batch, collecting all
outputs in order. It never oversamples, discards, duplicates or invents candidates
to meet the count. Native batch size is an execution setting, separate from the
scientific count and guidance. Failure does not select a different device,
checkpoint, precision or sampling algorithm. Cancellation terminates the child
process; recovery can verify an exact completed publication, not retry sampling.

## Sequence descriptors and scientific limits

Biopython computes GC fraction. SciPy computes base-composition Shannon entropy
in bits (0–2) and all unordered position-wise Hamming distances; the owner counts
identical strings and longest homopolymer runs. Distances count differing positions
out of 200, without alignment or reverse-complement equivalence. A singleton has
zero pairs and undefined distance summaries. Duplicate count means total minus
unique, while the candidate table reports each string's total multiplicity.

Entropy and homopolymers are simple low-complexity descriptors, not a repeat-family
annotation or a genome-mapping analysis. Pairwise diversity concerns this generated
batch, not novelty relative to training data. No arbitrary tissue, motif grammar,
functional cCRE identity, enhancer activity, endogenous regulation or biological
improvement is established. The training domain is human accessible DNA from the
four released sample conditions; native training-label order excludes chromosomes
1 and 2. Successful inference is not a new experimental validation study.

If explicitly requested, reuse the accepted `dna_sequences.v1` port for DHS64
prediction at an actually supported biosample condition. DHS64's native encoding
adds trailing zero vectors to 500 positions for these 200 real bases; it invents
no genomic flanks. Generator conditioning, model-predicted DNase signal and
observed biological activity remain distinct. Agreement is not generator validation.
DeepPBMC needs 2,114 real bases; these candidates cannot be padded with fabricated
DNA to make them eligible.

## Operator resource and runtime

ScientificResourceLibrary owns the `sequence_generator` resource with schema
`dna-diffusion-sequence-generator.v1`. It pins the released checkpoint, config and
qualified label metadata. The native source revision is
`fdc760716a06b20b51b5a8730601863784d3eadd`; the Hugging Face release is
`dd52f05cd60520c2cdb72f2a52234ca7e0d57924`. Source and weights remain outside Git.
Normal deployment uses the existing library and unchanged operator/launcher
configuration; researchers do not switch to an acceptance index.

Provision Python 3.12.16 separately from Agent with the
[qualified dependency lock](../../requirements/dna-diffusion-py312.lock). The native
package installs from its pinned upstream revision. The qualified combination is
Torch 2.7.1 and Transformers 4.52.4; the complete 156-package inventory and clean
dependency check are recorded by RSD-3 acceptance. Other combinations require
qualification. For example, in an isolated venv, install with `python -m pip install
-r requirements/dna-diffusion-py312.lock`. Use the existing `ScientificResourceLibrary.install` API with
`own_files=False` to index existing checkpoint/config/label files in place. Do not
replace the normal library index or downgrade Agent, DHS64, CREsted or EpiZoo.
Device, batch size and thread count belong to the operator runtime. Float32 with
TF32 disabled and no autocast is the current precision policy. The development
GPU model is not a scientific input restriction; new device/runtime combinations
need qualification and stochastic output may differ.

The [pinned upstream license](https://github.com/pinellolab/DNA-Diffusion/blob/fdc760716a06b20b51b5a8730601863784d3eadd/LICENSE.md)
provides academic/nonprofit use under Apache 2.0 and requires contacting the author
about profit-making use of the code **or generated sequences**. The checkpoint
model card's Apache metadata does not establish unrestricted commercial or hosted
rights. This deployment is for authorized local internal academic research;
no per-request licensing mechanism is added.

The [RSD-3 report](../../../agent-data/acceptance/regulatory-sequence-design/rsd-3/REPORT.md)
owns native parity, runtime portability, actual normal Firefox/OpenRouter use,
download/reuse evidence, preserved failures and the bounded readiness decision.

Native qualification, normal installation, two Firefox generation requests and
eight downloads pass. The later [Qwen integration](alibaba-qwen-provider.md) passes
genuine strict 31-tool planning and delivers the saved-batch comparison in normal
Firefox. After provider access was restored, normal Firefox continuation completed
native DHS64 prediction on the exact accepted 20 K562 candidates, four checked
prediction downloads and unsupported-condition clarification without science.
RSD-3 is complete within its bounded internal academic scope. The report preserves
earlier provider failures, withheld/incorrect generated prose and its later
read-only correction, plus the download-concurrency limit. Scores and their
canonical measurement metadata remain authoritative; generated explanations
are not universally reliable or biological validation.
