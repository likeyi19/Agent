# Alibaba Cloud Qwen in Agent Web

Alibaba Model Studio is an optional provider in the existing model factory
registry. Select **Qwen3.8-Max · Alibaba Cloud** in the ordinary **Conversation
model** menu. Each submitted interaction records its selected profile. The same
interpreter, 31-tool Registry planner, scientific owners, evidence review,
Session/Run/Revision storage and delivery routes serve every provider.
Existing OpenRouter/Groq profiles and the deployment's default remain unchanged.

The adapter requests `qwen3.8-max` through OpenAI-compatible Chat Completions,
with the unchanged Agent JSON Schema, `strict: true`, `enable_thinking: false`,
non-streaming responses and at most 4,096 completion tokens. It accepts only
nonempty assistant `content` from one completed response with the requested
model identity. Private `reasoning_content` is never read, saved or logged.
Incomplete, refused, wrong-model or transport-error responses retain the shared
failure semantics. The adapter performs no retry, model substitution or fallback;
existing bounded orchestration recovery remains its own responsibility.

Alibaba documents the [workspace-specific Singapore endpoint](https://www.alibabacloud.com/help/en/model-studio/model-calling-in-sub-workspace),
[Chat Completions protocol](https://www.alibabacloud.com/help/en/model-studio/qwen-api-via-openai-chat-completions)
and [structured output](https://help.aliyun.com/en/model-studio/qwen-structured-output).
Use the exact base URL shown for the workspace whose key was tested, ending in
`/compatible-mode/v1`. A public Beijing endpoint is not the qualified Singapore
workspace endpoint. Agent does not infer regions or workspace identifiers.

## Persistent credentials for the existing local launcher

The local deployment uses the existing `agent` Conda environment and private
operator configuration. Its launcher executes `conda run -p
/home/likeyi/anaconda3/envs/agent`; variables exported only in another terminal
are not sufficient. Credentials belong in that environment's existing private
`conda-meta/state`, with mode `0600`, outside the repository. This is the same
Conda mechanism already used by the deployment, not an Agent credential store.

Run the following **in an interactive terminal**. The key is entered with echo
disabled, never as a command argument or shell-history literal. Other saved
Conda variables are preserved. The endpoint prompt is for the tested Singapore
workspace base URL; do not include `/chat/completions`.

```bash
/home/likeyi/anaconda3/bin/python -c '
from getpass import getpass
from pathlib import Path
from urllib.parse import urlsplit
from conda.core.prefix_data import PrefixData
endpoint = input("Tested Singapore workspace base URL: ").strip()
key = getpass("Alibaba API key (hidden): ").strip()
u = urlsplit(endpoint)
assert (key and u.scheme == "https" and u.hostname
        and not u.username and not u.password and not u.query and not u.fragment
        and u.path.rstrip("/") == "/compatible-mode/v1"), "Invalid base URL or empty key"
PrefixData("/home/likeyi/anaconda3/envs/agent").set_environment_env_vars({
    "DASHSCOPE_BASE_URL": endpoint, "DASHSCOPE_API_KEY": key
})
Path("/home/likeyi/anaconda3/envs/agent/conda-meta/state").chmod(0o600)
print("Saved privately. Restart Agent through launch-uat.sh to load these values.")
'
```

Verify availability without printing values:

```bash
/home/likeyi/anaconda3/bin/conda run --no-capture-output \
  -p /home/likeyi/anaconda3/envs/agent python -c \
  'import os; print({k: bool(os.environ.get(k)) for k in ("DASHSCOPE_API_KEY", "DASHSCOPE_BASE_URL")})'
```

Stop the running Web server with Ctrl-C in its launch terminal, then restart it:

```bash
/home/likeyi/program/agent-data/config/launch-uat.sh --host 127.0.0.1 --port 8765
```

Open `http://127.0.0.1:8765`, reopen the desired Session, and select Qwen in
**Conversation model** before submitting. Existing processes do not acquire
later environment edits. `DASHSCOPE_MODEL` is not used: the selected Agent
profile owns the exact model identity. Neither the key nor the private endpoint
belongs in operator JSON, Git, reports or chat. Missing settings cause an
explicit `PLANNING_PROVIDER_CONFIGURATION_FAILED` prerequisite; a populated but
incorrect endpoint/key may instead produce a sanitized provider error.

## Model profile

The current private `/home/likeyi/program/agent-data/config/operator-uat.json`
contains this additional entry in `model_profiles`. For another deployment, add
it to that deployment's existing list and restart its server. Preserve other
profiles and its chosen default.

```json
{
  "profile_id": "alibaba-qwen3-8-max",
  "provider_id": "alibaba",
  "model_id": "qwen3.8-max",
  "display_label": "Qwen3.8-Max · Alibaba Cloud",
  "enabled": true,
  "supports_structured_output": true,
  "request_timeout_seconds": 120
}
```

The optional `openai` SDK is the existing compatible transport dependency.
No Alibaba-specific scientific inputs, schema copies or Web execution routes
are required. The [RSD-3 acceptance report](../../../agent-data/acceptance/regulatory-sequence-design/rsd-3/REPORT.md)
separates genuine provider/Firefox evidence from simulated regressions and records
remaining limits. One successful qualification does not establish universal
model reliability or biological validity. The earlier OpenRouter HTTP 402 remains
a separate billing observation.

Restored-access acceptance completes normal Firefox prediction from previously
generated DNA and unsupported-condition clarification. The recorded Alibaba 403
is historical; the user reported an account recharge before the successful access
check. Full milestone closeout is bounded by the report, including observed prose
errors, later read-only correction and a download-concurrency limit. No credential
or endpoint change was needed for this continuation.
