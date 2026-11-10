# Manual researcher Web testing on the current host

The prepared private Operator file is
`/home/likeyi/program/agent-data/config/operator-manual.json`. It uses the separate
workspace, `/home/likeyi/program/agent-data/workspaces/researcher-manual`, originally
provisioned without copied Sessions, inputs or scientific results. Its subsequent
manual-test Sessions and results remain there. The PBMC10k acceptance workspace
and its evidence remain separate. This guide starts a foreground process only.

The profile is `openai-gpt-5-6-sol`, provider `openai`, model `gpt-5.6-sol`.
The file contains this existing official profile, an empty `input_sets` list,
`scientific_resource_root` pointing at the shared library, `upload_root` pointing
at this workspace's `uploads`, `upload_max_bytes: 4294967296` and
`upload_max_concurrent: 1`. Credentials remain in the existing Conda environment;
the Operator JSON contains no API key. Activating Conda loads the installed native
runtime paths and existing credentials. Do not print the environment or keys.

## Start in a terminal on the host

Run this complete block in Bash. The existing launcher activates Conda and starts
`python -m agent.web`; its final `--config` argument selects the manual Operator
file instead of its UAT default. All flags below are supported by that entry point.

```bash
(
set -e
cd /home/likeyi/program/Agent
source /home/likeyi/anaconda3/etc/profile.d/conda.sh
conda activate agent

export PYTHONDONTWRITEBYTECODE=1
export AGENT_SCIENTIFIC_RESOURCE_ROOT=/home/likeyi/program/agent-data/resources/library
export AGENT_NATIVE_MEMORY_BYTES="$(python - <<'PY'
from pathlib import Path
fields = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
available = int(fields['MemAvailable'].split()[0]) * 1024
gib = 1024**3
budget = ((available - 32*gib) // gib) * gib
if budget <= 0:
    raise SystemExit('Available memory cannot retain the chosen 32 GiB host reserve.')
print(budget)
PY
)"
test -n "$AGENT_NATIVE_MEMORY_BYTES"
export AGENT_NATIVE_TIMEOUT_SECONDS=7200
export AGENT_ANNOTATION_R_TIMEOUT_SECONDS=7200

export AGENT_DELIVERY_SOURCE_BYTES=4294967296
export AGENT_DELIVERY_DELIVERED_BYTES=4294967296
export AGENT_DELIVERY_LOGICAL_BYTES=4294967296
export AGENT_DELIVERY_TEMPORARY_BYTES=17179869184
export AGENT_DELIVERY_TRANSFERS=1
export TMPDIR=/home/likeyi/program/agent-data/workspaces/researcher-manual/tmp

python - <<'PY'
import os, shutil, tempfile
from pathlib import Path
assert os.environ.get('OPENAI_API_KEY'), 'The activated agent environment has no configured OpenAI key.'
root = Path(os.environ['TMPDIR'])
assert root.is_dir(), 'The prepared manual temporary directory is missing.'
with tempfile.TemporaryFile(dir=root) as stream:
    stream.write(b'storage check')
free = shutil.disk_usage(root).free
assert free > 16*1024**3 + 64*1024**2, 'Insufficient free temporary storage for the configured delivery reservation and disk margin.'
print('OpenAI credential present; temporary directory writable.')
print('Native memory budget (GiB):', int(os.environ['AGENT_NATIVE_MEMORY_BYTES']) // 1024**3)
print('Temporary storage free (GiB):', free // 1024**3)
PY

/home/likeyi/program/agent-data/config/launch-uat.sh \
  --config /home/likeyi/program/agent-data/config/operator-manual.json \
  --host 127.0.0.1 --port 8000
)
```

The memory calculation uses current Linux `MemAvailable`, retains a chosen
32 GiB reserve for other host work, and rounds the remaining budget down to whole
GiB. It does not reuse the earlier 81 GiB limit. Recalculate at each startup;
later competing workloads still consume host memory. The existing user systemd
service/cgroup v2 boundary applies this aggregate budget to native Python/R
execution, disables swap for that unit, and retains cancellation/timeouts. Native
execution requires the working user systemd manager; it does not fall back to
uncontrolled execution. PCA fitting and its existing verifier use that boundary.

Both the JSON resource field and `AGENT_SCIENTIFIC_RESOURCE_ROOT` select the same
shared library: the application uses its scoped owner and isolated workers inherit
the explicit environment location. Native runtimes and qualified references stay
in their existing installation locations. Nothing is copied into the test workspace.

The upload limit is per file (4 GiB). Scientific delivery controls are separate:
maximum source file 4 GiB, maximum delivered file 4 GiB, logical HDF5 inspection
limit 4 GiB, aggregate reserved temporary-file capacity 16 GiB, one active
scientific download. `AGENT_DELIVERY_TRANSFERS` does not set browser upload
concurrency; `upload_max_concurrent` does. The existing 16 MiB transfer chunk
default is retained. Limits do not establish that every scientific format/input
is compatible. Temporary snapshots use the configured `TMPDIR`; completed browser
downloads are saved on the local computer according to its browser settings.

## Connect from the local computer

The host listens at **http://127.0.0.1:8000/**. On the local computer, run:

```bash
ssh -N -T -o ExitOnForwardFailure=yes \
  -L 127.0.0.1:8000:127.0.0.1:8000 \
  -p 22 likeyi@101.6.43.249
```

The destination and port above were observed from this host session's SSH
connection. Use the same SSH key/password or SSH configuration normally used to
reach this host. Keep this terminal open and browse **http://127.0.0.1:8000/**
on the local computer. If local port 8000 is occupied, change only the first
forwarded port to `8001` and open `http://127.0.0.1:8001/`; the host still listens
on 8000. Do not bind the Agent service to a public address.

The current Web application has no separate login/authentication system. SSH
authenticates access to this loopback service. The application rejects foreign
Origin headers for mutations; use the UI and API through the same forwarded URL.
No firewall change is needed.

The page should offer **GPT-5.6 Sol · OpenAI API** in **Conversation model**.
Use **Open by ID** beside recent conversations to reopen a known Session.
Start with **New conversation**. **Recent conversations · this browser** is browser-local history;
an old acceptance entry can remain in that list but is not migrated into this
new server workspace. A new Session starts without accepted results or uploaded
sample data. The `/api/v1/models` and `/api/v1/health` endpoints can confirm the
configured model and instance availability without a model call; the model
profile and workspace are fixed by the exact Operator file above.

## Stop

Press **Ctrl+C** in the host terminal running the launcher and wait for Uvicorn
shutdown. Press **Ctrl+C** in the local SSH-forwarding terminal afterward. Saved
Sessions remain in the manual workspace. If a scientific task is running, first
use its **Cancel turn** control and wait for it to stop before ending the server.
Do not use broad `pkill` commands on a shared host.

You can check that the host listener has closed with:

```bash
ss -ltnp 'sport = :8000'
```

The scientific figure interface now plots saved native QC measurements and profiles,
with their recorded populations and filtering thresholds. Missing measurements stay
unavailable; viewing these figures does not repeat QC or filtering. Finite reference
choices can be selected directly in the conversation. The same startup procedure
and workspace apply to these researcher-experience improvements. PBMC10k scientific
acceptance remains **CONDITIONAL GO**; this does not qualify a new full-data analysis.
