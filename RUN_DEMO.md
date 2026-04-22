# Psi0 SIMPLE Demo Run

This guide starts where [INSTALL_UPDATE.md](./INSTALL_UPDATE.md) leaves off.
It shows a complete demo flow for:

1. entering the working `psi + simple` environment
2. downloading the released Psi0 base checkpoints
3. wrapping those downloaded checkpoints into a real `.runs/...` checkpoint
4. serving that checkpoint with `psi0_serve_simple.py`
5. running SIMPLE evaluation against that served checkpoint

The default example task below is:

```bash
G1WholebodyXMovePickTeleop-v0
```

That task ends with `Teleop`, so the SIMPLE entrypoint and agent are:

- `eval_decoupled_wbc.py`
- `psi0_decoupled_wbc`

For `*MP*` tasks, the SIMPLE entrypoint and agent are different. This guide
shows both cases later.


## Assumptions

You have already completed [INSTALL_UPDATE.md](./INSTALL_UPDATE.md), including:

- the working top-level Nix dev shell
- `.venv-psi`
- `uv sync`
- `flash_attn`
- `.env`
- a working `third_party/SIMPLE` checkout

You should also already be using the updated
[scripts/fix_cuda_env.sh](./scripts/fix_cuda_env.sh), which exposes:

- `libcuda.so.1` for runtime loading
- `libcuda.so` for Triton / linker-based compile steps


## Terminal Layout

This demo is easiest with three terminals:

1. `Terminal A`: create the `.runs/...` checkpoint from the downloaded
   Psi0 checkpoints
2. `Terminal B`: serve the resulting checkpoint
3. `Terminal C`: run SIMPLE evaluation

If you are serving on a remote machine and evaluating on a different machine,
replace `localhost` below with the forwarded or remote host as needed.


## 1. Enter The Repo Environment

Run this in every terminal you use for the demo:

```bash
cd ~/Psi0-handless

env -u LD_PRELOAD -u LD_LIBRARY_PATH \
  nix --extra-experimental-features "nix-command flakes" develop -c bash
```

Once the new dev-shell prompt opens, run:

```bash
cd ~/Psi0-handless

source .venv-psi/bin/activate

set -a
source .env
set +a

source scripts/fix_cuda_env.sh
```

Optional sanity check:

```bash
python - <<'PY'
import torch
print("cuda_available =", torch.cuda.is_available())
print("device_count   =", torch.cuda.device_count())
PY
```

Expected shape:

- `cuda_available = True`
- `device_count >= 1`


## 2. Choose A Demo Task

Set the task once and reuse it across terminals:

```bash
export TASK=G1WholebodyXMovePickTeleop-v0
```


## 3. Download SIMPLE Task Data

Download the task dataset used by the wrapping run:

```bash
hf download USC-PSI-Lab/psi-data \
  "simple/$TASK.zip" \
  --local-dir="$PSI_HOME/data" \
  --repo-type=dataset

unzip -n "$PSI_HOME/data/simple/$TASK.zip" -d "$PSI_HOME/data/simple"
```

Some setups end up with the task directly under:

```bash
$PSI_HOME/data/simple/$TASK
```

while others end up under:

```bash
$PSI_HOME/data/simple/simple/$TASK
```

Detect the working root automatically:

```bash
export SIMPLE_DATA_ROOT="$PSI_HOME/data/simple"
[ -d "$SIMPLE_DATA_ROOT/$TASK" ] || export SIMPLE_DATA_ROOT="$PSI_HOME/data/simple/simple"

echo "$SIMPLE_DATA_ROOT"
ls "$SIMPLE_DATA_ROOT/$TASK"
```


## 4. Download The Released Psi0 Checkpoints

Create the local checkpoint cache if needed:

```bash
mkdir -p "$PSI_HOME/cache/checkpoints"
```

Download the released Psi0 VLM checkpoint:

```bash
hf download USC-PSI-Lab/psi-model \
  --include="psi0/pre.fast.1by1.2601091803.ckpt.ego200k.he30k/*" \
  --local-dir="$PSI_HOME/cache/checkpoints" \
  --repo-type=model
```

Download the released Psi0 action expert checkpoint:

```bash
hf download USC-PSI-Lab/psi-model \
  --include="psi0/postpre.1by1.pad36.2601131206.ckpt.he30k/*" \
  --local-dir="$PSI_HOME/cache/checkpoints" \
  --repo-type=model
```

Set the resulting paths:

```bash
export PRE_CKPT="$PSI_HOME/cache/checkpoints/psi0/pre.fast.1by1.2601091803.ckpt.ego200k.he30k"
export POST_CKPT="$PSI_HOME/cache/checkpoints/psi0/postpre.1by1.pad36.2601131206.ckpt.he30k"

ls "$PRE_CKPT"
ls "$POST_CKPT"
```


## 5. Create A Real `.runs/...` Checkpoint From The Downloaded Checkpoints

The released checkpoints above are not already in the exact deployable
`.runs/.../checkpoints/ckpt_<step>` format that `psi0_serve_simple.py`
expects.

The smallest reliable conversion path in this repo is a 1-step finetune run
with:

- `learning_rate=0.0`
- `max_training_steps=1`
- `checkpointing_steps=1`

That creates a real run directory and a real `ckpt_1` while leaving the loaded
weights effectively unchanged for demo purposes.

Run this in `Terminal A`:

```bash
export CUDA_VISIBLE_DEVICES=0

torchrun --standalone --nproc_per_node=1 scripts/train.py \
  finetune_simple_psi0_config \
  --debug \
  --seed=292285 \
  --exp=minwrap \
  --train.name=finetune \
  --train.output_dir=.runs \
  --train.data_parallel=ddp \
  --train.mixed_precision=bf16 \
  --train.num_workers=0 \
  --train.train_batch_size=1 \
  --train.gradient_accumulation_steps=1 \
  --train.learning_rate=0.0 \
  --train.lr_scheduler_type=constant \
  --train.lr_scheduler_kwargs.weight_decay=0.0 \
  --train.lr_scheduler_kwargs.betas 0.95 0.999 \
  --train.max_training_steps=1 \
  --train.warmup_steps=0 \
  --train.warmup_ratio=None \
  --train.checkpointing_steps=1 \
  --train.validation_steps=0 \
  --train.val_num_batches=0 \
  --train.max_checkpoints_to_keep=1 \
  --log.report_to=None \
  --data.root_dir="$SIMPLE_DATA_ROOT" \
  --data.train-repo-ids="$TASK" \
  --data.transform.repack.pad-action-dim=36 \
  --data.transform.repack.pad-state-dim=36 \
  --data.transform.field.stat-path=meta/stats_psi0.json \
  --data.transform.field.stat-action-key=action \
  --data.transform.field.stat-state-key=states \
  --data.transform.field.action_norm_type=bounds \
  --data.transform.field.no-use-norm-mask \
  --data.transform.field.normalize-state \
  --data.transform.field.pad-action-dim=36 \
  --data.transform.field.pad-state-dim=36 \
  --data.transform.model.resize.size 180 320 \
  --data.transform.model.center_crop.size 180 320 \
  --model.model_name_or_path="$PRE_CKPT" \
  --model.pretrained-action-header-path="$POST_CKPT" \
  --model.noise-scheduler=flow \
  --model.train-diffusion-steps=1000 \
  --model.n_conditions=0 \
  --model.action-chunk-size=30 \
  --model.action-dim=36 \
  --model.action-exec-horizon=30 \
  --model.observation-horizon=1 \
  --model.odim=36 \
  --model.view_feature_dim=2048 \
  --model.no-tune-vlm \
  --model.no-use_film \
  --model.no-combined_temb \
  --model.rtc \
  --model.max-delay=8
```

If that finishes successfully, capture the run directory:

```bash
export RUN_DIR="$(ls -td .runs/finetune/debug-minwrap* | head -n1)"
export CKPT_STEP=1

echo "$RUN_DIR"
ls "$RUN_DIR"
ls "$RUN_DIR/checkpoints/ckpt_$CKPT_STEP"
```

Expected files include:

- `$RUN_DIR/run_config.json`
- `$RUN_DIR/argv.txt`
- `$RUN_DIR/checkpoints/ckpt_1/model.safetensors`


## 6. Troubleshooting The Wrap Step

### `ld: cannot find -lcuda`

If training runs and then fails during checkpoint save with:

```text
ld: cannot find -lcuda
```

then the shell can see `libcuda.so.1` for runtime loading, but the linker still
cannot see the unversioned `libcuda.so`.

The updated helper should already fix this. Re-run:

```bash
source scripts/fix_cuda_env.sh
```

and verify:

```bash
python - <<'PY'
import ctypes
ctypes.CDLL("libcuda.so")
ctypes.CDLL("libcuda.so.1")
print("libcuda linker/runtime names both OK")
PY
```

Then rerun the same `torchrun ...` command.

### `torch.cuda.is_available()` is `False`

Go back to the CUDA troubleshooting section in
[INSTALL_UPDATE.md](./INSTALL_UPDATE.md).

### `ModuleNotFoundError: No module named 'gear_sonic'`

`gear_sonic` is a nested SIMPLE dependency used by the decoupled-WBC SIMPLE
evaluation path. If it is missing, the most common cause is that the nested
SIMPLE submodule checkout was incomplete when the top-level `uv sync` ran.

From the repo root:

```bash
cd ~/Psi0-handless

git -C third_party/SIMPLE \
  -c url."https://github.com/".insteadOf=git@github.com: \
  -c protocol.file.allow=always \
  submodule update --init third_party/gear_sonic

test -f third_party/SIMPLE/third_party/gear_sonic/pyproject.toml -o \
     -f third_party/SIMPLE/third_party/gear_sonic/setup.py && echo gear-sonic-meta-ok
```

Then re-enter the repo environment and resync the Python environment:

```bash
env -u LD_PRELOAD -u LD_LIBRARY_PATH \
  nix --extra-experimental-features "nix-command flakes" develop -c bash
```

Once the dev-shell prompt opens:

```bash
cd ~/Psi0-handless
source .venv-psi/bin/activate
GIT_LFS_SKIP_SMUDGE=1 uv sync --all-groups --index-strategy unsafe-best-match --active
python -c "import gear_sonic; print(gear_sonic.__file__)"
```

After that, retry the SIMPLE eval command.


## 7. Serve The Wrapped Checkpoint

Open `Terminal B`, re-enter the repo environment, and then serve the checkpoint:

```bash
cd ~/Psi0-handless

env -u LD_PRELOAD -u LD_LIBRARY_PATH \
  nix --extra-experimental-features "nix-command flakes" develop -c bash
```

Once the new dev-shell prompt opens, run:

```bash
cd ~/Psi0-handless

source .venv-psi/bin/activate

set -a
source .env
set +a

source scripts/fix_cuda_env.sh

export CUDA_VISIBLE_DEVICES=0
export RUN_DIR="$(ls -td .runs/finetune/debug-minwrap* | head -n1)"
export CKPT_STEP=1

bash scripts/deploy/serve_psi0_simple.sh "$RUN_DIR" "$CKPT_STEP"
```

That serves Psi0 on:

```text
http://0.0.0.0:22085
```

Health check from another terminal:

```bash
curl -i http://localhost:22085/health
```

If the serving machine is remote, forward the port first:

```bash
ssh -L 22085:localhost:22085 <user>@<remote-host>
```


## 8. Download SIMPLE Eval Data

Open `Terminal C`.

You can run the evaluation on the same machine if it has enough resources, but
Psi0 serving plus Isaac/SIMPLE on one workstation can be heavy. Using a
separate workstation for SIMPLE eval is often more comfortable.

If you evaluate from this repo checkout, use the already-built top-level
environment and run the SIMPLE CLI from `third_party/SIMPLE`.

First, enter the environment:

```bash
cd ~/Psi0-handless

env -u LD_PRELOAD -u LD_LIBRARY_PATH \
  nix --extra-experimental-features "nix-command flakes" develop -c bash
```

Once the new dev-shell prompt opens, run:

```bash
cd ~/Psi0-handless

source .venv-psi/bin/activate
source scripts/fix_cuda_env.sh
```

Then switch into the SIMPLE checkout and download eval data:

```bash
cd third_party/SIMPLE

export TASK=G1WholebodyXMovePickTeleop-v0

hf download USC-PSI-Lab/psi-data \
  "simple-eval/$TASK.zip" \
  --local-dir=data/evals \
  --repo-type=dataset

unzip -n "data/evals/simple-eval/$TASK.zip" -d data/evals/simple-eval
```

Important:

- environment variables from the server terminal do not automatically carry
  over to this SIMPLE terminal
- set `TASK`, `DR`, `ENTRY`, and `AGENT` again in this shell before running the
  SIMPLE CLI

Before running the decoupled-WBC SIMPLE evaluator, install SIMPLE's `sonic`
dependencies into the current `.venv-psi`. These are nested SIMPLE packages and
are not pulled in by the top-level repo's `uv sync --all-groups`:

```bash
cd ~/Psi0-handless

VIRTUAL_ENV=.venv-psi uv pip install \
  -e 'third_party/SIMPLE/third_party/decoupled_wbc[full]' \
  -e 'third_party/SIMPLE/third_party/gear_sonic[sim,teleop]' \
  -e third_party/SIMPLE/third_party/unitree_sdk2_python \
  -e third_party/SIMPLE/third_party/XRoboToolkit-PC-Service-Pybind_X86_and_ARM64

python -c "import gear_sonic; print(gear_sonic.__file__)"
```

Then return to the SIMPLE checkout:

```bash
cd ~/Psi0-handless/third_party/SIMPLE
```


## 9. Pick The SIMPLE Eval Entrypoint

Choose a domain-randomization level:

```bash
export DR=level-0
```

For `Teleop` tasks:

```bash
export ENTRY=eval_decoupled_wbc.py
export AGENT=psi0_decoupled_wbc
```

For `MP` tasks:

```bash
export ENTRY=eval.py
export AGENT=psi0
```

Automatic selection:

```bash
if [[ "$TASK" == *Teleop* ]]; then
  export ENTRY=eval_decoupled_wbc.py
  export AGENT=psi0_decoupled_wbc
else
  export ENTRY=eval.py
  export AGENT=psi0
fi

echo "$ENTRY"
echo "$AGENT"
```


## 10. Run SIMPLE Evaluation

Start evaluation:

```bash
export ACCEPT_EULA=Y
export OMNI_KIT_ACCEPT_EULA=YES
export SIMPLE_DISABLE_TUI=1
export TORCH_EXTENSIONS_DIR="$PWD/.torch_extensions"
mkdir -p "$TORCH_EXTENSIONS_DIR"

python "src/simple/cli/$ENTRY" \
  "simple/$TASK" \
  "$AGENT" \
  "$DR" \
  --host=localhost \
  --port=22085 \
  --sim-mode=mujoco_isaac \
  --headless \
  --data-format=lerobot \
  --data-dir="data/evals/simple-eval/$TASK/$DR"
```

Notes:

- `--host=localhost --port=22085` matches the local Psi0 serve script in this
  fork.
- If the policy server is on a remote machine and you are not using SSH port
  forwarding, replace `localhost` with that machine's reachable host.
- `--headless` is the safer default on remote or shell-only machines. A
  non-headless `mujoco_isaac` launch can crash inside native graphics /
  simulator libraries before Python raises a normal exception.
- `TORCH_EXTENSIONS_DIR` gives CuRobo and related JIT-built extensions a stable
  cache path instead of a short-lived shell temp directory.

For a single episode, SIMPLE eval can take several minutes.

Important GPU note:

- SIMPLE eval uses Isaac Sim for rendering in `mujoco_isaac` mode.
- Isaac Sim 4.5.0 does not support GPUs without RT cores, including
  `A100` and `H100`.
- On those GPUs, it is common to see native Isaac startup failures or
  segmentation faults even when:
  - `torch.cuda.is_available()` is `True`
  - the policy server is healthy
  - `--headless` is used
- If your eval host is an `A100` or `H100`, the most likely fix is to run
  SIMPLE eval on a supported RTX-class workstation instead, and keep the
  Psi0 policy server on the remote machine if desired.


## 11. Where To Find Results

The wrapped Psi0 run lives under:

```bash
.runs/finetune/debug-minwrap...
```

The SIMPLE rollout outputs are typically written under the SIMPLE checkout,
for example:

```bash
third_party/SIMPLE/data/evals/psi0
```

If you want to inspect the exact checkpoint that was served:

```bash
ls "$RUN_DIR/checkpoints/ckpt_$CKPT_STEP"
```


## 12. Copy-Paste Summary

If you just want the high-level flow:

1. enter the top-level Nix shell and activate `.venv-psi`
2. `source .env`
3. `source scripts/fix_cuda_env.sh`
4. download `simple/$TASK.zip`
5. download the released Psi0 VLM and action-head checkpoints
6. run the 1-step zero-LR wrap command to create `.runs/.../checkpoints/ckpt_1`
7. serve that checkpoint with `bash scripts/deploy/serve_psi0_simple.sh "$RUN_DIR" 1`
8. in `third_party/SIMPLE`, download `simple-eval/$TASK.zip`
9. pick `ENTRY` and `AGENT` based on whether the task is `Teleop` or `MP`
10. run the SIMPLE CLI against `localhost:22085`


## 13. Segfault Troubleshooting

If SIMPLE segfaults after the eval banner appears and the traceback ends in
Isaac Sim modules such as:

- `isaacsim.simulation_app`
- `isaacsim.asset.importer.urdf`
- `.../scripts/ui/ui_utils.py`

then the crash is most likely inside Isaac Sim startup, not in the Psi0
checkpoint, the HTTP serve path, or the CUDA shim helper.

Most common cause:

- the eval host is using an unsupported Isaac Sim GPU, especially `A100`
  or `H100`

What to do:

1. Keep serving Psi0 from the current machine if you want.
2. Move the SIMPLE evaluation terminal to a supported RTX-class workstation.
3. Re-run the same eval command there, pointing `--host` and `--port` at the
   running policy server.

If the GPU is supported and you still crash:

- retry from a clean shell with `--headless`
- keep `SIMPLE_DISABLE_TUI=1`
- keep a stable `TORCH_EXTENSIONS_DIR`
- consider resetting Isaac user state on that machine before retrying
