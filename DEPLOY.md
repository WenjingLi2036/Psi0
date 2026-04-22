Setup for psi-inference_rtc

conda env create -f psi_deploy_env.yaml
conda activate psi_deploy

Run deploy actor script
bash ./real/scripts/deploy_psi0-rtc.sh

Run VLA server
bash ./scripts/deploy/serve_psi0-rtc.sh

VLA Server

mkdir -p "$PSI_HOME/cache/checkpoints"

hf download USC-PSI-Lab/psi-model \
  --include="psi0/real-checkpoints/task1/argv.txt" \
  --include="psi0/real-checkpoints/task1/run_config.json" \
  --include="psi0/real-checkpoints/task1/checkpoints/ckpt_40000/**" \
  --local-dir="$PSI_HOME/cache/checkpoints" \
  --repo-type=model

export CHECKPOINT_DIR="$PSI_HOME/cache/checkpoints/psi0/real-checkpoints/task1"
export CHECKPOINT_STEP=40000

bash scripts/deploy/serve_psi0_simple.sh "$CHECKPOINT_DIR" 40000


1) Setup G1 Vision server
2) Start g1 and do L2 + B, then L2 + R2
3) Start