#!/usr/bin/env bash
set -euo pipefail

# ===============================
# Sweep launcher (GPU 3 & 6 only)
# ===============================

# ---- Quick config ----
ENTRYPOINT="${ENTRYPOINT:-FGSM_LAW_CIFAR100.py}"    # your training script filename
DATA_DIR="${DATA_DIR:-cifar-data}"        # dataset root
BASE_OUT="${BASE_OUT:-runs}"            # top-level output dir for all runs
MODEL="${MODEL:-ResNet18}"              # model name passed to your script
LAM_START="${LAM_START:-50}"            # fixed as requested
# Hardcode to use GPUs 3 and 6 only, with two concurrent jobs
GPU_LIST="3,6"
JOBS=2

# Optional: reduce CUDA memory fragmentation (helps some OOM cases)
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Always use stable PCI-index mapping
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# ---- Hyper-parameter grid ----
betas=(0.4 0.6)
lamdas=(40 42 44)               # NOTE: your script uses --lamda as the flag name
lam_scales=(0.09 0.1 0.11 0.12)
seeds=(0 1 2)                   # 3 runs per combo

# ---- GPU selection (fixed to 3 and 6) ----
IFS=',' read -r -a GPUS <<< "$GPU_LIST"
NGPUS="${#GPUS[@]}"
if [[ "$NGPUS" -ne 2 ]]; then
  echo "[ERROR] Expected exactly two GPUs (3 and 6). Got: $GPU_LIST"
  exit 1
fi

# Round-robin GPU allocator
idx=0
next_gpu () {
  local g="${GPUS[$((idx % NGPUS))]}"
  idx=$((idx+1))
  echo "$g"
}

running_jobs () { jobs -rp | wc -l | tr -d ' '; }

run_one () {
  local gpu="$1" beta="$2" lam="$3" lscale="$4" seed="$5"
  local outdir="${BASE_OUT}/beta_${beta}/lam_${lam}/lscale_${lscale}/seed_${seed}"
  mkdir -p "$outdir"

  echo "[INFO] GPU=${gpu} | beta=${beta} lam=${lam} lam_scale=${lscale} seed=${seed} → $outdir"

  # Each process only sees ONE physical device (the assigned one)
  CUDA_VISIBLE_DEVICES="$gpu" \
  python "$ENTRYPOINT" \
    --data-dir "$DATA_DIR" \
    --model "$MODEL" \
    --beta "$beta" \
    --lamda "$lam" \
    --lam_scale "$lscale" \
    --lam_start "$LAM_START" \
    --seed "$seed" \
    --out_dir "$outdir" \
    2>&1 | tee -a "${outdir}/console.log"
}

# ---- Launch sweep with at most 2 concurrent jobs (for GPUs 3 and 6) ----
for beta in "${betas[@]}"; do
  for lam in "${lamdas[@]}"; do
    for lscale in "${lam_scales[@]}"; do
      for seed in "${seeds[@]}"; do
        gpu="$(next_gpu)"
        run_one "$gpu" "$beta" "$lam" "$lscale" "$seed" &
        while [ "$(jobs -rp | wc -l | tr -d ' ')" -ge "$JOBS" ]; do
          # Wait for one job to finish before launching the next
          wait -n || true
        done
      done
    done
  done
done

wait
echo "[DONE] All runs finished."
