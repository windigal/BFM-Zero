#!/usr/bin/env bash
set -euo pipefail

GMR_PY="/home/hanwei/miniforge3/envs/gmr/bin/python"
BFM_ROOT="/home/hanwei/code/BFM-Zero"
TEXTOP_ROOT="/home/hanwei/code/TextOp"

BABEL_DIR="/home/hanwei/dataset/babel_v1-0_release/babel_v1.0_release"
AMASS_HF_DIR="/home/hanwei/dataset/AMASS"
AMASS_SMPLX_DIR="/home/hanwei/dataset/AMASS_babel_smplx"
AMASS_ROBOT_30_DIR="/home/hanwei/dataset/AMASS_robot_g1_29dof_30fps"
AMASS_ROBOT_50_DIR="/home/hanwei/dataset/AMASS_robot_g1_29dof_50fps"
TEXTOP_PACK_DIR="${TEXTOP_ROOT}/dataset/BABEL-AMASS-ROBOT-29dof-50fps-TEACH"
FINAL_OUTPUT_DIR="${BFM_ROOT}/artifacts/textop_babel_h2_f8_50fps"
ROBOT_CFG="${TEXTOP_ROOT}/TextOpRobotMDAR/robotmdar/config/skeleton/g1_29dof.yaml"

echo "[autocontinue] waiting for current AMASS download / retarget jobs to finish..."
while pgrep -af 'hf download tuguobin/AMASS|smplx_to_robot_dataset.py --src_folder /home/hanwei/dataset/AMASS_babel_smplx --tgt_folder /home/hanwei/dataset/AMASS_robot_g1_29dof_30fps' >/dev/null; do
  date
  du -sh "${AMASS_HF_DIR}" "${AMASS_SMPLX_DIR}" "${AMASS_ROBOT_30_DIR}" 2>/dev/null || true
  echo "retarget_pkl_count=$(find "${AMASS_ROBOT_30_DIR}" -type f -name '*.pkl' 2>/dev/null | wc -l)"
  sleep 300
done

echo "[autocontinue] download/retarget idle. Running final extraction + retarget + 50Hz + pack..."

cd "${BFM_ROOT}"
"${GMR_PY}" -m humanoidverse.scripts.extract_babel_required_amass_raw \
  --babel-dir "${BABEL_DIR}" \
  --amass-hf-dir "${AMASS_HF_DIR}" \
  --output-dir "${AMASS_SMPLX_DIR}"

cd "${TEXTOP_ROOT}"
"${GMR_PY}" dataset/smplx_to_robot_dataset.py \
  --src_folder "${AMASS_SMPLX_DIR}" \
  --tgt_folder "${AMASS_ROBOT_30_DIR}" \
  --robot unitree_g1 \
  --num_cpus 4

"${GMR_PY}" dataset/process_retarget_data.py \
  --input_dir "${AMASS_ROBOT_30_DIR}" \
  --output_dir "${AMASS_ROBOT_50_DIR}" \
  --robot_config "${ROBOT_CFG}" \
  --dof_layout full

"${GMR_PY}" dataset/pack_dataset.py \
  --amass_robot "${AMASS_ROBOT_50_DIR}" \
  --babel "${BABEL_DIR}" \
  --output_dir "${TEXTOP_PACK_DIR}"

cd "${BFM_ROOT}"
"${GMR_PY}" -m humanoidverse.scripts.build_textop_babel_dataset \
  --amass-robot-dir "${AMASS_ROBOT_50_DIR}" \
  --babel-dir "${BABEL_DIR}" \
  --output-dir "${FINAL_OUTPUT_DIR}" \
  --overwrite-output

echo "[autocontinue] pipeline finished."
