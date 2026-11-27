#!/usr/bin/env bash
# Copyright 2026 verl contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -eo pipefail

REPO_DIR=/root/work/tiankuan/shtskysmile/fix_partial_rollout/verl
BASELINE_PYTHON=/root/.local/conda/envs/verl/bin/python
CANN_ENV=/root/work/tiankuan/linxid/envs/ascend-20261106/Ascend/cann/set_env.sh
ATB_ENV=/root/work/tiankuan/linxid/envs/ascend-20261106/Ascend/nnal/nnal/atb/set_env.sh
# Reuse the compatibility library required by the reference CANN 9.1 stack.
PROFILER_SHIM=/root/work/tiankuan/shtskysmile/final_code/verl/jiutian/cross_dc/shim/libmsprof_shim.so

for required_file in "$BASELINE_PYTHON" "$CANN_ENV" "$ATB_ENV" "$PROFILER_SHIM"; do
    if [ ! -f "$required_file" ]; then
        echo "Missing Ascend environment file: $required_file" >&2
        exit 1
    fi
done

# Match the CANN/ATB activation and driver library setup of the reference launcher.
# The vendor activation scripts require nounset to be disabled.
set +u
unset PYTHONHOME RAY_ADDRESS
source "$CANN_ENV"
source "$ATB_ENV"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver"
export LD_PRELOAD="${LD_PRELOAD:+$LD_PRELOAD:}$PROFILER_SHIM"
export PATH="/root/.local/conda/envs/verl/bin:$PATH"
export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1 PIP_USER=0 PYTHONUNBUFFERED=1
export VERL_PLATFORM=huawei VERL_LOGGING_LEVEL=INFO
export ASCEND_GLOBAL_LOG_LEVEL=1 DENSE_ASCEND_LOG_LEVEL=1
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1 RAY_DEDUP_LOGS=0
export VLLM_ASCEND_ENABLE_NZ=0
export NO_PROXY='*' no_proxy='*'
export HCCL_SOCKET_IFNAME=eth0 GLOO_SOCKET_IFNAME=eth0 TP_SOCKET_IFNAME=eth0
export HCCL_CONNECT_TIMEOUT=3600 HCCL_EXEC_TIMEOUT=7200
export HCCL_HOST_SOCKET_PORT_RANGE=60000-60050 HCCL_NPU_SOCKET_PORT_RANGE=61000-61050

# Keep the reference launcher's physical-to-logical device mapping.
if [[ "${ASCEND_VISIBLE_DEVICES:-}" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    IFS=',' read -r -a physical_devices <<< "$ASCEND_VISIBLE_DEVICES"
    logical_devices=''
    for index in "${!physical_devices[@]}"; do
        logical_devices+="${logical_devices:+,}$index"
    done
    export ASCEND_RT_VISIBLE_DEVICES="$logical_devices"
else
    export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
fi
ulimit -n 32768 2>/dev/null || true

BASELINE_CONFIG_NAME=Omega_qwen25_7b_single_instance_4t4r
case "${1:-}" in
    true|false)
        BASELINE_CONFIG_NAME="${BASELINE_CONFIG_NAME}_partial_rollout_$1"
        export RUN_ID="omega_qwen25_7b_single_instance_4t4r_partial_rollout_${1}_r01"
        shift
        ;;
    debug_false_repeat|debug_false_trace|debug_false_token_is|debug_true_trace)
        BASELINE_CONFIG_NAME="${BASELINE_CONFIG_NAME}_$1"
        export RUN_ID="omega_qwen25_7b_single_instance_4t4r_${1}_r01"
        shift
        ;;
esac

export RUN_ID="${RUN_ID:-omega_qwen25_7b_single_instance_4t4r_r02}"
export VERL_FILE_LOGGER_ROOT="$REPO_DIR/outputs"
RUN_DIR="$REPO_DIR/outputs/partial_rollout_repro/$RUN_ID"
mkdir -p "$RUN_DIR"
cd "$REPO_DIR"

"$BASELINE_PYTHON" -u -m verl.trainer.main_ppo \
    --config-path "$REPO_DIR/config" \
    --config-name "$BASELINE_CONFIG_NAME" \
    "trainer.default_local_dir=$RUN_DIR/ckpts" \
    "$@" 2>&1 | tee "$RUN_DIR/train.log"
