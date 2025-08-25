# generate_config.py
#!/usr/bin/env python3
import argparse
import json
import configparser
import os
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("--out", default="facefusion_stream.ini")
parser.add_argument("--gpu-list", default="")
parser.add_argument("--gpu-count", type=int, default=0)
parser.add_argument("--threads", type=int, default=4)
parser.add_argument("--queue", type=int, default=4)
parser.add_argument("--memory-limit-gb", type=int, default=0)
args = parser.parse_args()

cfg = configparser.ConfigParser()
# minimal skeleton; user may expand
cfg["paths"] = {
    "temp_path": "/tmp/facefusion_temp",
    "jobs_path": "/tmp/facefusion_jobs",
    "source_paths": "",
    "target_path": "",
    "output_path": ""
}
cfg["execution"] = {
    "execution_device_id": args.gpu_list if args.gpu_list else "",
    "execution_providers": "cuda" if args.gpu_count>0 else "cpu",
    "execution_thread_count": str(args.threads),
    "execution_queue_count": str(args.queue)
}
cfg["misc"] = {
    "log_level": "INFO",
    "halt_on_error": "false"
}
if args.memory_limit_gb and args.memory_limit_gb>0:
    cfg["memory"] = {"system_memory_limit": str(args.memory_limit_gb)}

open(args.out,"w").write("\n".join([f"[{s}]\n"+"\n".join(f"{k} = {v}" for k,v in cfg[s].items()) for s in cfg.sections()]))
print(args.out)
bash
Копировать
Редактировать
# launch_facefusion.sh
#!/usr/bin/env bash
set -euo pipefail

# usage: ./launch_facefusion.sh <INPUT_SRT_PORT> <OUTPUT_SRT_PORT> [facefusion_cmd]
IN_PORT=${1:-1234}
OUT_PORT=${2:-1235}
FACEFUSION_CMD=${3:-"python3 facefusion.py"}

# detect GPUs
GPU_INFO=$(python3 gpu_detect.py)
GPU_COUNT=$(echo "$GPU_INFO" | python3 -c "import sys,json;print(json.load(sys.stdin)['gpu_count'])")
GPU_LIST=$(echo "$GPU_INFO" | python3 -c "import sys,json;print(','.join(str(g['index']) for g in json.load(sys.stdin)['gpus']))" || true)
CPU_COUNT=$(echo "$GPU_INFO" | python3 -c "import sys,json;print(json.load(sys.stdin)['cpu_count'])")

# compute threads/queue defaults
if [ "$GPU_COUNT" -gt 0 ]; then
  THREADS=$(( (CPU_COUNT / GPU_COUNT) > 2 ? (CPU_COUNT / GPU_COUNT) : 2 ))
else
  THREADS=$(( CPU_COUNT > 2 ? CPU_COUNT-1 : 2 ))
fi
QUEUE=4

# generate config
python3 generate_config.py --out facefusion_stream.ini --gpu-list "${GPU_LIST}" --gpu-count "${GPU_COUNT}" --threads "${THREADS}" --queue "${QUEUE}" > /dev/null

# start mediamtx if not running (optional minimal check)
if ! pgrep -f mediamtx >/dev/null 2>&1; then
  if command -v mediamtx >/dev/null 2>&1; then
    mediamtx &>/dev/null &
  fi
fi

# Launch facefusion with new flags --input-stream-url --output-stream-url
INPUT_URL="srt://0.0.0.0:${IN_PORT}"
OUTPUT_URL="srt://0.0.0.0:${OUT_PORT}"

# build command
CMD="${FACEFUSION_CMD} --config facefusion_stream.ini --input-stream-url \"${INPUT_URL}\" --output-stream-url \"${OUTPUT_URL}\" --fps-out 30 --max-queue-size ${QUEUE} --threads ${THREADS}"
if [ "$GPU_COUNT" -gt 0 ]; then
  CMD="${CMD} --gpus ${GPU_COUNT} --gpu-list \"${GPU_LIST}\" --use-fp16"
fi

echo "Launching: $CMD"
# run in background
bash -c "$CMD" &
disown