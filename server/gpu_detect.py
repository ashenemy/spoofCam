# gpu_detect.py
#!/usr/bin/env python3
import json
import sys
try:
    import torch
    has_torch = True
except Exception:
    has_torch = False

import subprocess

def nvidia_smi_list():
    try:
        out = subprocess.check_output(["nvidia-smi","--query-gpu=index,name,memory.total","--format=csv,noheader"], stderr=subprocess.DEVNULL)
        lines = out.decode().strip().splitlines()
        gpus=[]
        for l in lines:
            parts=[p.strip() for p in l.split(',')]
            if parts:
                gpus.append({"index": int(parts[0]), "name": parts[1], "memory": parts[2]})
        return gpus
    except Exception:
        return []

def main():
    gpus = nvidia_smi_list()
    if not gpus and has_torch:
        try:
            cnt = torch.cuda.device_count()
            for i in range(cnt):
                gpus.append({"index": i, "name": torch.cuda.get_device_name(i), "memory": ""})
        except Exception:
            pass
    info = {
        "torch_available": has_torch,
        "gpu_count": len(gpus),
        "gpus": gpus,
        "cpu_count": __import__("os").cpu_count()
    }
    print(json.dumps(info))

if __name__ == "__main__":
    main()
