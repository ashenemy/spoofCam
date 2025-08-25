# sender.py
#!/usr/bin/env python3
import argparse
import subprocess
import shlex
import sys
import os

parser = argparse.ArgumentParser()
parser.add_argument("--input", "-i", required=True, help="input file or device (camera)")
parser.add_argument("--host", required=True, help="target server host")
parser.add_argument("--port", type=int, required=True, help="target server port")
parser.add_argument("--fps", type=int, default=30)
parser.add_argument("--width", type=int, default=1280)
parser.add_argument("--height", type=int, default=720)
parser.add_argument("--srt_opts", default="pkt_size=1316", help="SRT options query string")
parser.add_argument("--hwaccel", action="store_true", help="use hwaccel if available")
args = parser.parse_args()

ffmpeg_cmd = [
    "ffmpeg",
    "-re",
    "-i", args.input,
    "-vf", f"scale={args.width}:{args.height},fps={args.fps}",
    "-vcodec", "libx264",
    "-preset", "ultrafast",
    "-tune", "zerolatency",
    "-g", str(max(1, args.fps*2)),
    "-f", "mpegts",
    f"srt://{args.host}:{args.port}?{args.srt_opts}"
]

if args.hwaccel:
    # best-effort; user system dependent
    ffmpeg_cmd = [
        "ffmpeg", "-re", "-hwaccel", "cuda", "-hwaccel_output_format", "cuda",
        "-i", args.input,
        "-vf", f"scale_cuda={args.width}:{args.height},fps={args.fps}",
        "-c:v", "h264_nvenc",
        "-preset", "llhp",
        "-rc", "vbr_hq",
        "-f", "mpegts",
        f"srt://{args.host}:{args.port}?{args.srt_opts}"
    ]

print("Running ffmpeg:", " ".join(shlex.quote(p) for p in ffmpeg_cmd), file=sys.stderr)
proc = subprocess.Popen(ffmpeg_cmd)
try:
    proc.wait()
except KeyboardInterrupt:
    proc.terminate()
    proc.wait()
