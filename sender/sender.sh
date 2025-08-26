# ./sender.sh -i input.mp4 --host 127.0.0.1 --port 9000 [--fps 30] [--width 1280] [--height 720] [--srt_opts pkt_size=1316] [--hwaccel]



fps=30
width=720
height=1280
srt_opts="pkt_size=1316&latency=50&rcvlatency=50&peerlatency=50&maxbw=0&transtype=live"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -i|--input) input="$2"; shift 2 ;;
    --stream) stream="$2"; shift 2 ;;
    *) echo "Unknown option $1"; exit 1 ;;
  esac
done

if [[ -z "$input" || -z "$stream" ]]; then
  echo "Usage: $0 -i INPUT --stream STREAM_URL"
  exit 1
fi

if [[ -f "$input" ]]; then
  ffmpeg_cmd=(ffmpeg -re -stream_loop -1 -i "$input" -vf "scale=${width}:${height},fps=${fps}" -vcodec libx264 -pix_fmt yuv420p -preset ultrafast -tune zerolatency -g $((fps*2)) -f mpegts "${stream}?${srt_opts}")
else
  ffmpeg_cmd=(ffmpeg -re -i "$input" -vf "scale=${width}:${height},fps=${fps}" -vcodec libx264 -pix_fmt yuv420p -preset ultrafast -tune zerolatency -g $((fps*2)) -f mpegts "${stream}?${srt_opts}")
fi

echo "Running ffmpeg: ${ffmpeg_cmd[*]}" >&2
"${ffmpeg_cmd[@]}"