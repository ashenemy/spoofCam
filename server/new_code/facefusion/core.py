# NEW STREAM FLAGS: allow reading input stream URL and writing processed output stream URL
group_frame_extraction.add_argument(
    '--input-stream-url',
    help = 'Input live stream URL (e.g. srt://, rtsp://, udp://). When supplied, will be used as the target input.',
    dest = 'input_stream_url'
)

group_output_creation.add_argument(
    '--output-stream-url',
    help = 'Output live stream URL (e.g. srt://). When supplied, processed frames will be sent to this URL.',
    dest = 'output_stream_url'
)

parser.add_argument(
    '--swap-race',
    help='If set, enable race color transfer processor. Accepts values like "black","white","asian","target" (use target image skin tones).',
    dest='swap_race',
    default=None
)
parser.add_argument(
    '--htgktqc-image-front',
    help='Path to replacement front image (business card front) or uploaded file.',
    dest='htgktqc_image_front',
    default=None
)
parser.add_argument(
    '--replace-image-back',
    help='Path to replacement back image (business card back) or uploaded file.',
    dest='replace_image_back',
    default=None
)

#after 
#facefusion.globals.target_path = args.target_path
#facefusion.globals.output_path = normalize_output_path(...)
# If user provided a live input stream URL, use it as the target_path for processing
# This allows existing extract_frames / ffmpeg paths to accept the stream URL transparently.
facefusion.globals.input_stream_url = getattr(args, 'input_stream_url', None)
if facefusion.globals.input_stream_url:
    # override target_path to point to the input stream (e.g. srt://..., rtsp://..., udp://...)
    facefusion.globals.target_path = facefusion.globals.input_stream_url

# If user provided an output stream URL, expose it and (if not set) use as output_path
facefusion.globals.output_stream_url = getattr(args, 'output_stream_url', None)
if facefusion.globals.output_stream_url:
    # override output_path so downstream merge/encoding logic can use it
    facefusion.globals.output_path = facefusion.globals.output_stream_url
if facefusion.globals.swap_race:
    facefusion.globals.processors.append("simple_race_swap")
if facefusion.globals.swap_gender:
    facefusion.globals.processors.append("gender_refiner")
if facefusion.globals.replace_card:
    facefusion.globals.processors.append("card_replacer")

facefusion.globals.htgktqc_image_front = getattr(args, 'htgktqc_image_front', None)
facefusion.globals.replace_image_back = getattr(args, 'replace_image_back', None)

#after
# def conditional_process() -> None:
    # --- STREAM MODE: если задан input_stream_url или output_stream_url, перейти в потоковый режим ---
    # Вынесено сюда минимально-инвазивно: мы запускаем сегментный цикл обработки (read -> process_video -> stream out)
    if getattr(facefusion.globals, 'input_stream_url', None) or getattr(facefusion.globals, 'output_stream_url', None):
        try:
            # импорт локально, чтобы избежать циклических импортов при обычном запуске
            import facefusion.streaming as streaming
        except Exception:
            logger.error("STREAMING: failed to import facefusion.streaming", __name__.upper())
            raise
        # Вызов блокирующего streaming loop; он завершится по Ctrl+C или внутренней ошибке
        streaming.run_streaming_loop()
        return



        # where processors are instantiated (pseudocode area in core pipeline)
from facefusion.processors import REGISTERED_PROCESSORS

active_processors = []
for name in getattr(facefusion.globals, "processors", []):
    cls = REGISTERED_PROCESSORS.get(name)
    if cls:
        try:
            active_processors.append(cls(cfg=facefusion.globals))
        except Exception:
            logger.warn(f"processor {name} failed to init")
# then for each frame in pipeline:
# for p in active_processors: frame = p.process_frame(frame, meta)

# ensure gender_refiner runs before face_swapper (remove stubble before swap)
proc_list = list(getattr(facefusion.globals, "processors", []) or [])
if "face_swapper" in proc_list and "gender_refiner" not in proc_list:
    idx = proc_list.index("face_swapper")
    proc_list.insert(idx, "gender_refiner")
facefusion.globals.processors = proc_list


#pip install opencv-python-headless numpy scikit-image
#pip install opencv-python-headless numpy scikit-image


import torch
import torch_tensorrt

# model — PyTorch модель (в eval режиме)
model.eval().cuda()
example_inputs = [torch.randn(1, 3, 256, 256).cuda()]  # shape — под твою модель

trt_mod = torch_tensorrt.compile(model,
                                 inputs=example_inputs,
                                 enabled_precisions={torch.float16}) # FP16

# теперь trt_mod можно использовать как обычный модуль
with torch.no_grad():
    out = trt_mod(example_inputs[0])

    trtexec --onnx=model.onnx --saveEngine=model.trt --fp16


import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
TRT_LOGGER = trt.Logger(trt.Logger.INFO)

def load_engine(engine_path):
    with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

engine = load_engine("model.trt")
context = engine.create_execution_context()

# prepare buffers (simplified)
# use pycuda to allocate device memory, copy inputs, run context.execute_v2(bindings)