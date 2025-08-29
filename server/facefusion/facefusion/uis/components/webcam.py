import os
import subprocess
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Deque, Generator, List, Optional

import cv2
import gradio
from tqdm import tqdm
import numpy as np
import time
import uuid
from pathlib import Path
import shlex
from subprocess import PIPE

from facefusion import ffmpeg_builder, logger, state_manager, wording
from facefusion.audio import create_empty_audio_frame
from facefusion.common_helper import is_windows
from facefusion.content_analyser import analyse_stream
from facefusion.face_analyser import get_average_face, get_many_faces
from facefusion.ffmpeg import open_ffmpeg
from facefusion.filesystem import filter_image_paths, is_directory
from facefusion.processors.core import get_processors_modules
from facefusion.types import Face, Fps, StreamMode, VisionFrame, WebcamMode
from facefusion.uis.core import get_ui_component
from facefusion.vision import normalize_frame_color, read_static_images, unpack_resolution

WEBCAM_CAPTURE : Optional[cv2.VideoCapture] = None
WEBCAM_IMAGE : Optional[gradio.Image] = None
WEBCAM_START_BUTTON : Optional[gradio.Button] = None
WEBCAM_STOP_BUTTON : Optional[gradio.Button] = None


def get_webcam_capture(webcam_device_id : int) -> Optional[cv2.VideoCapture]:
	global WEBCAM_CAPTURE

	if WEBCAM_CAPTURE is None:
		cv2.setLogLevel(0)
		if is_windows():
			webcam_capture = cv2.VideoCapture(webcam_device_id, cv2.CAP_DSHOW)
		else:
			webcam_capture = cv2.VideoCapture(webcam_device_id)
		cv2.setLogLevel(3)

		if webcam_capture and webcam_capture.isOpened():
			WEBCAM_CAPTURE = webcam_capture
	return WEBCAM_CAPTURE


def clear_webcam_capture() -> None:
	global WEBCAM_CAPTURE

	if WEBCAM_CAPTURE and WEBCAM_CAPTURE.isOpened():
		WEBCAM_CAPTURE.release()
	WEBCAM_CAPTURE = None


def render() -> None:
	global WEBCAM_IMAGE
	global WEBCAM_START_BUTTON
	global WEBCAM_STOP_BUTTON

	WEBCAM_IMAGE = gradio.Image(
		label = wording.get('uis.webcam_image')
	)
	WEBCAM_START_BUTTON = gradio.Button(
		value = wording.get('uis.start_button'),
		variant = 'primary',
		size = 'sm'
	)
	WEBCAM_STOP_BUTTON = gradio.Button(
		value = wording.get('uis.stop_button'),
		size = 'sm'
	)


def listen() -> None:
	webcam_device_id_dropdown = get_ui_component('webcam_device_id_dropdown')
	webcam_mode_radio = get_ui_component('webcam_mode_radio')
	webcam_resolution_dropdown = get_ui_component('webcam_resolution_dropdown')
	webcam_fps_slider = get_ui_component('webcam_fps_slider')
	source_image = get_ui_component('source_image')

	if webcam_device_id_dropdown and webcam_mode_radio and webcam_resolution_dropdown and webcam_fps_slider:
		start_event = WEBCAM_START_BUTTON.click(start, inputs = [ webcam_device_id_dropdown, webcam_mode_radio, webcam_resolution_dropdown, webcam_fps_slider ], outputs = WEBCAM_IMAGE)
		WEBCAM_STOP_BUTTON.click(stop, cancels = start_event, outputs = WEBCAM_IMAGE)

	if source_image:
		source_image.change(stop, cancels = start_event, outputs = WEBCAM_IMAGE)


def start(webcam_device_id : int, webcam_mode : WebcamMode, webcam_resolution : str, webcam_fps : Fps) -> Generator[VisionFrame, None, None]:
	state_manager.set_item('face_selector_mode', 'one')
	source_image_paths = filter_image_paths(state_manager.get_item('source_paths'))
	source_frames = read_static_images(source_image_paths)
	source_faces = get_many_faces(source_frames)
	source_face = get_average_face(source_faces)
	stream = None
	webcam_capture = None

	if webcam_mode in [ 'srt' ]:
		# open incoming SRT stream (as rawvideo pipe)
		stream = open_srt_input(stream_resolution=webcam_resolution, stream_fps=webcam_fps)
	webcam_width, webcam_height = unpack_resolution(webcam_resolution)

	if isinstance(webcam_device_id, int):
		webcam_capture = get_webcam_capture(webcam_device_id)

	# If we have a cv2 capture device, use existing path
	if webcam_capture and webcam_capture.isOpened():
		webcam_capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG')) #type:ignore[attr-defined]
		webcam_capture.set(cv2.CAP_PROP_FRAME_WIDTH, webcam_width)
		webcam_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, webcam_height)
		webcam_capture.set(cv2.CAP_PROP_FPS, webcam_fps)

		for capture_frame in multi_process_capture(source_face, webcam_capture, webcam_fps):
			capture_frame = normalize_frame_color(capture_frame)
			if webcam_mode == 'inline':
				yield capture_frame
			else:
				try:
					stream.stdin.write(capture_frame.tobytes())
				except Exception:
					clear_webcam_capture()
				yield None

	# If no cv2 device but we have an SRT stream opened as pipe, process frames from pipe
	if (not webcam_capture or not webcam_capture.isOpened()) and stream is not None:
		for capture_frame in multi_process_capture_from_pipe(source_face, stream, webcam_width, webcam_height, webcam_fps):
			capture_frame = normalize_frame_color(capture_frame)
			if webcam_mode == 'inline':
				yield capture_frame
			else:
				# If stream is an input SRT (we're substituting webcam), we yield None to UI
				yield None


def multi_process_capture(source_face : Face, webcam_capture : cv2.VideoCapture, webcam_fps : Fps) -> Generator[VisionFrame, None, None]:
	deque_capture_frames: Deque[VisionFrame] = deque()

	with tqdm(desc = wording.get('streaming'), unit = 'frame', disable = state_manager.get_item('log_level') in [ 'warn', 'error' ]) as progress:
		with ThreadPoolExecutor(max_workers = state_manager.get_item('execution_thread_count')) as executor:
			futures = []

			while webcam_capture and webcam_capture.isOpened():
				_, capture_frame = webcam_capture.read()
				if analyse_stream(capture_frame, webcam_fps):
					yield None
				future = executor.submit(process_stream_frame, source_face, capture_frame)
				futures.append(future)

				for future_done in [ future for future in futures if future.done() ]:
					capture_frame = future_done.result()
					deque_capture_frames.append(capture_frame)
					futures.remove(future_done)

				while deque_capture_frames:
					progress.update()
					yield deque_capture_frames.popleft()


def stop() -> gradio.Image:
	clear_webcam_capture()
	return gradio.Image(value = None)


def process_stream_frame(source_face : Face, target_vision_frame : VisionFrame) -> VisionFrame:
	source_audio_frame = create_empty_audio_frame()
	# optional frame dump directory (set env FACEFUSION_DUMP_DIR to enable)
	dump_dir = os.environ.get('FACEFUSION_DUMP_DIR')
	if dump_dir:
		try:
			Path(dump_dir).mkdir(parents=True, exist_ok=True)
		except Exception:
			dump_dir = None

	# save input frame for debugging if enabled
	if dump_dir and isinstance(target_vision_frame, (np.ndarray,)):
		try:
			in_name = f"in_{int(time.time()*1000)}_{uuid.uuid4().hex[:8]}.png"
			in_path = Path(dump_dir) / in_name
			cv2.imwrite(str(in_path), target_vision_frame)
		except Exception:
			pass

	for processor_module in get_processors_modules(state_manager.get_item('processors')):
		logger.disable()
		if processor_module.pre_process('stream'):
			# processors return processed frame
			target_vision_frame = processor_module.process_frame(
			{
				'source_face': source_face,
				'source_audio_frame': source_audio_frame,
				'target_vision_frame': target_vision_frame
			})
		logger.enable()

	# save output frame for debugging if enabled
	if dump_dir and isinstance(target_vision_frame, (np.ndarray,)):
		try:
			out_name = f"out_{int(time.time()*1000)}_{uuid.uuid4().hex[:8]}.png"
			out_path = Path(dump_dir) / out_name
			cv2.imwrite(str(out_path), target_vision_frame)
		except Exception:
			pass

	return target_vision_frame


def open_stream(stream_mode : StreamMode, stream_resolution : str, stream_fps : Fps) -> subprocess.Popen[bytes]:
	commands = ffmpeg_builder.chain(
		ffmpeg_builder.capture_video(),
		ffmpeg_builder.set_media_resolution(stream_resolution),
		ffmpeg_builder.set_input_fps(stream_fps)
	)

	if stream_mode == 'srt':
		commands.extend(ffmpeg_builder.set_input('-'))
		commands.extend(ffmpeg_builder.set_stream_mode('srt'))
		commands.extend(ffmpeg_builder.set_stream_quality(2000))
		commands.extend(ffmpeg_builder.set_output('srt://localhost:27000?pkt_size=1316'))

	return open_ffmpeg(commands)


def open_srt_input(stream_resolution: str, stream_fps: Fps, srt_uri: Optional[str] = None) -> subprocess.Popen[bytes]:
	"""Open incoming SRT stream and return ffmpeg process that outputs rawvideo to stdout (BGR24).

	If srt_uri is None, attempt to read `target_path` from state_manager or config.
	"""
	if srt_uri is None:
		# try runtime state then config
		srt_uri = state_manager.get_item('target_path') or None
		if not srt_uri:
			try:
				from facefusion import config as _config
				srt_uri = _config.get_str_value('paths', 'target_path', 'srt://0.0.0.0:27000?pkt_size=1316')
			except Exception:
				srt_uri = 'srt://0.0.0.0:27000?pkt_size=1316'
	"""Open incoming SRT stream and return ffmpeg process that outputs rawvideo to stdout (BGR24)."""
	width, height = unpack_resolution(stream_resolution)
	frame_size = width * height * 3

	cmd = [
		'ffmpeg', '-hide_banner', '-loglevel', 'error',
		'-i', srt_uri,
		'-vf', f'scale={width}:{height},fps={stream_fps}',
		'-pix_fmt', 'bgr24',
		'-f', 'rawvideo', '-'
	]

	proc = subprocess.Popen(cmd, stdout=PIPE, stderr=PIPE)
	# small wait to allow process to start
	time.sleep(0.1)
	return proc


def multi_process_capture_from_pipe(source_face: Face, proc: subprocess.Popen, width: int, height: int, webcam_fps: Fps) -> Generator[VisionFrame, None, None]:
	"""Read rawvideo frames from proc.stdout, process them in threadpool and yield processed frames."""
	frame_size = width * height * 3
	deque_capture_frames: Deque[VisionFrame] = deque()

	with tqdm(desc=wording.get('streaming'), unit='frame', disable=state_manager.get_item('log_level') in ['warn', 'error']) as progress:
		with ThreadPoolExecutor(max_workers=state_manager.get_item('execution_thread_count')) as executor:
			futures = []
			while True:
				try:
					raw = proc.stdout.read(frame_size)
					if not raw or len(raw) < frame_size:
						# attempt reconnect or break
						break
					frame = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3))
					# submit processing
					future = executor.submit(process_stream_frame, source_face, frame)
					futures.append(future)

					for future_done in [f for f in futures if f.done()]:
						capture_frame = future_done.result()
						deque_capture_frames.append(capture_frame)
						futures.remove(future_done)

					while deque_capture_frames:
						progress.update()
						yield deque_capture_frames.popleft()
				except Exception:
					break



def get_available_webcam_ids(webcam_id_start : int, webcam_id_end : int) -> List[int]:
	available_webcam_ids = []

	for index in range(webcam_id_start, webcam_id_end):
		webcam_capture = get_webcam_capture(index)

		if webcam_capture and webcam_capture.isOpened():
			available_webcam_ids.append(index)
			clear_webcam_capture()

	return available_webcam_ids
