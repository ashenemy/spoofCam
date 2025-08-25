
from functools import lru_cache
from facefusion.types import Fps, VisionFrame
STREAM_COUNTER = 0


@lru_cache(maxsize = None)

def create_static_model_set(download_scope):
	return {}

def get_inference_pool():
	return {}

def clear_inference_pool():
	pass

def resolve_execution_providers():
	return ['cpu']

def collect_model_downloads():
	return {}, {}

def pre_check():
	return True

def analyse_stream(vision_frame: VisionFrame, video_fps: Fps) -> bool:
	return False

def analyse_frame(vision_frame: VisionFrame) -> bool:
	return False

@lru_cache(maxsize=None)
def analyse_image(image_path: str) -> bool:
	return False

@lru_cache(maxsize=None)
def analyse_video(video_path: str, trim_frame_start: int, trim_frame_end: int) -> bool:
	return False
