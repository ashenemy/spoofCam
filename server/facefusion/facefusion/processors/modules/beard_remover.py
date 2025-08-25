from argparse import ArgumentParser
from typing import List, TypedDict

import cv2
import numpy as np

from facefusion.processors.typing import ExampleInputs
from facefusion.typing import ApplyStateItem, Args, Face, FaceSet, InferencePool, ProcessMode, UpdateProgress, VisionFrame

# --- Функция удаления щетины (размытие нижней части лица) ---
def remove_beard_from_face(frame: VisionFrame, face: Face = None) -> VisionFrame:
    img = frame.copy()
    h, w = img.shape[:2]
    # Если есть координаты лица, размываем только нижнюю часть
    if face and hasattr(face, 'landmarks'):
        # Пример: размываем область ниже носа
        nose_y = int(np.mean([pt[1] for pt in face.landmarks[27:36]]))  # 27-35 — нос
        beard_area = img[nose_y:h, :]
        beard_area = cv2.GaussianBlur(beard_area, (31, 31), 0)
        img[nose_y:h, :] = beard_area
    else:
        # Если нет разметки — размываем нижнюю треть
        img[h//2:h, :] = cv2.GaussianBlur(img[h//2:h, :], (31, 31), 0)
    return img

# --- Шаблонные функции процессора ---
def get_inference_pool() -> InferencePool:
    return {}

def clear_inference_pool() -> None:
    pass

def register_args(program: ArgumentParser) -> None:
    pass

def apply_args(args: Args, apply_state_item: ApplyStateItem) -> None:
    pass

def pre_check() -> bool:
    return True

def pre_process(mode: ProcessMode) -> bool:
    return True

def post_process() -> None:
    pass

def get_reference_frame(source_face: Face, target_face: Face, temp_vision_frame: VisionFrame) -> VisionFrame:
    # Удаляем щетину с target_vision_frame
    return remove_beard_from_face(temp_vision_frame, target_face)

def process_frame(inputs: ExampleInputs) -> VisionFrame:
    # Удаляем щетину с target_vision_frame
    return remove_beard_from_face(inputs['target_vision_frame'], inputs.get('target_face'))

def process_frames(source_path: str, temp_frame_paths: List[str], update_progress: UpdateProgress) -> None:
    for i, frame_path in enumerate(temp_frame_paths):
        img = cv2.imread(frame_path)
        img = remove_beard_from_face(img)
        cv2.imwrite(frame_path, img)
        update_progress(i + 1, len(temp_frame_paths))

def process_image(source_path: str, target_path: str, output_path: str) -> None:
    img = cv2.imread(target_path)
    img = remove_beard_from_face(img)
    cv2.imwrite(output_path, img)

def process_video(source_path: str, temp_frame_paths: List[str]) -> None:
    for frame_path in temp_frame_paths:
        img = cv2.imread(frame_path)
        img = remove_beard_from_face(img)
        cv2.imwrite(frame_path, img)