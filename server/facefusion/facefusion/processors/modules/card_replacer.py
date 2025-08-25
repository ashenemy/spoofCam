from argparse import ArgumentParser
from typing import List, Optional
import os

import cv2
import numpy as np


import onnxruntime as ort
from facefusion.filesystem import resolve_relative_path

from facefusion.processors.typing import ExampleInputs
from facefusion.typing import ApplyStateItem, Args, Face, FaceSet, InferencePool, ProcessMode, UpdateProgress, VisionFrame


card_front_path: Optional[str] = None
card_back_path: Optional[str] = None

# #00FF00 / #0000FF
def segment_card(img: np.ndarray) -> np.ndarray:
    model_path = resolve_relative_path('../../.assets/models/U-2-Net.onnx')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"U^2-Net ONNX model not found at {model_path}")
    session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    img_resized = cv2.resize(img, (320, 320))
    img_norm = img_resized.astype(np.float32) / 255.0
    img_norm = (img_norm - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    img_input = np.transpose(img_norm, (2, 0, 1))[None, ...]  # (1, 3, 320, 320)
    input_name = session.get_inputs()[0].name
    mask = session.run(None, {input_name: img_input})[0]
    mask = mask[0, 0]  # (320, 320)
    mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
    mask = (mask > 0.5).astype(np.uint8) * 255
    return mask

def register_args(program: ArgumentParser) -> None:
    program.add_argument('--card-front', type=str, help='Path to image for green card replacement')
    program.add_argument('--card-back', type=str, help='Path to image for blue card replacement')

def apply_args(args: Args, apply_state_item: ApplyStateItem) -> None:
    global card_front_path, card_back_path
    card_front_path = args.get('card_front')
    card_back_path = args.get('card_back')

def pre_check() -> bool:
    return True

def pre_process(mode: ProcessMode) -> bool:
    return True

def post_process() -> None:
    pass

def get_inference_pool() -> InferencePool:
    return {}

def clear_inference_pool() -> None:
    pass

def detect_card_color(img: np.ndarray, mask: np.ndarray) -> Optional[str]:
    mean_color = cv2.mean(img, mask=mask)[:3]
    b, g, r = mean_color
    if g > 1.2 * r and g > 1.2 * b:
        return 'green'
    if b > 1.2 * r and b > 1.2 * g:
        return 'blue'
    return None

def replace_card(img: np.ndarray, card_img_path: str, mask: np.ndarray) -> np.ndarray:
    card_img = cv2.imread(card_img_path)
    if card_img is None:
        return img
    x, y, w, h = cv2.boundingRect(mask)
    card_resized = cv2.resize(card_img, (w, h))
    center = (x + w // 2, y + h // 2)
    mask3 = np.zeros_like(img, dtype=np.uint8)
    mask3[y:y+h, x:x+w] = cv2.merge([mask[y:y+h, x:x+w]]*3)
    mixed = cv2.seamlessClone(card_resized, img, mask3[:,:,0], center, cv2.NORMAL_CLONE)
    return mixed

def process_frame(inputs: ExampleInputs) -> VisionFrame:
    global card_front_path, card_back_path
    frame = inputs['target_vision_frame']
    if not card_front_path and not card_back_path:
        return frame
    img = frame.copy()
    mask = segment_card(img)
    card_color = detect_card_color(img, mask)
    if card_color == 'green' and card_front_path:
        img = replace_card(img, card_front_path, mask)
    elif card_color == 'blue' and card_back_path:
        img = replace_card(img, card_back_path, mask)
    return img

def process_frames(source_path: str, temp_frame_paths: List[str], update_progress: UpdateProgress) -> None:
    for i, frame_path in enumerate(temp_frame_paths):
        img = cv2.imread(frame_path)
        mask = segment_card(img)
        card_color = detect_card_color(img, mask)
        if card_color == 'green' and card_front_path:
            img = replace_card(img, card_front_path, mask)
        elif card_color == 'blue' and card_back_path:
            img = replace_card(img, card_back_path, mask)
        cv2.imwrite(frame_path, img)
        update_progress(i + 1, len(temp_frame_paths))

def process_image(source_path: str, target_path: str, output_path: str) -> None:
    img = cv2.imread(target_path)
    mask = segment_card(img)
    card_color = detect_card_color(img, mask)
    if card_color == 'green' and card_front_path:
        img = replace_card(img, card_front_path, mask)
    elif card_color == 'blue' and card_back_path:
        img = replace_card(img, card_back_path, mask)
    cv2.imwrite(output_path, img)

def process_video(source_path: str, temp_frame_paths: List[str]) -> None:
    for frame_path in temp_frame_paths:
        img = cv2.imread(frame_path)
        mask = segment_card(img)
        card_color = detect_card_color(img, mask)
        if card_color == 'green' and card_front_path:
            img = replace_card(img, card_front_path, mask)
        elif card_color == 'blue' and card_back_path:
            img = replace_card(img, card_back_path, mask)
        cv2.imwrite(frame_path, img)