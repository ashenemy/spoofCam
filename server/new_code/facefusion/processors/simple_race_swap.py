# simple_race_swap.py
"""
Lightweight skin-tone / race color transfer processor.

Behavior:
- If facefusion provides a target image (globals.target_path) and target contains face, we extract skin regions from target
  and transfer color distribution to source frame skin region using skimage.exposure.match_histograms (preferred).
- If no target available or skimage missing, fall back to simple HSV-scale color transfer on detected skin mask.
- This is *color transfer only* — structure/features are preserved; minimal compute and no heavy NN inference.
"""
from __future__ import annotations
import os
import numpy as np
import cv2

import facefusion
from facefusion import logger

try:
    from skimage.exposure import match_histograms
    HAS_SKIMAGE = True
except Exception:
    HAS_SKIMAGE = False

def _skin_mask_bgr(img_bgr):
    # simple heuristic skin detection in HSV
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    # broad range for skin tones
    lower = np.array([0, 30, 40], dtype=np.uint8)
    upper = np.array([25, 255, 255], dtype=np.uint8)
    mask1 = cv2.inRange(hsv, lower, upper)
    lower2 = np.array([160, 30, 40], dtype=np.uint8)
    upper2 = np.array([180, 255, 255], dtype=np.uint8)
    mask2 = cv2.inRange(hsv, lower2, upper2)
    mask = cv2.bitwise_or(mask1, mask2)
    # morphological ops to clean
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return (mask>0).astype(np.uint8)

def _hist_match_skin(src_bgr, ref_bgr):
    # expects uint8 images BGR
    try:
        if HAS_SKIMAGE:
            # convert to RGB for skimage
            src_rgb = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2RGB)
            ref_rgb = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2RGB)
            matched = match_histograms(src_rgb, ref_rgb, multichannel=True)
            out = cv2.cvtColor(np.clip(matched,0,255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            return out
    except Exception as e:
        logger.warn(f"simple_race_swap: skimage match_histograms failed: {e}")

    # fallback: per-channel mean/var scaling on HSV (cheap)
    src_hsv = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    ref_hsv = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    for ch in (0,1,2):
        s_mean, s_std = src_hsv[:,:,ch].mean(), src_hsv[:,:,ch].std() + 1e-6
        r_mean, r_std = ref_hsv[:,:,ch].mean(), ref_hsv[:,:,ch].std() + 1e-6
        src_hsv[:,:,ch] = (src_hsv[:,:,ch] - s_mean) * (r_std / s_std) + r_mean
    out = cv2.cvtColor(np.clip(src_hsv,0,255).astype(np.uint8), cv2.COLOR_HSV2BGR)
    return out

class Processor:
    name = "simple_race_swap"
    description = "Skin tone transfer from target to source (cheap)."

    def __init__(self, cfg=None):
        self.cfg = cfg or {}
        # if swap_race == 'target' we use target image skin tone; otherwise we could implement presets (not heavy)
        self.swap_race = getattr(facefusion.globals, "swap_race", None)

    def process_frame(self, frame_bgr: np.ndarray, meta: dict) -> np.ndarray:
        """
        frame_bgr: current source frame (BGR uint8)
        meta: optional metadata (not required)
        returns modified frame_bgr
        """
        try:
            # quick bailouts
            if not self.swap_race:
                return frame_bgr

            # determine reference image
            ref_path = None
            if str(self.swap_race).lower() == "target":
                ref_path = getattr(facefusion.globals, "target_path", None)
            # else: could support presets, but for now require 'target' or explicit target_path
            if not ref_path or not os.path.exists(ref_path):
                # nothing to do
                return frame_bgr

            ref_img = cv2.imread(ref_path)
            if ref_img is None:
                return frame_bgr

            # detect skin masks
            src_mask = _skin_mask_bgr(frame_bgr)
            ref_mask = _skin_mask_bgr(ref_img)

            # crop to bounding boxes to speed up
            ys, xs = np.where(src_mask)
            if ys.size == 0:
                return frame_bgr
            y0, y1 = max(0, ys.min()-10), min(frame_bgr.shape[0], ys.max()+10)
            x0, x1 = max(0, xs.min()-10), min(frame_bgr.shape[1], xs.max()+10)
            src_crop = frame_bgr[y0:y1, x0:x1]

            # ref crop: take central face-like region if ref mask empty
            ry, rx = np.where(ref_mask)
            if ry.size==0:
                # fallback: center crop of ref
                h,w = ref_img.shape[:2]
                rh0, rh1 = h//4, 3*h//4
                rw0, rw1 = w//4, 3*w//4
                ref_crop = ref_img[rh0:rh1, rw0:rw1]
            else:
                ry0, ry1 = max(0, ry.min()-10), min(ref_img.shape[0], ry.max()+10)
                rx0, rx1 = max(0, rx.min()-10), min(ref_img.shape[1], rx.max()+10)
                ref_crop = ref_img[ry0:ry1, rx0:rx1]

            # perform histogram/color transfer on src_crop using ref_crop
            matched = _hist_match_skin(src_crop, ref_crop)

            # blend only on skin mask area
            mask_crop = src_mask[y0:y1, x0:x1][:, :, np.newaxis]
            blended = (matched.astype(np.float32) * mask_crop + src_crop.astype(np.float32) * (1 - mask_crop)).astype(np.uint8)
            out = frame_bgr.copy()
            out[y0:y1, x0:x1] = blended
            return out
        except Exception as e:
            logger.warn(f"simple_race_swap exception: {e}")
            return frame_bgr