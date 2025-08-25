# card_replacer.py
"""
Card replacer processor.
- Detects quadrilateral object (business card-like) in frame by contour detection and aspect ratio heuristics.
- Warps provided replacement image onto detected quad using perspective transform.
- If replace_card global points to a directory with front.png/back.png, we can choose front/back depending on detected color side (simple color heuristic).
"""
from __future__ import annotations
import os
import cv2
import numpy as np
import facefusion
from facefusion import logger

def _find_card_quad(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # use Canny to find edges
    edges = cv2.Canny(gray, 50, 150)
    # dilate to connect
    edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)), iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    h,w = img.shape[:2]
    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < (w*h)*0.001:  # too small
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02*peri, True)
        if len(approx) == 4:
            pts = approx.reshape(4,2)
            # compute aspect ratio
            rect = cv2.boundingRect(pts)
            rw,rh = rect[2], rect[3]
            if rw < 20 or rh < 10: continue
            ar = float(rw)/float(rh)
            if 0.5 < ar < 3.5:
                candidates.append((area, pts))
    if not candidates:
        return None
    # choose largest by area
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1].astype(np.float32)

def _order_quad(pts):
    # order points tl,tr,br,bl
    rect = np.zeros((4,2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

class Processor:
    name = "card_replacer"
    description = "Replace detected business card with provided image (perspective warp)."

    def __init__(self, cfg=None):

        self.cfg = cfg or {}
        self.replace_path = getattr(facefusion.globals, "replace_card", None)
        # load images (single image or front/back)
         self.front = None
        self.back = None
        # priority: CLI flags/replace_card param (existing) then explicit front/back globals
        # load from replace_card path if provided
        if self.replace_path:
            if os.path.isdir(self.replace_path):
                fp = os.path.join(self.replace_path, "front.png")
                bp = os.path.join(self.replace_path, "back.png")
                if os.path.exists(fp): self.front = cv2.imread(fp)
                if os.path.exists(bp): self.back = cv2.imread(bp)
            elif os.path.exists(self.replace_path):
                self.front = cv2.imread(self.replace_path)
        # override with explicit globals if set
        front_glob = getattr(facefusion.globals, "htgktqc_image_front", None)
        back_glob = getattr(facefusion.globals, "replace_image_back", None)
        try:
            if front_glob and os.path.exists(front_glob):
                self.front = cv2.imread(front_glob)
            if back_glob and os.path.exists(back_glob):
                self.back = cv2.imread(back_glob)
        except Exception:
            pass

    def process_frame(self, frame_bgr: np.ndarray, meta: dict) -> np.ndarray:
        try:
            if not self.replace_path or (self.front is None and self.back is None):
                return frame_bgr
            quad = _find_card_quad(frame_bgr)
            if quad is None:
                return frame_bgr
            quad = _order_quad(quad)
            # decide which side to use: simple color heuristic -> sample center color
            cx = int(quad[:,0].mean()); cy = int(quad[:,1].mean())
            sample = frame_bgr[max(0,cy-2):min(frame_bgr.shape[0],cy+3), max(0,cx-2):min(frame_bgr.shape[1],cx+3)]
            avg_color = sample.mean(axis=(0,1)) if sample.size else frame_bgr[cy,cx]
            # heuristic: if greenish > blueish -> treat as green side -> map to front if available
            use_front = True
            if self.front is not None and self.back is not None:
                b,g,r = avg_color
                if b > g and b > r:
                    use_front = False
                else:
                    use_front = True
            img_src = self.front if use_front else (self.back or self.front)
            if img_src is None:
                return frame_bgr
            # prepare warp
            h_src, w_src = img_src.shape[:2]
            dst_pts = quad
            src_pts = np.array([[0,0],[w_src-1,0],[w_src-1,h_src-1],[0,h_src-1]], dtype=np.float32)
            M = cv2.getPerspectiveTransform(src_pts, dst_pts)
            warped = cv2.warpPerspective(img_src, M, (frame_bgr.shape[1], frame_bgr.shape[0]), flags=cv2.INTER_LINEAR)
            # create mask of warped area
            mask = np.zeros((frame_bgr.shape[0], frame_bgr.shape[1]), dtype=np.uint8)
            cv2.fillConvexPoly(mask, dst_pts.astype(np.int32), 255)
            mask3 = cv2.merge([mask,mask,mask])
            invmask = cv2.bitwise_not(mask3)
            out = cv2.bitwise_and(frame_bgr, invmask)
            out = cv2.add(out, cv2.bitwise_and(warped, mask3))
            # small feather to blend edges
            k = int(max(3, min(frame_bgr.shape[:2]) * 0.005))
            if k % 2 == 0: k += 1
            blur = cv2.GaussianBlur(out, (k,k), 0)
            alpha = (cv2.GaussianBlur(mask.astype(np.float32)/255.0, (k,k),0))[:,:,np.newaxis]
            out = (out.astype(np.float32)*(1-alpha) + blur.astype(np.float32)*alpha).astype(np.uint8)
            return out
        except Exception as e:
            logger.warn(f"card_replacer exception: {e}")
            return frame_bgr