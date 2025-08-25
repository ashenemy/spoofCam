# facefusion/uis/layouts/stream.py
"""
Stream UI layout for FaceFusion.
Integrates settings from jobs/webcam + our streaming fields:
  - input_stream_url
  - output_stream_url
  - playback_url (for in-browser player)
Uses the repo's UI layout contract: pre_render(), render(ui), listen()
"""
from __future__ import annotations
import threading
import time
from typing import Any

import gradio as gr

import facefusion
import facefusion.globals as globals_
import facefusion.core as core
import facefusion.streaming as streaming
from facefusion.uis import helpers  # many layouts use helpers; keep if present

LAYOUT_NAME = "stream"

# state for the background worker thread
_stream_thread: threading.Thread | None = None

def pre_render() -> bool:
    # called by UI loader to decide whether to load this layout
    return True

def _start_stream_in_thread():
    global _stream_thread
    if _stream_thread and _stream_thread.is_alive():
        return False
    def _target():
        try:
            # call existing pipeline entrypoint; it will detect input_stream_url in globals
            core.conditional_process()
        except Exception as e:
            # rely on existing logging; ensure stop flag cleared on exit
            globals_.stop_streaming = False
            raise
    globals_.stop_streaming = False
    _stream_thread = threading.Thread(target=_target, daemon=True)
    _stream_thread.start()
    return True

def _stop_stream_thread():
    # signal streaming loop to stop; streaming.run_streaming_loop checks stop_streaming flag
    globals_.stop_streaming = True
    # also attempt to interrupt the background thread politely
    global _stream_thread
    if _stream_thread:
        # wait short time for thread to finish
        _stream_thread.join(timeout=5.0)
        if _stream_thread.is_alive():
            # leave it — streaming loop should check flag and exit soon
            return False
    return True

def render(ui: gr.Blocks) -> None:
    """
    Build UI. This follows the repo conventions used by other layouts.
    The layout places controls on left and preview/player on right.
    """
    with ui:
        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("### Stream — configuration (merged from Jobs + Webcam)")
                # Paths & general
                temp_path = gr.Textbox(label="temp_path", placeholder="/tmp/facefusion", value=globals_.temp_path or "")
                jobs_path = gr.Textbox(label="jobs_path", placeholder="", value=globals_.jobs_path or "")
                target_path = gr.Textbox(label="target_path (fallback)", placeholder="", value=globals_.target_path or "")
                output_path = gr.Textbox(label="output_path (fallback)", placeholder="", value=globals_.output_path or "")

                # Execution / resources
                execution_device_id = gr.Textbox(label="execution_device_id", placeholder="e.g. 0,1 or GPU list", value=str(getattr(globals_, "execution_device_id", "") or ""))
                execution_providers = gr.Textbox(label="execution_providers", placeholder="cuda/cpu/onnx", value=str(getattr(globals_, "execution_providers", "") or "cuda"))
                execution_thread_count = gr.Number(label="execution_thread_count", value=int(getattr(globals_, "execution_thread_count", 4)))
                execution_queue_count = gr.Number(label="execution_queue_count", value=int(getattr(globals_, "execution_queue_count", 4)))

                # Face detector / landmarker quick settings (merged)
                face_detector_model = gr.Textbox(label="face_detector_model", value=str(getattr(globals_, "face_detector_model", "") or ""))
                face_detector_size = gr.Number(label="face_detector_size", value=int(getattr(globals_, "face_detector_size", 512)))
                face_landmarker_model = gr.Textbox(label="face_landmarker_model", value=str(getattr(globals_, "face_landmarker_model", "") or ""))

                # Our streaming-specific fields
                input_stream_url = gr.Textbox(label="Input stream URL (input_stream_url)", placeholder="srt://server:port or rtsp://...", value=str(getattr(globals_, "input_stream_url", "") or ""))
                output_stream_url = gr.Textbox(label="Output stream URL (output_stream_url)", placeholder="srt://server:port or udp://...", value=str(getattr(globals_, "output_stream_url", "") or ""))
                segment_seconds = gr.Number(label="Segment length (seconds)", value=float(getattr(globals_, "segment_seconds", 2.0)))
                playback_url = gr.Textbox(label="Playback URL (for in-browser preview, e.g. HLS .m3u8)", value=str(getattr(globals_, "playback_url", "") or ""))

                # buttons
                with gr.Row():
                    start_btn = gr.Button("Start streaming", variant="primary")
                    stop_btn = gr.Button("Stop streaming", variant="stop")
                    save_cfg = gr.Button("Save config")
                    load_cfg = gr.File(label="Load config (json)")

                # hidden status
                status = gr.Textbox(label="Status", interactive=False, value="idle")

            with gr.Column(scale=1):
                gr.Markdown("### Output player / preview")
                video_player = gr.Video(value="", label="Output player (set Playback URL)", show_label=True, interactive=False)
                gr.Markdown("If your output stream is not HTTP/HLS it may not be playable in the browser. Use `playback_url` to point to an HLS/HTTP stream.")

    # listeners for interaction wired by listen()
    # return references so listen() can access them
    return {
        "temp_path": temp_path,
        "jobs_path": jobs_path,
        "target_path": target_path,
        "output_path": output_path,
        "execution_device_id": execution_device_id,
        "execution_providers": execution_providers,
        "execution_thread_count": execution_thread_count,
        "execution_queue_count": execution_queue_count,
        "face_detector_model": face_detector_model,
        "face_detector_size": face_detector_size,
        "face_landmarker_model": face_landmarker_model,
        "input_stream_url": input_stream_url,
        "output_stream_url": output_stream_url,
        "segment_seconds": segment_seconds,
        "playback_url": playback_url,
        "start_btn": start_btn,
        "stop_btn": stop_btn,
        "save_cfg": save_cfg,
        "load_cfg": load_cfg,
        "status": status,
        "video_player": video_player
    }

def listen(refs: dict[str, Any]) -> None:
    """
    Wire up controls to actions.
    This function is executed after render() by the UI loader (see other layouts).
    """
    temp_path = refs["temp_path"]
    jobs_path = refs["jobs_path"]
    target_path = refs["target_path"]
    output_path = refs["output_path"]
    execution_device_id = refs["execution_device_id"]
    execution_providers = refs["execution_providers"]
    execution_thread_count = refs["execution_thread_count"]
    execution_queue_count = refs["execution_queue_count"]
    face_detector_model = refs["face_detector_model"]
    face_detector_size = refs["face_detector_size"]
    face_landmarker_model = refs["face_landmarker_model"]
    input_stream_url = refs["input_stream_url"]
    output_stream_url = refs["output_stream_url"]
    segment_seconds = refs["segment_seconds"]
    playback_url = refs["playback_url"]
    start_btn = refs["start_btn"]
    stop_btn = refs["stop_btn"]
    save_cfg = refs["save_cfg"]
    load_cfg = refs["load_cfg"]
    status = refs["status"]
    video_player = refs["video_player"]

    def _save_cfg_action(_):
        # gather form into a dict and prompt download via UI helper if present
        cfg = {
            "paths": {"temp_path": temp_path.value, "jobs_path": jobs_path.value, "target_path": target_path.value, "output_path": output_path.value},
            "execution": {"execution_device_id": execution_device_id.value, "execution_providers": execution_providers.value, "execution_thread_count": execution_thread_count.value, "execution_queue_count": execution_queue_count.value},
            "face_detector": {"face_detector_model": face_detector_model.value, "face_detector_size": face_detector_size.value, "face_landmarker_model": face_landmarker_model.value},
            "streaming": {"input_stream_url": input_stream_url.value, "output_stream_url": output_stream_url.value, "segment_seconds": segment_seconds.value, "playback_url": playback_url.value}
        }
        # helper.save_config exists in other layouts; if not, fallback to file write via python
        try:
            helpers.save_config(cfg)
        except Exception:
            # best-effort: write to temp file and open file picker
            import json, tempfile, os
            p = tempfile.mktemp(suffix=".json")
            with open(p, "w") as f:
                json.dump(cfg, f, indent=2)
            return p

    def _load_cfg_action(file_obj):
        # file_obj is a temporary upload object that Gradio provides
        try:
            import json
            data = json.load(file_obj)
            # set UI fields if present
            if "paths" in data:
                temp_path.value = data["paths"].get("temp_path", temp_path.value)
                jobs_path.value = data["paths"].get("jobs_path", jobs_path.value)
                target_path.value = data["paths"].get("target_path", target_path.value)
                output_path.value = data["paths"].get("output_path", output_path.value)
            if "execution" in data:
                execution_device_id.value = data["execution"].get("execution_device_id", execution_device_id.value)
                execution_providers.value = data["execution"].get("execution_providers", execution_providers.value)
                execution_thread_count.value = data["execution"].get("execution_thread_count", execution_thread_count.value)
                execution_queue_count.value = data["execution"].get("execution_queue_count", execution_queue_count.value)
            if "streaming" in data:
                input_stream_url.value = data["streaming"].get("input_stream_url", input_stream_url.value)
                output_stream_url.value = data["streaming"].get("output_stream_url", output_stream_url.value)
                segment_seconds.value = data["streaming"].get("segment_seconds", segment_seconds.value)
                playback_url.value = data["streaming"].get("playback_url", playback_url.value)
        except Exception as e:
            status.value = f"load failed: {e}"

    def _start_action():
        # write settings into facefusion.globals
        globals_.temp_path = temp_path.value or globals_.temp_path
        globals_.jobs_path = jobs_path.value or globals_.jobs_path
        globals_.target_path = target_path.value or globals_.target_path
        globals_.output_path = output_path.value or globals_.output_path

        globals_.execution_device_id = execution_device_id.value
        globals_.execution_providers = execution_providers.value
        globals_.execution_thread_count = int(execution_thread_count.value or 4)
        globals_.execution_queue_count = int(execution_queue_count.value or 4)

        globals_.face_detector_model = face_detector_model.value or getattr(globals_, "face_detector_model", "")
        globals_.face_detector_size = int(face_detector_size.value or getattr(globals_, "face_detector_size", 512))
        globals_.face_landmarker_model = face_landmarker_model.value or getattr(globals_, "face_landmarker_model", "")

        # streaming specific
        globals_.input_stream_url = input_stream_url.value.strip() or None
        globals_.output_stream_url = output_stream_url.value.strip() or None
        globals_.segment_seconds = float(segment_seconds.value or 2.0)
        globals_.playback_url = playback_url.value.strip() or None

        # update UI player if playback_url provided
        if globals_.playback_url:
            try:
                video_player.value = globals_.playback_url
            except Exception:
                pass

        # start background pipeline
        ok = _start_stream_in_thread()
        if ok:
            status.value = "running"
        else:
            status.value = "already running"

    def _stop_action():
        ok = _stop_stream_thread()
        if ok:
            status.value = "stopping"
        else:
            status.value = "stop requested; waiting"

    # wire up
    start_btn.click(fn=_start_action, inputs=[], outputs=[status])
    stop_btn.click(fn=_stop_action, inputs=[], outputs=[status])
    save_cfg.click(fn=_save_cfg_action, inputs=[], outputs=[])
    load_cfg.upload(fn=_load_cfg_action, inputs=[load_cfg], outputs=[status])