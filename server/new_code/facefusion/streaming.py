# facefusion/streaming.py
"""
Простой сегментный streaming loop для FaceFusion.

Алгоритм:
- читаем короткие сегменты (segment_seconds) из input_stream_url (ffmpeg -t segment_seconds -i input ...)
- запускаем существующий pipeline обрабоки видео (core.process_video), перенастраивая facefusion.globals.target_path -> сегмент
- полученный result (facefusion.globals.output_path) стримим в output_stream_url (ffmpeg)
- повторяем
"""
from __future__ import annotations
import os
import time
import tempfile
import subprocess
import signal
import shutil

import facefusion
import facefusion.globals
from facefusion import logger, wording
# импортим core.lazy: core содержит process_video() и вспомогательные функции
import facefusion.core as core

# default segment length в секундах — можно менять, либо брать из globals если добавим параметр
SEGMENT_SECONDS_DEFAULT = 2.0

def _run_ffmpeg_record_segment(input_url: str, out_path: str, segment_seconds: float) -> bool:
    """
    Записать сегмент длиной segment_seconds из input_url в out_path (mp4).
    Возвращает True если команда завершилась успешно.
    """
    # используем copy контейнера (быстрее), если это не работает — ffmpeg выдаст ошибку и мы увидим лог.
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-y",
        "-i", input_url,
        "-t", str(segment_seconds),
        "-c", "copy",
        out_path
    ]
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        logger.debug(f"STREAMING: ffmpeg record failed: {e}", __name__.upper())
        return False

def _run_ffmpeg_stream_out(output_file: str, output_url: str) -> bool:
    """
    Отправить готовый видеофайл output_file в output_url через ffmpeg.
    Пытаемся использовать nvenc если доступно, иначе libx264.
    Блокирующий вызов.
    """
    encoder = facefusion.globals.output_video_encoder or "libx264"
    # если пользователь указал hw encoder like h264_nvenc, используем его
    preferred = encoder
    if preferred in ["h264_nvenc", "hevc_nvenc"]:
        vcodec = preferred
        extra = ["-preset", "llhp"]
    else:
        vcodec = "libx264"
        extra = ["-preset", "ultrafast", "-tune", "zerolatency"]

    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-re",
        "-i", output_file,
        "-c:v", vcodec
    ] + extra + [
        "-f", "mpegts",
        output_url
    ]
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"STREAMING: ffmpeg stream-out failed: {e}", __name__.upper())
        return False

def run_streaming_loop(segment_seconds: float | None = None) -> None:
    """
    Главный loop: читает сегменты из input_stream_url, обрабатывает их штатной pipeline process_video()
    и стримит результат в output_stream_url. Останавливаться по Ctrl+C.
    """
    seg = float(segment_seconds) if segment_seconds else SEGMENT_SECONDS_DEFAULT
    input_url = getattr(facefusion.globals, 'input_stream_url', None)
    output_url = getattr(facefusion.globals, 'output_stream_url', None)

    if not input_url:
        logger.error(wording.get('ffmpeg_not_installed'), __name__.upper())
        raise RuntimeError("STREAMING: input_stream_url is not set")

    # временные каталоги, один на всю сессию
    tmpdir = tempfile.mkdtemp(prefix="facefusion_stream_")
    seq = 0
    logger.info(f"STREAMING: starting loop seg={seg}s input={input_url} output={output_url}", __name__.upper())
    try:
        while True:
            if getattr(facefusion.globals, "stop_streaming", False):
                logger.info("STREAMING: stop_streaming flag set, exiting loop", __name__.upper())
                break
                
            seq += 1
            segment_in = os.path.join(tmpdir, f"segment_in_{seq:06d}.mp4")
            segment_out = os.path.join(tmpdir, f"segment_out_{seq:06d}.mp4")

            # 1) Записать сегмент из входного стрима
            ok = _run_ffmpeg_record_segment(input_url, segment_in, seg)
            if not ok:
                logger.warn(f"STREAMING: failed to record segment #{seq}, retrying...", __name__.upper())
                # краткая пауза и продолжить
                time.sleep(0.5)
                continue

            # 2) Подготовка globals — указываем target_path на сегмент и output_path на ожидаемый сегмент_out
            old_target = facefusion.globals.target_path
            old_output = facefusion.globals.output_path
            try:
                facefusion.globals.target_path = segment_in
                facefusion.globals.output_path = segment_out

                # 3) Запустить существующую pipeline обработки для файла (process_video использует globals)
                logger.info(f"STREAMING: processing segment #{seq}", __name__.upper())
                # process_video() находится в core — он вернет после обработки или залогирует ошибку
                core.process_video()

                # 4) Если output_stream_url задан — стримим результат
                if output_url:
                    if os.path.exists(segment_out) and os.path.getsize(segment_out) > 0:
                        logger.info(f"STREAMING: streaming processed segment #{seq} -> {output_url}", __name__.upper())
                        _run_ffmpeg_stream_out(segment_out, output_url)
                    else:
                        logger.warn(f"STREAMING: processed output missing for segment #{seq}", __name__.upper())

            finally:
                # вернуть globals как было
                facefusion.globals.target_path = old_target
                facefusion.globals.output_path = old_output

                # удалить старые файлы чтобы не заполнять диск (оставляем последний на случай отладки)
                try:
                    if os.path.exists(segment_in):
                        os.remove(segment_in)
                    if os.path.exists(segment_out):
                        os.remove(segment_out)
                except Exception:
                    pass

            # небольшой jitter, позволяем системе отдать ресурсы (а также контролируем CPU spike)
            time.sleep(0.01)
    except KeyboardInterrupt:
        logger.info("STREAMING: interrupted by user (KeyboardInterrupt)", __name__.upper())
    except Exception as e:
        logger.error(f"STREAMING: fatal exception: {e}", __name__.upper())
        raise
    finally:
        try:
            shutil.rmtree(tmpdir)
        except Exception:
            pass
        logger.info("STREAMING: loop finished, cleaned temporary files", __name__.upper())