"""
Background video runner.

Reads a video file (or a live camera / RTSP stream), takes one frame every
`sample_seconds` of video time, and passes it to the same process function the
image endpoints use. The dashboard just keeps polling /api/state, so the view
updates as the video plays.

sync_clock=True makes the seat timers follow VIDEO time instead of wall time:
a recording where a bag sits on a desk for 30 minutes turns that desk green at
the 30-minute mark of the video, even when played at 10x.
"""
import threading
import time

import cv2


def fmt_time(sec):
    sec = int(max(0, sec))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


class VideoRunner:
    def __init__(self, process_fn, advance_clock_fn):
        """
        process_fn(frame, source_name)   -> analyzes one frame
        advance_clock_fn(seconds)        -> moves the seat clock forward
        """
        self.process_fn = process_fn
        self.advance_clock = advance_clock_fn
        self._lock = threading.Lock()
        self._thread = None
        self._stop = threading.Event()
        self._paused = threading.Event()
        self._status = {"state": "idle"}

    # --------------------------------------------------------
    def status(self):
        with self._lock:
            return dict(self._status)

    def _set(self, **kw):
        with self._lock:
            self._status.update(kw)

    def is_running(self):
        return self._thread is not None and self._thread.is_alive()

    # --------------------------------------------------------
    def start(self, source, name, live=False, sample_seconds=2.0, speed=1.0,
              sync_clock=True, loop=False):
        self.stop()

        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise ValueError(f"Could not open video source: {name}")
        if live:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        fps = cap.get(cv2.CAP_PROP_FPS) or 0
        if fps <= 1 or fps > 240:
            fps = 25.0
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        duration = frame_count / fps if (frame_count > 0 and not live) else None

        sample_seconds = max(0.2, float(sample_seconds))
        speed = max(0.1, float(speed))

        self._stop.clear()
        self._paused.clear()
        with self._lock:
            self._status = {
                "state": "playing",
                "name": name,
                "live": live,
                "fps": round(fps, 2),
                "duration_s": duration,
                "position_s": 0.0,
                "frames_analyzed": 0,
                "sample_seconds": sample_seconds,
                "speed": speed,
                "sync_clock": bool(sync_clock) and not live,
                "loop": bool(loop),
                "error": None,
            }
        self._thread = threading.Thread(
            target=self._run,
            args=(cap, name, live, fps, sample_seconds, speed,
                  bool(sync_clock) and not live, bool(loop)),
            daemon=True,
        )
        self._thread.start()

    def pause(self):
        if self.is_running():
            self._paused.set()
            self._set(state="paused")

    def resume(self):
        if self.is_running():
            self._paused.clear()
            self._set(state="playing")

    def stop(self):
        if self.is_running():
            self._stop.set()
            self._paused.clear()
            self._thread.join(timeout=10)
        self._thread = None

    # --------------------------------------------------------
    def _run(self, cap, name, live, fps, sample_seconds, speed, sync_clock, loop):
        step = max(1, round(sample_seconds * fps))    # frames between analyses
        frame_idx = 0
        live_start = time.time()
        last_vpos = last_wall = None
        final_state = "finished"

        try:
            while not self._stop.is_set():
                if self._paused.is_set():
                    self._stop.wait(0.2)
                    # don't let time spent paused count as video time
                    last_vpos = last_wall = None
                    continue

                t0 = time.time()

                if live:
                    for _ in range(3):       # drop stale buffered frames
                        cap.grab()
                    ok, frame = cap.read()
                    vpos = time.time() - live_start
                else:
                    ok, frame = cap.read()
                    vpos = frame_idx / fps

                if not ok or frame is None:
                    if live:
                        raise RuntimeError("Camera stopped sending frames")
                    if loop and frame_idx > 0:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        frame_idx = 0
                        last_vpos = last_wall = None
                        continue
                    break

                self.process_fn(frame, f"{name} @ {fmt_time(vpos)}")

                # Make the seat clock follow video time (never backwards)
                now = time.time()
                if sync_clock and last_vpos is not None and vpos > last_vpos:
                    extra = (vpos - last_vpos) - (now - last_wall)
                    if extra > 0:
                        self.advance_clock(extra)
                last_vpos, last_wall = vpos, now

                with self._lock:
                    self._status["position_s"] = vpos
                    self._status["frames_analyzed"] += 1

                # Skip ahead to the next sample (grab = no decode, fast)
                if not live:
                    for _ in range(step - 1):
                        if not cap.grab():
                            break
                    frame_idx += step

                # Pace playback: sample_seconds of video per (sample_seconds / speed) real seconds
                wait = sample_seconds / (1.0 if live else speed) - (time.time() - t0)
                if wait > 0:
                    self._stop.wait(wait)

            if self._stop.is_set():
                final_state = "stopped"
        except Exception as e:  # keep the server alive, show error on the page
            final_state = "error"
            self._set(error=str(e))
            print(f"[video] error: {e}", flush=True)
        finally:
            cap.release()
            self._set(state=final_state)
            print(f"[video] {name}: {final_state}", flush=True)
