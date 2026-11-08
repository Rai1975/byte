"""Camera check: how much does the iris signal move when you look at the screen edges, per camera backend?

    python camera_check.py

Why: Windows camera pipelines can alter the picture (Windows Studio Effects "Eye contact" re-renders your
eyes to look at the camera, "Automatic framing" zooms, etc.). That shrinks the iris movement the tracker
relies on. This script records the same four glances through each capture mode and prints the signal
size ("span") and noise, so you can pick the mode that keeps the signal. Keep your head still and look
only with your eyes at the dot (it is read aloud). Takes about a minute. ESC aborts.
"""

import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import time

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision

import eye_cursor as ec
from speech import Speaker

MODES = [("DSHOW 1280x720", cv2.CAP_DSHOW, 1280, 720), ("MSMF 1280x720", cv2.CAP_MSMF, 1280, 720),
         ("DSHOW 640x480", cv2.CAP_DSHOW, 640, 480), ("MSMF 640x480", cv2.CAP_MSMF, 640, 480)]
TARGETS = [("LEFT", (0.04, 0.5)), ("RIGHT", (0.96, 0.5)), ("TOP", (0.5, 0.06)), ("BOTTOM", (0.5, 0.94)),
           ("CENTER", (0.5, 0.5))]
SETTLE, HOLD = 1.2, 1.5


def main():
    sw, sh, _ = ec.screen_info()
    voice = Speaker()
    opts = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=ec.MODEL_PATH), running_mode=vision.RunningMode.VIDEO,
        num_faces=1, output_face_blendshapes=True, output_facial_transformation_matrixes=True)
    results = []
    win = "camera check"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    voice.say("Camera check. It runs every mode twice, so it takes about three minutes. Keep your head still and move only your eyes. Get ready.")
    t_ready = time.time() + 7
    while time.time() < t_ready:
        img = np.full((sh, sw, 3), 32, np.uint8)
        cv2.putText(img, "Get ready: head still, move only your eyes", (sw // 2 - 420, sh // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (240, 240, 240), 2, cv2.LINE_AA)
        cv2.imshow(win, img)
        cv2.waitKey(30)
    for name, backend, w, h in MODES + MODES[::-1]:       # second pass in reverse order: cancels order/attention effects
        cap = cv2.VideoCapture(0, backend)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)    # force auto exposure (drivers remember manual settings)
        if not (cap.isOpened() and cap.read()[0]):
            print(f"{name}: could not open")
            cap.release()
            continue
        lm = vision.FaceLandmarker.create_from_options(opts)
        voice.say(f"Mode {name}. Keep your head still and look at the dot with your eyes only.")
        t_end = time.time() + 4
        while time.time() < t_end:
            cap.read()
        per_target, n_frames, t_start, last = {}, 0, time.time(), -1
        for label, (px, py) in TARGETS:
            voice.say(label.lower())
            feats, t0 = [], time.time()
            while time.time() - t0 < SETTLE + HOLD:
                ok, fr = cap.read()
                if not ok:
                    continue
                fr = cv2.flip(fr, 1)
                ts = max(int(time.perf_counter() * 1000), last + 1)
                last = ts
                res = lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB,
                                                   data=cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)), ts)
                info = ec.extract(res, fr.shape[1], fr.shape[0])
                n_frames += 1
                if info is not None and time.time() - t0 > SETTLE and info["blink"] < ec.BLINK_THRESH:
                    feats.append(info["feat"])
                img = np.full((sh, sw, 3), 32, np.uint8)
                cv2.circle(img, (int(px * sw), int(py * sh)), 16, (90, 220, 120), -1, cv2.LINE_AA)
                cv2.putText(img, f"{name}: look at the {label.lower()} dot", (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                            (240, 240, 240), 2, cv2.LINE_AA)
                cv2.imshow(win, img)
                if cv2.waitKey(1) & 0xFF == 27:
                    voice.close()
                    cv2.destroyAllWindows()
                    return
            per_target[label] = np.array(feats)
        fps = n_frames / (time.time() - t_start)
        cap.release()
        lm.close()
        if any(len(v) < 5 for v in per_target.values()):
            print(f"{name}: face not tracked well enough")
            continue
        med = {k: np.median(v, axis=0) for k, v in per_target.items()}
        noise = np.mean([v[:, :2].std(0) for v in per_target.values()], axis=0)
        span_x = med["RIGHT"][0] - med["LEFT"][0]
        span_y = med["BOTTOM"][1] - med["TOP"][1]
        results.append((name, fps, span_x, span_y, noise, med["RIGHT"][9] - med["LEFT"][9] if len(med["RIGHT"]) > 9 else 0))
    voice.close()
    cv2.destroyAllWindows()

    print("\nsignal = how far the iris offset moves between edges (bigger is better); noise = frame-to-frame sd")
    print(f"{'mode':18s}{'fps':>6s}{'x span':>9s}{'y span':>9s}{'noise x':>9s}{'noise y':>9s}{'x SNR':>8s}{'y SNR':>8s}")
    for name, fps, sx, sy, nz, _ in results:
        print(f"{name:18s}{fps:6.0f}{sx:9.3f}{sy:9.3f}{nz[0]:9.4f}{nz[1]:9.4f}{abs(sx) / nz[0]:8.1f}{abs(sy) / nz[1]:8.1f}")
    print("\naverage over the two passes:")
    for name, *_ in MODES:
        rr = [r for r in results if r[0] == name]
        if rr:
            sx, sy = np.mean([r[2] for r in rr]), np.mean([r[3] for r in rr])
            nx, ny = np.mean([r[4][0] for r in rr]), np.mean([r[4][1] for r in rr])
            print(f"{name:18s}{np.mean([r[1] for r in rr]):6.0f}{sx:9.3f}{sy:9.3f}{nx:9.4f}{ny:9.4f}{abs(sx) / nx:8.1f}{abs(sy) / ny:8.1f}")
    print("\nA mode whose spans are several times smaller than the others is altering the image "
          "(e.g. an eye-contact effect). Use the mode with the largest SNR.")


if __name__ == "__main__":
    main()
