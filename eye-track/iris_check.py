"""Iris check: how big are your eyes in the image and how many pixels does the iris really move?

    python iris_check.py

Prints numbers only (nothing is saved). Look at the LEFT, RIGHT and CENTER dots with your eyes, head still.
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

TARGETS = [("CENTER", (0.5, 0.5)), ("LEFT", (0.04, 0.5)), ("RIGHT", (0.96, 0.5)), ("CENTER", (0.5, 0.5)),
           ("TOP", (0.5, 0.06)), ("BOTTOM", (0.5, 0.94))]
SETTLE, HOLD = 1.3, 2.0


def main():
    sw, sh, _ = ec.screen_info()
    voice = Speaker()
    cap = cv2.VideoCapture(0, cv2.CAP_MSMF)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)    # force auto exposure (drivers remember manual settings)
    lm = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=ec.MODEL_PATH), running_mode=vision.RunningMode.VIDEO,
        num_faces=1, output_face_blendshapes=True, output_facial_transformation_matrixes=True))
    win = "iris check"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    voice.say("Iris check. Keep your head still and look at each dot with your eyes. Get ready.")
    t_end = time.time() + 6
    while time.time() < t_end:
        cap.read()
        img = np.full((sh, sw, 3), 32, np.uint8)
        cv2.putText(img, "Get ready: head still, move only your eyes", (sw // 2 - 420, sh // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (240, 240, 240), 2, cv2.LINE_AA)
        cv2.imshow(win, img)
        cv2.waitKey(30)

    rows, last, shape = {}, -1, None
    for label, (px, py) in TARGETS:
        voice.say(label.lower())
        t0, acc = time.time(), []
        while time.time() - t0 < SETTLE + HOLD:
            ok, fr = cap.read()
            if not ok:
                continue
            fr = cv2.flip(fr, 1)
            shape = fr.shape
            ts = max(int(time.perf_counter() * 1000), last + 1)
            last = ts
            res = lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB,
                                               data=cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)), ts)
            info = ec.extract(res, fr.shape[1], fr.shape[0])
            img = np.full((sh, sw, 3), 32, np.uint8)
            cv2.circle(img, (int(px * sw), int(py * sh)), 16, (90, 220, 120), -1, cv2.LINE_AA)
            cv2.putText(img, f"look at the {label.lower()} dot", (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (240, 240, 240), 2, cv2.LINE_AA)
            cv2.imshow(win, img)
            if cv2.waitKey(1) & 0xFF == 27:
                return
            if info is None or time.time() - t0 < SETTLE or info["blink"] > ec.BLINK_THRESH:
                continue
            gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
            sharp, bright = [], []
            for e in info["eyes"]:
                x, y = e["c"].astype(int)
                r = int(e["w"])
                crop = gray[max(y - r // 2, 0):y + r // 2, max(x - r, 0):x + r]
                if crop.size:
                    sharp.append(cv2.Laplacian(crop, cv2.CV_64F).var())
                    bright.append(crop.mean())
            acc.append(dict(ix=info["ix"], iy=info["iy"], w=np.mean([e["w"] for e in info["eyes"]]),
                            r=np.mean([e["r"] for e in info["eyes"]]),
                            iris=np.mean([e["iris"] for e in info["eyes"]], axis=0),
                            c=np.mean([e["c"] for e in info["eyes"]], axis=0),
                            sharp=np.mean(sharp) if sharp else 0, bright=np.mean(bright) if bright else 0,
                            open=info["eyes"][0]["open"], yaw=info["feat"][3], pitch=info["feat"][4]))
        rows.setdefault(label, []).append(acc)
    voice.close()
    cap.release()
    cv2.destroyAllWindows()

    def med(label, key, idx=0):
        a = [d[key] for d in rows[label][idx]]
        return np.median(a, axis=0) if a else None

    print(f"\nframe {shape[1]}x{shape[0]}")
    allf = [d for v in rows.values() for acc in v for d in acc]
    if not allf:
        print("no face tracked")
        return
    w = np.median([d["w"] for d in allf]); r = np.median([d["r"] for d in allf])
    print(f"eye width      : {w:.0f} px   (iris radius {r:.1f} px)   <- below ~60 px the iris barely moves; sit closer")
    print(f"eye region     : brightness {np.median([d['bright'] for d in allf]):.0f}/255, "
          f"sharpness {np.median([d['sharp'] for d in allf]):.0f} (low = dim or blurry)")
    print(f"eye openness   : {np.median([d['open'] for d in allf]):.2f}   (lid gap / eye width; <0.25 = eyes half closed, iris partly hidden)")
    L, R = med("LEFT", "iris"), med("RIGHT", "iris")
    cL, cR = med("LEFT", "c"), med("RIGHT", "c")
    print("\nhow far the iris moved between the LEFT and RIGHT dots:")
    print(f"  iris centre in the image : {np.linalg.norm(R - L):5.1f} px")
    print(f"  relative to eye corners  : {abs(med('RIGHT', 'ix') - med('LEFT', 'ix')) * w:5.1f} px "
          f"(= {abs(med('RIGHT', 'ix') - med('LEFT', 'ix')):.3f} of eye width)")
    print(f"  eye corners moved        : {np.linalg.norm(cR - cL):5.1f} px  (head/eye-corner shift between the two glances)")
    print(f"  head yaw change          : {np.degrees(abs(med('RIGHT', 'yaw') - med('LEFT', 'yaw'))):.1f} deg")
    T, B = med("TOP", "iy"), med("BOTTOM", "iy")
    print(f"  TOP -> BOTTOM iris-y     : {abs(B - T):.3f} of eye width ({abs(B - T) * w:.1f} px)")
    c1, c2 = med("CENTER", "ix", 0), med("CENTER", "ix", 1)
    print(f"  CENTER repeat (should match): {c1:+.3f} vs {c2:+.3f}")
    print("\nA healthy setup moves the iris about 0.15 to 0.25 of an eye width from left to right.")


if __name__ == "__main__":
    main()
