"""
Dumper Driver Fatigue Monitor - PC prototype
---------------------------------------------
Detects: eye closure (EAR), PERCLOS, yawning, head nodding.
Alert levels:
  L1 - soft beep (early signs)
  L2 - loud beep (eyes closed > 1.5 s)
  L3 - simulated fleet/control-room alert (logged to fleet_alerts.csv)

Install:  pip install opencv-python mediapipe numpy
          (use Python 3.8 - 3.12)
Run:      python driver_fatigue_monitor.py
Keys:     q = quit, c = recalibrate (look at camera, eyes open, for 5 s)
"""

import csv
import os
import platform
import threading
import time
from collections import deque

import cv2
import mediapipe as mp
import numpy as np

try:
    import serial          # pip install pyserial
except ImportError:
    serial = None

# ----------------------------- CONFIG ---------------------------------
SERIAL_PORT = "COM7"       # Windows: COM3, COM4...  Linux: /dev/ttyUSB0  Mac: /dev/cu.usbserial-*
BAUD_RATE = 9600           # set to None to run without Arduino
TRUCK_ID = "DUMPER-07"
DRIVER_ID = "DRV-101"
GPS = (21.1458, 79.0882)           # dummy coordinates (replace with GPS later)
CAMERA_INDEX = 0

CALIBRATION_SECONDS = 5
EAR_RATIO = 0.72                   # closed if EAR < baseline * this
EYE_CLOSED_L2_SEC = 1.5            # loud alarm
EYE_CLOSED_L3_SEC = 3.0            # fleet alert
PERCLOS_WINDOW_SEC = 60
PERCLOS_L1 = 0.15                  # 15 % closed in the last minute
PERCLOS_L3 = 0.30

MAR_YAWN = 0.60                    # mouth aspect ratio for yawn
YAWN_MIN_SEC = 1.5
NOD_DELTA = 0.18                   # head pitch change vs baseline
NOD_MIN_SEC = 1.0

FLEET_ALERT_COOLDOWN = 15          # seconds between fleet alerts
LOG_FILE = "fleet_alerts.csv"

# MediaPipe landmark indices
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
MOUTH = {"top": 13, "bottom": 14, "left": 78, "right": 308}
NOSE, CHIN, L_EYE_OUT, R_EYE_OUT = 1, 152, 263, 33


# ----------------------------- HELPERS --------------------------------
def dist(a, b):
    return float(np.linalg.norm(np.array(a) - np.array(b)))


def eye_aspect_ratio(pts):
    p1, p2, p3, p4, p5, p6 = pts
    return (dist(p2, p6) + dist(p3, p5)) / (2.0 * dist(p1, p4) + 1e-6)


def mouth_aspect_ratio(lm):
    v = dist(lm[MOUTH["top"]], lm[MOUTH["bottom"]])
    h = dist(lm[MOUTH["left"]], lm[MOUTH["right"]])
    return v / (h + 1e-6)


def head_pitch_proxy(lm):
    """Ratio that grows when the head tilts down (nose moves toward chin)."""
    eye_mid_y = (lm[L_EYE_OUT][1] + lm[R_EYE_OUT][1]) / 2.0
    return (lm[NOSE][1] - eye_mid_y) / (lm[CHIN][1] - eye_mid_y + 1e-6)


_last_beep = 0.0


def beep(level):
    """Non-blocking beep. Windows uses winsound, others use terminal bell."""
    global _last_beep
    now = time.time()
    gap = 1.5 if level == 1 else 0.6
    if now - _last_beep < gap:
        return
    _last_beep = now

    def _play():
        if platform.system() == "Windows":
            import winsound
            winsound.Beep(1000 if level == 1 else 2500, 200 if level == 1 else 500)
        else:
            print("\a", end="", flush=True)

    threading.Thread(target=_play, daemon=True).start()


def send_fleet_alert(reason, score):
    """Placeholder for the control-room alert. Replace with MQTT/HTTP/SMS later."""
    new_file = not os.path.exists(LOG_FILE)
    with open(LOG_FILE, "a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["time", "truck", "driver", "lat", "lon", "reason", "fatigue_score"])
        w.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), TRUCK_ID, DRIVER_ID,
                    GPS[0], GPS[1], reason, round(score, 1)])
    print(f"[FLEET ALERT] {TRUCK_ID} | {reason} | score={score:.0f}")


# ----------------------------- MAIN -----------------------------------
def main():
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1, refine_landmarks=True,
        min_detection_confidence=0.5, min_tracking_confidence=0.5)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("Cannot open camera.")
        return

    # Arduino link (optional)
    ser, last_sent = None, 0.0
    if serial is not None and BAUD_RATE:
        try:
            ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0)
            time.sleep(2)      # Arduino resets when the port opens
            print(f"Arduino connected on {SERIAL_PORT}")
        except Exception as e:
            print(f"Arduino not connected ({e}). Running without LEDs.")

    # calibration state
    calibrating, calib_start = True, time.time()
    calib_ear, calib_pitch = [], []
    ear_thresh, pitch_base = 0.2, 0.5

    # event state
    closed_since = yawn_since = nod_since = None
    perclos_buf = deque()              # (timestamp, is_closed)
    yawn_times, nod_times = deque(), deque()
    yawn_active = nod_active = False
    last_fleet = 0.0
    fps_t, fps = time.time(), 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        now = time.time()

        res = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        level, reasons = 0, []
        ear = mar = 0.0
        face_found = bool(res.multi_face_landmarks)

        if face_found:
            lm = [(p.x * w, p.y * h) for p in res.multi_face_landmarks[0].landmark]
            ear = (eye_aspect_ratio([lm[i] for i in LEFT_EYE]) +
                   eye_aspect_ratio([lm[i] for i in RIGHT_EYE])) / 2.0
            mar = mouth_aspect_ratio(lm)
            pitch = head_pitch_proxy(lm)

            # ---------- calibration ----------
            if calibrating:
                calib_ear.append(ear)
                calib_pitch.append(pitch)
                if now - calib_start >= CALIBRATION_SECONDS:
                    ear_thresh = float(np.median(calib_ear)) * EAR_RATIO
                    pitch_base = float(np.median(calib_pitch))
                    calibrating = False
                    print(f"Calibrated: EAR threshold={ear_thresh:.3f}, pitch base={pitch_base:.3f}")
            else:
                # ---------- eye closure ----------
                is_closed = ear < ear_thresh
                perclos_buf.append((now, is_closed))
                if is_closed:
                    closed_since = closed_since or now
                else:
                    closed_since = None
                closed_dur = (now - closed_since) if closed_since else 0.0

                if closed_dur >= EYE_CLOSED_L3_SEC:
                    level = max(level, 3); reasons.append("EYES CLOSED >3s")
                elif closed_dur >= EYE_CLOSED_L2_SEC:
                    level = max(level, 2); reasons.append("EYES CLOSED")

                # ---------- yawn ----------
                if mar > MAR_YAWN:
                    yawn_since = yawn_since or now
                    if now - yawn_since >= YAWN_MIN_SEC and not yawn_active:
                        yawn_active = True
                        yawn_times.append(now)
                else:
                    yawn_since, yawn_active = None, False

                # ---------- head nod ----------
                if pitch - pitch_base > NOD_DELTA:
                    nod_since = nod_since or now
                    if now - nod_since >= NOD_MIN_SEC and not nod_active:
                        nod_active = True
                        nod_times.append(now)
                else:
                    nod_since, nod_active = None, False

        # ---------- rolling windows ----------
        while perclos_buf and now - perclos_buf[0][0] > PERCLOS_WINDOW_SEC:
            perclos_buf.popleft()
        while yawn_times and now - yawn_times[0] > 300:
            yawn_times.popleft()
        while nod_times and now - nod_times[0] > 300:
            nod_times.popleft()

        perclos = (sum(c for _, c in perclos_buf) / len(perclos_buf)) if perclos_buf else 0.0

        if not calibrating and face_found:
            if perclos >= PERCLOS_L3:
                level = max(level, 3); reasons.append("HIGH PERCLOS")
            elif perclos >= PERCLOS_L1:
                level = max(level, 1); reasons.append("PERCLOS rising")
            if len(yawn_times) >= 2:
                level = max(level, 1); reasons.append("Frequent yawning")
            if nod_active:
                level = max(level, 2); reasons.append("HEAD NOD")
            elif len(nod_times) >= 2:
                level = max(level, 1); reasons.append("Repeated nodding")

        # no face for a while = driver looking away / absent
        if not face_found and not calibrating:
            reasons.append("NO FACE")

        # ---------- fatigue score (0-100) ----------
        score = (60 * min(perclos / PERCLOS_L3, 1.0) +
                 20 * min(len(yawn_times) / 3.0, 1.0) +
                 20 * min(len(nod_times) / 3.0, 1.0))

        # ---------- send level to Arduino (also acts as heartbeat) ----------
        if ser is not None and now - last_sent > 0.2:
            try:
                ser.write(str(level).encode())
            except Exception:
                ser = None
                print("Arduino disconnected.")
            last_sent = now

        # ---------- actions ----------
        if level in (1, 2, 3):
            beep(1 if level == 1 else 2)
        if level == 3 and now - last_fleet > FLEET_ALERT_COOLDOWN:
            send_fleet_alert(", ".join(reasons), score)
            last_fleet = now

        # ---------- overlay ----------
        colors = {0: (0, 200, 0), 1: (0, 220, 255), 2: (0, 130, 255), 3: (0, 0, 255)}
        names = {0: "NORMAL", 1: "LEVEL 1: WARNING", 2: "LEVEL 2: ALARM",
                 3: "LEVEL 3: FLEET ALERT"}
        if calibrating:
            left = max(0, CALIBRATION_SECONDS - (now - calib_start))
            cv2.putText(frame, f"CALIBRATING... keep eyes open ({left:.0f}s)", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        else:
            cv2.rectangle(frame, (0, 0), (w, 50), colors[level], -1)
            cv2.putText(frame, names[level], (15, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
            info = [f"EAR: {ear:.2f} (thr {ear_thresh:.2f})",
                    f"MAR: {mar:.2f}",
                    f"PERCLOS(60s): {perclos * 100:.0f}%",
                    f"Yawns(5m): {len(yawn_times)}  Nods(5m): {len(nod_times)}",
                    f"Fatigue score: {score:.0f}/100",
                    f"{TRUCK_ID} / {DRIVER_ID}"]
            for i, t in enumerate(info):
                cv2.putText(frame, t, (15, 80 + 28 * i),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            if reasons:
                cv2.putText(frame, " | ".join(reasons), (15, h - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        fps = 0.9 * fps + 0.1 / max(now - fps_t, 1e-6)
        fps_t = now
        cv2.putText(frame, f"FPS {fps:.0f}", (w - 110, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.imshow("Driver Fatigue Monitor", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("c"):
            calibrating, calib_start = True, time.time()
            calib_ear.clear(); calib_pitch.clear()
            perclos_buf.clear(); yawn_times.clear(); nod_times.clear()

    if ser is not None:
        try:
            ser.write(b"0")
            ser.close()
        except Exception:
            pass
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
