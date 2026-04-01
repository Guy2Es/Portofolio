#!/usr/bin/env python3
import json
import math
import time

import cv2
import numpy as np
import paho.mqtt.client as mqtt

BROKER = "localhost"
PORT = 1883

TOPIC_CAMERA_FRAME = "camera/frame"
TOPIC_ATT = "iris/attitude"          # roll/pitch/yaw in degrees
TOPIC_PVN = "iris/pvn"               # n/e/d in meters (NED) (optional)
TOPIC_ARUCO = "aruco/world_pose"     # output

# --- Marker ---
ARUCO_DICT_NAME = "DICT_4X4_50"
TARGET_ID = 0

# Ton box dans le .world: <size>0.5 0.5 0.02</size> -> côté du marker = 0.5 m
MARKER_SIZE_M = 0.5

# Camera SDF: <horizontal_fov>1.8</horizontal_fov>
HFOV_RAD = 1.8

# down_camera_link pose: <pose>0 0 -0.05 0 0 0</pose>
# NED body: +Z = down -> caméra 5 cm sous le drone => +0.05
CAMERA_OFFSET_BODY_NED = np.array([0.0, 0.0, 0.05], dtype=np.float64)

# Low-pass smoothing
LPF_ALPHA = 0.25  # 0..1 (plus grand = moins filtré)

SHOW_DEBUG = True

# ---------------- ArUco compat layer ----------------
DICT_MAP = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
}

dict_id = DICT_MAP.get(ARUCO_DICT_NAME, cv2.aruco.DICT_4X4_50)

# getPredefinedDictionary (new) / Dictionary_get (old)
if hasattr(cv2.aruco, "getPredefinedDictionary"):
    aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
else:
    aruco_dict = cv2.aruco.Dictionary_get(dict_id)

# DetectorParameters (new) / DetectorParameters_create (old)
if hasattr(cv2.aruco, "DetectorParameters"):
    aruco_params = cv2.aruco.DetectorParameters()
else:
    aruco_params = cv2.aruco.DetectorParameters_create()

# ArucoDetector (new) / detectMarkers (old)
if hasattr(cv2.aruco, "ArucoDetector"):
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

    def detect_markers(gray_img):
        return detector.detectMarkers(gray_img)
else:
    def detect_markers(gray_img):
        return cv2.aruco.detectMarkers(gray_img, aruco_dict, parameters=aruco_params)

# ---------------- Geometry ----------------
# Conversion optical -> NED (camera down), comme dans PrecisionLand.cpp :
# optical: X right, Y down, Z away from lens
# NED:     X forward(north), Y right(east), Z down
R_ned_opt = np.array([
    [0.0, -1.0, 0.0],
    [1.0,  0.0, 0.0],
    [0.0,  0.0, 1.0],
], dtype=np.float64)

last_frame = None
last_att = {"have": False, "roll": 0.0, "pitch": 0.0, "yaw": 0.0}  # radians
last_pvn = {"have": False, "n": 0.0, "e": 0.0, "d": 0.0}

rel_ned_filt = np.array([0.0, 0.0, 0.0], dtype=np.float64)
rel_filt_init = False


def euler_to_R_ned_body(roll, pitch, yaw):
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)

    # R = Rz(yaw) * Ry(pitch) * Rx(roll)
    Rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    Ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=np.float64)
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=np.float64)
    return Rz @ Ry @ Rx


def build_camera_matrix(w, h):
    fx = (w / 2.0) / math.tan(HFOV_RAD / 2.0)
    vfov = 2.0 * math.atan(math.tan(HFOV_RAD / 2.0) * (h / w))
    fy = (h / 2.0) / math.tan(vfov / 2.0)
    cx, cy = (w / 2.0), (h / 2.0)

    K = np.array([[fx, 0.0, cx],
                  [0.0, fy, cy],
                  [0.0, 0.0, 1.0]], dtype=np.float64)
    dist = np.zeros((5, 1), dtype=np.float64)
    return K, dist


def on_message(client, userdata, msg):
    global last_frame, last_att, last_pvn

    if msg.topic == TOPIC_CAMERA_FRAME:
        jpg = np.frombuffer(msg.payload, dtype=np.uint8)
        frame = cv2.imdecode(jpg, cv2.IMREAD_COLOR)
        if frame is not None:
            last_frame = frame
        return

    payload = msg.payload.decode(errors="replace")
    try:
        data = json.loads(payload)
    except Exception:
        return

    if msg.topic == TOPIC_ATT:
        try:
            last_att["roll"] = math.radians(float(data.get("roll_deg", 0.0)))
            last_att["pitch"] = math.radians(float(data.get("pitch_deg", 0.0)))
            last_att["yaw"] = math.radians(float(data.get("yaw_deg", 0.0)))
            last_att["have"] = True
        except Exception:
            pass

    elif msg.topic == TOPIC_PVN:
        try:
            last_pvn["n"] = float(data.get("n", 0.0))
            last_pvn["e"] = float(data.get("e", 0.0))
            last_pvn["d"] = float(data.get("d", 0.0))
            last_pvn["have"] = True
        except Exception:
            pass


def main():
    global rel_ned_filt, rel_filt_init

    client = mqtt.Client()
    client.on_message = on_message
    client.connect(BROKER, PORT, 60)
    client.subscribe(TOPIC_CAMERA_FRAME)
    client.subscribe(TOPIC_ATT)
    client.subscribe(TOPIC_PVN)
    client.loop_start()

    try:
        while True:
            if last_frame is None:
                time.sleep(0.01)
                continue

            frame = last_frame
            h, w = frame.shape[:2]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            corners, ids, _rej = detect_markers(gray)

            out = {
                "have_tag": False,
                "target_id": int(TARGET_ID),
                "ts": time.time(),
            }

            if ids is not None and len(ids) > 0:
                ids_flat = ids.flatten().tolist()
                if TARGET_ID in ids_flat:
                    idx = ids_flat.index(TARGET_ID)
                    c = corners[idx]

                    K, dist = build_camera_matrix(w, h)

                    # Pose marker dans frame caméra (optical)
                    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers([c], MARKER_SIZE_M, K, dist)
                    rvec = rvecs[0].reshape(3, 1)
                    tvec_opt = tvecs[0].reshape(3, 1)

                    # optical -> NED(cam)
                    t_ned_cam = (R_ned_opt @ tvec_opt).reshape(3)

                    # add camera offset
                    rel_body_ned = (CAMERA_OFFSET_BODY_NED + t_ned_cam).reshape(3)

                    if last_att["have"]:
                        # body -> world NED
                        R_ned_body = euler_to_R_ned_body(last_att["roll"], last_att["pitch"], last_att["yaw"])
                        rel_world_ned = (R_ned_body @ rel_body_ned).reshape(3)
                        rel_world_ned[1] *= -1.0

                        # low-pass
                        if not rel_filt_init:
                            rel_ned_filt = rel_world_ned.copy()
                            rel_filt_init = True
                        else:
                            rel_ned_filt = LPF_ALPHA * rel_world_ned + (1.0 - LPF_ALPHA) * rel_ned_filt

                        out.update({
                            "have_tag": True,
                            "rel_n": float(rel_ned_filt[0]),
                            "rel_e": float(rel_ned_filt[1]),
                            "rel_d": float(rel_ned_filt[2]),
                            # Always publish body-frame relative position too (for OFFBOARD centering)
                            "rel_body_x": float(rel_body_ned[0]),
                            "rel_body_y": float(rel_body_ned[1]),
                            "rel_body_z": float(rel_body_ned[2]),
                        })

                        if last_pvn["have"]:
                            out.update({
                                "veh_n": float(last_pvn["n"]),
                                "veh_e": float(last_pvn["e"]),
                                "veh_d": float(last_pvn["d"]),
                                "tag_n": float(last_pvn["n"] + rel_ned_filt[0]),
                                "tag_e": float(last_pvn["e"] + rel_ned_filt[1]),
                                "tag_d": float(last_pvn["d"] + rel_ned_filt[2]),
                            })
                    else:
                        out.update({
                            "have_tag": True,
                            "rel_body_x": float(rel_body_ned[0]),
                            "rel_body_y": float(rel_body_ned[1]),
                            "rel_body_z": float(rel_body_ned[2]),
                        })

                    if SHOW_DEBUG:
                        cv2.aruco.drawDetectedMarkers(frame, [c], np.array([[TARGET_ID]], dtype=np.int32))
                        cv2.drawFrameAxes(frame, K, dist, rvec, tvecs[0], 0.25)

            client.publish(TOPIC_ARUCO, json.dumps(out))

            if SHOW_DEBUG:
                cv2.imshow("aruco_world_pose", frame)
                if (cv2.waitKey(1) & 0xFF) == 27:
                    break

            time.sleep(0.02)

    finally:
        client.loop_stop()
        client.disconnect()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
