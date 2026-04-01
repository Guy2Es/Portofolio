#!/usr/bin/env python3
"""
Mission intégrée:
1) Décollage stabilisé sur ArUco (OFFBOARD cmd_vel)
2) Transition ArUco -> ligne (avance douce jusqu’à détecter la ligne)
3) Aller/retour sur ligne (logique inspirée de 123.py, mais on déplace le drone, pas un viewport)
4) Atterrissage stabilisé sur ArUco (centrage + descente), puis LAND/DISARM via px4_bridge

Scripts à lancer en parallèle (ou via run_mission_all.py):
- px4_bridge.py        -> topics iris/*
- camera_publisher.py  -> topic camera/frame
- aruco_world_pose.py  -> topic aruco/world_pose
"""

import json
import math
import time
from enum import Enum

import cv2
import numpy as np
import paho.mqtt.client as mqtt


# =========================
# MQTT
# =========================
BROKER = "localhost"
PORT = 1883

TOPIC_CMD_MODE   = "iris/cmd_mode"
TOPIC_CMD_VEL    = "iris/cmd_vel"
TOPIC_ALT        = "iris/altitude"
TOPIC_ATT        = "iris/attitude"
TOPIC_ARMED      = "iris/armed"
TOPIC_ARUCO      = "aruco/world_pose"
TOPIC_CAMERA     = "camera/frame"


# =========================
# Vision thresholds (HSV)
# =========================
YELLOW_LO, YELLOW_HI = (20, 100, 100), (35, 255, 255)
GREEN_LO,  GREEN_HI  = (35,  50,  50), (85, 255, 255)
BLUE_LO,   BLUE_HI   = (90,  80,  80), (140, 255, 255)

# =========================
# Mission params
# =========================
LOOP_DT = 0.05

TARGET_ALT_M = 1.5
ALT_TOL_M = 0.15

# ✅ Altitude hold (OFFBOARD): keep ~TARGET_ALT_M during the whole mission (except landing descent)
ALT_HOLD_KP = 1.0               # P gain: higher = more aggressive
ALT_HOLD_MAX_VZ = 0.45          # clamp |vz| (m/s)
ALT_HOLD_DEADBAND_M = 0.04      # no correction within ±deadband

# Takeoff/centering control
CLIMB_VZ = -0.6               # vz body NED, +down => climb = negative
MAX_VXY_TAKEOFF = 0.6
Kp_xy = 0.8
Ki_xy = 0.15
TAG_TIMEOUT_S = 0.25

# Yaw control for 180° turn
Kp_yaw = 1.6
MAX_YAW_RATE = math.radians(60)
YAW_TOL = math.radians(6)


# ✅ Hold after 180° turn
TURN_HOLD_AFTER_180_S = 4.0   # seconds to hover in place once yaw is within tolerance
# Line following
V_FWD = 0.3
MAX_VY = 0.4
Kp_line = 0.9
Kp_line_yaw = 3.0
MAX_YAW_RATE_LINE = math.radians(50)
SCAN_Y1_FRAC_out = 0.55
SCAN_Y2_FRAC_out = 0.75
SCAN_Y1_FRAC_back = 0.10
SCAN_Y2_FRAC_back = 0.30
SCAN_BAND = 4
MIN_PIX_ON_SCAN = 10

# Transition ArUco -> Line
FIND_LINE_VX = 0.35           # avance douce
FIND_LINE_MAX_T = 12.0        # safety: si pas de ligne trouvée, on bascule quand même
FIND_LINE_STABLE_FRAMES = 5  # nb de frames "ok" requis pour considérer la ligne trouvée

# Blue handling
BLUE_ARM_DELAY_AFTER_ALT_S = 45.0  # ✅ demandé: activer la reconnaissance BLEU 30s après avoir atteint 1.5m
BLUE_STABLE_FRAMES = 6
BLUE_CLOSE_AREA = 9000             # ajuster selon caméra
BLUE_WAIT_S = 0.1

# Blue bottom-third trigger
BLUE_BOTTOM_Y_FRAC = 0.5      # bas de l’image (dernier tiers)
BLUE_BOTTOM_STABLE_FRAMES = 6     # stabilité avant stop+attente

# Landing
LAND_TAG_TRIGGER_DIST = 1.2   # meters (horizontal distance)
LAND_DESCEND_VZ = 0.35        # vz positive -> descend
LAND_FINAL_ALT = 0.30         # when below => LAND mode
LAND_HOLD_CENTER_ALT = 0.8    # keep centering until this alt

# ✅ Requested: stabilize centered above ArUco before descending
LAND_STABILIZE_S = 0.5        # time (s) to hold centered before starting descent
LAND_CENTER_TOL_M = 0.12      # |rel_n| and |rel_e| threshold to be considered "well centered" (meters)
LAND_HOVER_ALT_KP = 0.9       # simple P hold around the entry altitude (vz command)
LAND_HOVER_ALT_MAX_VZ = 0.35  # max |vz| during hover stabilization

SHOW_DEBUG_CV = True


# =========================
# Helpers
# =========================
def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x

def wrap_pi(a):
    while a <= -math.pi: a += 2 * math.pi
    while a > math.pi: a -= 2 * math.pi
    return a

def hsv_mask(frame_bgr, lo, hi):
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    lo = np.array(lo, dtype=np.uint8)
    hi = np.array(hi, dtype=np.uint8)
    mask = cv2.inRange(hsv, lo, hi)
    # Robustify thin / noisy lines
    k = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)
    mask = cv2.dilate(mask, k, iterations=1)
    return mask

def scanline_x(mask, y, x_min=0, x_max=None, band=SCAN_BAND):
    """Return mean x of non-zero pixels around row y (±band) within [x_min,x_max]."""
    h, w = mask.shape[:2]
    if x_max is None: x_max = w
    y = int(clamp(y, 0, h - 1))
    y0 = max(0, y - int(band))
    y1 = min(h, y + int(band) + 1)
    roi = mask[y0:y1, x_min:x_max]
    ys, xs = np.where(roi > 0)
    if xs.size < MIN_PIX_ON_SCAN:
        return None
    return float(xs.mean() + x_min)

def blob_center_and_area(mask):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, 0.0
    c = max(cnts, key=cv2.contourArea)
    area = float(cv2.contourArea(c))
    if area <= 1.0:
        return None, area
    M = cv2.moments(c)
    if abs(M["m00"]) < 1e-6:
        return None, area
    cx = float(M["m10"] / M["m00"])
    cy = float(M["m01"] / M["m00"])
    return (cx, cy), area


# =========================
# Shared state from topics
# =========================
last_alt = {"have": False, "rel": 0.0}
last_att = {"have": False, "yaw": 0.0}
last_armed = {"have": False, "armed": False}
last_tag = {"have": False, "ts": 0.0, "rel_n": 0.0, "rel_e": 0.0, "rel_d": 0.0}
last_frame = {"have": False, "ts": 0.0, "img": None}

# PI integrators for ArUco centering
i_n = 0.0
i_e = 0.0

# Yaw target
yaw_target = None

# Lane width estimate (pixels)
lane_w_px = None

# Timing
alt_reached_t0 = None   # ✅ moment où l'altitude 1.5m est atteinte (référence pour le délai BLEU)

# Landing stabilization timers
land_hover_alt = None       # target altitude during LAND_STABILIZE (m)
land_centered_t0 = None     # start time once centered above ArUco

# Turn hold timer (after 180° yaw)
turn_hold_t0 = None


# =========================
# MQTT publish
# =========================
client = mqtt.Client()

def publish_mode(mode: str):
    client.publish(TOPIC_CMD_MODE, json.dumps({"mode": mode}))

def publish_vel(vx, vy, vz, yaw_rate):
    client.publish(
        TOPIC_CMD_VEL,
        json.dumps(
            {
                "vx": float(vx),              # +forward (m/s)
                "vy": float(vy),              # +right   (m/s)
                "vz": float(vz),              # +down    (m/s)
                "yaw_rate": float(yaw_rate),  # rad/s
            }
        ),
    )


# =========================
# Control blocks
# =========================
def tag_ok(now: float) -> bool:
    return bool(last_tag["have"]) and ((now - float(last_tag["ts"])) < TAG_TIMEOUT_S)

def compute_vxy_from_tag():
    """Body-frame vx/vy command that drives the tag offset -> 0 (centering).
    Prefers BODY-frame offsets (rel_body_x/rel_body_y) when available; falls back to world rel_n/rel_e.
    """
    global i_n, i_e
    if ("rel_body_x" in last_tag) and ("rel_body_y" in last_tag):
        en = float(last_tag["rel_body_x"])  # forward error
        ee = float(last_tag["rel_body_y"])  # right error
    else:
        en = float(last_tag.get("rel_body_x", last_tag["rel_n"]))
        ee = float(last_tag.get("rel_body_y", last_tag["rel_e"]))

    i_n = clamp(i_n + en * LOOP_DT, -2.0, 2.0)
    i_e = clamp(i_e + ee * LOOP_DT, -2.0, 2.0)

    vn = Kp_xy * en + Ki_xy * i_n
    ve = Kp_xy * ee + Ki_xy * i_e

    # Using BODY offsets (forward/right) when available
    vx = clamp(vn, -MAX_VXY_TAKEOFF, +MAX_VXY_TAKEOFF)
    vy = clamp(ve, -MAX_VXY_TAKEOFF, +MAX_VXY_TAKEOFF)
    return vx, vy

def compute_yaw_rate():
    if not last_att["have"] or yaw_target is None:
        return 0.0
    ey = wrap_pi(yaw_target - float(last_att["yaw"]))
    return clamp(Kp_yaw * ey, -MAX_YAW_RATE, +MAX_YAW_RATE)

def line_follow_cmd_out(frame_bgr, swapped: bool):
    """
    Suivi du couloir (jaune/vert) + pivot pour suivre une polyligne.
    Inspiré de 123.py: on estime l'orientation via 2 scanlines.
      - vy : recentrage (milieu du couloir)
      - yaw_rate : tourne vers la direction du couloir
    """
    global lane_w_px

    h, w = frame_bgr.shape[:2]
    cx = w / 2.0
    y1 = int(h * SCAN_Y1_FRAC_out)  # look-ahead
    y2 = int(h * SCAN_Y2_FRAC_out)  # near
    dy = max(1.0, float(y2 - y1))

    mask_y = hsv_mask(frame_bgr, YELLOW_LO, YELLOW_HI)
    mask_g = hsv_mask(frame_bgr, GREEN_LO,  GREEN_HI)

    # Measure on two scanlines
    yx1 = scanline_x(mask_y, y1)
    yx2 = scanline_x(mask_y, y2)
    gx1 = scanline_x(mask_g, y1)
    gx2 = scanline_x(mask_g, y2)

    # Update lane width estimate when both visible (near line)
    if (yx2 is not None) and (gx2 is not None):
        dpx = abs(gx2 - yx2)
        lane_w_px = dpx if lane_w_px is None else (0.85 * lane_w_px + 0.15 * dpx)
    if lane_w_px is None:
        lane_w_px = 220.0

    yellow_is_left = not swapped

    def complete_pair(yx, gx):
        # If only one side is visible, synthesize the other using lane width
        if yx is not None and gx is None:
            gx = yx + lane_w_px if yellow_is_left else yx - lane_w_px
        elif gx is not None and yx is None:
            yx = gx - lane_w_px if yellow_is_left else gx + lane_w_px
        return yx, gx

    yx1, gx1 = complete_pair(yx1, gx1)
    yx2, gx2 = complete_pair(yx2, gx2)

    ok1 = (yx1 is not None) and (gx1 is not None)
    ok2 = (yx2 is not None) and (gx2 is not None)

    dbg = None
    if SHOW_DEBUG_CV:
        dbg = frame_bgr.copy()
        cv2.line(dbg, (0, y1), (w, y1), (255, 255, 255), 1)
        cv2.line(dbg, (0, y2), (w, y2), (255, 255, 255), 1)
        if ok1:
            cv2.circle(dbg, (int(yx1), y1), 6, (0, 255, 255), -1)
            cv2.circle(dbg, (int(gx1), y1), 6, (0, 255, 0), -1)
        if ok2:
            cv2.circle(dbg, (int(yx2), y2), 6, (0, 255, 255), -1)
            cv2.circle(dbg, (int(gx2), y2), 6, (0, 255, 0), -1)

    if not ok1 and not ok2:
        return 0.0, 0.0, 0.0, dbg, {"ok": False}

    mid1 = 0.5 * (yx1 + gx1) if ok1 else None
    mid2 = 0.5 * (yx2 + gx2) if ok2 else None
    if mid1 is None and mid2 is not None:
        mid1 = mid2
    if mid2 is None and mid1 is not None:
        mid2 = mid1

    # Lateral centering (use near scanline)
    err_lat = (mid2 - cx) / max(cx, 1.0)
    vy = clamp(Kp_line * err_lat, -MAX_VY, +MAX_VY)

    # Orientation of corridor (like 123.py phi)
    dx = (mid1 - mid2)
    ang = math.atan2(dx, dy)  # rad
    yaw_rate = clamp(Kp_line_yaw * ang, -MAX_YAW_RATE_LINE, +MAX_YAW_RATE_LINE)

    # Forward speed: slow down when turning or off-center
    turn_factor = clamp(1.0 - 1.6 * abs(ang), 0.25, 1.0)
    lat_factor = clamp(1.0 - 1.2 * abs(err_lat), 0.30, 1.0)
    vx = V_FWD * min(turn_factor, lat_factor)

    if SHOW_DEBUG_CV and dbg is not None:
        cv2.circle(dbg, (int(mid2), y2), 6, (0, 0, 255), -1)
        cv2.circle(dbg, (int(mid1), y1), 6, (0, 0, 255), -1)
        cv2.line(dbg, (int(mid2), y2), (int(mid1), y1), (0, 0, 255), 2)
        cv2.putText(dbg, f"lat={err_lat:.2f} vy={vy:.2f} ang={math.degrees(ang):.1f} yaw={math.degrees(yaw_rate):.1f} vx={vx:.2f}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    return vx, vy, yaw_rate, dbg, {"ok": True, "ang": ang, "err_lat": err_lat}

def line_follow_cmd_back(frame_bgr, swapped: bool):
    """
    Suivi du couloir (jaune/vert) + pivot pour suivre une polyligne.
    Inspiré de 123.py: on estime l'orientation via 2 scanlines.
      - vy : recentrage (milieu du couloir)
      - yaw_rate : tourne vers la direction du couloir
    """
    global lane_w_px

    h, w = frame_bgr.shape[:2]
    cx = w / 2.0
    y1 = int(h * SCAN_Y1_FRAC_back)  # look-ahead
    y2 = int(h * SCAN_Y2_FRAC_back)  # near
    dy = max(1.0, float(y2 - y1))

    mask_y = hsv_mask(frame_bgr, YELLOW_LO, YELLOW_HI)
    mask_g = hsv_mask(frame_bgr, GREEN_LO,  GREEN_HI)

    # Measure on two scanlines
    yx1 = scanline_x(mask_y, y1)
    yx2 = scanline_x(mask_y, y2)
    gx1 = scanline_x(mask_g, y1)
    gx2 = scanline_x(mask_g, y2)

    # Update lane width estimate when both visible (near line)
    if (yx2 is not None) and (gx2 is not None):
        dpx = abs(gx2 - yx2)
        lane_w_px = dpx if lane_w_px is None else (0.85 * lane_w_px + 0.15 * dpx)
    if lane_w_px is None:
        lane_w_px = 220.0

    yellow_is_left = not swapped

    def complete_pair(yx, gx):
        # If only one side is visible, synthesize the other using lane width
        if yx is not None and gx is None:
            gx = yx + lane_w_px if yellow_is_left else yx - lane_w_px
        elif gx is not None and yx is None:
            yx = gx - lane_w_px if yellow_is_left else gx + lane_w_px
        return yx, gx

    yx1, gx1 = complete_pair(yx1, gx1)
    yx2, gx2 = complete_pair(yx2, gx2)

    ok1 = (yx1 is not None) and (gx1 is not None)
    ok2 = (yx2 is not None) and (gx2 is not None)

    dbg = None
    if SHOW_DEBUG_CV:
        dbg = frame_bgr.copy()
        cv2.line(dbg, (0, y1), (w, y1), (255, 255, 255), 1)
        cv2.line(dbg, (0, y2), (w, y2), (255, 255, 255), 1)
        if ok1:
            cv2.circle(dbg, (int(yx1), y1), 6, (0, 255, 255), -1)
            cv2.circle(dbg, (int(gx1), y1), 6, (0, 255, 0), -1)
        if ok2:
            cv2.circle(dbg, (int(yx2), y2), 6, (0, 255, 255), -1)
            cv2.circle(dbg, (int(gx2), y2), 6, (0, 255, 0), -1)

    if not ok1 and not ok2:
        return 0.0, 0.0, 0.0, dbg, {"ok": False}

    mid1 = 0.5 * (yx1 + gx1) if ok1 else None
    mid2 = 0.5 * (yx2 + gx2) if ok2 else None
    if mid1 is None and mid2 is not None:
        mid1 = mid2
    if mid2 is None and mid1 is not None:
        mid2 = mid1

    # Lateral centering (use near scanline)
    err_lat = (mid2 - cx) / max(cx, 1.0)
    vy = clamp(Kp_line * err_lat, -MAX_VY, +MAX_VY)

    # Orientation of corridor (like 123.py phi)
    dx = (mid1 - mid2)
    ang = math.atan2(dx, dy)  # rad
    yaw_rate = clamp(Kp_line_yaw * ang, -MAX_YAW_RATE_LINE, +MAX_YAW_RATE_LINE)

    # Forward speed: slow down when turning or off-center
    turn_factor = clamp(1.0 - 1.6 * abs(ang), 0.25, 1.0)
    lat_factor = clamp(1.0 - 1.2 * abs(err_lat), 0.30, 1.0)
    vx = V_FWD * min(turn_factor, lat_factor)

    if SHOW_DEBUG_CV and dbg is not None:
        cv2.circle(dbg, (int(mid2), y2), 6, (0, 0, 255), -1)
        cv2.circle(dbg, (int(mid1), y1), 6, (0, 0, 255), -1)
        cv2.line(dbg, (int(mid2), y2), (int(mid1), y1), (0, 0, 255), 2)
        cv2.putText(dbg, f"lat={err_lat:.2f} vy={vy:.2f} ang={math.degrees(ang):.1f} yaw={math.degrees(yaw_rate):.1f} vx={vx:.2f}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    return vx, vy, yaw_rate, dbg, {"ok": True, "ang": ang, "err_lat": err_lat}

def blue_logic(frame_bgr):
    """Detect blue marker; return (seen, center, area)."""
    mask_b = hsv_mask(frame_bgr, BLUE_LO, BLUE_HI)
    center, area = blob_center_and_area(mask_b)
    return (center is not None and area > 800.0), center, area


# =========================
# Mission state machine
# =========================
class State(Enum):
    INIT = 0
    ARM = 1
    TAKEOFF = 2
    OFFBOARD = 3
    TAKEOFF_CENTER = 4
    FIND_LINE = 5
    LINE_OUT = 6
    BLUE_APPROACH = 7
    BLUE_WAIT = 8
    TURN_AROUND = 9
    LINE_BACK = 10

    # Landing: hover centered above ArUco -> controlled descent -> PX4 LAND -> DISARM
    LAND_STABILIZE = 11
    LAND_CENTER_DESCEND = 12
    LAND_FINAL = 13
    DISARM = 14
    DONE = 15


state = State.INIT
t_state = 0.0

blue_seen_cnt = 0
blue_armed = False
swapped = False

find_line_ok_cnt = 0


def on_connect(_client, _userdata, _flags, rc):
    print("[MISSION] MQTT connected rc=", rc)
    client.subscribe(TOPIC_ALT)
    client.subscribe(TOPIC_ATT)
    client.subscribe(TOPIC_ARUCO)
    client.subscribe(TOPIC_CAMERA)
    client.subscribe(TOPIC_ARMED)

def on_message(_client, _userdata, msg):
    try:
        data = json.loads(msg.payload.decode(errors="replace"))
    except Exception:
        data = None

    if msg.topic == TOPIC_ALT and data and "rel_alt_m" in data:
        last_alt["rel"] = float(data["rel_alt_m"])
        last_alt["have"] = True

    elif msg.topic == TOPIC_ATT and data and "yaw_deg" in data:
        last_att["yaw"] = math.radians(float(data["yaw_deg"]))
        last_att["have"] = True

    elif msg.topic == TOPIC_ARUCO and data:
        # IMPORTANT:
        # - Do NOT overwrite the last good tag with have_tag=False frames.
        #   We keep the last valid measurement until TAG_TIMEOUT_S expires (see tag_ok()).
        if bool(data.get("have_tag", False)):
            last_tag["have"] = True
            last_tag["ts"] = float(data.get("ts", 0.0))

            # World-frame (optional, mainly for logging)
            if "rel_n" in data and "rel_e" in data and "rel_d" in data:
                last_tag["rel_n"] = float(data["rel_n"])
                last_tag["rel_e"] = float(data["rel_e"])
                last_tag["rel_d"] = float(data["rel_d"])

            # Body-frame (recommended for control: forward/right/down)
            if "rel_body_x" in data and "rel_body_y" in data and "rel_body_z" in data:
                last_tag["rel_body_x"] = float(data["rel_body_x"])
                last_tag["rel_body_y"] = float(data["rel_body_y"])
                last_tag["rel_body_z"] = float(data["rel_body_z"])

    elif msg.topic == TOPIC_ARMED and data is not None:
        last_armed["armed"] = bool(data.get("armed", False))
        last_armed["have"] = True

    elif msg.topic == TOPIC_CAMERA:
        jpg = np.frombuffer(msg.payload, dtype=np.uint8)
        frame = cv2.imdecode(jpg, cv2.IMREAD_COLOR)
        if frame is not None:
            last_frame["img"] = frame
            last_frame["ts"] = time.time()
            last_frame["have"] = True


def set_state(s: State):
    global state, t_state, blue_seen_cnt, find_line_ok_cnt
    global land_hover_alt, land_centered_t0, i_n, i_e

    state = s
    t_state = time.time()
    blue_seen_cnt = 0
    find_line_ok_cnt = 0

    # Reset landing timers/integrators when entering landing phases
    if s == State.LAND_STABILIZE:
        land_hover_alt = None
        land_centered_t0 = None
        i_n = 0.0
        i_e = 0.0
    elif s == State.LAND_CENTER_DESCEND:
        land_hover_alt = None
        land_centered_t0 = None

    print("[MISSION] ->", state.name)


def main():

    global yaw_target, blue_seen_cnt, blue_armed, swapped, find_line_ok_cnt, alt_reached_t0, turn_hold_t0
    global land_hover_alt, land_centered_t0

    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(BROKER, PORT, 60)
    client.loop_start()

    try:
        set_state(State.INIT)

        # Safety: push a zero setpoint quickly
        publish_vel(0, 0, 0, 0)

        while True:
            now = time.time()

            # Default commands
            vx, vy, vz, yaw_rate = 0.0, 0.0, 0.0, 0.0
            dbg = None

            # -------- STATES --------
            if state == State.INIT:
                if (now - t_state) > 0.5:
                    set_state(State.ARM)

            elif state == State.ARM:
                publish_mode("ARM")
                # Attendre confirmation armement via iris/armed
                if last_armed.get("have", False) and last_armed.get("armed", False):
                    set_state(State.TAKEOFF)

            elif state == State.TAKEOFF:
                # Décollage fiable via action.takeoff dans le bridge
                publish_mode("TAKEOFF")
                publish_vel(0, 0, 0, 0)
                # Dès qu'on est proche de l'altitude cible, passer en OFFBOARD pour centrage fin
                if last_alt["have"] and last_alt["rel"] >= (TARGET_ALT_M - 0.25):
                    set_state(State.OFFBOARD)

            elif state == State.OFFBOARD:
                publish_mode("OFFBOARD")
                if last_att["have"] and yaw_target is None:
                    yaw_target = float(last_att["yaw"])
                # envoyer quelques setpoints neutres pour sécuriser Offboard START
                publish_vel(0, 0, vz, 0)
                if (now - t_state) > 2.0:
                    set_state(State.TAKEOFF_CENTER)

            elif state == State.TAKEOFF_CENTER:
                # climb to target alt while centering on tag
                yaw_rate = 0.0

                if last_alt["have"] and last_alt["rel"] < (TARGET_ALT_M - ALT_TOL_M):
                    vz = CLIMB_VZ
                else:
                    vz = alt_hold_vz()
                if tag_ok(now):
                    vx, vy = compute_vxy_from_tag()
                else:
                    vx, vy = 0.0, 0.0

                # reached altitude -> transition to line (and start blue delay reference)
                if last_alt["have"] and abs(last_alt["rel"] - TARGET_ALT_M) <= ALT_TOL_M and (now - t_state) > 1.5:
                    alt_reached_t0 = now  # ✅ référence pour activer BLEU après 15s
                    blue_armed = False
                    swapped = False
                    set_state(State.FIND_LINE)

            elif state == State.FIND_LINE:
                # Après 1.5 m: se diriger vers la ligne jaune/verte (inspiré de 123.py)
                # Idée: avancer doucement, et si on voit la ligne (même partiellement) -> vy + yaw_rate pour l'amener au centre.
                # Si on ne voit rien -> scan en yaw pour retrouver la ligne.
                vz = alt_hold_vz()
                # Base forward motion
                vx = FIND_LINE_VX
                vy = 0.0
                yaw_rate = 0.0

                if last_frame["have"]:
                    frame = last_frame["img"]
                    vx_l, vy_l, yr_l, dbg, info = line_follow_cmd_out(frame, swapped=False)

                    if info.get("ok", False):
                        # Dès qu'on détecte la ligne, on steer vers elle
                        find_line_ok_cnt += 1
                        vy = clamp(vy_l, -0.6, 0.6)
                        yaw_rate = clamp(yr_l, -MAX_YAW_RATE_LINE, +MAX_YAW_RATE_LINE)
                        # On avance un peu moins si on tourne beaucoup
                        vx = max(0.20, min(vx, vx_l))
                    else:
                        # Pas de ligne -> pattern de recherche (yaw scan), on garde un peu d'avance
                        find_line_ok_cnt = max(0, find_line_ok_cnt - 1)
                        t = now - t_state
                        # oscillation : +, -, + ...
                        sgn = 1.0 if int(t / 2.0) % 2 == 0 else -1.0
                        yaw_rate = sgn * math.radians(18)
                        vx = 0.20
                        vy = 0.0

                    if SHOW_DEBUG_CV and dbg is not None:
                        cv2.putText(dbg, f"STATE: FIND_LINE ok_cnt={find_line_ok_cnt}/{FIND_LINE_STABLE_FRAMES}",
                                    (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                    if find_line_ok_cnt >= FIND_LINE_STABLE_FRAMES:
                        set_state(State.LINE_OUT)

                # Safety timeout: si pas trouvé, on bascule quand même
                if (now - t_state) >= FIND_LINE_MAX_T:
                    set_state(State.LINE_OUT)

            elif state == State.LINE_OUT:
                yaw_rate = 0.0
                vz = alt_hold_vz()
                # ✅ activation BLEU uniquement 15s après avoir atteint l'altitude
                if alt_reached_t0 is not None and (now - alt_reached_t0) >= BLUE_ARM_DELAY_AFTER_ALT_S:
                    blue_armed = True

                if last_frame["have"]:
                    frame = last_frame["img"]

                    # vision line
                    vx_line, vy_line, yaw_rate, dbg, info = line_follow_cmd_out(frame, swapped=False)
                    vx, vy = vx_line, vy_line

                    # vision blue
                    seen, bcenter, barea = blue_logic(frame)
                    # ✅ Condition demandée: continuer à avancer jusqu’à ce que le bleu soit dans le dernier tiers de l’image (bas)
                    h, w = frame.shape[:2]
                    blue_in_last_third = False
                    if seen and (bcenter is not None):
                        blue_in_last_third = (float(bcenter[1]) >= (BLUE_BOTTOM_Y_FRAC * h))

                    if blue_armed and seen and blue_in_last_third:
                        blue_seen_cnt += 1
                    else:
                        blue_seen_cnt = max(0, blue_seen_cnt - 1)

                    # ✅ Quand le bleu est stable dans le dernier tiers: STOP + attente 4s, puis 180° et retour
                    if blue_armed and blue_seen_cnt >= BLUE_BOTTOM_STABLE_FRAMES:
                        set_state(State.BLUE_WAIT)

                    if SHOW_DEBUG_CV and dbg is not None:
                        cv2.putText(dbg, f"STATE: LINE_OUT blue_armed={int(blue_armed)} blue_cnt={blue_seen_cnt}",
                                    (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                else:
                    vx, vy = 0.0, 0.0

            elif state == State.BLUE_APPROACH:
                # Center on blue blob; move forward slowly until "close"
                vz = alt_hold_vz()
                yaw_rate = 0.0

                if last_frame["have"]:
                    frame = last_frame["img"]
                    seen, bcenter, barea = blue_logic(frame)
                    h, w = frame.shape[:2]
                    cx = w / 2.0

                    if seen and bcenter is not None:
                        err = (bcenter[0] - cx) / max(cx, 1.0)
                        vy = clamp(0.8 * err, -0.5, +0.5)
                        vx = 0.35
                        if barea >= BLUE_CLOSE_AREA:
                            vx, vy = 0.0, 0.0
                            set_state(State.BLUE_WAIT)
                    else:
                        # lost blue -> resume line out
                        set_state(State.LINE_OUT)

                    if SHOW_DEBUG_CV:
                        dbg = frame.copy()
                        if bcenter is not None:
                            cv2.circle(dbg, (int(bcenter[0]), int(bcenter[1])), 10, (255, 0, 0), 2)
                        cv2.putText(dbg, f"STATE: BLUE_APPROACH area={barea:.0f}", (10, 25),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                else:
                    vx, vy = 0.0, 0.0

            elif state == State.BLUE_WAIT:
                # ✅ FIX: plus de blue_wait_t0 -> on utilise t_state
                vx, vy, vz, yaw_rate = 0.0, 0.0, alt_hold_vz(), 0.0
                if (now - t_state) >= BLUE_WAIT_S:
                    if last_att["have"]:
                        yaw_target = wrap_pi(float(last_att["yaw"]) + math.pi)
                    set_state(State.TURN_AROUND)

            elif state == State.TURN_AROUND:
                vx, vy, vz = 0.0, 0.0, alt_hold_vz()
                yaw_rate = compute_yaw_rate()

                if last_att["have"] and yaw_target is not None:
                    ey = wrap_pi(yaw_target - float(last_att["yaw"]))

                    # ✅ Hold 4s once the 180° yaw is achieved (must stay within YAW_TOL continuously)
                    if abs(ey) <= YAW_TOL and (now - t_state) > 0.8:
                        if turn_hold_t0 is None:
                            turn_hold_t0 = now
                        if (now - turn_hold_t0) >= TURN_HOLD_AFTER_180_S:
                            swapped = True
                            set_state(State.LINE_BACK)
                    else:
                        turn_hold_t0 = None

            elif state == State.LINE_BACK:
                yaw_rate = 0.0
                vz = alt_hold_vz()
                if last_frame["have"]:
                    frame = last_frame["img"]
                    vx_line, vy_line, yaw_rate, dbg, info = line_follow_cmd_back(frame, swapped=True)
                    vx, vy = vx_line, vy_line
                else:
                    vx, vy = 0.0, 0.0

                # Trigger landing when ArUco tag is close enough
                if tag_ok(now):
                    dn = float(last_tag.get("rel_body_x", last_tag["rel_n"]))
                    de = float(last_tag.get("rel_body_y", last_tag["rel_e"]))
                    dist = math.hypot(dn, de)
                    if dist <= LAND_TAG_TRIGGER_DIST:
                        set_state(State.LAND_STABILIZE)

            elif state == State.LAND_STABILIZE:
                yaw_rate = 0.0

                # Hold altitude (around the altitude where we entered landing)
                if land_hover_alt is None:
                    if last_alt["have"]:
                        land_hover_alt = TARGET_ALT_M
                    else:
                        land_hover_alt = TARGET_ALT_M

                if last_alt["have"]:
                    e_alt = float(last_alt["rel"]) - float(land_hover_alt)
                    vz = clamp(LAND_HOVER_ALT_KP * e_alt, -LAND_HOVER_ALT_MAX_VZ, +LAND_HOVER_ALT_MAX_VZ)
                else:
                    vz = 0.0

                centered = False
                if tag_ok(now):
                    vx, vy = compute_vxy_from_tag()
                    en = float(last_tag.get("rel_body_x", last_tag["rel_n"]))
                    ee = float(last_tag.get("rel_body_y", last_tag["rel_e"]))
                    centered = (abs(en) <= LAND_CENTER_TOL_M) and (abs(ee) <= LAND_CENTER_TOL_M)
                else:
                    vx, vy = 0.0, 0.0
                    centered = False

                # Start the 0.5s timer only when we are well centered; reset if we drift or lose tag
                if centered:
                    if land_centered_t0 is None:
                        land_centered_t0 = now
                    if (now - land_centered_t0) >= LAND_STABILIZE_S:
                        set_state(State.LAND_CENTER_DESCEND)
                else:
                    land_centered_t0 = None

            elif state == State.LAND_CENTER_DESCEND:
                yaw_rate = 0.0

                if tag_ok(now):
                    vx, vy = compute_vxy_from_tag()
                else:
                    vx, vy = 0.0, 0.0

                if last_alt["have"]:
                    alt = float(last_alt["rel"])
                    if alt > LAND_FINAL_ALT:
                        vz = LAND_DESCEND_VZ
                        if alt <= LAND_HOLD_CENTER_ALT:
                            vz = clamp(LAND_DESCEND_VZ * 0.8, 0.15, 0.40)
                    else:
                        vz = 0.0
                        set_state(State.LAND_FINAL)
                else:
                    vz = 0.2

            elif state == State.LAND_FINAL:
                publish_mode("LAND")
                publish_vel(0, 0, 0, 0)
                if (now - t_state) > 4.0:
                    set_state(State.DISARM)

            elif state == State.DISARM:
                publish_mode("DISARM")
                publish_vel(0, 0, 0, 0)
                if (now - t_state) > 1.5:
                    set_state(State.DONE)

            elif state == State.DONE:
                publish_vel(0, 0, 0, 0)
                print("[MISSION] DONE")
                break

            # Publish cmd_vel for OFFBOARD control (even if some states publish_mode too)
            publish_vel(vx, vy, vz, yaw_rate)

            # Debug window
            if SHOW_DEBUG_CV and last_frame["have"]:
                if dbg is None:
                    dbg = last_frame["img"].copy()

                alt_txt = f"ALT={last_alt['rel']:.2f}m" if last_alt["have"] else "ALT=?"
                tag_txt = "TAG=Y" if tag_ok(now) else "TAG=N"
                blue_txt = f"BLUE_ARM={int(blue_armed)}"
                if alt_reached_t0 is not None and not blue_armed:
                    blue_txt += f" (in {max(0.0, BLUE_ARM_DELAY_AFTER_ALT_S-(now-alt_reached_t0)):.1f}s)"
                cv2.putText(dbg, f"{alt_txt}  {tag_txt}  {blue_txt}",
                            (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                # Landing stabilization feedback
                land_txt = ""
                if state == State.LAND_STABILIZE:
                    if land_centered_t0 is None:
                        land_txt = f"LAND_STAB: centering (tol={LAND_CENTER_TOL_M:.2f}m) {LAND_STABILIZE_S:.0f}s"
                    else:
                        land_txt = f"LAND_STAB: {min(LAND_STABILIZE_S, now-land_centered_t0):.1f}/{LAND_STABILIZE_S:.0f}s"
                elif state in (State.LAND_CENTER_DESCEND, State.LAND_FINAL):
                    land_txt = f"LANDING: {state.name}"
                if land_txt:
                    cv2.putText(dbg, land_txt, (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

                cv2.imshow("mission_debug", dbg)
                if (cv2.waitKey(1) & 0xFF) == 27:
                    print("[MISSION] ESC -> stop")
                    break

            time.sleep(LOOP_DT)

    finally:
        try:
            publish_vel(0, 0, 0, 0)
            publish_mode("LAND")
        except Exception:
            pass
        client.loop_stop()
        client.disconnect()
        cv2.destroyAllWindows()



def alt_hold_vz(target_alt_m: float = TARGET_ALT_M) -> float:
    """Return vz (body NED, +down) to hold altitude around target_alt_m using rel_alt_m."""
    if not last_alt.get("have", False):
        return 0.0
    cur = float(last_alt.get("rel", 0.0))  # meters, positive up
    err = target_alt_m - cur               # >0 => below target => need CLIMB (vz negative)
    if abs(err) <= ALT_HOLD_DEADBAND_M:
        return 0.0
    vz = -ALT_HOLD_KP * err
    return clamp(vz, -ALT_HOLD_MAX_VZ, +ALT_HOLD_MAX_VZ)

if __name__ == "__main__":
    main()
