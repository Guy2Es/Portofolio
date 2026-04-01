#!/usr/bin/env python3
"""
Lance tous les scripts nécessaires (dans des sous-processus) depuis le dossier courant.
Usage:
  python3 run_mission_all.py
"""

import subprocess
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent  # ✅ dossier où est run_mission_all.py

PROCS = [
    [sys.executable, str(BASE / "px4_bridge.py")],
    [sys.executable, str(BASE / "camera_publisher.py")],
    [sys.executable, str(BASE / "aruco_world_pose.py")],
    [sys.executable, str(BASE / "mission_full_aruco_line_return_land_patched.py")],
]

def main():
    ps = []
    try:
        for i, cmd in enumerate(PROCS):
            print("[RUN] start:", " ".join(cmd))
            ps.append(subprocess.Popen(cmd))
            time.sleep(0.4)

        print("[RUN] Tous les process sont lancés. Ctrl+C pour arrêter.")
        while True:
            rc = ps[-1].poll()  # mission = dernier
            if rc is not None:
                print("[RUN] Mission terminée avec code", rc)
                break
            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\n[RUN] Arrêt demandé (Ctrl+C).")
    finally:
        for p in ps[::-1]:
            if p.poll() is None:
                p.terminate()
        time.sleep(0.8)
        for p in ps[::-1]:
            if p.poll() is None:
                p.kill()
        print("[RUN] Stoppé.")

if __name__ == "__main__":
    main()
